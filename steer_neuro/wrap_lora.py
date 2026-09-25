"""Wraps a frozen backbones_lite.ConvNet with PaLoRA dual-branch (R/U) adapters
at one of three scopes, and gives it a `forward_mix(x, w_r, w_u)` preference-
conditioned forward pass. nn.Sequential.forward can't be reused directly once
some children need `forward_mix` instead of `forward`, so LoRAConditionedEncoder
walks the encoder's children manually.

Scopes (see STEER-neuro plan):
  bottleneck  — only the final Linear (flat conv features -> embedding)
  last_stage  — + every conv{step}* layer in the encoder's last downsampling stage
  full        — + every conv{step}* layer in the encoder

Works unmodified on both Champollion (3 stages) and ALMA (2 stages) because it
finds "last stage" programmatically from each encoder's own layer names,
rather than hardcoding a stage count.
"""
import re

import torch.nn as nn

from backbones_lite import Conv3dSame
from networks import LoRADualConv3d, LoRADualLinear

_CONV_RE = re.compile(r"^conv(\d+)[a-e]?$")

SCOPES = ("bottleneck", "last_stage", "full")


def _conv_steps(encoder_seq: nn.Sequential):
    """{step_index: [layer_name, ...]} for every conv{step}{letter} child."""
    steps = {}
    for name in encoder_seq._modules:
        m = _CONV_RE.match(name)
        if m:
            steps.setdefault(int(m.group(1)), []).append(name)
    return steps


def _layer_rank(out_width: int, rank: int, rank_ratio: float) -> int:
    """rank_ratio (fraction of this layer's own output width) takes precedence
    over the fixed `rank` when given -- this is what makes "50%" mean the same
    thing on a 256-d bottleneck Linear and a 32-channel conv layer, rather than
    an absolute rank that's a wildly different fraction on each."""
    if rank_ratio is not None:
        return max(1, round(rank_ratio * out_width))
    return rank


def wrap_lora_(encoder_seq: nn.Sequential, scope: str, rank: int = 4,
               rank_ratio: float = None, alpha: float = None):
    """Mutates `encoder_seq` in place, replacing selected children with LoRA wrappers.
    Returns {layer_name: rank_used} (for logging)."""
    assert scope in SCOPES, f"scope must be one of {SCOPES}, got {scope!r}"
    wrapped = {}

    if scope in ("last_stage", "full"):
        steps = _conv_steps(encoder_seq)
        if scope == "last_stage":
            target_names = steps[max(steps)] if steps else []
        else:
            target_names = [n for names in steps.values() for n in names]
        for name in target_names:
            child = encoder_seq._modules[name]
            assert isinstance(child, (nn.Conv3d, Conv3dSame)), (name, type(child))
            r = _layer_rank(child.out_channels, rank, rank_ratio)
            setattr(encoder_seq, name, LoRADualConv3d(child, rank=r, alpha=alpha))
            wrapped[name] = r

    if "Linear" in encoder_seq._modules:
        child = encoder_seq._modules["Linear"]
        r = _layer_rank(child.out_features, rank, rank_ratio)
        setattr(encoder_seq, "Linear", LoRADualLinear(child, rank=r, alpha=alpha))
        wrapped["Linear"] = r

    return wrapped


class LoRAConditionedEncoder(nn.Module):
    """Frozen ConvNet + PaLoRA adapters at `scope`, with a preference-conditioned
    forward pass `forward_mix(x, w_r, w_u)`. `forward(x)` (no preference) runs the
    unconditioned encoder (dW=0 everywhere), i.e. exactly the pretrained model."""

    def __init__(self, convnet: nn.Module, scope: str, rank: int = 4,
                 rank_ratio: float = None, alpha: float = None):
        super().__init__()
        self.convnet = convnet
        self.scope = scope
        self.wrapped_layers = wrap_lora_(convnet.encoder, scope=scope, rank=rank,
                                          rank_ratio=rank_ratio, alpha=alpha)

    def forward_mix(self, x, w_r: float, w_u: float):
        h = x
        for child in self.convnet.encoder._modules.values():
            if isinstance(child, (LoRADualLinear, LoRADualConv3d)):
                h = child.forward_mix(h, w_r, w_u)
            else:
                h = child(h)
        return h

    def forward(self, x):
        return self.convnet(x)

    def branch_norms(self):
        """{layer_name: (rho_R, rho_U)} for every wrapped layer."""
        return {
            name: child.branch_norms()
            for name, child in self.convnet.encoder._modules.items()
            if isinstance(child, (LoRADualLinear, LoRADualConv3d))
        }
