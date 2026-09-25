"""PaLoRA dual-branch (R / U) adapters for frozen pretrained encoders.

Differs from pareto_ssl/networks.py::LoRADualLayer in one respect: that class
creates and *trains* its own nn.Linear (used when STEER trains an encoder from
scratch, as on MOSEI). Here the base Conv3d/Linear layers come from a frozen,
already-pretrained encoder (Champollion or ALMA) and must never be touched —
only the low-rank A/B branches are trainable. Both classes implement the same
`forward_mix(h, w_r, w_u)` weight-space conditioning (independent,
non-summing coefficients):

    theta(w_r, w_u) = W0 + (alpha/rank) * (w_r * A_r @ B_r + w_u * A_u @ B_u)

`forward(x)` (no lambda) returns the frozen base's own forward, i.e. dW=0 —
used for any layer *outside* the adapted scope so the model composes cleanly
regardless of which of the three regimes (bottleneck / last_stage / full) is
active for a given encoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRADualLinear(nn.Module):
    """Wraps a frozen nn.Linear with two low-rank branches (R and U)."""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        out_dim, in_dim = base.weight.shape
        self.A_r = nn.Parameter(torch.zeros(out_dim, rank))
        self.B_r = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.A_u = nn.Parameter(torch.zeros(out_dim, rank))
        self.B_u = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.rank = rank
        self.scale = (alpha / rank) if alpha is not None else 1.0

    def forward_mix(self, h: torch.Tensor, w_r: float, w_u: float) -> torch.Tensor:
        dW = self.scale * (w_r * (self.A_r @ self.B_r) + w_u * (self.A_u @ self.B_u))
        return F.linear(h, self.base.weight + dW, self.base.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.base(h)

    @torch.no_grad()
    def branch_norms(self):
        """(rho_R, rho_U) = ||scale * A_. @ B_.||_F / ||W0||_F -- the same
        adapter-magnitude diagnostic ("ΔW/W") used in the MOSEI STEER work to
        confirm both branches are actually active (not starved) during training."""
        dR = self.scale * (self.A_r @ self.B_r)
        dU = self.scale * (self.A_u @ self.B_u)
        w0_norm = self.base.weight.norm().clamp_min(1e-12)
        return (dR.norm() / w0_norm).item(), (dU.norm() / w0_norm).item()


class LoRADualConv3d(nn.Module):
    """Wraps a frozen Conv3d (or Conv3dSame) with two rank-`rank` bottleneck branches.

    Each branch factorises the additive weight update as two convs rather than
    a literal low-rank tensor decomposition of the 5D conv kernel (the standard
    practical Conv-LoRA construction):
        B_*: same kernel/stride/padding as the base layer, in_ch -> rank, no bias
        A_*: 1x1x1 conv, rank -> out_ch, no bias
    so B_*'s output already has the base layer's exact output spatial shape and
    A_* only remixes channels — the sum is spatially well-defined by construction.
    """

    def __init__(self, base_conv: nn.Conv3d, rank: int = 4, alpha: float = None):
        super().__init__()
        self.base = base_conv
        for p in self.base.parameters():
            p.requires_grad_(False)

        conv_cls = type(base_conv)  # nn.Conv3d or backbones_lite.Conv3dSame
        in_ch, out_ch = base_conv.in_channels, base_conv.out_channels
        kwargs = dict(kernel_size=base_conv.kernel_size, stride=base_conv.stride,
                      padding=base_conv.padding, bias=False)

        def make_branch():
            B = conv_cls(in_ch, rank, **kwargs)
            A = nn.Conv3d(rank, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
            nn.init.zeros_(A.weight)
            nn.init.normal_(B.weight, std=0.01)
            return B, A

        self.B_r, self.A_r = make_branch()
        self.B_u, self.A_u = make_branch()
        self.rank = rank
        self.scale = (alpha / rank) if alpha is not None else 1.0

    def forward_mix(self, x: torch.Tensor, w_r: float, w_u: float) -> torch.Tensor:
        out = self.base(x)
        dR = self.A_r(self.B_r(x))
        dU = self.A_u(self.B_u(x))
        return out + self.scale * (w_r * dR + w_u * dU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    @torch.no_grad()
    def branch_norms(self):
        """(rho_R, rho_U), same definition as LoRADualLinear.branch_norms: composes
        the two-conv branch (1x1x1 A after kxkxk B) into a single effective
        kxkxk weight tensor, comparable in shape to the base conv's own weight."""
        def eff_weight(A, B):
            a = A.weight.view(A.weight.shape[0], A.weight.shape[1])  # (out_ch, rank)
            b = B.weight  # (rank, in_ch, k, k, k)
            return torch.einsum("or,ri...->oi...", a, b)

        dR = self.scale * eff_weight(self.A_r, self.B_r)
        dU = self.scale * eff_weight(self.A_u, self.B_u)
        w0_norm = self.base.weight.norm().clamp_min(1e-12)
        return (dR.norm() / w0_norm).item(), (dU.norm() / w0_norm).item()


def lora_parameters(module: nn.Module):
    """Yields only the trainable A/B branch parameters of every LoRA*-wrapped
    submodule (never the frozen `base`), for building the adapter optimizer."""
    for m in module.modules():
        if isinstance(m, (LoRADualLinear, LoRADualConv3d)):
            yield m.A_r
            yield m.B_r
            yield m.A_u
            yield m.B_u
