"""Frozen pretrained encoder reconstruction for Champollion (sulci) and ALMA (diffusion).

Both checkpoints are PyTorch Lightning `state_dict`s wrapping a plain ConvNet
+ ProjectionHead (see backbones_lite.py). We only ever need the frozen
encoder's forward pass, so we load straight into the plain nn.Module classes
and never touch Lightning.

Region: S.C.-sylv., UKB, ALMA modality WholeBrain. Side (left/right) is
selected via STEER_NEURO_SIDE (default "right", so the original right-side
pass keeps working unchanged) -- the two sides are NOT mirror images of the
same architecture, each has its own independently-fit crop bounding box and
(for ALMA) its own hyperparameter-search result:
  - Champollion: .hydra/config.yaml under
    .../Champollion_V1_after_ablation_latent_256/SC-sylv_{right,left}/.../
    (the 256-d latent variant, chosen over the 32-d one to avoid a hard,
    frozen 32-d ceiling well below ALMA's embedding width -- same conv-trunk
    architecture and 256-d bottleneck both sides, only in_shape differs:
    confirmed via each side's own crop array shape + hydra input_size).
  - ALMA: cross-checked directly against each checkpoint's own embedded
    `hyper_parameters` dict. Left is NOT just right's in_shape swapped in --
    block_depth (3 vs 2), initial_kernel_size (7 vs 9), and
    num_representation_features (512 vs 128) all differ too, i.e. ALMA is the
    WIDER encoder on the left side, the opposite of the right-side asymmetry
    (confirmed 2026-09-08 against the left checkpoint's hyper_parameters).
"""
import os

import torch

from backbones_lite import ConvNet, ProjectionHead

SIDE = os.environ.get("STEER_NEURO_SIDE", "right")
assert SIDE in ("left", "right"), f"STEER_NEURO_SIDE must be 'left' or 'right', got {SIDE!r}"

# Checkpoint locations are site-specific and are supplied through the
# environment; there is no portable default.
#   STEER_NEURO_CHAMPO_CKPT  Champollion encoder weights for this hemisphere
#   STEER_NEURO_ALMA_CKPT    ALMA encoder weights for this hemisphere
_CHAMPO_CKPT_DEFAULT = None
_ALMA_CKPT_DEFAULT = None

# Overridable via env var so the exact same code runs unchanged on Jean Zay
# (or anywhere else) once the checkpoints are staged there -- neurospin paths
# are NOT reachable from Jean Zay compute nodes, this is not just convenience.
CHAMPO_CKPT = os.environ.get("STEER_NEURO_CHAMPO_CKPT", _CHAMPO_CKPT_DEFAULT)
ALMA_CKPT = os.environ.get("STEER_NEURO_ALMA_CKPT", _ALMA_CKPT_DEFAULT)
for _v, _p in (("STEER_NEURO_CHAMPO_CKPT", CHAMPO_CKPT), ("STEER_NEURO_ALMA_CKPT", ALMA_CKPT)):
    if not _p:
        raise RuntimeError(f"set ${_v} to the encoder checkpoint for side={SIDE!r}")

_CHAMPO_IN_SHAPE = {"right": (1, 42, 34, 49), "left": (1, 38, 36, 49)}[SIDE]
CHAMPO_CFG = dict(
    in_channels=1, in_shape=_CHAMPO_IN_SHAPE, encoder_depth=3, filters=[32, 64, 128],
    block_depth=4, initial_kernel_size=7, initial_stride=1,
    num_representation_features=256, linear=True, drop_rate=0.05,
    adaptive_pooling=None, max_pool=False,
)
CHAMPO_PROJ_LAYERS = [256, 256, 256, 256]

_ALMA_CFG_BY_SIDE = {
    "right": dict(
        in_channels=1, in_shape=(1, 88, 72, 104), encoder_depth=2, filters=[32, 64],
        block_depth=2, initial_kernel_size=9, initial_stride=2,
        num_representation_features=128, linear=True, drop_rate=0.2254310902803914,
        adaptive_pooling=None, max_pool=False,
    ),
    "left": dict(
        in_channels=1, in_shape=(1, 80, 72, 104), encoder_depth=2, filters=[32, 64],
        block_depth=3, initial_kernel_size=7, initial_stride=2,
        num_representation_features=512, linear=True, drop_rate=0.2776570348972516,
        adaptive_pooling=None, max_pool=False,
    ),
}
ALMA_CFG = _ALMA_CFG_BY_SIDE[SIDE]
ALMA_PROJ_LAYERS = {"right": [128, 256, 256, 256], "left": [512, 1024, 1024, 1024]}[SIDE]


def _freeze(module: torch.nn.Module) -> torch.nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    return module


def build_champollion_encoder(ckpt_path: str = CHAMPO_CKPT, freeze: bool = True):
    """Returns (encoder, embed_dim). Discards the pretrained projection head."""
    enc = ConvNet(**CHAMPO_CFG)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    enc_sd = {k[len("backbones.0."):]: v
              for k, v in sd.items() if k.startswith("backbones.0.encoder.")}
    missing, unexpected = enc.load_state_dict(enc_sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    if freeze:
        _freeze(enc)
    return enc, CHAMPO_CFG["num_representation_features"]


def build_alma_encoder(ckpt_path: str = ALMA_CKPT, freeze: bool = True):
    """Returns (encoder, embed_dim). Discards the pretrained projection head."""
    enc = ConvNet(**ALMA_CFG)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    # cross-check against the checkpoint's own embedded hyperparameters when present
    hp = ckpt.get("hyper_parameters")
    if hp:
        for k in ("in_shape", "encoder_depth", "block_depth", "filters",
                   "initial_kernel_size", "initial_stride", "num_representation_features"):
            expected = ALMA_CFG.get(k) if k != "in_shape" else tuple(ALMA_CFG["in_shape"])
            got = hp.get(k)
            got = tuple(got) if k == "in_shape" and got is not None else got
            assert got == expected, f"ALMA_CFG[{k}]={expected!r} != checkpoint hparams {got!r}"
    enc_sd = {k[len("encoder."):]: v
              for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = enc.load_state_dict(enc_sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    if freeze:
        _freeze(enc)
    return enc, ALMA_CFG["num_representation_features"]


if __name__ == "__main__":
    enc_c, dim_c = build_champollion_encoder()
    enc_a, dim_a = build_alma_encoder()
    with torch.no_grad():
        hc = enc_c(torch.zeros(2, *CHAMPO_CFG["in_shape"]))
        ha = enc_a(torch.zeros(2, *ALMA_CFG["in_shape"]))
    print(f"Champollion: embed_dim={dim_c}, forward shape={tuple(hc.shape)}")
    print(f"ALMA:        embed_dim={dim_a}, forward shape={tuple(ha.shape)}")
    print("Both frozen encoders load and run correctly.")
