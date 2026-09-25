"""Small architecture pieces for the 5 multimodal-SSL baselines (GMC, CoMM,
CLIP, Cross+Self, FactorCL), ported from pareto_ssl/networks.py and
pareto_ssl/multibench/benchmark_multibench.py's FusionEncoder -- kept
self-contained here rather than cross-imported, same reasoning as
losses.py's own vendoring: don't let this drift silently with that project.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleProjectionHead(nn.Module):
    """L2-normalising projection head -- Linear(in,in) -> ReLU -> Linear(in,out).

    normalize=False drops the L2 step, leaving `net` byte-for-byte the
    official FactorCL mlp_head(dim_in, feat_dim) (that repo normalises
    nowhere). Not the same class as backbones_lite.ProjectionHead (STEER's
    own R/U heads use a different, more general layers_shapes interface) --
    kept separate and minimal here so it's directly auditable against the
    pareto_ssl reference architecture it's reproducing.
    """

    def __init__(self, in_dim: int, out_dim: int, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.net(z)
        return F.normalize(h, dim=-1) if self.normalize else h


class FusionEncoder(nn.Module):
    """Joint encoder for GMC/CoMM: concat(z_v, z_t) -> head -> L2-norm.

    hidden=None (default) is a single Linear(2*in_dim, out_dim) -- the
    original GMC/CoMM architecture, minimal head capacity. Passing `hidden`
    gives it an MLP head (Linear->ReLU->Linear) mirroring
    SimpleProjectionHead's shape, so head capacity can be a controlled
    variable rather than a confound when comparing against methods that DO
    have a 2-layer head (CLIP/Cross+Self/FactorCL all use SimpleProjectionHead).
    Not used by default here -- STEER's own readout has no analogous "extra
    head capacity" either, so the matched-budget comparison keeps hidden=None
    unless deliberately testing the capacity-controlled variant.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = None):
        super().__init__()
        self.hidden = hidden
        if hidden:
            self.net = nn.Sequential(
                nn.Linear(2 * in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        else:
            self.fc = nn.Linear(2 * in_dim, out_dim)

    def forward(self, z_v: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_v, z_t], dim=-1)
        return F.normalize(self.net(x) if self.hidden else self.fc(x), dim=-1)
