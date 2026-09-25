"""Encoder architectures for Pareto-SSL benchmark.

Backbone: AlexNet (same as CoMM trifeatures notebook), expects 224×224 RGB input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import AlexNet


class AlexNetEncoder(AlexNet):
    """
    AlexNet backbone → L2-normalised latent_dim vector.

    Identical to CoMM's AlexNetEncoder (models/alexnet.py) with global_pool='avg'.
    Requires 224×224 input (AlexNet conv features → 256×6×6 → 9216-d → latent_dim).
    """
    def __init__(self, latent_dim: int = 512, dropout: float = 0.5):
        super().__init__(dropout=dropout)
        self.classifier = nn.Linear(256 * 6 * 6, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (B, 256, 6, 6) for 224×224 input
        x = self.avgpool(x)           # adaptive avg → (B, 256, 6, 6)
        x = torch.flatten(x, 1)       # (B, 9216)
        return F.normalize(self.classifier(x), dim=-1)


class FusionHead(nn.Module):
    """
    Lightweight MLP that fuses two L2-normalised latent vectors into one.
    Used as joint encoder for GMC and CoMM.

    Matches the spirit of CoMM's MMFusion but without the full transformer:
    concatenate → MLP → L2-normalise.
    """
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(torch.cat([z1, z2], dim=-1)), dim=-1)


class ProjectionHead(nn.Module):
    """L2-normalising projection head — for SimCLR and cross-modal InfoNCE.

    normalize=False drops the L2 step, leaving `net` byte-for-byte the official
    FactorCL `mlp_head(dim_in, feat_dim)` = Linear -> ReLU -> Linear. That repo
    normalises nowhere, so --factorcl_official needs it. `normalize` holds no
    parameters, so the state_dict is identical either way and a wrong setting
    loads SILENTLY: the probe must read it from train_meta, never from a CLI flag.
    """
    def __init__(self, in_dim: int = 512, proj_dim: int = 256, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.net(z)
        return F.normalize(h, dim=-1) if self.normalize else h


class SpatialFiLM(nn.Module):
    """FiLM conditioning for spatial conv feature maps (B, C, H, W).

    Initialised to identity (gamma=1, beta=0 for any lambda) so training
    starts from the same baseline as a plain AlexNet.
    """
    def __init__(self, num_channels: int, film_hidden: int = 64):
        super().__init__()
        self.gamma = nn.Sequential(
            nn.Linear(1, film_hidden), nn.ReLU(),
            nn.Linear(film_hidden, num_channels),
        )
        self.beta = nn.Sequential(
            nn.Linear(1, film_hidden), nn.ReLU(),
            nn.Linear(film_hidden, num_channels),
        )
        # Identity init: gamma output = 1, beta output = 0 for any input
        nn.init.zeros_(self.gamma[-1].weight)
        nn.init.ones_(self.gamma[-1].bias)
        nn.init.zeros_(self.beta[-1].weight)
        nn.init.zeros_(self.beta[-1].bias)

    def forward(self, x: torch.Tensor, lam_t: torch.Tensor) -> torch.Tensor:
        # lam_t: (1,1) — broadcasts over (B, C, H, W)
        g = self.gamma(lam_t).view(1, -1, 1, 1)
        b = self.beta(lam_t).view(1, -1, 1, 1)
        return g * x + b


class HyperNetEncoder(nn.Module):
    """
    AlexNet encoder with FiLM conditioning injected after every ReLU in the
    convolutional stack.  The encoder output changes with lambda, so information
    is extracted at the feature level rather than only at the projection head.

    AlexNet features layout (indices into self.features):
      0  Conv1   1  ReLU ← FiLM(64)    2  MaxPool
      3  Conv2   4  ReLU ← FiLM(192)   5  MaxPool
      6  Conv3   7  ReLU ← FiLM(384)
      8  Conv4   9  ReLU ← FiLM(256)
      10 Conv5  11  ReLU ← FiLM(256)  12  MaxPool
    """
    _FILM_AFTER = {1: 0, 4: 1, 7: 2, 9: 3, 11: 4}   # feature index → film index
    _CHANNELS   = [64, 192, 384, 256, 256]

    def __init__(self, latent_dim: int = 512, film_hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        base = AlexNet(dropout=dropout)
        self.features   = base.features
        self.avgpool    = base.avgpool
        self.classifier = nn.Linear(256 * 6 * 6, latent_dim)
        self.film = nn.ModuleList(
            [SpatialFiLM(c, film_hidden) for c in self._CHANNELS]
        )
        # Disable inplace ReLU — inplace ops on tensors that FiLM reads afterwards
        # can cause autograd version-counter errors during backward.
        for mod in self.features.modules():
            if isinstance(mod, nn.ReLU):
                mod.inplace = False

    def forward(self, x: torch.Tensor, lam: float) -> torch.Tensor:
        lam_t = torch.tensor([[lam]], dtype=x.dtype, device=x.device)
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self._FILM_AFTER:
                x = self.film[self._FILM_AFTER[i]](x, lam_t)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return F.normalize(self.classifier(x), dim=-1)


class FiLMLayer(nn.Module):
    """Linear layer with FiLM modulation conditioned on a scalar lambda (used by mmimdb pipeline)."""
    def __init__(self, in_dim: int, out_dim: int, film_hidden: int = 64):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.gamma = nn.Sequential(
            nn.Linear(1, film_hidden), nn.ReLU(),
            nn.Linear(film_hidden, film_hidden), nn.ReLU(),
            nn.Linear(film_hidden, out_dim),
        )
        self.beta = nn.Sequential(
            nn.Linear(1, film_hidden), nn.ReLU(),
            nn.Linear(film_hidden, film_hidden), nn.ReLU(),
            nn.Linear(film_hidden, out_dim),
        )
        nn.init.zeros_(self.gamma[-1].weight); nn.init.ones_(self.gamma[-1].bias)
        nn.init.zeros_(self.beta[-1].weight);  nn.init.zeros_(self.beta[-1].bias)

    def forward(self, h: torch.Tensor, lam_t: torch.Tensor) -> torch.Tensor:
        h = self.linear(h)
        return self.gamma(lam_t) * h + self.beta(lam_t)


class FiLMProjectionHead(nn.Module):
    """λ-conditioned projection head (used by mmimdb pipeline)."""
    def __init__(self, input_dim: int, embed_dim: int = 128):
        super().__init__()
        self.layer1 = FiLMLayer(input_dim, 256)
        self.layer2 = FiLMLayer(256, 128)
        self.layer3 = FiLMLayer(128, embed_dim)

    def forward(self, h: torch.Tensor, lam: float) -> torch.Tensor:
        lam_t = torch.tensor([[lam]], dtype=h.dtype, device=h.device)
        h = F.relu(self.layer1(h, lam_t))
        h = F.relu(self.layer2(h, lam_t))
        h = self.layer3(h, lam_t)
        return F.normalize(h, dim=-1)


class DualFiLMLayer(nn.Module):
    """
    Shared linear layer with two structurally separated FiLM branches (R and U).

    Branch R is updated only by L_R (cross-modal, shared objective).
    Branch U is updated only by L_U (within-modal, unique objective).
    The shared linear weight W is updated by both.

    At inference, forward(h, lam) blends the two modulations smoothly:
      γ(λ) = λ·γ_R + (1-λ)·γ_U,  β(λ) = λ·β_R + (1-λ)·β_U
    Because the blend happens before the ReLU in each layer, z(λ) is a
    non-linear function of λ — genuinely different from endpoint blending.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear  = nn.Linear(in_dim, out_dim)
        self.gamma_r = nn.Parameter(torch.ones(out_dim))
        self.beta_r  = nn.Parameter(torch.zeros(out_dim))
        self.gamma_u = nn.Parameter(torch.ones(out_dim))
        self.beta_u  = nn.Parameter(torch.zeros(out_dim))

    def forward_r(self, h: torch.Tensor) -> torch.Tensor:
        return self.gamma_r * self.linear(h) + self.beta_r

    def forward_u(self, h: torch.Tensor) -> torch.Tensor:
        return self.gamma_u * self.linear(h) + self.beta_u

    def forward(self, h: torch.Tensor, lam: float) -> torch.Tensor:
        g = lam * self.gamma_r + (1.0 - lam) * self.gamma_u
        b = lam * self.beta_r  + (1.0 - lam) * self.beta_u
        return g * self.linear(h) + b


class DualFiLMProjectionHead(nn.Module):
    """
    3-layer projection head with structurally separated R/U FiLM branches.

    forward_r(h) : used during training by L_R — updates gamma_r, beta_r, shared W
    forward_u(h) : used during training by L_U — updates gamma_u, beta_u, shared W
    forward(h, lam): inference — blends R and U modulations per layer before ReLU

    No manual gradient blocking needed: the separate forward methods naturally
    route gradients only to the appropriate FiLM parameters.
    """
    def __init__(self, input_dim: int, embed_dim: int = 128):
        super().__init__()
        self.layer1 = DualFiLMLayer(input_dim, 256)
        self.layer2 = DualFiLMLayer(256, 128)
        self.layer3 = DualFiLMLayer(128, embed_dim)

    def forward_r(self, h: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.layer1.forward_r(h))
        h = F.relu(self.layer2.forward_r(h))
        return F.normalize(self.layer3.forward_r(h), dim=-1)

    def forward_u(self, h: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.layer1.forward_u(h))
        h = F.relu(self.layer2.forward_u(h))
        return F.normalize(self.layer3.forward_u(h), dim=-1)

    def forward(self, h: torch.Tensor, lam: float) -> torch.Tensor:
        h = F.relu(self.layer1(h, lam))
        h = F.relu(self.layer2(h, lam))
        return F.normalize(self.layer3(h, lam), dim=-1)


class DualFiLMProjectionHeadPreNorm(nn.Module):
    """
    Variant of DualFiLMProjectionHead where FiLM is applied ONCE to the final
    pre-norm embedding instead of inside each layer.

    Motivation: in DualFiLMProjectionHead the shared W dominates all layers,
    and L2 normalization at the end washes out the per-layer FiLM modulations
    (scalar gamma on a vector whose direction is already set by W).
    Here the shared backbone produces a rich embedding, and the FiLM parameters
    (gamma_r/u, beta_r/u) modulate it per-dimension BEFORE normalization —
    directly rotating the direction on the unit sphere.

    forward_r(h): backbone(h) → gamma_r * z + beta_r → L2_norm
    forward_u(h): backbone(h) → gamma_u * z + beta_u → L2_norm
    forward(h,λ): backbone(h) → blend(gamma, beta, λ)  → L2_norm
    """
    def __init__(self, input_dim: int, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, embed_dim),
        )
        # Per-dimension FiLM for R and U — applied to pre-norm output; identity init
        self.gamma_r = nn.Parameter(torch.ones(embed_dim))
        self.beta_r  = nn.Parameter(torch.zeros(embed_dim))
        self.gamma_u = nn.Parameter(torch.ones(embed_dim))
        self.beta_u  = nn.Parameter(torch.zeros(embed_dim))

    def forward_r(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.gamma_r * self.net(h) + self.beta_r, dim=-1)

    def forward_u(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.gamma_u * self.net(h) + self.beta_u, dim=-1)

    def forward(self, h: torch.Tensor, lam: float) -> torch.Tensor:
        z = self.net(h)
        g = lam * self.gamma_r + (1.0 - lam) * self.gamma_u
        b = lam * self.beta_r  + (1.0 - lam) * self.beta_u
        return F.normalize(g * z + b, dim=-1)


class LoRADualLayer(nn.Module):
    """
    Linear layer with two low-rank adaptation branches (R and U) — 2-branch linear ablation.

    Training (endpoint branches, never mixed):
      forward_r(h): (W + A_R @ B_R) h  — A_R/B_R touched only by L_R
      forward_u(h): (W + A_U @ B_U) h  — A_U/B_U touched only by L_U

    Inference — linear blend of effective weight matrices (no cross-terms):
      W(λ) = W + λΔW_R + (1-λ)ΔW_U,  exact at λ=0 and λ=1.

    Init: A=0, B~N(0,0.01) so ΔW≈0 at start.
    """
    def __init__(self, in_dim: int, out_dim: int, rank: int = 4, alpha: float = None):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.A_r = nn.Parameter(torch.zeros(out_dim, rank))
        self.B_r = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.A_u = nn.Parameter(torch.zeros(out_dim, rank))
        self.B_u = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        # PaLoRA LoRA scaling α/r (Fix 8). alpha=None → scale 1.0 (base runs unchanged).
        self.scale = (alpha / rank) if alpha is not None else 1.0

    def forward_r(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.linear.weight + self.scale * (self.A_r @ self.B_r), self.linear.bias)

    def forward_u(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.linear.weight + self.scale * (self.A_u @ self.B_u), self.linear.bias)

    def forward(self, h: torch.Tensor, lam: float) -> torch.Tensor:
        # θ(λ) = W + (α/r) · [λ · A_R B_R + (1-λ) · A_U B_U]
        # The α/r scale multiplies the FULL weighted sum (not the loss).
        dW = self.scale * (lam * (self.A_r @ self.B_r) + (1.0 - lam) * (self.A_u @ self.B_u))
        return F.linear(h, self.linear.weight + dW, self.linear.bias)

    def forward_mix(self, h: torch.Tensor, w_r: float, w_u: float) -> torch.Tensor:
        """Simplex mixing — independent coefficients (they need NOT sum to 1).

        θ(w) = W + (α/r) · [w_R · A_R B_R + w_U · A_U B_U]

        Used by LoRA-Simplex, where the preference lives on the 2-simplex
        (λ_R, λ_U1, λ_U2) and each modality's head sees (λ_R, λ_U{m}) — so its two
        coefficients sum to 1 - λ_U{other}, not to 1. forward(h, lam) is the
        special case w_r=lam, w_u=1-lam.
        """
        dW = self.scale * (w_r * (self.A_r @ self.B_r) + w_u * (self.A_u @ self.B_u))
        return F.linear(h, self.linear.weight + dW, self.linear.bias)

    def orth_loss(self) -> torch.Tensor:
        """Frobenius penalty: forces R and U LoRA subspaces to be orthogonal.
        Gram matrices are (rank×rank) — O(rank²), no batch dependency."""
        gram_A = self.A_r.T @ self.A_u  # (rank, rank) column-space overlap
        gram_B = self.B_r @ self.B_u.T  # (rank, rank) row-space overlap
        return gram_A.pow(2).sum() + gram_B.pow(2).sum()


class LoRADualProjectionHead(nn.Module):
    """
    3-layer projection head with low-rank dual branches for approach 4.

    forward_r(h) : trained by L_R — updates A_r, B_r (+ shared W)
    forward_u(h) : trained by L_U — updates A_u, B_u (+ shared W)
    forward(h, λ): inference — blends as W + (λA_r+(1-λ)A_u)(λB_r+(1-λ)B_u)

    The quadratic λ-dependence creates genuinely different subspaces at each λ,
    unlike FiLM (which only scales/shifts after the same W·h).
    Output is L2-normalised.
    """
    def __init__(self, input_dim: int, embed_dim: int = 128, rank: int = 4, alpha: float = None):
        super().__init__()
        self.layer1 = LoRADualLayer(input_dim, 256, rank, alpha=alpha)
        self.layer2 = LoRADualLayer(256, 128, rank, alpha=alpha)
        self.layer3 = LoRADualLayer(128, embed_dim, rank, alpha=alpha)

    def forward_mix(self, h: torch.Tensor, w_r: float, w_u: float) -> torch.Tensor:
        """LoRA-Simplex forward with independent (w_R, w_U) coefficients."""
        h = F.relu(self.layer1.forward_mix(h, w_r, w_u))
        h = F.relu(self.layer2.forward_mix(h, w_r, w_u))
        return F.normalize(self.layer3.forward_mix(h, w_r, w_u), dim=-1)

    def forward_r(self, h: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.layer1.forward_r(h))
        h = F.relu(self.layer2.forward_r(h))
        return F.normalize(self.layer3.forward_r(h), dim=-1)

    def forward_u(self, h: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.layer1.forward_u(h))
        h = F.relu(self.layer2.forward_u(h))
        return F.normalize(self.layer3.forward_u(h), dim=-1)

    def forward(self, h: torch.Tensor, lam: float) -> torch.Tensor:
        h = F.relu(self.layer1(h, lam))
        h = F.relu(self.layer2(h, lam))
        return F.normalize(self.layer3(h, lam), dim=-1)

    def orth_loss(self) -> torch.Tensor:
        return self.layer1.orth_loss() + self.layer2.orth_loss() + self.layer3.orth_loss()


class LoRATriLayer(nn.Module):
    """
    Linear layer with three low-rank branches: R (endpoint), U (endpoint), M (curvature).

    Inference weight: W(λ) = W + λΔW_R + (1-λ)ΔW_U + λ(1-λ)ΔW_M
      — exact at endpoints (λ=0 → W+ΔW_U, λ=1 → W+ΔW_R)
      — quadratic curvature from ΔW_M, which peaks at λ=0.5

    Training: call forward(h, lam) for ALL loss computations at the sampled λ.
    Natural gradient routing via the Jacobian (no manual routing needed):
      ∂L/∂ΔW_R ∝ λ,      ∂L/∂ΔW_U ∝ (1-λ),      ∂L/∂ΔW_M ∝ λ(1-λ)
    Over uniform λ: E[λ]=0.5, E[1-λ]=0.5, E[λ(1-λ)]=1/6 → M gets ~1/3 gradient.
    Compensate with a higher lora_lr_mult_m (default 3×).

    rank_m defaults to rank; set higher for more curvature capacity.
    Init: all A=0, B~N(0,0.01).
    """
    def __init__(self, in_dim: int, out_dim: int, rank: int = 4, rank_m: int = None, alpha: float = None):
        super().__init__()
        rank_m = rank_m if rank_m is not None else rank
        self.linear = nn.Linear(in_dim, out_dim)
        self.A_r = nn.Parameter(torch.zeros(out_dim, rank))
        self.B_r = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.A_u = nn.Parameter(torch.zeros(out_dim, rank))
        self.B_u = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.A_m = nn.Parameter(torch.zeros(out_dim, rank_m))
        self.B_m = nn.Parameter(torch.randn(rank_m, in_dim) * 0.01)
        # PaLoRA LoRA scaling α/r (Fix 8). alpha=None → scale 1.0 (base runs unchanged).
        self.scale = (alpha / rank) if alpha is not None else 1.0

    def forward_r(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.linear.weight + self.scale * (self.A_r @ self.B_r), self.linear.bias)

    def forward_u(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.linear.weight + self.scale * (self.A_u @ self.B_u), self.linear.bias)

    def forward(self, h: torch.Tensor, lam: float) -> torch.Tensor:
        # θ(λ) = W + (α/r) · [λ A_R B_R + (1-λ) A_U B_U + λ(1-λ) A_M B_M]
        dW = self.scale * (lam * (self.A_r @ self.B_r)
                           + (1.0 - lam) * (self.A_u @ self.B_u)
                           + lam * (1.0 - lam) * (self.A_m @ self.B_m))
        return F.linear(h, self.linear.weight + dW, self.linear.bias)

    def orth_loss(self) -> torch.Tensor:
        """Frobenius penalty: forces R and U LoRA subspaces to be orthogonal.
        Gram matrices are (rank×rank) — O(rank²), no batch dependency."""
        gram_A = self.A_r.T @ self.A_u  # (rank, rank) column-space overlap
        gram_B = self.B_r @ self.B_u.T  # (rank, rank) row-space overlap
        return gram_A.pow(2).sum() + gram_B.pow(2).sum()


class LoRATriProjectionHead(nn.Module):
    """
    3-layer projection head with three LoRA branches: R, U, M (curvature).

    Used for approach 4 (3-branch). Training calls forward(h, lam) at every
    sampled λ — the curve is genuinely trained at every point, not just endpoints.

    forward_r / forward_u are evaluation-only (λ=1 and λ=0 specialisations).
    Output is L2-normalised.
    """
    def __init__(self, input_dim: int, embed_dim: int = 128,
                 rank: int = 4, rank_m: int = None, alpha: float = None):
        super().__init__()
        self.layer1 = LoRATriLayer(input_dim, 256, rank, rank_m, alpha=alpha)
        self.layer2 = LoRATriLayer(256, 128, rank, rank_m, alpha=alpha)
        self.layer3 = LoRATriLayer(128, embed_dim, rank, rank_m, alpha=alpha)

    def forward_r(self, h: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.layer1.forward_r(h))
        h = F.relu(self.layer2.forward_r(h))
        return F.normalize(self.layer3.forward_r(h), dim=-1)

    def forward_u(self, h: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.layer1.forward_u(h))
        h = F.relu(self.layer2.forward_u(h))
        return F.normalize(self.layer3.forward_u(h), dim=-1)

    def forward(self, h: torch.Tensor, lam: float) -> torch.Tensor:
        h = F.relu(self.layer1(h, lam))
        h = F.relu(self.layer2(h, lam))
        return F.normalize(self.layer3(h, lam), dim=-1)

    def orth_loss(self) -> torch.Tensor:
        return self.layer1.orth_loss() + self.layer2.orth_loss() + self.layer3.orth_loss()


class _LoRATransformerLayer(nn.Module):
    """Pre-norm TransformerEncoderLayer with LoRA on FFN (linear1, linear2).
    Attention weights are shared (no LoRA) — only FFN adapts per λ.
    """
    def __init__(self, d_model: int, nhead: int, dim_ff: int,
                 lora_cls, rank: int, rank_m, alpha: float = None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        if lora_cls is LoRADualLayer:
            self.ff1 = LoRADualLayer(d_model, dim_ff, rank, alpha=alpha)
            self.ff2 = LoRADualLayer(dim_ff, d_model, rank, alpha=alpha)
        else:
            self.ff1 = LoRATriLayer(d_model, dim_ff, rank, rank_m, alpha=alpha)
            self.ff2 = LoRATriLayer(dim_ff, d_model, rank, rank_m, alpha=alpha)

    def _sa(self, src):
        out, _ = self.self_attn(src, src, src, need_weights=False)
        return self.dropout(out)

    def forward_mix(self, src, w_r: float, w_u: float):
        """Simplex mixing: independent (w_R, w_U) on both FFN LoRA layers."""
        h = src + self._sa(self.norm1(src))
        n = self.norm2(h)
        ff = self.ff2.forward_mix(F.relu(self.ff1.forward_mix(n, w_r, w_u)), w_r, w_u)
        return h + self.dropout(ff)

    def forward_r(self, src):
        src = src + self._sa(self.norm1(src))
        return src + self.dropout(self.ff2.forward_r(F.relu(self.ff1.forward_r(self.norm2(src)))))

    def forward_u(self, src):
        src = src + self._sa(self.norm1(src))
        return src + self.dropout(self.ff2.forward_u(F.relu(self.ff1.forward_u(self.norm2(src)))))

    def forward(self, src, lam: float):
        src = src + self._sa(self.norm1(src))
        return src + self.dropout(self.ff2(F.relu(self.ff1(self.norm2(src), lam)), lam))


class LoRATransformerEncoder(nn.Module):
    """
    CoMM Transformer backbone with LoRA on the initial projection + all 5 FFN layers.
    Mirrors LoRADualProjectionHead / LoRATriProjectionHead but for the encoder.

    forward_r(x)    → h at R endpoint (λ=1), L2-normalized, shape (B, 40)
    forward_u(x)    → h at U endpoint (λ=0), L2-normalized, shape (B, 40)
    forward(x, lam) → h at blended λ,         L2-normalized, shape (B, 40)

    Input x: (B, L, n_features) — same format as TransformerEncoder.
    Attention weights are shared across λ; only FFN and the input projection use LoRA.
    """
    _D   = 40    # ENC_DIM_XFMR — fixed by CoMM Transformer
    _DFF = 2048  # default nn.TransformerEncoderLayer dim_feedforward
    _NH  = 5     # nhead
    _NL  = 5     # num_layers

    def __init__(self, n_features: int, branches: int = 2,
                 rank: int = 4, rank_m: int = None, alpha: float = None):
        super().__init__()
        D, DFF = self._D, self._DFF
        lora_cls = LoRADualLayer if branches == 2 else LoRATriLayer
        if branches == 2:
            self.inp_proj = LoRADualLayer(n_features, D, rank, alpha=alpha)
        else:
            self.inp_proj = LoRATriLayer(n_features, D, rank, rank_m, alpha=alpha)
        self.layers = nn.ModuleList([
            _LoRATransformerLayer(D, self._NH, DFF, lora_cls, rank, rank_m, alpha=alpha)
            for _ in range(self._NL)
        ])

    def _run(self, x: torch.Tensor, method: str, lam: float = None):
        B, L, _ = x.shape
        h = x.reshape(B * L, -1)
        if method == 'r':
            h = self.inp_proj.forward_r(h).reshape(B, L, self._D)
            for layer in self.layers:
                h = layer.forward_r(h)
        elif method == 'u':
            h = self.inp_proj.forward_u(h).reshape(B, L, self._D)
            for layer in self.layers:
                h = layer.forward_u(h)
        else:
            h = self.inp_proj(h, lam).reshape(B, L, self._D)
            for layer in self.layers:
                h = layer(h, lam)
        return F.normalize(h[:, -1], dim=-1)

    def forward_r(self, x: torch.Tensor) -> torch.Tensor:
        return self._run(x, 'r')

    def forward_u(self, x: torch.Tensor) -> torch.Tensor:
        return self._run(x, 'u')

    def forward(self, x: torch.Tensor, lam: float) -> torch.Tensor:
        return self._run(x, 'lam', lam)

    def forward_mix(self, x: torch.Tensor, w_r: float, w_u: float) -> torch.Tensor:
        """LoRA-Simplex forward: the input projection and every FFN layer use the
        independent coefficients (w_R, w_U) instead of a single scalar λ, so the
        REPRESENTATION (not just the head) is preference-specific."""
        B, L, _ = x.shape
        h = self.inp_proj.forward_mix(x.reshape(B * L, -1), w_r, w_u).reshape(B, L, self._D)
        for layer in self.layers:
            h = layer.forward_mix(h, w_r, w_u)
        return F.normalize(h[:, -1], dim=-1)


class LoRAAlexNetEncoder(AlexNet):
    """AlexNet backbone with LoRA on the classifier linear (9216 → latent_dim).

    Approach-5 analogue for the trifeature (image) setting: same dual/tri
    structure as LoRATransformerEncoder but applied to AlexNet's single FC
    classifier instead of a Transformer FFN.  The conv features are shared.

    branches=2 → LoRADualLayer  (linear ablation, approach 4/5 linear)
    branches=3 → LoRATriLayer   (curve with curvature branch ΔW_M)
    """
    def __init__(self, latent_dim: int = 512, branches: int = 2,
                 rank: int = 4, rank_m: int = None, dropout: float = 0.5,
                 alpha: float = None):
        super().__init__(dropout=dropout)
        rank_m = rank_m or rank
        if branches == 2:
            self.classifier = LoRADualLayer(256 * 6 * 6, latent_dim, rank, alpha=alpha)
        else:
            self.classifier = LoRATriLayer(256 * 6 * 6, latent_dim, rank, rank_m, alpha=alpha)

    def _feat(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward_r(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.classifier.forward_r(self._feat(x)), dim=-1)

    def forward_u(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.classifier.forward_u(self._feat(x)), dim=-1)

    def forward(self, x: torch.Tensor, lam: float) -> torch.Tensor:
        return F.normalize(self.classifier(self._feat(x), lam), dim=-1)

    # Multi-preference support (fix8 / LoRA-Simplex)
    # The conv trunk is λ-blind: only `classifier` carries LoRA. So features are
    # computed ONCE per batch and only the (cheap) classifier is rerun for each
    # preference — same trick as the λ-blind-encoder path, without losing the
    # extra R/U capacity that encoder-LoRA provides.
    def feat(self, x: torch.Tensor) -> torch.Tensor:
        """λ-independent conv features — compute once, reuse for every preference."""
        return self._feat(x)

    def head_from_feat(self, f: torch.Tensor, lam: float) -> torch.Tensor:
        return F.normalize(self.classifier(f, lam), dim=-1)

    def head_mix_from_feat(self, f: torch.Tensor, w_r: float, w_u: float) -> torch.Tensor:
        """Simplex mixing with independent coefficients (see LoRADualLayer.forward_mix)."""
        return F.normalize(self.classifier.forward_mix(f, w_r, w_u), dim=-1)


class CoMMProjectionHead(nn.Module):
    """
    Exact projection head from CoMM (Dufumier et al., ICLR 2025).
    3-layer MLP with BatchNorm — matches CoMM._build_mlp(512, 512, 256).
    Used for gmc and comm training.
    Output is NOT pre-normalised; L2-norm is applied in the loss.
    """
    def __init__(self, in_dim: int = 512, mlp_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # caller applies F.normalize
