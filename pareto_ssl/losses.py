"""SSL loss functions for Pareto-SSL benchmark.

Position on redundant ↔ unique axis:
    clip      — pure cross-modal alignment → maximises R, kills U
    gmc       — each mod vs joint encoder
    comm      — gmc + within-modal negatives
    simclr    — within-modal only → maximises U, ignores R
    factorcl  — explicit R + conditional U → aims to capture both
"""

import torch
import torch.nn.functional as F


# Within-modality losses

def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """SimCLR NT-Xent. z1, z2: (B, D) L2-normalised."""
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    sim.fill_diagonal_(float("-inf"))
    labels = torch.cat([torch.arange(B, 2 * B),
                        torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, labels)


# Cross-modal losses

def infonce_cross(zi: torch.Tensor, zj: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Symmetric CLIP-style cross-modal InfoNCE. zi, zj: (B, D) L2-normalised."""
    B = zi.shape[0]
    logits = torch.mm(zi, zj.T) / temperature
    labels = torch.arange(B, device=zi.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def clip_loss(zi: torch.Tensor, zj: torch.Tensor,
              logit_scale: torch.Tensor) -> torch.Tensor:
    """CLIP (Radford et al., 2021) with a LEARNED temperature.

    Identical to infonce_cross except the fixed 1/temperature is replaced by a
    trainable scalar. `logit_scale` holds log(1/tau); it is exponentiated here and
    clamped at 100, as in the reference implementation, which keeps the logits from
    running away early in training. Initialise it to log(1/0.07) (scale 14.29).

    zi, zj: (B, D) L2-normalised.
    """
    B = zi.shape[0]
    logits = logit_scale.exp().clamp(max=100.0) * (zi @ zj.T)
    labels = torch.arange(B, device=zi.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def cross_self_loss(z1_a: torch.Tensor, z1_b: torch.Tensor,
                    z2_a: torch.Tensor, z2_b: torch.Tensor,
                    temperature: float = 0.1, ssl_scale: float = 1.0) -> torch.Tensor:
    """Cross+Self (CoMM repo, losses/cross_self.py; FactorCL's baseline category 2).

        loss = InfoNCE(m1, m2) + ssl_scale * 0.5 * (InfoNCE(m1_a, m1_b)
                                                    + InfoNCE(m2_a, m2_b))

    Cross-modal CL plus a unimodal (Xi, Xi') CL term per modality. The reference
    implementation applies the SAME symmetric InfoNCE to all three pairs -- one
    `self.infonce` called on (mod1, mod2), (mod1_aug1, mod1_aug2) and
    (mod2_aug1, mod2_aug2) -- so infonce_cross is used throughout rather than
    swapping in NT-Xent for the unimodal terms, which would add within-view
    negatives the reference does not have. ssl_scale=1.0 is the repo default.

    z*: (B, D) L2-normalised. _a / _b are the two augmented views.
    """
    cross = infonce_cross(z1_a, z2_a, temperature)
    ssl1  = infonce_cross(z1_a, z1_b, temperature)
    ssl2  = infonce_cross(z2_a, z2_b, temperature)
    return cross + ssl_scale * 0.5 * (ssl1 + ssl2)


def comm_infonce(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.1, INF: float = 1e8) -> torch.Tensor:
    """
    CoMM InfoNCE (Dufumier & Castillo-Navarro et al., ICLR 2025).
    Extends cross-modal InfoNCE with within-modal negatives.
    z1, z2: (B, D) L2-normalised.
    """
    N = z1.shape[0]
    sim_zij = (z1 @ z2.T) / temperature                                    # (N, N) cross-modal
    sim_zii = (z1 @ z1.T) / temperature - INF * torch.eye(N, device=z1.device)  # (N, N) within-M1
    sim_zjj = (z2 @ z2.T) / temperature - INF * torch.eye(N, device=z1.device)  # (N, N) within-M2
    # (2N, 2N) block matrix — positive pairs sit on the diagonal
    sim_Z = torch.cat([
        torch.cat([sim_zij,   sim_zii  ], dim=1),
        torch.cat([sim_zjj,   sim_zij.T], dim=1),
    ], dim=0)
    return -F.log_softmax(sim_Z, dim=1).diagonal().mean()


def gmc_loss(z_mods: list, z_joint: torch.Tensor, temperature: float = 0.1,
             weights=None) -> torch.Tensor:
    """
    GMC loss (Poklukar et al., ICML 2023).
    NT-Xent between each unimodal representation and the joint encoder.
    z_mods: list of (B, D) L2-normalised tensors.
    z_joint: (B, D) L2-normalised joint representation.

    `weights` (one per modality, summing to 1) re-weights the per-modality terms
    instead of averaging them. This exists for the "why not just train N models with
    different loss weights?" control: it is the only preference axis GMC HAS, since
    the objective contains no unique/uniqueness term to trade shared information off
    against. weights=None reproduces the published uniform average exactly.
    """
    B = z_joint.shape[0]
    w = [1.0 / len(z_mods)] * len(z_mods) if weights is None else list(weights)
    total = torch.tensor(0.0, device=z_joint.device)
    for wi, z_mod in zip(w, z_mods):
        out = torch.cat([z_joint, z_mod], dim=0)            # (2B, D)
        sim_exp = torch.exp(torch.mm(out, out.T) / temperature)
        mask    = ~torch.eye(2 * B, dtype=torch.bool, device=out.device)
        denom   = (sim_exp * mask).sum(dim=-1)              # (2B,) negatives sum
        pos     = torch.exp((z_joint * z_mod).sum(dim=-1) / temperature)  # (B,)
        pos     = torch.cat([pos, pos])                      # (2B,)
        total   = total + wi * (-torch.log(pos / denom)).mean()
    return total


def comm_loss(z_mods: list, z_joint: torch.Tensor, temperature: float = 0.1,
              weights=None) -> torch.Tensor:
    """
    Single-aug comm_infonce between each modality and the joint (same augmentation).
    z_mods: list of (B, D) L2-normalised tensors.
    z_joint: (B, D) L2-normalised joint representation.

    `weights`: see gmc_loss. None reproduces the published uniform average.
    """
    w = [1.0 / len(z_mods)] * len(z_mods) if weights is None else list(weights)
    total = torch.tensor(0.0, device=z_joint.device)
    for wi, z_mod in zip(w, z_mods):
        total = total + wi * comm_infonce(z_mod, z_joint, temperature)
    return total


def comm_loss_dual(z1_list: list, z2_list: list, temperature: float = 0.1) -> torch.Tensor:
    """
    Exact CoMM dual-augmentation loss (Dufumier et al., ICLR 2025).
    z1_list = [z1_v, z1_t, z1_j]  from aug1
    z2_list = [z2_v, z2_t, z2_j]  from aug2  (last element = joint = prototype)
    Each modality from aug1 is contrasted against the joint from aug2, and vice versa.
    """
    z1_j, z2_j = z1_list[-1], z2_list[-1]
    total = torch.tensor(0.0, device=z1_j.device)
    for z1_i, z2_i in zip(z1_list, z2_list):
        total = total + (comm_infonce(z1_i, z2_j, temperature) +
                         comm_infonce(z2_i, z1_j, temperature)) / 2
    return total / len(z1_list)


# FactorCL

def _cond_nce(z_a: torch.Tensor, z_b: torch.Tensor, cond_z: torch.Tensor,
              tau1: float = 0.1, tau2: float = 0.1) -> torch.Tensor:
    """
    Conditional NCE: within-modal InfoNCE conditioned on the other modality.

    For each anchor z_a_i:
      positive = z_b_i  (different augmentation of the same sample)
      negative = z_b_j  (other samples)
      conditioning: negatives j that share the cross-modal signal cond_z_j ≈ cond_z_i
                    are downweighted by subtracting sim(cond_i, cond_j)/tau2

    CondNCE_i = sim(z_a_i, z_b_i)/tau1
                - log Σ_{j≠i} exp(sim(z_a_i, z_b_j)/tau1 - sim(cond_i, cond_j)/tau2)

    z_a, z_b : (B, D) L2-normalised — two augmented views of the same modality
    cond_z   : (B, D) L2-normalised — conditioning modality representations
    """
    pos  = (z_a * z_b).sum(-1) / tau1                         # (B,) positive sims
    sim_z = z_a @ z_b.T / tau1                                # (B, B) within-modal
    sim_c = (cond_z @ cond_z.T / tau2).clamp(min=0)            # (B, B) conditioning (only downweight)
    adj   = sim_z - sim_c                                      # subtract shared signal
    adj.fill_diagonal_(float("-inf"))                          # exclude self
    log_denom = torch.logsumexp(adj, dim=1)                    # (B,)
    return -(pos - log_denom).mean()


def factorcl(z1_a: torch.Tensor, z1_b: torch.Tensor,
             z2_a: torch.Tensor, z2_b: torch.Tensor,
             cond1: torch.Tensor = None, cond2: torch.Tensor = None,
             z1_r: torch.Tensor = None, z2_r: torch.Tensor = None,
             tau_r: float = 0.1, tau_u: float = 0.1,
             lam: float = 1.0) -> torch.Tensor:
    """
    FactorCL loss (Liang et al., NeurIPS 2023).

    L = L_R + lam * (L_U1 + L_U2)
      L_R  = cross-modal InfoNCE(z1_r, z2_r)     — shared head, L2-normalised
      L_U1 = CondNCE(z1_a, z1_b | cond1)         — unique M1, conditioned on z2_r (detached)
      L_U2 = CondNCE(z2_a, z2_b | cond2)         — unique M2, conditioned on z1_r (detached)

    Separate shared and unique projection heads prevent the collapse where L_R
    forces the conditioning to become identical to the unique representation.
    """
    if z1_r is None:
        z1_r = z1_a
    if z2_r is None:
        z2_r = z2_a
    if cond1 is None:
        cond1 = z2_r
    if cond2 is None:
        cond2 = z1_r
    l_r  = infonce_cross(z1_r, z2_r, tau_r)
    l_u1 = _cond_nce(z1_a, z1_b, cond1, tau_u, tau_r)
    l_u2 = _cond_nce(z2_a, z2_b, cond2, tau_u, tau_r)
    return l_r + lam * (l_u1 + l_u2)
