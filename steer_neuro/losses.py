"""SSL loss functions, vendored from pareto_ssl/losses.py and pareto_ssl/factorcl.py
(kept self-contained rather than cross-imported so steer_neuro doesn't couple to
that project's evolution). Do not let these drift silently from the originals.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """SimCLR NT-Xent. z1, z2: (B, D) L2-normalised."""
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    sim.fill_diagonal_(float("-inf"))
    labels = torch.cat([torch.arange(B, 2 * B),
                        torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, labels)


def infonce_cross(zi: torch.Tensor, zj: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Symmetric CLIP-style cross-modal InfoNCE. zi, zj: (B, D) L2-normalised."""
    B = zi.shape[0]
    logits = torch.mm(zi, zj.T) / temperature
    labels = torch.arange(B, device=zi.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def clip_loss(zi: torch.Tensor, zj: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """CLIP (Radford et al., 2021) with a LEARNED temperature. Identical to
    infonce_cross except the fixed 1/temperature is a trainable scalar.
    logit_scale holds log(1/tau); exponentiated + clamped at 100 here, as in
    the reference implementation (keeps logits from running away early in
    training). Init to log(1/0.07) (scale 14.29), own zero-decay param group.
    zi, zj: (B, D) L2-normalised."""
    B = zi.shape[0]
    logits = logit_scale.exp().clamp(max=100.0) * (zi @ zj.T)
    labels = torch.arange(B, device=zi.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def cross_self_loss(z1_a: torch.Tensor, z1_b: torch.Tensor,
                    z2_a: torch.Tensor, z2_b: torch.Tensor,
                    temperature: float = 0.1, ssl_scale: float = 1.0) -> torch.Tensor:
    """Cross+Self (CoMM repo, losses/cross_self.py; FactorCL's baseline
    category 2): loss = InfoNCE(m1,m2) + ssl_scale*0.5*(InfoNCE(m1_a,m1_b) +
    InfoNCE(m2_a,m2_b)). Same symmetric InfoNCE (infonce_cross) on all three
    pairs, matching the reference -- not NT-Xent for the unimodal terms
    (that would add within-view negatives the reference doesn't have).
    z*: (B, D) L2-normalised. _a/_b are the two augmented views."""
    cross = infonce_cross(z1_a, z2_a, temperature)
    ssl1 = infonce_cross(z1_a, z1_b, temperature)
    ssl2 = infonce_cross(z2_a, z2_b, temperature)
    return cross + ssl_scale * 0.5 * (ssl1 + ssl2)


def comm_infonce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1, INF: float = 1e8) -> torch.Tensor:
    """CoMM InfoNCE (Dufumier & Castillo-Navarro et al., ICLR 2025). Extends
    cross-modal InfoNCE with within-modal negatives. z1, z2: (B, D) L2-normalised."""
    N = z1.shape[0]
    sim_zij = (z1 @ z2.T) / temperature
    sim_zii = (z1 @ z1.T) / temperature - INF * torch.eye(N, device=z1.device)
    sim_zjj = (z2 @ z2.T) / temperature - INF * torch.eye(N, device=z1.device)
    sim_Z = torch.cat([
        torch.cat([sim_zij, sim_zii], dim=1),
        torch.cat([sim_zjj, sim_zij.T], dim=1),
    ], dim=0)
    return -F.log_softmax(sim_Z, dim=1).diagonal().mean()


def gmc_loss(z_mods: list, z_joint: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """GMC loss (Poklukar et al., ICML 2023): NT-Xent between each unimodal
    representation and the joint encoder. z_mods: list of (B,D) L2-normalised
    tensors. z_joint: (B,D) L2-normalised joint representation."""
    B = z_joint.shape[0]
    total = torch.tensor(0.0, device=z_joint.device)
    for z_mod in z_mods:
        out = torch.cat([z_joint, z_mod], dim=0)
        sim_exp = torch.exp(torch.mm(out, out.T) / temperature)
        mask = ~torch.eye(2 * B, dtype=torch.bool, device=out.device)
        denom = (sim_exp * mask).sum(dim=-1)
        pos = torch.exp((z_joint * z_mod).sum(dim=-1) / temperature)
        pos = torch.cat([pos, pos])
        total = total + (-torch.log(pos / denom)).mean()
    return total / len(z_mods)


def comm_loss(z_mods: list, z_joint: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Single-aug comm_infonce between each modality and the joint (same
    augmentation). z_mods: list of (B,D) L2-normalised. z_joint: (B,D) L2-normalised."""
    total = torch.tensor(0.0, device=z_joint.device)
    for z_mod in z_mods:
        total = total + comm_infonce(z_mod, z_joint, temperature)
    return total / len(z_mods)


class InfoNCECritic(nn.Module):
    """Concat critic with the InfoNCE (NCE) objective -- vendored from
    pareto_ssl/factorcl.py, sibling of CLUBInfoNCECritic below but for the
    NON-CLUB (maximise, not minimise, MI) terms in FactorCL's 6-loss
    formulation (InfoNCE(v,t), InfoNCE(v,v_aug), InfoNCE(t,t_aug),
    InfoNCE(conditional))."""

    def __init__(self, A_dim, B_dim, hidden_dim, layers, activation, **extra_kwargs):
        super().__init__()
        self._f = _mlp(A_dim + B_dim, hidden_dim, 1, layers, activation)

    def forward(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]
        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))
        T0 = self._f(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self._f(torch.cat([x_tile, y_tile], dim=-1))
        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - torch.log(torch.tensor(float(sample_size))))
        return -lower_bound


def _mlp(dim, hidden_dim, output_dim, layers, activation):
    activation = {'relu': nn.ReLU, 'tanh': nn.Tanh}[activation]
    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*seq)


class CLUBInfoNCECritic(nn.Module):
    """Variational CLUB upper bound on I(x;y), InfoNCE-style critic.
    Vendored from pareto_ssl/factorcl.py — used to decorrelate each unique
    head from its own (stop-gradient) shared head.
    """

    def __init__(self, A_dim, B_dim, hidden_dim, layers, activation, **extra_kwargs):
        super().__init__()
        self._f = _mlp(A_dim + B_dim, hidden_dim, 1, layers, activation)

    def forward(self, x_samples, y_samples):
        """CLUB upper bound estimate on I(x;y)."""
        sample_size = y_samples.shape[0]
        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))
        T0 = self._f(torch.cat([y_samples, x_samples], dim=-1))
        T1 = self._f(torch.cat([y_tile, x_tile], dim=-1))
        return T0.mean() - T1.mean()

    def learning_loss(self, x_samples, y_samples):
        """InfoNCE loss to fit the critic (call on detached inputs)."""
        sample_size = y_samples.shape[0]
        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))
        T0 = self._f(torch.cat([y_samples, x_samples], dim=-1))
        T1 = self._f(torch.cat([y_tile, x_tile], dim=-1))
        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - torch.log(torch.tensor(float(sample_size))))
        return -lower_bound
