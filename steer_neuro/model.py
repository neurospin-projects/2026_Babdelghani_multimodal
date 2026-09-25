"""Full STEER-neuro model: two frozen pretrained encoders (Champollion sulci,
ALMA diffusion), each PaLoRA-conditioned at a configurable scope, feeding 4
fresh plain heads (r_champo, r_alma, u_champo, u_alma) and 2 CLUB critics.

Preference routing :
    lambda = (lambda_R, lambda_U_champo, lambda_U_alma), sums to 1
    Champollion encoder sees (lambda_R, lambda_U_champo)
    ALMA        encoder sees (lambda_R, lambda_U_alma)

Losses, per preference (view 'a' for L_R and the CLUB term, views a/b for L_U):
    L_R  = infonce_cross(r_champo(h_c^a), r_alma(h_a^a))
    L_U_champo = nt_xent(u_champo(h_c^a), u_champo(h_c^b))
               + beta * clamp(CLUB(u_champo(h_c^a), sg[r_champo(h_c^a)]), min=0)
    L_U_alma   = nt_xent(u_alma(h_a^a),   u_alma(h_a^b))
               + beta * clamp(CLUB(u_alma(h_a^a),   sg[r_alma(h_a^a)]),   min=0)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones_lite import ProjectionHead
from encoders import build_champollion_encoder, build_alma_encoder
from losses import nt_xent, infonce_cross, CLUBInfoNCECritic
from wrap_lora import LoRAConditionedEncoder, SCOPES


class SteerNeuroModel(nn.Module):
    def __init__(self, scope: str = "bottleneck", rank: int = 4,
                 rank_champo: int = None, rank_alma: int = None,
                 rank_ratio: float = None, rank_ratio_champo: float = None,
                 rank_ratio_alma: float = None, alpha: float = None,
                 alpha_champo: float = None, alpha_alma: float = None,
                 proj_dim: int = 64, head_hidden: int = 128,
                 club_hidden: int = 64, club_layers: int = 1, club_activation: str = "relu",
                 beta_club: float = 0.5):
        """Two ways to set LoRA rank, `rank_ratio` takes precedence when given:
          - absolute: `rank` (shared) / `rank_champo` / `rank_alma` -- a fixed
            integer rank per wrapped layer, regardless of that layer's width.
          - fractional: `rank_ratio` (shared) / `rank_ratio_champo` /
            `rank_ratio_alma` -- rank = round(ratio * layer_out_width), computed
            PER WRAPPED LAYER. This is what makes e.g. 0.5 mean "half of each
            layer's own width" consistently across the 256-d Champollion
            bottleneck, the 128-d ALMA bottleneck, and (at last_stage/full
            scope) conv layers as narrow as 32 channels -- an absolute rank
            that's 50% of one is a wildly different fraction of another.
        `alpha_champo`/`alpha_alma` override the shared `alpha` per encoder --
        needed because at the same rank_ratio, ALMA's narrower bottleneck
        (128-d vs Champollion's 256-d) gets a smaller absolute rank at its
        Linear layer, so the SAME alpha gives it LESS scale=alpha/rank damping
        (confirmed 2026-09-05: a real Jean Zay run showed ALMA's LoRA branch
        magnitude still climbing unbounded past 4x the base weight norm at
        alpha=4, while Champollion's had settled to a healthy ~0.3-1.0 range).
        """
        super().__init__()
        assert scope in SCOPES
        self.beta_club = beta_club
        self.proj_dim = proj_dim
        rank_champo = rank_champo if rank_champo is not None else rank
        rank_alma = rank_alma if rank_alma is not None else rank
        ratio_champo = rank_ratio_champo if rank_ratio_champo is not None else rank_ratio
        ratio_alma = rank_ratio_alma if rank_ratio_alma is not None else rank_ratio
        alpha_champo = alpha_champo if alpha_champo is not None else alpha
        alpha_alma = alpha_alma if alpha_alma is not None else alpha

        enc_c, dim_c = build_champollion_encoder()
        enc_a, dim_a = build_alma_encoder()
        self.enc_champo = LoRAConditionedEncoder(enc_c, scope=scope, rank=rank_champo,
                                                  rank_ratio=ratio_champo, alpha=alpha_champo)
        self.enc_alma = LoRAConditionedEncoder(enc_a, scope=scope, rank=rank_alma,
                                                rank_ratio=ratio_alma, alpha=alpha_alma)

        self.r_champo = ProjectionHead(dim_c, [dim_c, head_hidden, proj_dim], activation="relu")
        self.r_alma = ProjectionHead(dim_a, [dim_a, head_hidden, proj_dim], activation="relu")
        self.u_champo = ProjectionHead(dim_c, [dim_c, head_hidden, proj_dim], activation="relu")
        self.u_alma = ProjectionHead(dim_a, [dim_a, head_hidden, proj_dim], activation="relu")

        self.club_champo = CLUBInfoNCECritic(proj_dim, proj_dim, club_hidden, club_layers, club_activation)
        self.club_alma = CLUBInfoNCECritic(proj_dim, proj_dim, club_hidden, club_layers, club_activation)

    # -- encoding -----------------------------------------------------------
    def encode_champo(self, x, w_r: float, w_u: float):
        return self.enc_champo.forward_mix(x, w_r, w_u)

    def encode_alma(self, x, w_r: float, w_u: float):
        return self.enc_alma.forward_mix(x, w_r, w_u)

    # -- adapter / head parameter groups (base encoders stay frozen) --------
    def adapter_and_head_parameters(self):
        for m in (self.enc_champo, self.enc_alma, self.r_champo, self.r_alma,
                  self.u_champo, self.u_alma):
            for p in m.parameters():
                if p.requires_grad:
                    yield p

    def champo_parameters(self):
        """Everything that only ever sees Champollion's own encoder output --
        enc_champo's LoRA adapters + r_champo + u_champo. Lets
        train_steer_neuro.py give Champollion and ALMA separate learning
        rates (--lr_champo/--lr_alma), a lever never tried before: every
        prior fix for the champo/alma asymmetry (alpha_champo/alpha_alma,
        rank_ratio_champo/rank_ratio_alma) controlled LoRA SCALE, never the
        optimizer's own step size."""
        for m in (self.enc_champo, self.r_champo, self.u_champo):
            for p in m.parameters():
                if p.requires_grad:
                    yield p

    def alma_parameters(self):
        """Same as champo_parameters() but for ALMA's side -- enc_alma +
        r_alma + u_alma."""
        for m in (self.enc_alma, self.r_alma, self.u_alma):
            for p in m.parameters():
                if p.requires_grad:
                    yield p

    def critic_parameters(self):
        for m in (self.club_champo, self.club_alma):
            yield from m.parameters()

    def mean_branch_norms(self):
        """Mean (rho_R, rho_U) across all wrapped layers, per encoder -- the
        "is either branch starved" check from the MOSEI work. Observed there:
        both LoRA branches stay active, rho growing from ~0.05 to ~0.4-0.63,
        with no branch starvation."""
        out = {}
        for name, enc in (("champo", self.enc_champo), ("alma", self.enc_alma)):
            norms = list(enc.branch_norms().values())
            if norms:
                rs, us = zip(*norms)
                out[f"rho_R_{name}"] = sum(rs) / len(rs)
                out[f"rho_U_{name}"] = sum(us) / len(us)
        return out

    # -- one preference's losses, given a batch of 4 raw crops --------------
    def preference_losses(self, champo_a, champo_b, alma_a, alma_b, lam, return_raw=False):
        w_r, w_uc, w_ua = lam

        h_c_a = self.encode_champo(champo_a, w_r, w_uc)
        h_c_b = self.encode_champo(champo_b, w_r, w_uc)
        h_a_a = self.encode_alma(alma_a, w_r, w_ua)
        h_a_b = self.encode_alma(alma_b, w_r, w_ua)

        r_c_a = F.normalize(self.r_champo(h_c_a), dim=-1)
        r_a_a = F.normalize(self.r_alma(h_a_a), dim=-1)
        u_c_a = F.normalize(self.u_champo(h_c_a), dim=-1)
        u_c_b = F.normalize(self.u_champo(h_c_b), dim=-1)
        u_a_a = F.normalize(self.u_alma(h_a_a), dim=-1)
        u_a_b = F.normalize(self.u_alma(h_a_b), dim=-1)

        L_R = infonce_cross(r_c_a, r_a_a)

        club_c = self.club_champo(u_c_a, r_c_a.detach())
        club_a = self.club_alma(u_a_a, r_a_a.detach())
        L_Uc = nt_xent(u_c_a, u_c_b) + self.beta_club * torch.clamp(club_c, min=0.0)
        L_Ua = nt_xent(u_a_a, u_a_b) + self.beta_club * torch.clamp(club_a, min=0.0)

        loss = w_r * L_R + w_uc * L_Uc + w_ua * L_Ua
        diagnostics = dict(L_R=L_R.item(), L_Uc=L_Uc.item(), L_Ua=L_Ua.item(),
                            club_c=club_c.item(), club_a=club_a.item())
        if return_raw:
            # still-differentiable L_Uc/L_Ua tensors (NOT .item()'d) -- lets a
            # caller isolate d(L_Uc)/d(u_champo params) and d(L_Ua)/d(u_alma
            # params) via torch.autograd.grad(..., retain_graph=True) BEFORE
            # the real combined-loss backward() consumes the graph. Each of
            # L_Uc/L_Ua already includes its own beta_club*clamp(club, min=0)
            # term (see above) -- that's this method's own definition of
            # "L_Uc"/"L_Ua" throughout the codebase, so the gradient norm this
            # enables measures the combined NT-Xent + CLUB-penalty signal
            # actually reaching each head's parameters, not the bare NT-Xent
            # term in isolation.
            return loss, diagnostics, dict(L_Uc=L_Uc, L_Ua=L_Ua)
        return loss, diagnostics

    def critic_learning_loss(self, champo_a, alma_a, lam):
        """Critic InfoNCE fit step — call with detached encoder/head outputs,
        BEFORE the encoder step, on the same sampled preference(s)."""
        w_r, w_uc, w_ua = lam
        with torch.no_grad():
            h_c_a = self.encode_champo(champo_a, w_r, w_uc)
            h_a_a = self.encode_alma(alma_a, w_r, w_ua)
            r_c_a = F.normalize(self.r_champo(h_c_a), dim=-1)
            r_a_a = F.normalize(self.r_alma(h_a_a), dim=-1)
            u_c_a = F.normalize(self.u_champo(h_c_a), dim=-1)
            u_a_a = F.normalize(self.u_alma(h_a_a), dim=-1)
        loss_c = self.club_champo.learning_loss(u_c_a, r_c_a)
        loss_a = self.club_alma.learning_loss(u_a_a, r_a_a)
        return loss_c + loss_a

    # -- probe-time readout: assembled (R, U_champo, U_alma) blocks ---------
    @torch.no_grad()
    def embed_blocks(self, champo, alma, w_r: float, w_uc: float, w_ua: float):
        h_c = self.encode_champo(champo, w_r, w_uc)
        h_a = self.encode_alma(alma, w_r, w_ua)
        r = F.normalize(0.5 * (F.normalize(self.r_champo(h_c), dim=-1)
                                + F.normalize(self.r_alma(h_a), dim=-1)), dim=-1)
        u_c = F.normalize(self.u_champo(h_c), dim=-1)
        u_a = F.normalize(self.u_alma(h_a), dim=-1)
        return r, u_c, u_a


@torch.no_grad()
def endpoint_cka_diagnostic(model: "SteerNeuroModel", champo_probe, alma_probe, total_dim: int = None):
    """The training-time non-degeneracy check: dim-readout z(lambda) at the 3
    pure vertices vs the centroid, on a FIXED held-out probe batch. Call this
    periodically during training (not every step -- it's 4 extra forward
    passes) to watch whether the family is separating as tau anneals toward 1,
    the same "endpoint CKA" trend the MOSEI work tracked across training.
    Collapsed (~0.9-1.0) is expected early (tau~0, everything maps near the
    centroid by construction); it should fall as annealing progresses if the
    lambda axis is live. See eval_steer_neuro.py for the full post-hoc version."""
    from simplex import dim_readout, linear_cka

    total_dim = total_dim or model.proj_dim  # "dim" readout width is D, not 3*D
    vertices = {"R": (1.0, 0.0, 0.0), "U_champo": (0.0, 1.0, 0.0), "U_alma": (0.0, 0.0, 1.0)}
    centroid = (1 / 3, 1 / 3, 1 / 3)

    def z(lam):
        r, uc, ua = model.embed_blocks(champo_probe, alma_probe, *lam)
        return dim_readout([r, uc, ua], list(lam), total_dim=total_dim)

    z_centroid = z(centroid)
    return {f"cka_{name}_vs_centroid": linear_cka(z(lam), z_centroid) for name, lam in vertices.items()}
