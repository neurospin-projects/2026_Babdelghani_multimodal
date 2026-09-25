"""5 multimodal-SSL baselines (GMC, CoMM, CLIP, Cross+Self, FactorCL) on the
SAME frozen Champollion/ALMA encoders + LoRA regime as SteerNeuroModel, so
comparisons against STEER isolate the training OBJECTIVE, not architecture,
adapter capacity, or data. Ported from pareto_ssl/losses.py, factorcl.py,
networks.py, and pareto_ssl/multibench/benchmark_multibench.py's training
branches (is_joint / clip+cross_self / vanilla factorcl) -- see that
project's own head-capacity-control comment (FusionEncoder docstring,
benchmark_multibench.py:368) for why matching representation width across
methods matters, same reasoning applied here.

Representation-dimensionality budget, matched to STEER's own default
readout width (proj_dim=64, see the method description sec:readout):
  gmc / comm   : FusionEncoder -> 64-d joint representation
  clip / cross_self : concat(proj_v(32), proj_t(32)) -> 64-d
  factorcl     : 10 heads x 7-d = 70-d (10*D_out can't hit exactly 64; 7 is
                 the closest clean per-head width, same choice pareto_ssl's
                 own width-matching comment settled on for this exact
                 problem -- see benchmark_multibench.py:1470-1485)

LoRA regime: BOTH encoders get STEER's own LoRAConditionedEncoder (dual R/U
PaLoRA branches) at the SAME scope/rank_ratio/alpha as whatever STEER config
this is being compared against -- giving every baseline the IDENTICAL
trainable-adapter parameter budget STEER has, not a smaller single-branch
adapter. Since baselines have no preference/lambda axis at all, the encoder
is always called via forward_mix(x, 0.5, 0.5) -- a fixed, neutral 50/50
blend of the two branches, never swept.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_networks import SimpleProjectionHead, FusionEncoder
from encoders import build_champollion_encoder, build_alma_encoder
from losses import (nt_xent, infonce_cross, clip_loss, cross_self_loss,
                     gmc_loss, comm_loss, InfoNCECritic, CLUBInfoNCECritic)
from wrap_lora import LoRAConditionedEncoder, SCOPES

METHODS = ("gmc", "comm", "clip", "cross_self", "factorcl")


class BaselineModel(nn.Module):
    def __init__(self, method: str, scope: str = "last_stage", rank: int = 4,
                 rank_ratio: float = 0.15, alpha_champo: float = 4.0, alpha_alma: float = 0.25,
                 total_dim: int = 64, factorcl_head_dim: int = 7,
                 club_hidden: int = 64, club_layers: int = 1, club_activation: str = "relu",
                 temperature: float = 0.1, ssl_scale: float = 1.0, clip_learned_temp: bool = True,
                 factorcl_head_norm: bool = True):
        super().__init__()
        assert method in METHODS, f"method must be one of {METHODS}, got {method!r}"
        assert scope in SCOPES
        self.method = method
        self.temperature = temperature
        self.ssl_scale = ssl_scale

        enc_c, dim_c = build_champollion_encoder()
        enc_a, dim_a = build_alma_encoder()
        self.dim_c, self.dim_a = dim_c, dim_a
        # SAME LoRA mechanism/capacity as SteerNeuroModel -- only ever called
        # with a fixed neutral blend (0.5, 0.5), since baselines have no
        # preference axis to sweep.
        self.enc_champo = LoRAConditionedEncoder(enc_c, scope=scope, rank=rank,
                                                  rank_ratio=rank_ratio, alpha=alpha_champo)
        self.enc_alma = LoRAConditionedEncoder(enc_a, scope=scope, rank=rank,
                                                rank_ratio=rank_ratio, alpha=alpha_alma)

        if method in ("gmc", "comm"):
            # Champollion/ALMA have DIFFERENT raw widths (unlike MultiBench's
            # Transformer encoders, which share one adim for both modalities)
            # -- bridge each to a common width before fusion, matching the
            # reference's mod_proj role but per-modality since ours must be.
            self.bridge_champo = nn.Linear(dim_c, total_dim)
            self.bridge_alma = nn.Linear(dim_a, total_dim)
            self.fusion = FusionEncoder(total_dim, total_dim, hidden=None)
        elif method in ("clip", "cross_self"):
            half = total_dim // 2
            self.proj_champo = SimpleProjectionHead(dim_c, half, normalize=True)
            self.proj_alma = SimpleProjectionHead(dim_a, half, normalize=True)
            if method == "clip" and clip_learned_temp:
                import math
                self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
            else:
                self.logit_scale = None
        else:  # factorcl
            D = factorcl_head_dim
            self.factorcl_head_dim = D
            # factorcl_head_norm=True is the original port (L2-normalised heads); False matches the
            # official FactorCL mlp_head, which normalises nowhere (see SimpleProjectionHead docstring)
            _PH = lambda in_dim: SimpleProjectionHead(in_dim, D, normalize=factorcl_head_norm)
            self.head_r_c, self.head_r_a = _PH(dim_c), _PH(dim_a)      # cross-modal InfoNCE (shared/redundant)
            self.head_cl_c, self.head_cl_a = _PH(dim_c), _PH(dim_a)    # cross-modal CLUB
            self.head_u_c, self.head_u_a = _PH(dim_c), _PH(dim_a)      # within-modal InfoNCE (unique)
            self.head_cr_c, self.head_cr_a = _PH(dim_c), _PH(dim_a)    # conditional InfoNCE
            self.head_ccl_c, self.head_ccl_a = _PH(dim_c), _PH(dim_a)  # conditional CLUB
            self.critic_r = InfoNCECritic(D, D, club_hidden, club_layers, club_activation)
            self.critic_u_c = InfoNCECritic(D, D, club_hidden, club_layers, club_activation)
            self.critic_u_a = InfoNCECritic(D, D, club_hidden, club_layers, club_activation)
            self.critic_cond = InfoNCECritic(2 * D, 2 * D, club_hidden, club_layers, club_activation)
            self.critic_cl = CLUBInfoNCECritic(D, D, club_hidden, club_layers, club_activation)
            self.critic_ccl = CLUBInfoNCECritic(2 * D, 2 * D, club_hidden, club_layers, club_activation)

    # -- encoding (fixed neutral LoRA blend, no preference axis) -------------
    def _encode(self, champo_a, champo_b, alma_a, alma_b):
        h_ca = self.enc_champo.forward_mix(champo_a, 0.5, 0.5)
        h_cb = self.enc_champo.forward_mix(champo_b, 0.5, 0.5)
        h_aa = self.enc_alma.forward_mix(alma_a, 0.5, 0.5)
        h_ab = self.enc_alma.forward_mix(alma_b, 0.5, 0.5)
        return h_ca, h_cb, h_aa, h_ab

    # -- readout (what gets extracted for downstream probing) ---------------
    def embed(self, champo, alma):
        """(B, total_dim) L2-normalised representation, single (non-augmented)
        view -- what extract_baseline_embeddings.py calls for evaluation."""
        h_c = self.enc_champo.forward_mix(champo, 0.5, 0.5)
        h_a = self.enc_alma.forward_mix(alma, 0.5, 0.5)
        if self.method in ("gmc", "comm"):
            z_v = F.normalize(self.bridge_champo(h_c), dim=-1)
            z_t = F.normalize(self.bridge_alma(h_a), dim=-1)
            return self.fusion(z_v, z_t)
        elif self.method in ("clip", "cross_self"):
            return torch.cat([self.proj_champo(h_c), self.proj_alma(h_a)], dim=-1)
        else:  # factorcl -- official get_embedding(): concat all 5 single-view heads per modality
            parts_c = [self.head_r_c(h_c), self.head_cl_c(h_c), self.head_u_c(h_c),
                      self.head_cr_c(h_c), self.head_ccl_c(h_c)]
            parts_a = [self.head_r_a(h_a), self.head_cl_a(h_a), self.head_u_a(h_a),
                      self.head_cr_a(h_a), self.head_ccl_a(h_a)]
            return torch.cat(parts_c + parts_a, dim=-1)

    # -- parameter groups (base encoders stay frozen) ------------------------
    def adapter_and_head_parameters(self):
        mods = [self.enc_champo, self.enc_alma]
        if self.method in ("gmc", "comm"):
            mods += [self.bridge_champo, self.bridge_alma, self.fusion]
        elif self.method in ("clip", "cross_self"):
            mods += [self.proj_champo, self.proj_alma]
        else:
            mods += [self.head_r_c, self.head_r_a, self.head_cl_c, self.head_cl_a,
                     self.head_u_c, self.head_u_a, self.head_cr_c, self.head_cr_a,
                     self.head_ccl_c, self.head_ccl_a,
                     self.critic_r, self.critic_u_c, self.critic_u_a, self.critic_cond]
        for m in mods:
            for p in m.parameters():
                if p.requires_grad:
                    yield p
        if self.method == "clip" and self.logit_scale is not None:
            yield self.logit_scale

    def critic_parameters(self):
        """Only FactorCL has a separate CLUB-critic optimizer step (the two
        MI-MINIMISING critics, critic_cl/critic_ccl) -- empty for every other
        method, so train_baselines_neuro.py's 2-optimizer loop (mirroring
        train_steer_neuro.py's own opt_main/opt_critic split) is a true no-op
        on the critic side for gmc/comm/clip/cross_self, no special-casing
        needed in the training script itself."""
        if self.method != "factorcl":
            return
        for m in (self.critic_cl, self.critic_ccl):
            yield from m.parameters()

    # -- losses ----------------------------------------------------------------
    def compute_loss(self, champo_a, champo_b, alma_a, alma_b):
        """Main-step loss (everything except FactorCL's 2 CLUB critics, which
        get their own learning_loss() below). Returns (loss, diag_dict)."""
        h_ca, h_cb, h_aa, h_ab = self._encode(champo_a, champo_b, alma_a, alma_b)

        if self.method in ("gmc", "comm"):
            z_v = F.normalize(self.bridge_champo(h_ca), dim=-1)
            z_t = F.normalize(self.bridge_alma(h_aa), dim=-1)
            z_j = self.fusion(z_v, z_t)
            loss = (gmc_loss([z_v, z_t], z_j, self.temperature) if self.method == "gmc"
                    else comm_loss([z_v, z_t], z_j, self.temperature))
            return loss, dict(loss=loss.item())

        if self.method == "clip":
            z_v, z_t = self.proj_champo(h_ca), self.proj_alma(h_aa)
            loss = (clip_loss(z_v, z_t, self.logit_scale) if self.logit_scale is not None
                    else infonce_cross(z_v, z_t, self.temperature))
            diag = dict(loss=loss.item())
            if self.logit_scale is not None:
                diag["logit_scale"] = float(self.logit_scale.exp().clamp(max=100.0))
            return loss, diag

        if self.method == "cross_self":
            z_va, z_ta = self.proj_champo(h_ca), self.proj_alma(h_aa)
            z_vb, z_tb = self.proj_champo(h_cb), self.proj_alma(h_ab)
            loss = cross_self_loss(z_va, z_vb, z_ta, z_tb, self.temperature, self.ssl_scale)
            return loss, dict(loss=loss.item())

        # factorcl -- 4 of the 6 official losses (InfoNCE terms); the 2 CLUB
        # terms enter here too (forward(), the MI upper-bound estimate used
        # to PENALISE the encoder/heads) but are fit separately via
        # critic_learning_loss() below, same detached-critic split STEER's
        # own club_champo/club_alma already use.
        z_r_c, z_r_a = self.head_r_c(h_ca), self.head_r_a(h_aa)
        z_cl_c, z_cl_a = self.head_cl_c(h_ca), self.head_cl_a(h_aa)
        z_u_ca, z_u_cb = self.head_u_c(h_ca), self.head_u_c(h_cb)
        z_u_aa, z_u_ab = self.head_u_a(h_aa), self.head_u_a(h_ab)
        z_cr_ca, z_cr_cb = self.head_cr_c(h_ca), self.head_cr_c(h_cb)
        z_cr_aa, z_cr_ab = self.head_cr_a(h_aa), self.head_cr_a(h_ab)
        z_cc_ca, z_cc_cb = self.head_ccl_c(h_ca), self.head_ccl_c(h_cb)
        z_cc_aa, z_cc_ab = self.head_ccl_a(h_aa), self.head_ccl_a(h_ab)
        z_cr_c, z_cr_a = torch.cat([z_cr_ca, z_cr_cb], -1), torch.cat([z_cr_aa, z_cr_ab], -1)
        z_cc_c, z_cc_a = torch.cat([z_cc_ca, z_cc_cb], -1), torch.cat([z_cc_aa, z_cc_ab], -1)

        l_r = self.critic_r(z_r_c, z_r_a)
        l_cl = self.critic_cl(z_cl_c, z_cl_a)          # MI upper bound -- penalise, not fit, here
        l_u_c = self.critic_u_c(z_u_ca, z_u_cb)
        l_u_a = self.critic_u_a(z_u_aa, z_u_ab)
        l_cond = self.critic_cond(z_cr_c, z_cr_a)
        l_ccl = self.critic_ccl(z_cc_c, z_cc_a)        # MI upper bound -- penalise, not fit, here
        loss = l_r + l_cl + l_u_c + l_u_a + l_cond + l_ccl
        return loss, dict(loss=loss.item(), l_r=l_r.item(), l_cl=l_cl.item(),
                          l_u_c=l_u_c.item(), l_u_a=l_u_a.item(),
                          l_cond=l_cond.item(), l_ccl=l_ccl.item())

    def critic_learning_loss(self, champo_a, champo_b, alma_a, alma_b):
        """FactorCL's 2 CLUB critics' OWN fitting objective (InfoNCE lower
        bound on detached activations) -- 0.0 no-op for every other method,
        same role as SteerNeuroModel.critic_learning_loss()."""
        if self.method != "factorcl":
            return torch.tensor(0.0)
        h_ca, h_cb, h_aa, h_ab = self._encode(champo_a, champo_b, alma_a, alma_b)
        z_cl_c = self.head_cl_c(h_ca).detach()
        z_cl_a = self.head_cl_a(h_aa).detach()
        z_cc_c = torch.cat([self.head_ccl_c(h_ca), self.head_ccl_c(h_cb)], -1).detach()
        z_cc_a = torch.cat([self.head_ccl_a(h_aa), self.head_ccl_a(h_ab)], -1).detach()
        return self.critic_cl.learning_loss(z_cl_c, z_cl_a) + self.critic_ccl.learning_loss(z_cc_c, z_cc_a)
