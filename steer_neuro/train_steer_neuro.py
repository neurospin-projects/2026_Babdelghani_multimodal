"""STEER-neuro training loop: Champollion (sulci) x ALMA (diffusion), region
S.C.-sylv. right / WholeBrain. Preference routing:

  - every batch samples M preferences from the 15-point grid (M=15 => full
    grid, matching the paper; M<15 is the design doc's cost-control default),
    annealed centroid -> nominal grid over training (gamma = step_frac / Q)
  - per preference: one CLUB-critic InfoNCE fit step (detached), then one
    encoder+heads step through preference_losses()
  - only PaLoRA adapters + 4 heads train; both base encoders stay frozen

Logging (the MOSEI-style instrumentation,):
  - `metrics.csv`, every --log_every steps: losses, critic loss, and the
    branch-magnitude ratios rho_R/rho_U per encoder (||scale*A.B||_F / ||W0||_F)
    -- confirms neither the R nor the U branch is starved during training.
  - `cka.csv`, every --cka_every steps, on a FIXED held-out probe batch (the
    last --probe_limit paired subjects, excluded from training): endpoint CKA
    of each pure-vertex dim-readout embedding vs the centroid, PLUS the three
    quality diagnostics in diagnostics.py -- cka_u_r (is the unique head
    actually unique?), pairwise endpoint CKA among the 3 vertices (does
    lambda change anything between any pair?), and the edge-interpolation
    ratio per edge (is the curve learned or merely interpolated?). Tracks
    whether the lambda family is separating as annealing (tau) progresses --
    the same quantities eval_steer_neuro.py computes post-hoc, but as a
    training curve.

Usage (smoke test):
  python3 train_steer_neuro.py --scope bottleneck --epochs 1 --limit 8 --batch_size 2 --num_prefs 4

Usage (full run, hand off to the user / a slurm job rather than launching directly):
  python3 train_steer_neuro.py --scope bottleneck --epochs 100 --batch_size 16 --num_prefs 4 \
    --rank_ratio 0.5 --save_dir <out>
"""
import argparse
import csv
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from diagnostics import unique_shared_cka, endpoint_cka_pairwise, all_edge_interpolation_ratios
from model import SteerNeuroModel, endpoint_cka_diagnostic
from paired_dataset import PairedSkeletonDataset, paired_subjects
from simplex import simplex_grid, anneal, sample_preferences
from wrap_lora import SCOPES


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--scope", choices=SCOPES, required=True)
    p.add_argument("--rank", type=int, default=4, help="shared absolute LoRA rank (overridden per-encoder below if set; ignored if any rank_ratio is set)")
    p.add_argument("--rank_champo", type=int, default=None, help="absolute rank override, Champollion (256-d) encoder")
    p.add_argument("--rank_alma", type=int, default=None, help="absolute rank override, ALMA (128-d) encoder")
    p.add_argument("--rank_ratio", type=float, default=None, help="shared fractional rank: rank = round(ratio * layer_out_width), PER WRAPPED LAYER. Takes precedence over --rank/--rank_champo/--rank_alma when set. Recommended starting point: --rank_ratio 0.15 (not the default, so absolute --rank stays available when this flag is simply omitted)")
    p.add_argument("--rank_ratio_champo", type=float, default=None, help="fractional-rank override, Champollion encoder")
    p.add_argument("--rank_ratio_alma", type=float, default=None, help="fractional-rank override, ALMA encoder")
    p.add_argument("--alpha", type=float, default=None, help="shared PaLoRA alpha (scale=alpha/rank per layer)")
    p.add_argument("--alpha_champo", type=float, default=None, help="override alpha for Champollion encoder")
    p.add_argument("--alpha_alma", type=float, default=None, help="override alpha for ALMA encoder -- recommended smaller than alpha_champo (e.g. half), since ALMA's narrower 128-d bottleneck gets a smaller absolute rank at the same rank_ratio, so the same alpha under-damps it")
    p.add_argument("--proj_dim", type=int, default=64)
    p.add_argument("--head_hidden", type=int, default=128)
    p.add_argument("--club_hidden", type=int, default=64)
    p.add_argument("--beta_club", type=float, default=0.5)
    p.add_argument("--grid_spacing", type=int, default=4,
                   help="simplex_grid() subdivisions -- 4 (default) = 15 points incl. 3 "
                        "asymmetric interior points, step 1/4; 3 = 10 points (3 vertices, 6 "
                        "edge points at 1/3 & 2/3, the exact centroid), step 1/3, but NO "
                        "asymmetric interior point (every point has a zero or is the exact "
                        "centroid) -- cheaper training population, but loses interior 3-way "
                        "tradeoff coverage")
    p.add_argument("--num_prefs", type=int, default=4, help="preferences sampled per batch (<= grid size)")
    p.add_argument("--anneal_Q", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_champo", type=float, default=None,
                   help="override --lr for Champollion's own params (enc_champo+r_champo+u_champo) "
                        "only. None (default) = use --lr for both encoders, original flat-LR "
                        "behavior. Never tried before 2026-09-13: every prior fix for the "
                        "champo/alma asymmetry (alpha_champo/alpha_alma, rank_ratio_*) controlled "
                        "LoRA scale, never the optimizer's own step size per encoder.")
    p.add_argument("--lr_alma", type=float, default=None, help="override --lr for ALMA's own params, see --lr_champo")
    p.add_argument("--critic_lr", type=float, default=1e-3)
    p.add_argument("--lr_schedule", choices=["none", "cosine"], default="none",
                   help="'none' (default): flat lr/critic_lr for the whole run -- the original "
                        "behavior. 'cosine': cosine-anneal BOTH opt_main and opt_critic down to "
                        "~0 over the full run (T_max=total_steps), one scheduler.step() per "
                        "optimizer.step(). Motivated by late-training oscillation observed in "
                        "L_Ua/club_a on a 20-epoch last_stage/club1.0 run (2026-09-10) -- both "
                        "optimizers ran at a constant lr=1e-3 for all 20 epochs with no decay at "
                        "all, a plausible cause of a model that never settles rather than "
                        "converging as training gets longer.")
    p.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay on adapter+head params -- a rank-independent restoring force against unbounded LoRA branch growth (see rho_R/rho_U in metrics.csv), independent of --alpha")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="subject-count cap on the TRAIN split, for smoke tests")
    p.add_argument("--probe_limit", type=int, default=16, help="held-out subject count for the periodic CKA diagnostic")
    p.add_argument("--log_every", type=int, default=10, help="steps between metrics.csv rows")
    p.add_argument("--cka_every", type=int, default=50, help="steps between cka.csv rows (4 extra forward passes each)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_from", default=None,
                   help="warm-start from an existing steer_neuro_final.pt's model_state_dict "
                        "(e.g. to add more epochs on top of an already-finished run without "
                        "re-learning from random init). Only the WEIGHTS resume -- checkpoints "
                        "don't save optimizer state or the step counter, so the AdamW optimizers "
                        "and the tau-annealing schedule (which depends on step) both restart from "
                        "scratch here, not a byte-identical continuation of the original run. "
                        "--scope/--rank*/--alpha*/--proj_dim/--head_hidden/--club_hidden/--beta_club "
                        "must match the checkpoint's own architecture or load_state_dict will fail.")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--max_steps", type=int, default=None, help="hard cap on optimizer steps, for smoke tests")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class CsvLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    def log(self, row):
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._file.flush()

    def close(self):
        self._file.close()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    all_subjects = paired_subjects()
    probe_subjects = all_subjects[-args.probe_limit:]
    train_pool = all_subjects[:-args.probe_limit]
    train_subjects = train_pool[:args.limit] if args.limit is not None else train_pool
    print(f"[data] {len(train_subjects)} train subjects, {len(probe_subjects)} held-out probe subjects", flush=True)

    dataset = PairedSkeletonDataset(subjects=train_subjects)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers, drop_last=True,
                         persistent_workers=args.num_workers > 0)

    probe_ds = PairedSkeletonDataset(subjects=probe_subjects)
    probe_loader = DataLoader(probe_ds, batch_size=len(probe_ds), shuffle=False, num_workers=0)
    probe_champo_a, _, probe_alma_a, _, _ = next(iter(probe_loader))
    probe_champo_a, probe_alma_a = probe_champo_a.to(args.device), probe_alma_a.to(args.device)

    model = SteerNeuroModel(scope=args.scope, rank=args.rank,
                             rank_champo=args.rank_champo, rank_alma=args.rank_alma,
                             rank_ratio=args.rank_ratio, rank_ratio_champo=args.rank_ratio_champo,
                             rank_ratio_alma=args.rank_ratio_alma,
                             alpha=args.alpha, alpha_champo=args.alpha_champo, alpha_alma=args.alpha_alma,
                             proj_dim=args.proj_dim, head_hidden=args.head_hidden,
                             club_hidden=args.club_hidden, beta_club=args.beta_club).to(args.device)
    print(f"[model] champo wrapped layers -> rank: {model.enc_champo.wrapped_layers}", flush=True)
    print(f"[model] alma   wrapped layers -> rank: {model.enc_alma.wrapped_layers}", flush=True)

    if args.resume_from is not None:
        prev = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        model.load_state_dict(prev["model_state_dict"])
        print(f"[resume] warm-started weights from {args.resume_from} "
              f"(prev run: scope={prev['args']['scope']}, epochs={prev['args']['epochs']}) -- "
              f"optimizer state and step counter still start fresh, see --resume_from help", flush=True)

    lr_champo = args.lr_champo if args.lr_champo is not None else args.lr
    lr_alma = args.lr_alma if args.lr_alma is not None else args.lr
    opt_main = torch.optim.AdamW([
        {"params": list(model.champo_parameters()), "lr": lr_champo},
        {"params": list(model.alma_parameters()), "lr": lr_alma},
    ], weight_decay=args.weight_decay)
    opt_critic = torch.optim.AdamW(model.critic_parameters(), lr=args.critic_lr)

    # total_steps needed here (not just below, for tau-annealing) so the
    # cosine schedule's T_max spans the WHOLE run, not just what's left when
    # this line used to appear further down.
    total_steps = args.epochs * (len(loader) if args.max_steps is None
                                  else min(len(loader), args.max_steps))
    sched_main = sched_critic = None
    if args.lr_schedule == "cosine":
        sched_main = torch.optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=total_steps)
        sched_critic = torch.optim.lr_scheduler.CosineAnnealingLR(opt_critic, T_max=total_steps)

    metrics_log = CsvLogger(os.path.join(args.save_dir, "metrics.csv"),
                             ["step", "epoch", "tau", "loss", "critic_loss",
                              "L_R", "L_Uc", "L_Ua", "club_c", "club_a",
                              "rho_R_champo", "rho_U_champo", "rho_R_alma", "rho_U_alma",
                              "grad_norm_Uc", "grad_norm_Ua", "lr_main"])
    cka_log = CsvLogger(os.path.join(args.save_dir, "cka.csv"),
                         ["step", "epoch", "tau", "cka_R_vs_centroid",
                          "cka_U_champo_vs_centroid", "cka_U_alma_vs_centroid",
                          "cka_u_r_champo", "cka_u_r_alma",
                          "pw_cka_R_U_champo", "pw_cka_R_U_alma", "pw_cka_U_champo_U_alma",
                          "edge_R_Uchampo_ratio", "edge_R_Ualma_ratio", "edge_Uchampo_Ualma_ratio"])
    # Same numbers as metrics.csv/cka.csv, mirrored live -- `tensorboard --logdir
    # <save_dir>/tb` (or point it at the parent of several save_dirs to compare
    # runs side by side). CSVs stay the source of truth for post-hoc analysis
    # scripts (scratch_effrank.py etc.); this is just for watching training live.
    tb = SummaryWriter(os.path.join(args.save_dir, "tb"))

    grid = simplex_grid(spacing=args.grid_spacing)
    step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        for batch_idx, (champo_a, champo_b, alma_a, alma_b, subs) in enumerate(loader):
            champo_a, champo_b = champo_a.to(args.device), champo_b.to(args.device)
            alma_a, alma_b = alma_a.to(args.device), alma_b.to(args.device)

            tau = min(step / max(total_steps - 1, 1), 1.0)
            prefs = sample_preferences(grid, args.num_prefs)
            annealed = [anneal(p, tau=tau, Q=args.anneal_Q) for p in prefs]

            # 1) critic step (detached), pooled across the sampled preferences
            opt_critic.zero_grad()
            critic_loss = sum(model.critic_learning_loss(champo_a, alma_a, lam) for lam in annealed) / len(annealed)
            critic_loss.backward()
            opt_critic.step()
            if sched_critic is not None:
                sched_critic.step()

            # 2) encoder + heads step, averaged over the sampled preferences
            want_grad_norms = (step % args.log_every == 0)
            opt_main.zero_grad()
            total_loss = 0.0
            diag_agg = {}
            L_Uc_sum = L_Ua_sum = 0.0
            for lam in annealed:
                if want_grad_norms:
                    loss, diag, raw = model.preference_losses(
                        champo_a, champo_b, alma_a, alma_b, lam, return_raw=True)
                    L_Uc_sum = L_Uc_sum + raw["L_Uc"] / len(annealed)
                    L_Ua_sum = L_Ua_sum + raw["L_Ua"] / len(annealed)
                else:
                    loss, diag = model.preference_losses(champo_a, champo_b, alma_a, alma_b, lam)
                total_loss = total_loss + loss / len(annealed)
                for k, v in diag.items():
                    diag_agg[k] = diag_agg.get(k, 0.0) + v / len(annealed)

            grad_norms = {}
            if want_grad_norms:
                # Isolated d(L_Uc)/d(u_champo params), d(L_Ua)/d(u_alma params) --
                # retain_graph=True so this doesn't consume the graph the real
                # backward() below still needs. L_Uc/L_Ua already include their
                # own beta_club*clamp(club, min=0) term (see
                # preference_losses' return_raw docstring), so this is the
                # combined NT-Xent+CLUB signal actually reaching each head,
                # not the bare NT-Xent term alone. Cheap: autograd.grad with
                # explicit `inputs` only backprops as far as those leaf
                # params, never re-entering the frozen+LoRA encoder's own
                # (much more expensive) backward.
                g_uc = torch.autograd.grad(L_Uc_sum, list(model.u_champo.parameters()),
                                           retain_graph=True, allow_unused=True)
                g_ua = torch.autograd.grad(L_Ua_sum, list(model.u_alma.parameters()),
                                           retain_graph=True, allow_unused=True)
                grad_norms = dict(
                    grad_norm_Uc=torch.sqrt(sum((g ** 2).sum() for g in g_uc if g is not None)).item(),
                    grad_norm_Ua=torch.sqrt(sum((g ** 2).sum() for g in g_ua if g is not None)).item(),
                )

            total_loss.backward()
            opt_main.step()
            if sched_main is not None:
                sched_main.step()

            if step % args.log_every == 0:
                branch = model.mean_branch_norms()
                row = dict(step=step, epoch=epoch, tau=round(tau, 4),
                           loss=total_loss.item(), critic_loss=critic_loss.item(), **diag_agg, **branch,
                           **grad_norms, lr_main=opt_main.param_groups[0]["lr"])
                metrics_log.log(row)
                for k, v in row.items():
                    if k not in ("step", "epoch"):
                        tb.add_scalar(f"metrics/{k}", v, step)
                elapsed = time.time() - t0
                print(f"[step {step}/{total_steps}] epoch={epoch} batch={batch_idx} tau={tau:.3f} "
                      f"loss={total_loss.item():.4f} critic_loss={critic_loss.item():.4f} "
                      f"diag={ {k: round(v, 4) for k, v in diag_agg.items()} } "
                      f"branch={ {k: round(v, 4) for k, v in branch.items()} } "
                      f"grad_norms={ {k: round(v, 4) for k, v in grad_norms.items()} } "
                      f"elapsed={elapsed:.1f}s", flush=True)

            if step % args.cka_every == 0:
                cka = endpoint_cka_diagnostic(model, probe_champo_a, probe_alma_a)
                usc = unique_shared_cka(model, probe_champo_a, probe_alma_a, grid=grid)
                pw = endpoint_cka_pairwise(model, probe_champo_a, probe_alma_a)
                edges = all_edge_interpolation_ratios(model, probe_champo_a, probe_alma_a)
                cka_row = dict(
                    step=step, epoch=epoch, tau=round(tau, 4),
                    cka_R_vs_centroid=cka["cka_R_vs_centroid"],
                    cka_U_champo_vs_centroid=cka["cka_U_champo_vs_centroid"],
                    cka_U_alma_vs_centroid=cka["cka_U_alma_vs_centroid"],
                    cka_u_r_champo=usc["cka_u_r_champo"], cka_u_r_alma=usc["cka_u_r_alma"],
                    pw_cka_R_U_champo=pw["cka_R_U_champo"], pw_cka_R_U_alma=pw["cka_R_U_alma"],
                    pw_cka_U_champo_U_alma=pw["cka_U_champo_U_alma"],
                    edge_R_Uchampo_ratio=edges["R_U_champo"]["ratio"],
                    edge_R_Ualma_ratio=edges["R_U_alma"]["ratio"],
                    edge_Uchampo_Ualma_ratio=edges["U_champo_U_alma"]["ratio"],
                )
                cka_log.log(cka_row)
                for k, v in cka_row.items():
                    if k not in ("step", "epoch"):
                        tb.add_scalar(f"cka/{k}", v, step)
                print(f"[cka step {step}] endpoint={cka} u_r={usc} pairwise={pw}", flush=True)

            step += 1
            if args.max_steps is not None and batch_idx + 1 >= args.max_steps:
                break

    metrics_log.close()
    cka_log.close()
    tb.close()
    ckpt_path = os.path.join(args.save_dir, "steer_neuro_final.pt")
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"[done] saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
