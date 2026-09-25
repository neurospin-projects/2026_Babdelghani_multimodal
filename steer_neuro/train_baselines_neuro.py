"""Baseline multimodal-SSL training loop: same Champollion x ALMA setup,
same frozen-encoder + LoRA regime, same data pipeline as train_steer_neuro.py
-- only the objective differs, one of {gmc, comm, clip, cross_self, factorcl}
(see model_baselines.py's own docstring for the full design rationale and
dimensionality-matching budget).

No preference/lambda sampling here -- these methods have no preference axis,
so each step is a single loss computation (or, for factorcl, a critic-fit
step + a main step, mirroring train_steer_neuro.py's own opt_main/opt_critic
split for its CLUB critics).

Usage (smoke test):
  python3 train_baselines_neuro.py --method gmc --scope bottleneck --epochs 1 --limit 8 --batch_size 2

Usage (full run, hand off rather than launching directly):
  python3 train_baselines_neuro.py --method factorcl --scope last_stage --rank_ratio 0.15 \\
    --alpha_champo 4 --alpha_alma 0.25 --epochs 20 --batch_size 16 --save_dir <out>
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_baselines import BaselineModel, METHODS
from paired_dataset import PairedSkeletonDataset, paired_subjects
from train_steer_neuro import CsvLogger  # same tiny CSV-logging helper, no need to duplicate
from wrap_lora import SCOPES


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--scope", choices=SCOPES, default="last_stage")
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--rank_ratio", type=float, default=0.15,
                   help="SAME meaning/value as train_steer_neuro.py's --rank_ratio -- pass the "
                        "identical value used for the STEER config being compared against, so "
                        "adapter capacity is matched, not just architecture.")
    p.add_argument("--alpha_champo", type=float, default=4.0)
    p.add_argument("--alpha_alma", type=float, default=0.25)
    p.add_argument("--total_dim", type=int, default=64,
                   help="gmc/comm/clip/cross_self readout width (factorcl uses factorcl_head_dim*10 instead)")
    p.add_argument("--factorcl_head_dim", type=int, default=7,
                   help="per-head width for factorcl's 10 heads (10*7=70, the closest clean "
                        "match to the 64-dim budget every other method uses -- see "
                        "model_baselines.py's own docstring)")
    p.add_argument("--factorcl_head_norm", type=int, choices=[0, 1], default=1,
                   help="factorcl only: 1 = L2-normalise the 10 heads (original port, default so past "
                        "runs reproduce); 0 = raw heads, as in the official FactorCL mlp_head")
    p.add_argument("--tag", default="", help="free-text run label (factorcl variant screens); stored in args.json/ckpt, used by probe_baselines.py for the config label")
    p.add_argument("--grad_clip", type=float, default=1.0, help="max grad norm on adapters+heads; 0 disables")
    p.add_argument("--club_hidden", type=int, default=64)
    p.add_argument("--club_layers", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--ssl_scale", type=float, default=1.0, help="cross_self only")
    p.add_argument("--clip_learned_temp", action="store_true", default=True)
    p.add_argument("--no_clip_learned_temp", dest="clip_learned_temp", action="store_false")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--critic_lr", type=float, default=1e-3, help="factorcl's CLUB critics only")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="subject-count cap on the TRAIN split, for smoke tests")
    p.add_argument("--probe_limit", type=int, default=16, help="held-out subjects excluded from training (unused for now, kept for parity)")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=None, help="hard cap on optimizer steps, for smoke tests")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", required=True)
    return p.parse_args()


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
    print(f"[data] {len(train_subjects)} train subjects (method={args.method})", flush=True)

    dataset = PairedSkeletonDataset(subjects=train_subjects)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        persistent_workers=args.num_workers > 0)

    model = BaselineModel(method=args.method, scope=args.scope, rank=args.rank,
                          rank_ratio=args.rank_ratio, alpha_champo=args.alpha_champo,
                          alpha_alma=args.alpha_alma, total_dim=args.total_dim,
                          factorcl_head_dim=args.factorcl_head_dim,
                          club_hidden=args.club_hidden, club_layers=args.club_layers,
                          temperature=args.temperature, ssl_scale=args.ssl_scale,
                          clip_learned_temp=args.clip_learned_temp,
                          factorcl_head_norm=bool(args.factorcl_head_norm)).to(args.device)

    opt_main = torch.optim.AdamW(model.adapter_and_head_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    critic_params = list(model.critic_parameters())
    opt_critic = torch.optim.AdamW(critic_params, lr=args.critic_lr) if critic_params else None
    n_main = sum(p.numel() for p in model.adapter_and_head_parameters())
    n_critic = sum(p.numel() for p in critic_params)
    print(f"[model] method={args.method} scope={args.scope} rank_ratio={args.rank_ratio} "
          f"main_params={n_main:,} critic_params={n_critic:,}", flush=True)

    fieldnames = ["step", "epoch", "loss"] + (
        ["l_r", "l_cl", "l_u_c", "l_u_a", "l_cond", "l_ccl", "critic_loss"] if args.method == "factorcl"
        else ["logit_scale"] if args.method == "clip" else [])
    metrics_log = CsvLogger(os.path.join(args.save_dir, "metrics.csv"), fieldnames)
    tb = SummaryWriter(os.path.join(args.save_dir, "tb"))

    total_steps = args.epochs * (len(loader) if args.max_steps is None else min(len(loader), args.max_steps))
    step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        for batch_idx, (champo_a, champo_b, alma_a, alma_b, subs) in enumerate(loader):
            champo_a, champo_b = champo_a.to(args.device), champo_b.to(args.device)
            alma_a, alma_b = alma_a.to(args.device), alma_b.to(args.device)

            # 1) critic step (detached) -- no-op unless method == "factorcl"
            #    (model.critic_parameters() is then empty, opt_critic is None)
            critic_loss_val = None
            if opt_critic is not None:
                opt_critic.zero_grad()
                critic_loss = model.critic_learning_loss(champo_a, champo_b, alma_a, alma_b)
                critic_loss.backward()
                opt_critic.step()
                critic_loss_val = critic_loss.item()

            # 2) main step
            opt_main.zero_grad()
            loss, diag = model.compute_loss(champo_a, champo_b, alma_a, alma_b)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(model.adapter_and_head_parameters()), args.grad_clip)
            opt_main.step()

            if step % args.log_every == 0:
                row = dict(step=step, epoch=epoch, **diag)
                if critic_loss_val is not None:
                    row["critic_loss"] = critic_loss_val
                metrics_log.log(row)
                for k, v in row.items():
                    if k not in ("step", "epoch"):
                        tb.add_scalar(f"metrics/{k}", v, step)
                elapsed = time.time() - t0
                print(f"[step {step}/{total_steps}] epoch={epoch} {diag} elapsed={elapsed:.1f}s", flush=True)

            step += 1
            if args.max_steps is not None and batch_idx + 1 >= args.max_steps:
                break

    metrics_log.close()
    tb.close()
    torch.save(dict(model_state_dict=model.state_dict(), args=vars(args)),
              os.path.join(args.save_dir, "baseline_final.pt"))
    print(f"[done] saved {os.path.join(args.save_dir, 'baseline_final.pt')}", flush=True)


if __name__ == "__main__":
    main()
