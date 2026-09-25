"""Probe the 5 multimodal-SSL baselines (gmc/comm/clip/cross_self/factorcl) on
the same 3 downstream tasks STEER is evaluated on -- prematurity/cognition/
isomap. Unlike STEER, baselines have no preference grid: one embedding per
checkpoint, so this mirrors probe_external_tasks.py's --reference code path
(extract once, score once per task) rather than its grid-sweep machinery.

Usage:
  python3 probe_baselines.py --ckpt <baseline_final.pt> [--ckpt ...] --task all \\
      --out_csv results/external_probe_baselines.csv
  python3 probe_baselines.py --ckpt results/Output_steer_baselines/sc_left --task all \\
      --out_csv results/external_probe_baselines.csv   # directory: finds every baseline_final.pt
"""
import argparse
import glob
import os
import sys

import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from extract_baseline_embeddings import load_model
from probe_external_tasks import (
    TASK_COHORT, run_prematurity, run_cognition, run_isomap, _row, RESULT_FIELDS,
)
from paired_dataset import paired_subjects


def detect_side(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # baselines' bridge/proj heads read straight off ALMA's own encoder width,
    # so this is a cheap, reliable side signal without touching model internals.
    for k, v in ckpt["model_state_dict"].items():
        if "enc_alma.convnet.encoder.Linear.base.weight" in k:
            return "left" if v.shape[0] == 512 else "right"
    raise ValueError(f"couldn't detect side from {ckpt_path}")


def extract_one(ckpt_path, side, cohort, out_dir, device, batch_size, num_workers, limit=None):
    method = torch.load(ckpt_path, map_location="cpu", weights_only=False)["args"]["method"]
    out_path = os.path.join(out_dir, f"{method}.csv")
    if os.path.exists(out_path):
        n_rows = sum(1 for _ in open(out_path)) - 1
        print(f"[extract] {out_path}: already exists ({n_rows} subjects), skipping", flush=True)
        return out_path
    env = dict(os.environ, STEER_NEURO_SIDE=side, STEER_NEURO_COHORT=cohort)
    cmd = [sys.executable, os.path.join(HERE, "extract_baseline_embeddings.py"),
          "--ckpt", ckpt_path, "--out_dir", out_dir, "--batch_size", str(batch_size),
          "--num_workers", str(num_workers), "--device", device]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    print(f"[extract] method={method} side={side} cohort={cohort} -> {out_path}", flush=True)
    import subprocess
    subprocess.run(cmd, env=env, check=True)
    return out_path


def resolve_checkpoints(ckpt_args):
    resolved = []
    for c in ckpt_args:
        if os.path.isdir(c):
            found = sorted(glob.glob(os.path.join(c, "**", "baseline_final.pt"), recursive=True))
            resolved.extend(found)
        else:
            resolved.append(c)
    return resolved


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ckpt", action="append", required=True)
    p.add_argument("--task", choices=["prematurity", "cognition", "isomap", "all"], default="all")
    p.add_argument("--out_csv", default=os.path.join(HERE, "results", "external_probe_baselines.csv"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_permutations", type=int, default=200)
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--n_jobs", type=int, default=8)
    p.add_argument("--n_permutations_cognition", type=int, default=0,
                   help="opt-in permutation test for cognition's mean_r (same flag/meaning as "
                        "probe_external_tasks.py); 0 = off. Use the same value as the STEER probes "
                        "so cognition p-values are comparable across methods.")
    return p.parse_args()


def main():
    args = parse_args()
    tasks = ["prematurity", "cognition", "isomap"] if args.task == "all" else [args.task]
    ckpts = resolve_checkpoints(args.ckpt)
    print(f"{len(ckpts)} checkpoint(s) to probe", flush=True)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    done = set()
    header_written = [False]
    if os.path.exists(args.out_csv):
        prior = pd.read_csv(args.out_csv)
        done = set(zip(prior["config"], prior["task"]))
        header_written[0] = len(prior) > 0
        print(f"[resume] {len(done)} (config, task) rows already in {args.out_csv}", flush=True)
    out_f = open(args.out_csv, "a")

    def _emit(config, task, side, res, metric_name):
        if (config, task) in done:
            print(f"  [{task}] {config}: already done, skipping", flush=True)
            return
        row = _row(config, task, side, None, res, metric_name)
        pd.DataFrame([row]).to_csv(out_f, index=False, header=not header_written[0])
        out_f.flush()
        header_written[0] = True

    for ckpt_path in ckpts:
        _targs = torch.load(ckpt_path, map_location="cpu", weights_only=False)["args"]
        # config label carries the training seed: without it, seeds 43-46 of the same
        # method collide with seed 42 on (config, task) and get skipped as "already done"
        _nn = "_nonorm" if _targs["method"] == "factorcl" and _targs.get("factorcl_head_norm", 1) == 0 else ""
        _tag = f"_{_targs['tag']}" if _targs.get("tag") else ""
        method = f"{_targs['method']}{_tag or _nn}_seed{_targs['seed']}"
        side = detect_side(ckpt_path)
        print(f"\n=== {method} ({ckpt_path}, side={side}) ===", flush=True)
        for task in tasks:
            cohort = TASK_COHORT[task]
            out_dir = os.path.join(os.path.dirname(ckpt_path), f"embeddings_{cohort}")
            csv_path = extract_one(ckpt_path, side, cohort, out_dir, args.device,
                                   args.batch_size, args.num_workers, args.limit)
            emb_df = pd.read_csv(csv_path, dtype={"subject": str})
            if (method, task) in done:
                print(f"  [{task}] {method}: already done, skipping", flush=True)
                continue
            if task == "prematurity":
                res = run_prematurity(emb_df, n_permutations=args.n_permutations, n_jobs=args.n_jobs)
                metric_name = "auc_oof"
            elif task == "cognition":
                res = run_cognition(emb_df, n_seeds=args.n_seeds, n_jobs=args.n_jobs,
                                n_permutations=args.n_permutations_cognition)
                metric_name = "pearson_r"
            else:
                res = run_isomap(emb_df, side, n_seeds=args.n_seeds, n_jobs=args.n_jobs)
                metric_name = "pearson_r"
            print(f"  [{task}] {method}: {res}", flush=True)
            _emit(method, task, side, res, metric_name)

    out_f.close()
    print(f"\nsaved -> {args.out_csv}")


if __name__ == "__main__":
    main()
