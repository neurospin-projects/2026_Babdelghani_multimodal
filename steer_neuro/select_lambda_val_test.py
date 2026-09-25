"""Rigorous consensus-lambda selection: val/test split, never touching test
until lambda* is already fixed.

Everything in probe_external_tasks.py (and today's consensus_lambda run) picks
lambda* using the SAME cross-validated scores that then get reported as the
final number -- real protection (each score is itself CV'd; consensus
requires agreement across independent seeds) but not full protection against
optimism leaking into lambda selection. This script splits each task's
downstream cohort ONCE into a val portion (used for every seed's 15-point
curve + consensus_lambda) and a test portion (scored exactly once, at
lambda* only, after lambda* is already fixed) -- the protocol the original
select_lambda.py plan called for.

Reuses everything from probe_external_tasks.py: run_prematurity/run_cognition
/run_isomap (now subjects_filter-aware), load_grid, simplex.consensus_lambda/
simplex_grid. Does NOT re-extract embeddings -- operates on whatever
embeddings_{cohort}_grid/ directories already exist on disk from prior probe
runs (extraction is checkpoint-only, not val/test-split-dependent, so the
existing caches are exactly what's needed here).

Usage:
  python3 select_lambda_val_test.py --ckpt <seed42 ckpt> --ckpt <seed43 ckpt> \\
      --ckpt <seed44 ckpt> --task prematurity --out_csv results/lambda_star.csv
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from probe_external_tasks import (
    HERE, TASK_COHORT, detect_side, load_grid,
    run_prematurity, run_cognition, run_isomap,
    PREMATURITY_LABELS, COG_SUB_LIST, ISOMAP_LABELS, ISOMAP_TRAIN_VAL_SPLIT, ISOMAP_TEST_SPLIT,
    _hcp_cog_meta,
)
from simplex import simplex_grid, consensus_lambda

SPLIT_DIR = os.path.join(HERE, "results", "downstream_splits")


def _task_labeled_population(task, side, any_grid_csv):
    """Subjects with BOTH an extracted embedding (any_grid_csv's own subject
    column) AND a usable label for this task -- the population the val/test
    split is carved from. Deliberately light: just enough merging to know who
    has a label, not the full CV/permutation pipeline in run_*()."""
    emb_subs = set(pd.read_csv(any_grid_csv, usecols=["subject"], dtype={"subject": str})["subject"])
    if task == "prematurity":
        labels_df = pd.read_csv(PREMATURITY_LABELS, low_memory=False)
        labels_df["src_subject_id"] = labels_df["src_subject_id"].str.replace("_", "")
        return emb_subs & set(labels_df["src_subject_id"])
    elif task == "cognition":
        meta, _ = _hcp_cog_meta()
        if meta is None:
            return set()
        return emb_subs & set(meta["Subject"])
    else:  # isomap
        labels = pd.read_csv(ISOMAP_LABELS[side], dtype={"Subject": str})

        def _read_split(path):
            with open(path) as f:
                return {line.strip().strip('"') for line in f if line.strip()}
        pop_subs = _read_split(ISOMAP_TRAIN_VAL_SPLIT[side]) | _read_split(ISOMAP_TEST_SPLIT[side])
        return emb_subs & set(labels["Subject"]) & pop_subs


def get_or_create_split(task, side, any_grid_csv, val_frac=0.8, seed=0):
    val_path = os.path.join(SPLIT_DIR, f"{task}_{side}_val_subjects.csv")
    test_path = os.path.join(SPLIT_DIR, f"{task}_{side}_test_subjects.csv")
    if os.path.exists(val_path) and os.path.exists(test_path):
        val_subs = set(pd.read_csv(val_path, dtype=str)["subject"])
        test_subs = set(pd.read_csv(test_path, dtype=str)["subject"])
        print(f"[split] {task}/{side}: reusing persisted split "
              f"({len(val_subs)} val / {len(test_subs)} test)", flush=True)
        return val_subs, test_subs

    pop = sorted(_task_labeled_population(task, side, any_grid_csv))
    val_subs, test_subs = train_test_split(pop, train_size=val_frac, random_state=seed)
    os.makedirs(SPLIT_DIR, exist_ok=True)
    pd.DataFrame({"subject": val_subs}).to_csv(val_path, index=False)
    pd.DataFrame({"subject": test_subs}).to_csv(test_path, index=False)
    print(f"[split] {task}/{side}: NEW split created and persisted "
          f"({len(val_subs)} val / {len(test_subs)} test, from {len(pop)} labeled+embedded subjects)", flush=True)
    return set(val_subs), set(test_subs)


def _score(task, emb_df, side, subjects_filter, n_permutations, n_jobs, n_seeds):
    if task == "prematurity":
        res = run_prematurity(emb_df, n_permutations=n_permutations, n_jobs=n_jobs, subjects_filter=subjects_filter)
    elif task == "cognition":
        res = run_cognition(emb_df, n_seeds=n_seeds, n_jobs=n_jobs, subjects_filter=subjects_filter)
    else:
        res = run_isomap(emb_df, side, n_seeds=n_seeds, n_jobs=n_jobs, subjects_filter=subjects_filter)
    if res.get("skipped"):
        return float("nan")
    return res.get("auc_oof", res.get("mean_r"))


def resolve_checkpoints(ckpt_args):
    resolved = []
    for c in ckpt_args:
        if os.path.isdir(c):
            found = sorted(glob.glob(os.path.join(c, "**", "steer_neuro_final.pt"), recursive=True))
            resolved.extend(found)
        else:
            resolved.append(c)
    return resolved


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ckpt", action="append", required=True,
                   help="repeatable, one per seed (or a directory containing several seeds' checkpoints)")
    p.add_argument("--task", choices=["prematurity", "cognition", "isomap"], required=True)
    p.add_argument("--val_frac", type=float, default=0.8)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--n_permutations_val", type=int, default=100,
                   help="reduced rigor for the 15-point val curve (speed matters more than precision here -- it's only for RANKING lambda points, not the reported number)")
    p.add_argument("--n_seeds_val", type=int, default=5)
    p.add_argument("--n_permutations_test", type=int, default=1000,
                   help="full rigor for the once-only test-set score at lambda*")
    p.add_argument("--n_seeds_test", type=int, default=60)
    p.add_argument("--n_jobs", type=int, default=8)
    p.add_argument("--out_csv", default=os.path.join(HERE, "results", "lambda_star.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    ckpts = resolve_checkpoints(args.ckpt)
    print(f"{len(ckpts)} checkpoint(s) (one per seed)", flush=True)

    cohort = TASK_COHORT[args.task]
    grid = simplex_grid()
    grid_names = ["_".join(f"{x:.2f}" for x in lam) for lam in grid]

    seed_grids = {}  # model_seed -> {lam_name: emb_df}
    side = None
    for ckpt_path in ckpts:
        import torch
        train_seed = torch.load(ckpt_path, map_location="cpu", weights_only=False)["args"]["seed"]
        side = detect_side(ckpt_path)
        grid_dir = os.path.join(os.path.dirname(ckpt_path), f"embeddings_{cohort}_grid")
        if not os.path.isdir(grid_dir):
            raise SystemExit(f"no cached {grid_dir} for {ckpt_path} -- run probe_external_tasks.py on it first")
        seed_grids[train_seed] = load_grid(grid_dir)

    any_csv = glob.glob(os.path.join(os.path.dirname(ckpts[0]), f"embeddings_{cohort}_grid", "*.csv"))[0]
    val_subs, test_subs = get_or_create_split(args.task, side, any_csv, args.val_frac, args.split_seed)

    print(f"\n[val] scoring all {len(grid_names)} grid points x {len(seed_grids)} seeds "
          f"(reduced rigor: n_permutations={args.n_permutations_val}, n_seeds={args.n_seeds_val})...", flush=True)
    scores_by_seed = {}
    for train_seed, grid_map in seed_grids.items():
        curve = []
        for name in grid_names:
            s = _score(args.task, grid_map[name], side, val_subs,
                       args.n_permutations_val, args.n_jobs, args.n_seeds_val)
            curve.append(s)
        scores_by_seed[train_seed] = curve
        print(f"  seed {train_seed}: best val={max(curve):.4f} @ {grid_names[int(np.nanargmax(curve))]}", flush=True)

    if len(scores_by_seed) == 1:
        only_seed = next(iter(scores_by_seed))
        curve = scores_by_seed[only_seed]
        lam_star = grid[int(np.nanargmax(curve))]
        info = dict(epsilon=None, acceptance_count=1, n_seeds=1, worst_rank=1.0, mean_rank=1.0)
        print("\n[WARNING] only 1 seed -- this is that seed's own argmax, NOT a validated consensus.", flush=True)
    else:
        lam_star, info = consensus_lambda(scores_by_seed, grid)
    lam_star_name = "_".join(f"{x:.2f}" for x in lam_star)
    print(f"\n[consensus] lambda* = {tuple(round(x,2) for x in lam_star)}  "
          f"({lam_star_name})  info={info}", flush=True)

    print(f"\n[test] scoring lambda* ONCE per seed on the held-out test split "
          f"(full rigor: n_permutations={args.n_permutations_test}, n_seeds={args.n_seeds_test})...", flush=True)
    test_scores = {}
    for train_seed, grid_map in seed_grids.items():
        s = _score(args.task, grid_map[lam_star_name], side, test_subs,
                   args.n_permutations_test, args.n_jobs, args.n_seeds_test)
        test_scores[train_seed] = s
        print(f"  seed {train_seed}: test score={s:.4f}", flush=True)

    vals = np.array(list(test_scores.values()))
    print(f"\n[FINAL] {args.task} @ lambda*={lam_star_name}: "
          f"mean={np.nanmean(vals):.4f} std={np.nanstd(vals):.4f} (n_seeds={len(vals)})", flush=True)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    rows = [dict(task=args.task, seed=s, lambda_star=lam_star_name, test_score=v) for s, v in test_scores.items()]
    header = not os.path.exists(args.out_csv)
    pd.DataFrame(rows).to_csv(args.out_csv, mode="a", index=False, header=header)
    print(f"saved -> {args.out_csv}")


if __name__ == "__main__":
    main()
