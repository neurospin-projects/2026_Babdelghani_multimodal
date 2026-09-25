"""Consensus-lambda operating-point selection for STEER-neuro, ported from the
MOSEI probe (probe.py:_get_all_z + the consensus operating-point rule).

Per trained seed: compute z(lambda) at all 15 grid points on the validation
split, score each lambda via a FIXED internal 80/20 split of that validation
set (fit a logistic-regression probe on 80%, score AUC on the held-out 20%)
-- one 15-point curve per seed. The consensus lambda* is then chosen across
seeds' curves via simplex.py::consensus_lambda (epsilon = mean seed-to-seed
std; a seed accepts lambda if its regret <= epsilon; pick lambda with max
acceptance, tie-break worst-then-mean rank). Only after lambda* is fixed is
the reported number produced: the probe is refit on the FULL validation split
at lambda*, scored once on the held-out test split.

If only one checkpoint/seed is given, there is no seed-to-seed std to define
epsilon from, so this falls back to that seed's own argmax -- loudly flagged
as NOT validated consensus (train more seeds and re-run before trusting the
result as an operating point; training itself is never launched by this
script -- see feedback_no_auto_run / feedback_no_jeanzay_ssh).

At deployment nothing of this selection machinery remains: inference only
ever needs lambda* fixed above, one forward pass through the encoders at that
point, and the PaLoRA adapters could in principle be folded into the frozen
base weights for zero runtime overhead -- not implemented here, since this
script is about *choosing* lambda*, not about shipping a folded checkpoint.

Usage:
  python3 select_lambda.py --ckpt results/Output_steer_neuro/last_stage/steer_neuro_final.pt
  # multiple seeds of the *same* scope -> real consensus selection:
  python3 select_lambda.py --ckpt .../seed42/steer_neuro_final.pt \
    --ckpt .../seed43/steer_neuro_final.pt --ckpt .../seed44/steer_neuro_final.pt
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from eval_downstream import HAND_FILE, load_handedness
from extract_embeddings import PairedInferenceDataset, load_model
from paired_dataset import paired_subjects
from simplex import consensus_lambda, dim_readout, simplex_grid
from torch.utils.data import DataLoader

SPLIT_SEED = 0          # top-level val/test split -- fixed, persisted to disk
PROBE_SPLIT_SEED = 0     # internal val 80/20 split -- fixed, identical across seeds/scopes


def _labeled_paired_subjects(hand_file, limit=None):
    hand = load_handedness(hand_file)
    hand_by_subject = dict(zip(hand["subject"], hand["label"]))
    subjects = [s for s in paired_subjects(limit=None) if s.replace("sub-", "") in hand_by_subject]
    if limit is not None:
        subjects = subjects[:limit]
    y = np.array([hand_by_subject[s.replace("sub-", "")] for s in subjects])
    return subjects, y


def get_or_make_split(hand_file, split_dir, val_frac, limit=None):
    """Fixed subject-level val/test split of the labeled paired cohort,
    persisted so every seed/scope reuses the exact same partition. Test is
    only ever read once the final lambda* is fixed (see main())."""
    os.makedirs(split_dir, exist_ok=True)
    val_path = os.path.join(split_dir, "handedness_val_subjects.csv")
    test_path = os.path.join(split_dir, "handedness_test_subjects.csv")
    if os.path.exists(val_path) and os.path.exists(test_path) and limit is None:
        val_subjects = pd.read_csv(val_path)["subject"].tolist()
        test_subjects = pd.read_csv(test_path)["subject"].tolist()
        print(f"[split] reusing persisted split: {val_path} ({len(val_subjects)}), "
              f"{test_path} ({len(test_subjects)})")
        return val_subjects, test_subjects

    subjects, y = _labeled_paired_subjects(hand_file, limit=limit)
    val_subjects, test_subjects = train_test_split(
        subjects, train_size=val_frac, random_state=SPLIT_SEED, stratify=y)
    if limit is None:
        pd.DataFrame({"subject": val_subjects}).to_csv(val_path, index=False)
        pd.DataFrame({"subject": test_subjects}).to_csv(test_path, index=False)
        print(f"[split] created and saved: {val_path} ({len(val_subjects)}), "
              f"{test_path} ({len(test_subjects)})")
    else:
        print(f"[split] --limit set -- one-off split, not persisted "
              f"({len(val_subjects)} val / {len(test_subjects)} test)")
    return val_subjects, test_subjects


def compute_all_z(model, subjects, grid, total_dim, device, batch_size, num_workers=4):
    """z(lambda) for every grid point, one encoder pass per lambda per batch
    (SteerNeuroModel.embed_blocks conditions the encoder itself via
    forward_mix, so this can't be done with a single cached forward pass).
    Returns ({lam: (N, total_dim) ndarray}, subject_order)."""
    ds = PairedInferenceDataset(subjects)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    accum = {lam: [] for lam in grid}
    subject_order = []
    with torch.no_grad():
        for champo, alma, subs in loader:
            champo, alma = champo.to(device), alma.to(device)
            subject_order.extend(subs)
            for lam in grid:
                r, uc, ua = model.embed_blocks(champo, alma, *lam)
                z = dim_readout([r, uc, ua], list(lam), total_dim=total_dim)
                accum[lam].append(z.cpu().numpy())
    z_by_lam = {lam: np.concatenate(chunks, axis=0) for lam, chunks in accum.items()}
    return z_by_lam, subject_order


def _labels_for(subject_order, hand_file):
    hand = load_handedness(hand_file)
    hand_by_subject = dict(zip(hand["subject"], hand["label"]))
    return np.array([hand_by_subject[s.replace("sub-", "")] for s in subject_order])


def build_seed_curve(ckpt_path, val_subjects, grid, total_dim, hand_file, device, batch_size):
    """One 15-point AUC curve for this checkpoint's seed: an internal 80/20
    split of `val_subjects` (fixed across every seed/scope via
    PROBE_SPLIT_SEED), probe fit on 80%, scored on the held-out 20%."""
    model, train_args = load_model(ckpt_path, device)
    total_dim = total_dim or train_args["proj_dim"]  # "dim" readout width is D, not 3*D
    z_by_lam, subject_order = compute_all_z(model, val_subjects, grid, total_dim, device, batch_size)
    y = _labels_for(subject_order, hand_file)

    idx = np.arange(len(subject_order))
    probe_train, probe_holdout = train_test_split(
        idx, train_size=0.8, random_state=PROBE_SPLIT_SEED, stratify=y)

    curve = []
    for lam in grid:
        z = z_by_lam[lam]
        clf = LogisticRegression(max_iter=1000).fit(z[probe_train], y[probe_train])
        proba = clf.predict_proba(z[probe_holdout])[:, 1]
        curve.append(roc_auc_score(y[probe_holdout], proba))
    return curve, train_args["seed"], total_dim


def final_test_auc(ckpt_path, lam_star, val_subjects, test_subjects, total_dim, hand_file, device, batch_size):
    """Test touched exactly once: probe refit on the FULL validation split at
    the already-fixed lambda*, scored on the held-out test split."""
    model, train_args = load_model(ckpt_path, device)
    total_dim = total_dim or train_args["proj_dim"]  # "dim" readout width is D, not 3*D

    z_val, val_order = compute_all_z(model, val_subjects, [lam_star], total_dim, device, batch_size)
    z_test, test_order = compute_all_z(model, test_subjects, [lam_star], total_dim, device, batch_size)
    y_val = _labels_for(val_order, hand_file)
    y_test = _labels_for(test_order, hand_file)

    clf = LogisticRegression(max_iter=1000).fit(z_val[lam_star], y_val)
    proba = clf.predict_proba(z_test[lam_star])[:, 1]
    return roc_auc_score(y_test, proba), train_args["seed"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ckpt", action="append", required=True,
                   help="one steer_neuro_final.pt per seed, repeatable; all must share the same scope")
    p.add_argument("--hand_file", default=HAND_FILE)
    p.add_argument("--val_frac", type=float, default=0.8)
    p.add_argument("--total_dim", type=int, default=None, help="defaults to proj_dim, per checkpoint (the 'dim' readout width, see the method description sec:readout)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--split_dir", default=os.path.join(os.path.dirname(__file__), "results", "downstream_splits"))
    p.add_argument("--limit", type=int, default=None, help="subject-count cap, for quick dry runs (not persisted)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_csv", default="select_lambda_result.csv")
    return p.parse_args()


def main():
    args = parse_args()
    grid = simplex_grid()

    val_subjects, test_subjects = get_or_make_split(args.hand_file, args.split_dir, args.val_frac, limit=args.limit)
    print(f"[data] {len(val_subjects)} validation / {len(test_subjects)} test subjects (labeled, paired)")

    scores_by_seed = {}
    total_dim = args.total_dim
    scopes_seen = set()
    for ckpt_path in args.ckpt:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        scopes_seen.add(ckpt["args"]["scope"])
        del ckpt
    assert len(scopes_seen) == 1, f"all --ckpt must share the same scope, got {scopes_seen}"

    for ckpt_path in args.ckpt:
        curve, seed, total_dim = build_seed_curve(
            ckpt_path, val_subjects, grid, total_dim, args.hand_file, args.device, args.batch_size)
        scores_by_seed[seed] = curve
        print(f"[seed {seed}] curve built from {ckpt_path}")

    print(f"\n{'lambda (R, U_champo, U_alma)':30s} " + "  ".join(f"seed{s:>6}" for s in scores_by_seed))
    for i, lam in enumerate(grid):
        vals = "  ".join(f"{scores_by_seed[s][i]:9.4f}" for s in scores_by_seed)
        print(f"{str(tuple(round(x, 2) for x in lam)):30s} {vals}")

    n_seeds = len(scores_by_seed)
    if n_seeds == 1:
        only_seed = next(iter(scores_by_seed))
        curve = scores_by_seed[only_seed]
        lam_star = grid[int(np.argmax(curve))]
        print(f"\nWARNING: only 1 seed ({only_seed}) -- epsilon (seed-to-seed std) is undefined, "
              f"so this is NOT validated consensus selection, just that seed's own best point. "
              f"Train additional seeds of this scope and re-run before trusting this as an "
              f"operating point (training is not launched by this script).")
        info = None
    else:
        lam_star, info = consensus_lambda(scores_by_seed, grid)
        print(f"\n[consensus] lambda* = {lam_star}  "
              f"epsilon={info['epsilon']:.4f}  acceptance={info['acceptance_count']}/{n_seeds} seeds  "
              f"worst_rank={info['worst_rank']:.0f}  mean_rank={info['mean_rank']:.1f}")

    print(f"\n[final] refitting at lambda*={lam_star} on the FULL validation split, "
          f"scoring once on the held-out test split ({len(test_subjects)} subjects)")
    rows = []
    test_aucs = []
    for ckpt_path in args.ckpt:
        auc, seed = final_test_auc(ckpt_path, lam_star, val_subjects, test_subjects,
                                    total_dim, args.hand_file, args.device, args.batch_size)
        test_aucs.append(auc)
        print(f"  seed {seed}: test AUC = {auc:.4f}")
        rows.append(dict(seed=seed, lam_R=lam_star[0], lam_U_champo=lam_star[1],
                          lam_U_alma=lam_star[2], test_auc=auc,
                          **{f"val_curve_{i}": scores_by_seed[seed][i] for i in range(len(grid))}))
    if n_seeds > 1:
        print(f"\n[final] test AUC across {n_seeds} seeds: {np.mean(test_aucs):.4f} +/- {np.std(test_aucs):.4f}")

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\nsaved -> {args.out_csv}")


if __name__ == "__main__":
    main()
