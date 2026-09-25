"""Compare ALMA's own (residualized) embeddings against STEER-neuro embeddings
on a real downstream label, reporting BOTH:
  - the existing significance test (statsmodels Logit: likelihood-ratio
    p-value + McFadden pseudo-R2), same methodology as
    DeepLearning_Tracto/ALMA_UKB_HAND_LOGIT.py, for direct comparability, and
  - a proper cross-validated AUC (repeated StratifiedKFold, logistic
    regression) -- the "is this actually predictive out of sample, in a
    publishable form" metric the existing ALMA scripts don't report (they
    fit in-sample on the full data and stop at the p-value).

Default label is UKB handedness (right=0/left=1, ambidextrous excluded),
matching ALMA_UKB_HAND_LOGIT.py exactly. Point this at a different label file
(with columns `subject`, some outcome) to reuse for other downstream tasks.

Usage:
  python3 extract_embeddings.py --ckpt <ckpt> --out_dir <dir>   # first
  python3 eval_downstream.py --steer_dir <dir> [--alma_res_csv ...] \
    [--out_csv downstream_comparison.csv]
"""
import argparse
import glob
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from paired_dataset import champo_subjects

# All three are neurospin-only paths by default -- env-var overridable (same
# STEER_NEURO_* pattern paired_dataset.py already uses for the champo/alma
# data paths) so this one script works unmodified on Jean Zay too: point the
# env var at a copy synced up there instead of maintaining a separate version.
HAND_FILE = os.environ.get(
    "STEER_NEURO_HAND_FILE",
    "",
)
ALMA_RES_DEFAULT = os.environ.get(
    "STEER_NEURO_ALMA_RES_CSV",
    "",
)
# configs/dataset/julien/MICCAI_2024/evaluation/SC-sylv_right_interruption_UKB40.yaml points
# subject_labels_file at interrupted_SC_labels.csv (an explicit 0/1 column) -- that file no
# longer exists on disk (confirmed, not just missing from one directory listing). What does
# exist is this bare positive-only list (414 subjects); absence from it is treated as
# confirmed negative, the same positive-list convention load_handedness() below uses.
INTERRUPTED_SC_FILE = os.environ.get(
    "STEER_NEURO_INTERRUPTED_SC_FILE",
    "",
)


def load_handedness(hand_file=HAND_FILE):
    """Same recoding as ALMA_UKB_HAND_LOGIT.py: right=0, left=1, ambidextrous excluded."""
    df = pd.read_csv(hand_file, usecols=["participant.eid", "participant.p1707_i0"])
    df = df.rename(columns={"participant.eid": "subject", "participant.p1707_i0": "label"})
    df["subject"] = df["subject"].astype(str)
    df = df[df["label"].isin([1.0, 2.0])].copy()
    df["label"] = (df["label"] == 2.0).astype(int)
    return df.reset_index(drop=True)


def load_interruption(all_subjects, interrupted_file=INTERRUPTED_SC_FILE):
    """Interruption_SC_right, binary. `all_subjects`: bare-ID universe to label
    (e.g. paired_dataset.champo_subjects(), "sub-" stripped) -- every one of
    them not in the positive list is labeled 0."""
    interrupted = set(pd.read_csv(interrupted_file)["Subject"].astype(str)
                       .str.replace("sub-", "", regex=False))
    df = pd.DataFrame({"subject": [s.replace("sub-", "") for s in all_subjects]})
    df["label"] = df["subject"].isin(interrupted).astype(int)
    return df.reset_index(drop=True)


def run_logit(X, y):
    """In-sample statsmodels Logit -- llr_pvalue + McFadden pseudo-R2.

    lbfgs/bfgs first, not newton: newton-raphson computes the full 193x193
    Hessian from all ~37k rows EVERY iteration (~1.4B FLOPs/iter) and
    statsmodels doesn't raise on non-convergence (just warns), so a
    non-converging newton fit silently burns all `maxiter` iterations before
    this function can even try the next method -- that was the actual
    multi-hour-runtime bug (not the data size). lbfgs/bfgs are quasi-Newton
    (no exact Hessian per step) and are usually far faster for the same MLE.

    BUT: statsmodels' own `res.mle_retvals['converged']` flag is not
    trustworthy on its own -- observed lbfgs report converged=True while
    landing at a likelihood far WORSE than the null model (extreme class
    imbalance flattens the gradient almost everywhere except near the true
    optimum), and bfgs report converged=False while actually finding the
    right answer. The only check that's actually valid: a real MLE can never
    do worse than the null model (llf >= llnull, i.e. pseudo_r2 >= 0) since
    the null is a nested special case of the same family. Reject and fall
    through to the next method if that's violated."""
    Xc = sm.add_constant(X, has_constant="add")
    for method, maxiter in (("lbfgs", 200), ("bfgs", 200), ("newton", 35)):
        try:
            res = sm.Logit(y, Xc).fit(method=method, disp=False, maxiter=maxiter)
            if res.llf < res.llnull - 1e-6:
                continue
            return res.llr_pvalue, res.prsquared
        except Exception:
            continue
    return np.nan, np.nan


def run_cv_auc(X, y, n_seeds=5, n_splits=5):
    """Repeated stratified k-fold logistic regression AUC -- the
    out-of-sample, publishable metric. Features standardised per fold (fit
    on train only) to help convergence; statsmodels above is left unscaled
    to match the existing ALMA script's own convention."""
    aucs = []
    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            scaler = StandardScaler().fit(X[tr])
            Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
            clf = LogisticRegression(max_iter=1000).fit(Xtr, y[tr])
            proba = clf.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(y[te], proba))
    return float(np.mean(aucs)), float(np.std(aucs)), len(aucs)


def evaluate_source(name, emb_df, label_df, n_seeds, n_splits, skip_logit=False):
    merged = emb_df.merge(label_df, on="subject", how="inner").dropna()
    feat_cols = [c for c in merged.columns if c not in ("subject", "label")]
    X = merged[feat_cols].values.astype(float)
    y = merged["label"].values.astype(float)
    n = len(y)
    if n < 50:
        return dict(source=name, n=n, llr_pvalue=np.nan, pseudo_r2=np.nan,
                    auc_mean=np.nan, auc_std=np.nan, n_folds=0)
    llr_p, pr2 = (np.nan, np.nan) if skip_logit else run_logit(X, y)
    auc_mean, auc_std, n_folds = run_cv_auc(X, y, n_seeds=n_seeds, n_splits=n_splits)
    return dict(source=name, n=n, llr_pvalue=llr_p, pseudo_r2=pr2,
                auc_mean=auc_mean, auc_std=auc_std, n_folds=n_folds)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--alma_res_csv", default=ALMA_RES_DEFAULT)
    p.add_argument("--no_default_alma_baseline", action="store_true",
                   help="skip the automatic ALMA_baseline source (used when ALMA_baseline is "
                        "being dispatched as its own separate job elsewhere, e.g. per-config "
                        "parallel dispatch on Jean Zay, so it isn't redundantly recomputed by "
                        "every other job too)")
    p.add_argument("--steer_dir", action="append", default=[],
                   help="LABEL=path, repeatable -- each dir holds steer_neuro_ukb_lam-*.csv "
                        "from extract_embeddings.py for one checkpoint (e.g. "
                        "--steer_dir yesterday_bottleneck=/tmp/embed_y_bn --steer_dir "
                        "today_bottleneck=/tmp/embed_t_bn). LABEL prefixes the resulting "
                        "source names so multiple checkpoints/runs can be compared side by side.")
    p.add_argument("--csv", action="append", default=[],
                   help="LABEL=path, repeatable -- a single embeddings CSV (subject + feature "
                        "columns) added as its own source, for anything not shaped like the "
                        "steer_neuro_ukb_lam-* family (e.g. --csv "
                        "Champollion_reference=results/reference_embeddings/champollion_reference.csv)")
    p.add_argument("--label", choices=["handedness", "interruption"], default="handedness",
                   help="downstream label to evaluate against")
    p.add_argument("--hand_file", default=HAND_FILE, help="only used with --label handedness")
    p.add_argument("--interrupted_file", default=INTERRUPTED_SC_FILE,
                   help="only used with --label interruption")
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--skip_logit", action="store_true",
                   help="skip the slow statsmodels Logit p-value/pseudo-R2 fit, report AUC only "
                        "-- for fast screening across many sources (e.g. all 15 grid points x "
                        "many checkpoints), not for the final reported numbers")
    p.add_argument("--out_csv", default="downstream_comparison.csv")
    return p.parse_args()


def main():
    args = parse_args()
    if args.label == "handedness":
        label_df = load_handedness(args.hand_file)
        print(f"handedness: {len(label_df)} subjects (right={(label_df.label == 0).sum()}, "
              f"left={(label_df.label == 1).sum()})")
    else:
        label_df = load_interruption(champo_subjects(), args.interrupted_file)
        print(f"interruption: {len(label_df)} subjects (interrupted={(label_df.label == 1).sum()}, "
              f"not={(label_df.label == 0).sum()})")

    sources = {}
    if args.no_default_alma_baseline:
        pass
    elif os.path.exists(args.alma_res_csv):
        df = pd.read_csv(args.alma_res_csv)
        df["subject"] = df["subject"].astype(str)
        sources["ALMA_baseline"] = df
    else:
        print(f"WARNING: ALMA baseline not found at {args.alma_res_csv}")

    for spec in args.steer_dir:
        label, _, steer_dir = spec.partition("=")
        if not steer_dir:
            label, steer_dir = os.path.basename(os.path.normpath(spec)), spec
        steer_files = sorted(glob.glob(os.path.join(steer_dir, "steer_neuro_ukb_lam-*.csv")))
        if not steer_files:
            print(f"WARNING: no steer_neuro_ukb_lam-*.csv found in {steer_dir} "
                  f"(label={label}) -- run extract_embeddings.py first")
        for f in steer_files:
            point = os.path.basename(f)[len("steer_neuro_ukb_lam-"):-len(".csv")]
            name = f"{label}_{point}"
            df = pd.read_csv(f)
            df["subject"] = df["subject"].astype(str)
            sources[name] = df

    for spec in args.csv:
        label, _, csv_path = spec.partition("=")
        if not csv_path:
            label, csv_path = os.path.splitext(os.path.basename(spec))[0], spec
        if not os.path.exists(csv_path):
            print(f"WARNING: --csv {label}={csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df["subject"] = df["subject"].astype(str)
        sources[label] = df

    if not sources:
        print("Nothing to evaluate.")
        return

    # Written incrementally (one row per source, flushed immediately) so a
    # kill partway through never loses already-computed sources. Resumable:
    # if --out_csv already has rows (from a prior killed run), those source
    # names are skipped rather than recomputed.
    n_tests = len(sources)
    rows = []
    done = set()
    if os.path.exists(args.out_csv):
        prior = pd.read_csv(args.out_csv)
        done = set(prior["source"])
        rows = prior.to_dict("records")
        print(f"[resume] {len(done)} sources already done in {args.out_csv}, skipping those", flush=True)

    with open(args.out_csv, "a" if done else "w") as f:
        header_written = bool(done)
        for i, (name, df) in enumerate(sources.items()):
            if name in done:
                continue
            r = evaluate_source(name, df, label_df, args.n_seeds, args.n_splits, skip_logit=args.skip_logit)
            r["p_bonf"] = min(r["llr_pvalue"] * n_tests, 1.0) if not np.isnan(r["llr_pvalue"]) else np.nan
            rows.append(r)
            row_df = pd.DataFrame([r])
            row_df.to_csv(f, index=False, header=not header_written)
            header_written = True
            f.flush()
            print(f"[{i + 1}/{n_tests}] {name}: n={r['n']} auc={r['auc_mean']:.4f} "
                  f"p_bonf={r['p_bonf']:.2e}", flush=True)

    out = pd.DataFrame(rows)

    print(f"\n{'source':20s} {'n':>6s}  {'llr_p':>10s} {'p_bonf':>10s}  {'pseudo_R2':>9s}  "
          f"{'AUC':>16s}  n_folds")
    for _, row in out.sort_values("auc_mean", ascending=False).iterrows():
        sig = "***" if row["p_bonf"] < 0.001 else ("**" if row["p_bonf"] < 0.01
              else ("*" if row["p_bonf"] < 0.05 else ""))
        print(f"{row['source']:20s} {int(row['n']):6d}  {row['llr_pvalue']:10.2e} "
              f"{row['p_bonf']:10.2e}{sig:3s} {row['pseudo_r2']:9.4f}  "
              f"{row['auc_mean']:.4f} +/- {row['auc_std']:.4f}  {int(row['n_folds'])}")

    print(f"\nsaved -> {args.out_csv} (written incrementally, one row per source as it completed)")


if __name__ == "__main__":
    main()
