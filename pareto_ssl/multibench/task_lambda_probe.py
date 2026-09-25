#!/usr/bin/env python
"""
High-precision task-vs-λ probe: 5-fold CV + AUC selection + multiple probe seeds,
followed by a blocked-permutation test on task-explained variance.

WHY THIS EXISTS
---------------
The standard probe scored each λ on a single 80/20 split of the validation set —
373 samples for mosei_multitask. At ~60% accuracy the binomial standard error is
sqrt(.6*.4/373) = 2.5 points, while the entire λ curve spans 1.6-2.9 points. The
measurement noise was larger than the effect, so every "flat curve" verdict may
have described the instrument rather than the model.

Three changes, all variance reduction — none of them biases the result toward
finding structure:

  1. 5-fold stratified CV over the FULL validation set (not one 80/20 split), so
     every sample is scored once per repeat instead of 20% of them.
  2. AUC instead of thresholded accuracy. Threshold-free, uses the whole ranking,
     and immune to the class-imbalance failure that made `disgust` look
     unlearnable (it scored 51% balanced accuracy under an unweighted classifier
     and 63% / AUC 67 once weighted).
  3. R independent probe seeds (different CV partitions), averaged.

Together these cut per-λ noise by roughly 2-3x.

STATISTICS
----------
Each (task, training-seed) contributes one selected preference λ* — a scalar for a
1-D λ, a 3-vector on the simplex. Treating those as points in preference space:

  SS_total   = Σ_i ||x_i - grand_mean||²
  SS_within  = Σ_tasks Σ_{i∈task} ||x_i - task_mean||²
  SS_between = SS_total - SS_within

  R²    = SS_between / SS_total          fraction of λ* variance explained by TASK
  ratio = (SS_between/(k-1)) / (SS_within/(n-k))     F-like between/within ratio

This is a multivariate one-way ANOVA (PERMANOVA), so it handles the simplex
without collapsing it to one coordinate.

BLOCKED PERMUTATION. Training seed is a nuisance factor: all tasks within a seed
share an encoder, so their λ* are correlated. Permuting task labels globally would
break that structure and give an anticonservative p. Labels are therefore shuffled
WITHIN each training seed (seed = block), which is the exact null of "task identity
carries no information, given the run."

  p = fraction of blocked permutations reaching the observed R²

Note the floor: with S training seeds and k tasks there are (k!)^S distinct
relabelings, but p is also bounded below by 1/(n_perm+1).

USAGE
  python pareto_ssl/multibench/task_lambda_probe.py \
      --runs "$SCRATCH/Output/multibench/mosei_multitask/approach4/*filmsimplex*seed*" \
      --dataset mosei_multitask --approach 4 --film_mode simplex \
      --probe_seeds 5 --folds 5 --save_json $SCRATCH/Output/task_lambda_probe.json
"""
import argparse
import glob
import math
import json
import os
import random
import statistics as st
import sys
import warnings
from collections import defaultdict

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pareto_ssl.multibench import probe as P
from pareto_ssl.multibench.multitask_registry import (
    is_multitask_dataset, usable_tasks, label_names, dataset_class,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
LR_CS = (0.01, 0.1, 1.0)


# λ evaluation

# Selection/reporting metric. AUC is threshold-free and prior-insensitive, which
# is what a cross-TASK comparison needs when positive rates range from 18%
# (disgust) to 54% (happy). Balanced accuracy is the interpretable counterpart --
# the mean of sensitivity and specificity, so a majority-class classifier scores
# 50 on every task regardless of prevalence -- and is what the MOSEI emotion
# literature reports. Whichever is chosen is used for BOTH the lambda selection
# and the reported number, and for the baselines identically; mixing them would
# make the columns incomparable.
_METRIC = ["auc"]


def _score(y_true, prob, pred):
    m = _METRIC[0]
    if m == "auc":
        return roc_auc_score(y_true, prob)
    if m == "balanced_acc":
        return balanced_accuracy_score(y_true, pred)
    if m == "macro_f1":
        return f1_score(y_true, pred, average="macro")
    raise ValueError(f"unknown metric {m!r}")


# Every C the probe selects, so the grid can be checked for saturation: a C pinned
# to an endpoint means the optimum lies outside the grid and the probe is
# mis-regularised, which would confound any A/B run on top of it.
_LAST_C = [None]
_C_HIST = []


def _fit_best(a, y_tr, ps):
    """Fit at each C, keep the one best on the FITTING data under the active
    metric. Selecting C by the same metric that is reported avoids a model chosen
    to maximise ranking then scored on thresholded decisions, or vice versa."""
    best, best_v, best_C = None, -np.inf, LR_CS[0]
    for C in LR_CS:
        clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced",
                                 random_state=ps)
        clf.fit(a, y_tr)
        v = _score(y_tr, clf.predict_proba(a)[:, 1], clf.predict(a))
        if v > best_v:
            best_v, best, best_C = v, clf, C
    _LAST_C[0] = best_C
    _C_HIST.append(best_C)
    return best


def _auc_cv(z, y, folds, probe_seeds):
    """Mean score over `probe_seeds` repeats of stratified `folds`-fold CV.

    class_weight='balanced' throughout: an unweighted logistic loss on a skewed
    label collapses toward the majority class and hides real signal.
    """
    vals = []
    for ps in range(probe_seeds):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=ps)
        for tr, te in skf.split(z, y):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            sc = P._scaler()
            a = sc.fit_transform(z[tr]); b = sc.transform(z[te])
            clf = _fit_best(a, y[tr], ps)
            vals.append(_score(y[te], clf.predict_proba(b)[:, 1], clf.predict(b)))
    return (float(np.mean(vals)) * 100 if vals else float("nan"),
            float(np.std(vals)) * 100 if vals else float("nan"))


# statistics

def _ss(points, groups):
    """-> (SS_between, SS_within, SS_total) for points grouped by `groups`."""
    X = np.asarray(points, dtype=float)
    g = np.asarray(groups)
    grand = X.mean(0)
    ss_tot = float(((X - grand) ** 2).sum())
    ss_w = 0.0
    for t in np.unique(g):
        sub = X[g == t]
        ss_w += float(((sub - sub.mean(0)) ** 2).sum())
    return ss_tot - ss_w, ss_w, ss_tot


def _stats(points, groups):
    ssb, ssw, sst = _ss(points, groups)
    k = len(np.unique(groups))
    n = len(points)
    r2 = ssb / sst if sst > 0 else float("nan")
    ratio = ((ssb / (k - 1)) / (ssw / (n - k))
             if k > 1 and n > k and ssw > 0 else float("inf"))
    within = float(np.sqrt(ssw / max(n - k, 1)))
    between = float(np.sqrt(ssb / max(k - 1, 1)))
    return dict(r2=r2, ratio=ratio, within=within, between=between,
                ss_between=ssb, ss_within=ssw, ss_total=sst)


def _blocked_perm_p(points, groups, blocks, obs_r2, n_perm=20000, seed=0):
    """Shuffle task labels WITHIN each training seed, recompute R².

    Blocking on the training seed matters: tasks inside one run share an encoder,
    so their λ* are correlated. A global shuffle would destroy that dependence and
    return an anticonservative p-value.
    """
    rng = random.Random(seed)
    groups = list(groups)
    blocks = list(blocks)
    idx_by_block = defaultdict(list)
    for i, b in enumerate(blocks):
        idx_by_block[b].append(i)
    hits = 0
    perm = list(groups)
    for _ in range(n_perm):
        for _b, idxs in idx_by_block.items():
            lab = [groups[i] for i in idxs]
            rng.shuffle(lab)
            for i, l in zip(idxs, lab):
                perm[i] = l
        if _stats(points, perm)["r2"] >= obs_r2 - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


# main



def _auc_holdout(z_tr, y_tr, z_te, y_te, probe_seeds):
    """Fit on the fitting split, score on the held-out split, under the active metric.

    Same estimator as _auc_cv -- balanced logistic regression, C chosen on the
    FITTING data only -- but the score comes from data never used for fitting or for
    selecting lambda. Repeated over probe_seeds; only the estimator's own randomness
    differs between repeats, so the spread here is small by construction and is not a
    substitute for the across-training-seed sd.
    """
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return float("nan"), float("nan")
    vals = []
    for ps in range(probe_seeds):
        sc = P._scaler()
        a = sc.fit_transform(z_tr); b = sc.transform(z_te)
        clf = _fit_best(a, y_tr, ps)
        vals.append(_score(y_te, clf.predict_proba(b)[:, 1], clf.predict(b)))
    return float(np.mean(vals)) * 100, float(np.std(vals)) * 100


def _baseline_z(enc_dir, args, device, split=None):
    """Representation of a lambda-blind baseline, mirroring probe.run_fixed.

    gmc / comm       : z = fusion(enc_v(v), enc_t(t))   -- the joint encoder's output
    factorcl         : z = concat(norm(proj_v_r(...)), norm(proj_t_r(...)))
    clip / cross_self: z = concat(proj_v(...), proj_t(...))  -- NOT normalised, matching
                       probe.run_fixed; these save proj_{v,t}.pth and NO fusion.pth, so
                       without this branch they fell into the gmc/comm path and died on
                       a missing fusion.pth.

    The fusion head is rebuilt at the shape recorded in train_meta: a
    capacity-matched CoMM/GMC has a different output dim and loading it into the
    default Linear(2*adim, adim) fails on every key.

    NOTE the dataset used for LABELS is args.dataset (the multitask build), while the
    architecture comes from train_meta (what the checkpoint was trained as). These
    differ on purpose -- SSL training is label-free, so a mosei-trained baseline is
    validly read out against mosei_multitask labels.
    """
    import torch, numpy as np, torch.nn.functional as Fn
    from pareto_ssl.networks import ProjectionHead
    from pareto_ssl.multibench.benchmark_multibench import _make_encoders, FusionEncoder, ENC_DIM_XFMR
    meta_f = os.path.join(enc_dir, "train_meta.json")
    meta = json.load(open(meta_f)) if os.path.exists(meta_f) else {}
    method = meta.get("method", "gmc")
    # Same trap as probe.run_fixed: _make_encoders reads benchmark_multibench's
    # module-level width at call time, so a capacity-swept checkpoint (--enc_width
    # 70) must set it BEFORE any encoder is built or every layer fails to load.
    P._apply_enc_width(meta, args.approach)
    adim = meta.get("enc_dim", ENC_DIM_XFMR)
    pdim = meta.get("proj_dim", args.proj_dim)

    def _ld(path, mod):
        mod.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        mod.eval()
        return mod

    enc_v, enc_t = _make_encoders(args.approach, args.dataset, args.enc_dim, device,
                                  film_mode="none")
    _ld(os.path.join(enc_dir, "enc_v.pth"), enc_v)
    _ld(os.path.join(enc_dir, "enc_t.pth"), enc_t)

    fusion = pvr = ptr = None
    _norm = True
    if method == "factorcl":
        pvr = _ld(os.path.join(enc_dir, "proj_v_r.pth"), ProjectionHead(adim, pdim).to(device))
        ptr = _ld(os.path.join(enc_dir, "proj_t_r.pth"), ProjectionHead(adim, pdim).to(device))
    elif method in ("clip", "cross_self"):
        pvr = _ld(os.path.join(enc_dir, "proj_v.pth"), ProjectionHead(adim, pdim).to(device))
        ptr = _ld(os.path.join(enc_dir, "proj_t.pth"), ProjectionHead(adim, pdim).to(device))
        _norm = False          # probe.run_fixed concatenates these RAW
    else:
        fusion = _ld(os.path.join(enc_dir, "fusion.pth"),
                     FusionEncoder(adim, meta.get("fusion_out") or adim,
                                   hidden=meta.get("fusion_hidden")).to(device))

    zs = []
    with torch.no_grad():
        for X, _ in P.mt_probe_loader(args.dataset, P._image_root(args.dataset), split or args.split,
                                      args.batch_size, task=P._MT_TASK[0],
                                      modalities=P.MODALITIES):
            v = X[0].float().to(device); t = X[1].float().to(device)
            if fusion is not None:
                zs.append(fusion(enc_v(v), enc_t(t)).cpu().numpy())
            else:
                _zv, _zt = pvr(enc_v(v)), ptr(enc_t(t))
                if _norm:
                    _zv, _zt = Fn.normalize(_zv, dim=-1), Fn.normalize(_zt, dim=-1)
                zs.append(torch.cat([_zv, _zt], dim=-1).cpu().numpy())
    return np.concatenate(zs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="glob of TRAINING run dirs (one per training seed)")
    ap.add_argument("--dataset", default="mosei_multitask")
    ap.add_argument("--approach", type=int, default=4)
    ap.add_argument("--film_mode", default="simplex")
    ap.add_argument("--test_all_lam", action="store_true",
                    help="with --eval_test, also score EVERY grid preference on test "
                         "and store the curve, so a consensus lambda can be applied "
                         "post-hoc without re-running the encoders.")
    ap.add_argument("--eval_test", action="store_true",
                    help="Select lambda on --split (valid) but REPORT on the test split: "
                         "fit the probe on valid at the selected lambda, score AUC on test. "
                         "Without this, selection and reporting share one split and the "
                         "reported AUC is optimistically biased by choosing 1 of 15 "
                         "preferences on the same data. Baselines have no lambda to select, "
                         "so for them this is simply fit-on-valid -> score-test.")
    ap.add_argument("--baseline", action="store_true",
                    help="Score a LAMBDA-BLIND baseline (gmc/comm/factorcl) instead of a "
                         "lambda-method. There is no preference grid: one representation is "
                         "extracted (fusion(enc_v,enc_t) for gmc/comm, concat of the R heads "
                         "for factorcl) and scored per task by the SAME _auc_cv used for the "
                         "lambda-methods. Without this the baselines are scored by probe.py's "
                         "matched protocol, whose AUC is a single unweighted valid->test fit "
                         "rather than balanced 5-fold CV -- not comparable.")
    ap.add_argument("--pca_shared", action="store_true",
                    help="dim_pca: one PCA basis per block shared across preferences. "
                         "Default is a separate basis per (lambda, block), matching "
                         "the reported numbers.")
    ap.add_argument("--narrow_c", action="store_true",
                    help="restore the legacy narrow C grid (0.01, 0.1, 1.0)")
    ap.add_argument("--wide_c", action="store_true",
                    help="widened C grid (1 .. 1e6). The default grid saturates: "
                         "~70%% of fits pin at its top endpoint. Check lr_C_hist.")
    ap.add_argument("--scaler", choices=["standard", "center", "none"],
                    default="standard",
                    help="probe preprocessing (see probe.py). 'standard' whitens a PCA "
                         "basis; 'center' centres only. Same setting for every arm.")
    ap.add_argument("--metric", choices=["auc", "balanced_acc", "macro_f1"],
                    default="auc",
                    help="metric used for BOTH lambda selection and reporting, and "
                         "for the baselines identically. balanced_acc is the "
                         "interpretable choice (50 = majority classifier on every "
                         "task regardless of prevalence).")
    ap.add_argument("--readout",
                    choices=["amp", "dim", "dim_rand", "dim_pca", "none"],
                    default="amp",
                    help="decomp_R readout (see probe.py). Under the default 'amp' a "
                         "linear probe can undo the sqrt(lambda) block scaling, so only "
                         "which blocks are non-zero matters and the lambda axis is "
                         "largely inert; 'dim' allocates dimensions instead.")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--probe_fit", default="cv_valid", choices=["cv_valid", "train"],
                    help="how each preference is SCORED for lambda selection. "
                         "cv_valid (default, historical): 5-fold CV inside --split. "
                         "train: fit the probe on the TRAIN split and score on --split, "
                         "i.e. the standard linear-evaluation protocol used by the "
                         "single-label tables (PROBE_FIT=train). The dispersion statistics "
                         "downstream are unchanged; only the per-(lambda, task) score is.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--probe_seeds", type=int, default=5)
    ap.add_argument("--enc_dim", type=int, default=128)
    ap.add_argument("--proj_dim", type=int, default=P.DEFAULT_PROJ_DIM)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="subset of tasks (default: USABLE_TASKS)")
    ap.add_argument("--n_perm", type=int, default=20000)
    ap.add_argument("--save_json", default=None)
    args = ap.parse_args()
    P._READOUT[0] = args.readout
    _METRIC[0] = args.metric
    P._SCALER[0] = args.scaler
    P._PCA_SHARED[0] = args.pca_shared
    # Regularisation grid. The published grid (0.01, 0.1, 1.0) saturates: ~70% of
    # fits pin at its top endpoint, so the optimum lies above it. --wide_c reaches
    # 1e6; check lr_C_hist peaks INSIDE whichever grid is used.
    if args.wide_c:
        globals()["LR_CS"] = (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4)
    elif args.scaler != "standard":
        globals()["LR_CS"] = (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dirs = sorted(d for d in glob.glob(args.runs) if os.path.isdir(d))
    if not run_dirs:
        print(f"No run dirs matched {args.runs}")
        return
    tasks = args.tasks or usable_tasks(args.dataset)
    _LN = label_names(args.dataset)

    print(f"\n{'='*100}")
    print(f"  TASK-vs-LAMBDA PROBE   {args.dataset} ap{args.approach} film={args.film_mode}")
    print(f"  {len(run_dirs)} training seeds | {len(tasks)} tasks | "
          f"{args.folds}-fold CV x {args.probe_seeds} probe seeds | metric = AUC"
          + f"  | probe_fit={args.probe_fit}"
          + ("  | SELECT on " + args.split + ", REPORT on test" if args.eval_test
             else f"  | select and report both on {args.split}"))
    print(f"{'='*100}")

    # labels for every task, in dataset order (loader uses shuffle=False)
    path = P._image_root(args.dataset)
    _DS = dataset_class(args.dataset)
    Y = _DS(path, args.split).labels       # (N, n_labels) on the SELECTION split
    # probe_fit=train fits on the TRAIN split and scores on --split, so the training
    # labels are needed too. Selection still happens on --split; nothing reads test.
    Y_fit = _DS(path, "train").labels if args.probe_fit == "train" else None
    Y_te = _DS(path, "test").labels if args.eval_test else None

    records, curves = [], {}
    for rd in run_dirs:
        tag = os.path.basename(rd)
        # Seed token normally sits in the run dir's own name. For an epoch-ladder
        # sweep the path is <...>_seed42/checkpoints/ep0200, whose basename carries
        # no seed -- every run would then be labelled "ep0200", collapsing all five
        # seeds onto one curves[] key and blocking the permutation test on a single
        # bogus block. So walk up the path until a seed token appears.
        seed = None
        for part in reversed(os.path.normpath(rd).split(os.sep)):
            seed = next((q.replace("seed", "") for q in part.split("_")
                         if q.startswith("seed")), None)
            if seed is not None:
                break
        if seed is None:
            seed = tag
        print(f"\n--- training seed {seed}   ({tag})")
        P._MT_TASK[0] = tasks[0]                      # any task: z does not depend on it
        z_te_by_lam = None
        if args.baseline:
            z_by_lam = {None: _baseline_z(rd, args, device)}
            if args.eval_test:
                z_te_by_lam = {None: _baseline_z(rd, args, device, split="test")}
        else:
            models = P._load_models(rd, args.approach, args.dataset, args.enc_dim,
                                    args.proj_dim, device, None, None, args.film_mode)
            enc_v, enc_t, pvr, ptr, pvu, ptu = models
            z_by_lam, _ = P._get_all_z(enc_v, enc_t, pvr, pvu, ptr, ptu,
                                       args.dataset, args.split, args.film_mode,
                                       args.batch_size, device, modality="both")
            z_fit_by_lam = None
            if args.probe_fit == "train":
                z_fit_by_lam, _ = P._get_all_z(enc_v, enc_t, pvr, pvu, ptr, ptu,
                                               args.dataset, "train", args.film_mode,
                                               args.batch_size, device, modality="both")
            if args.eval_test:
                z_te_by_lam, _ = P._get_all_z(enc_v, enc_t, pvr, pvu, ptr, ptu,
                                              args.dataset, "test", args.film_mode,
                                              args.batch_size, device, modality="both")
        grid = sorted(z_by_lam.keys(), key=lambda x: tuple(x) if isinstance(x, tuple) else (x,))
        n = len(next(iter(z_by_lam.values())))
        if n != len(Y):
            print(f"    [warn] z has {n} rows but labels have {len(Y)} — skipping")
            continue

        for t in tasks:
            y = (Y[:, _LN.index(t)] > 0).astype(int)
            if len(np.unique(y)) < 2:
                print(f"    {t:10s} single-class — skipped")
                continue
            if args.baseline:
                # No preference grid: one representation, one AUC per task. The
                # lambda-selection and dispersion machinery below is meaningless for a
                # lambda-blind method, so it is skipped rather than fed a placeholder.
                if args.eval_test:
                    y_te = (Y_te[:, _LN.index(t)] > 0).astype(int)
                    auc = _auc_holdout(z_by_lam[None], y, z_te_by_lam[None], y_te,
                                       args.probe_seeds)[0]
                elif args.probe_fit == "train":
                    y_fit = (Y_fit[:, _LN.index(t)] > 0).astype(int)
                    auc = _auc_holdout(_baseline_z(rd, args, device, split="train"), y_fit,
                                       z_by_lam[None], y, args.probe_seeds)[0]
                else:
                    auc = _auc_cv(z_by_lam[None], y, args.folds, args.probe_seeds)[0]
                records.append(dict(task=t, seed=seed, lam=None, auc=auc, spread=0.0))
                print(f"    {t:10s} AUC={auc:6.2f}")
                continue
            if args.probe_fit == "train":
                y_fit = (Y_fit[:, _LN.index(t)] > 0).astype(int)
                aucs = [_auc_holdout(z_fit_by_lam[l], y_fit, z_by_lam[l], y,
                                     args.probe_seeds)[0] for l in grid]
            else:
                aucs = [_auc_cv(z_by_lam[l], y, args.folds, args.probe_seeds)[0] for l in grid]
            best = int(np.argmax(aucs))
            lam = grid[best]
            if args.eval_test:
                # lambda chosen on the selection split above; the REPORTED number is
                # fit on that split and scored on test, so it is not inflated by having
                # picked 1 of 15 preferences on the same data.
                y_te = (Y_te[:, _LN.index(t)] > 0).astype(int)
                auc_te = _auc_holdout(z_by_lam[lam], y, z_te_by_lam[lam], y_te,
                                      args.probe_seeds)[0]
            vec = list(lam) if isinstance(lam, (tuple, list)) else [float(lam)]
            spread = max(aucs) - min(aucs)
            records.append(dict(task=t, seed=seed, lam=vec,
                                auc=(auc_te if args.eval_test else aucs[best]),
                                auc_select=aucs[best], spread=spread))
            curves[f"{seed}/{t}"] = {"grid": [list(l) if isinstance(l, tuple) else l
                                              for l in grid], "auc": aucs}
            if args.eval_test and args.test_all_lam:
                # Test AUC at EVERY preference, not only at this seed's argmax. The
                # encoder forwards are already paid for, so this is a few thousand
                # logistic fits; it is what lets a consensus lambda (one fixed
                # preference shared by all seeds) be scored on test afterwards
                # without launching a second job.
                curves[f"{seed}/{t}"]["auc_test"] = [
                    _auc_holdout(z_by_lam[l], y, z_te_by_lam[l], y_te,
                                 args.probe_seeds)[0] for l in grid]
            lam_s = ("(" + ",".join(f"{c:.2f}" for c in vec) + ")" if len(vec) > 1
                     else f"{vec[0]:.2f}")
            if args.eval_test:
                print(f"    {t:10s} lam*={lam_s:>18s}  AUC(test)={auc_te:6.2f}  "
                      f"[select {aucs[best]:.2f}]  curve span={spread:5.2f}")
            else:
                print(f"    {t:10s} lam*={lam_s:>18s}  AUC={aucs[best]:6.2f}  "
                      f"curve span={spread:5.2f}  (min {min(aucs):.2f})")

    if args.baseline:
        print(f"\n{'='*100}\n  BASELINE AUC PER TASK  (same _auc_cv as the lambda-methods: "
              f"{args.folds}-fold x {args.probe_seeds} probe seeds, class_weight=balanced)\n{'='*100}")
        print(f"  {'task':12s}{'n':>3s}{'AUC':>18s}")
        for t in tasks:
            v = [r["auc"] for r in records if r["task"] == t]
            if not v:
                continue
            m = float(np.mean(v))
            sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            print(f"  {t:12s}{len(v):>3d}{f'{m:.2f} ± {sd:.2f}':>18s}")
        if args.save_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
            from collections import Counter as _Ctr
            _cd = {str(k): v for k, v in sorted(_Ctr(_C_HIST).items())}
            print(f"\n  logistic C chosen (grid {list(LR_CS)}): {_cd}")
            json.dump({"config": vars(args), "records": records,
                       "lr_C_hist": _cd, "lr_C_grid": list(LR_CS)},
                      open(args.save_json, "w"), indent=2, default=str)
            print(f"\nsaved -> {args.save_json}")
        return

    if len(records) < 4:
        print("\nNot enough (task, seed) observations for the variance test.")
        return

    pts = [r["lam"] for r in records]
    grp = [r["task"] for r in records]
    blk = [r["seed"] for r in records]
    S = _stats(pts, grp)
    p = _blocked_perm_p(pts, grp, blk, S["r2"], n_perm=args.n_perm)

    print(f"\n{'='*100}\n  DISPERSION OF λ* IN PREFERENCE SPACE\n{'='*100}")
    print(f"  {'task':10s} {'n':>2s}  mean λ*                         within-task sd")
    for t in tasks:
        sub = [r["lam"] for r in records if r["task"] == t]
        if not sub:
            continue
        A = np.asarray(sub)
        m = ", ".join(f"{v:.2f}" for v in A.mean(0))
        sd = float(np.sqrt(((A - A.mean(0)) ** 2).sum(1).mean()))
        print(f"  {t:10s} {len(sub):2d}  ({m})".ljust(46) + f"  {sd:.3f}")

    print(f"\n  BETWEEN-task dispersion : {S['between']:.4f}")
    print(f"  WITHIN-task  dispersion : {S['within']:.4f}")
    print(f"  between/within ratio    : {S['ratio']:.3f}   (F-like; ~1 under the null)")
    _n, _k = len(records), len(set(grp))
    _null_r2 = (_k - 1) / (_n - 1) if _n > 1 else float("nan")
    print(f"  task-explained variance : R² = {S['r2']:.4f}"
          f"   ({100*S['r2']:.1f}% of λ* variance attributable to task)")
    print(f"    expected R² under the null: {_null_r2:.4f}  "
          f"-- R² above this is NOT evidence on its own; read the permutation p")
    print(f"  blocked permutation p   : {p:.4f}   "
          f"({args.n_perm} shuffles within training seed)")
    floor = 1.0 / (args.n_perm + 1)
    k, nseeds = len(set(grp)), len(set(blk))
    comb_floor = 1.0 / (math.factorial(k) ** nseeds) if k <= 8 else 0.0
    print(f"  (p floor: {max(floor, comb_floor):.5f} from {k}!^{nseeds} relabelings"
          f" and {args.n_perm} draws)")
    print(f"\n  VERDICT: {'TASKS SELECT DIFFERENT PREFERENCES (p < 0.05)' if p < 0.05 else 'NOT DISTINGUISHABLE FROM SEED NOISE'}")

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        from collections import Counter as _Ctr
        _cd = {str(k): v for k, v in sorted(_Ctr(_C_HIST).items())}
        print(f"\n  logistic C chosen (grid {list(LR_CS)}): {_cd}")
        if _C_HIST and (min(_C_HIST) == LR_CS[0] or max(_C_HIST) == LR_CS[-1]):
            print("  [warn] C reached a grid endpoint - the optimum may lie outside it")
        json.dump({"config": vars(args), "records": records, "stats": S,
                   "blocked_perm_p": p, "curves": curves,
                   "lr_C_hist": _cd, "lr_C_grid": list(LR_CS)},
                  open(args.save_json, "w"), indent=2, default=str)
        print(f"\nsaved -> {args.save_json}")


if __name__ == "__main__":
    main()
