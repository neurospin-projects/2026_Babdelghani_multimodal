#!/usr/bin/env python3
"""Model PID of STEER across the preference simplex on the REAL benchmarks.

Same question as run_model_pid_steer.py (Trifeature), estimator swapped: the inputs here
are continuous and high-dimensional, so the discrete convex program is replaced by BATCH
(Liang et al., NeurIPS 2023), the estimator the project already validated on synthetic
gates in run_pid_all.py.

For every preference lambda on the simplex and every task label:

    z(lambda)  from probe.py's own _load_models / _get_all_z, `dim` read-out
    yhat       prediction of the same linear probe protocol the results tables use
    PID        BATCH decomposition of I(X1, X2 ; yhat) -> R / U_vision / U_text / S

X1, X2 are the RAW modality features (flattened padded sequences for the affect sets,
flattened images for AV-MNIST), i.e. the model's inputs -- so the decomposition says which
interaction of the inputs the model's prediction actually uses at that preference.

Read the output ordinally. On real data there is no ground truth to calibrate against, so
what carries meaning is the TREND across lambda and the ranking of components, never the
absolute bits. Two things make the absolute numbers untrustworthy in particular:
  - BATCH is a neural estimator; its value depends on discriminator fit and sample size.
  - the finite-sample floor is real here too. --null-perms 1 measures it by decomposing a
    SHUFFLED yhat, but each extra run costs a full estimation, so it defaults to off.

Expect a flat U_vision on MOSEI / MOSI / UR-FUNNY: this project's own dataset-level PID
puts their second unique axis at ~0.000-0.002 bits, so there is no vision-unique
information for any preference to trade. That is a prediction of the method, not a failure
of it, and it is the reason Trifeature is the positive control.

Usage (Jean-Zay, where the data and checkpoints live):
  python pareto_ssl/multibench/pid_analysis/run_model_pid_multibench.py \
      --enc-dir $SCRATCH/Output/multibench/mosei/approach5/simclr_single_per_batch_filmsimplex_enc_decomp_R_loralinear_lc0_base_seed42 \
      --dataset mosei --out results/steer_mosei_s42_lc0.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (ROOT, os.path.join(ROOT, "CoMM"), os.path.join(HERE, "estimator")):
    sys.path.insert(0, p)

from pareto_ssl.multibench import probe as _probe                     # noqa: E402
from ce_alignment_information import critic_ce_alignment, MultimodalDataset  # noqa: E402
import ce_alignment_information as _cea                               # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG2 = float(np.log(2.0))


def _znorm(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean(0, keepdims=True)) / x.std(0, keepdims=True).clip(min=1e-6)


def batch_pid(x1, x2, y, num_labels, seed=42, discrim_epochs=40, ce_epochs=10):
    """One BATCH run -> dict of R / U1 / U2 / S in bits (U1 = vision, U2 = text)."""
    torch.manual_seed(seed); np.random.seed(seed)
    x1, x2 = _znorm(x1), _znorm(x2)
    _cea.batch_size = max(16, min(64, len(x1) // 10))
    t1 = torch.tensor(x1); t2 = torch.tensor(x2)
    ty = torch.tensor(np.asarray(y), dtype=torch.long).unsqueeze(-1)
    ds = MultimodalDataset([t1, t2], ty)
    res, _, _ = critic_ce_alignment(t1, t2, ty, num_labels, ds, ds,
                                    discrim_epochs=discrim_epochs, ce_epochs=ce_epochs)
    r, u1, u2, s = (torch.mean(res, dim=0).cpu() / LOG2).tolist()
    return dict(R=r, U1=u1, U2=u2, S=s)


def _stack_flat(seqs):
    """Stack per-sample arrays and flatten, zero-padding a ragged time axis.

    The affect loaders pad each BATCH to its own longest sequence, so batches come back
    with different T and a naive per-batch flatten cannot be concatenated (MOSEI: 1750
    features in one batch, 1540 in another). Padding to the split-wide maximum keeps the
    full sequence the reference pipeline expects, with a consistent feature layout.
    """
    shapes = {s.shape for s in seqs}
    if len(shapes) == 1:                      # images, or already-uniform sequences
        return np.stack(seqs).reshape(len(seqs), -1)
    T = max(s.shape[0] for s in seqs)
    D = seqs[0].shape[-1]
    out = np.zeros((len(seqs), T, D), dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i, :s.shape[0]] = s
    return out.reshape(len(seqs), -1)


def raw_sources(dataset, split, batch_size, max_n=None, seed=0, pool=False):
    """Flattened raw modality features for one split: (X1 vision, X2 text, labels).

    The reference pipeline (the reference BATCH implementation) feeds the FULL padded
    sequence, not a mean-pooled summary, so that is what is used here. Samples are kept
    individually until the subsample is chosen, so padding is applied once, to the
    selected rows only.
    """
    V, T, Y = [], [], []
    for item in _probe._raw_batches(dataset, split, batch_size):
        X, y = ((item[0], item[1]), item[2]) if len(item) == 3 else item
        V.extend(np.asarray(X[0], dtype=np.float32))
        T.extend(np.asarray(X[1], dtype=np.float32))
        Y.append(np.asarray(y).reshape(-1))
    Y = np.concatenate(Y)
    idx = (np.random.default_rng(seed).choice(len(Y), max_n, replace=False)
           if max_n and len(Y) > max_n else np.arange(len(Y)))
    _prep = ((lambda seqs: np.stack([s.reshape(len(s), -1).mean(0) if s.ndim > 1 else s
                                     for s in seqs]))
             if pool else _stack_flat)
    return (_prep([V[i] for i in idx]), _prep([T[i] for i in idx]), Y[idx], idx)


def fit_probe(zfit, yfit, zte, Cs=(1e-2, 0.1, 1.0, 10.0), seed=42):
    sc = StandardScaler()
    tr, te = sc.fit_transform(zfit), sc.transform(zte)
    rng = np.random.default_rng(42)
    val = rng.choice(len(tr), size=max(1, len(tr) // 5), replace=False)
    fit = np.setdiff1d(np.arange(len(tr)), val)
    best_C, best = Cs[0], -1.0
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=1000, random_state=seed).fit(tr[fit], yfit[fit])
        v = clf.score(tr[val], yfit[val])
        if v > best:
            best, best_C = v, C
    clf = LogisticRegression(C=best_C, max_iter=1000, random_state=seed).fit(tr, yfit)
    return clf.predict(te), float(best_C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc-dir", required=True)
    ap.add_argument("--dataset", required=True,
                    choices=["mosei", "mosi", "humor", "mustard", "avmnist", "enrico",
                             "mosei_multitask", "chsims"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--approach", type=int, default=5)
    ap.add_argument("--enc-dim", type=int, default=40)
    ap.add_argument("--proj-dim", type=int, default=64)
    ap.add_argument("--probe-fit", default="valid", choices=["valid", "train"],
                    help="split the probe is fitted on; must match the results table")
    ap.add_argument("--task", default=None, help="multitask datasets: label to probe")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-n", type=int, default=2000,
                    help="cap on test samples fed to the estimator")
    ap.add_argument("--lambdas", default="all", help="'all' or indices into the grid")
    ap.add_argument("--discrim-epochs", type=int, default=40)
    ap.add_argument("--ce-epochs", type=int, default=10)
    ap.add_argument("--null-perms", type=int, default=0,
                    help="shuffled-yhat runs per cell (finite-sample floor); each costs "
                         "a full BATCH estimation")
    ap.add_argument("--target", default="yhat", choices=["yhat", "true"],
                    help="what to decompose. 'true' ignores the model and decomposes the "
                         "LABEL -- the reference row. On MOSEI it must come out "
                         "text-dominated with vision-unique ~0 (vision is at chance for "
                         "sentiment); if it instead reads mostly redundant, the estimator "
                         "is inflating R and no model row can be trusted.")
    ap.add_argument("--pool", action="store_true",
                    help="mean-pool the source sequences instead of flattening them. "
                         "Flattened MOSEI text is ~15000-dim against 2000 samples, which "
                         "lets the discriminators fit anything; pooling is the control.")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    _probe._READOUT[0] = "dim"                 # the read-out every reported number uses
    if a.task:
        _probe._MT_TASK[0] = a.task

    with open(os.path.join(a.enc_dir, "train_meta.json")) as f:
        meta = json.load(f)
    film = meta.get("film_mode", "simplex_enc_decomp_R")
    if "decomp_R" not in film:
        raise SystemExit(f"{a.enc_dir} is film_mode {film!r}, not a STEER (decomp_R) run")
    print(f"checkpoint {a.enc_dir}\n  dataset={a.dataset} film={film} "
          f"lam_club={meta.get('lam_club')} proj_dim={meta.get('proj_dim', a.proj_dim)} "
          f"probe_fit={a.probe_fit} device={DEVICE}", flush=True)

    enc_v, enc_t, proj_v_r, proj_t_r, proj_v_u, proj_t_u = _probe._load_models(
        a.enc_dir, a.approach, a.dataset, a.enc_dim, a.proj_dim, DEVICE, None, None, film)

    fit_split = "valid" if a.probe_fit == "valid" else "train"
    z_fit, y_fit = _probe._get_all_z(enc_v, enc_t, proj_v_r, proj_v_u, proj_t_r, proj_t_u,
                                     a.dataset, fit_split, film, a.batch_size, DEVICE)
    z_te, y_te = _probe._get_all_z(enc_v, enc_t, proj_v_r, proj_v_u, proj_t_r, proj_t_u,
                                   a.dataset, "test", film, a.batch_size, DEVICE)

    X1, X2, y_raw, idx = raw_sources(a.dataset, "test", a.batch_size, a.max_n, a.seed,
                                     pool=a.pool)
    if not np.array_equal(np.asarray(y_raw).reshape(-1), np.asarray(y_te)[idx].reshape(-1)):
        raise SystemExit("raw-source labels do not match the probe's label order -- the "
                         "sources and yhat would be misaligned, aborting")
    num_labels = int(np.max(np.concatenate([np.asarray(y_fit), np.asarray(y_te)]))) + 1
    print(f"  sources: vision {X1.shape}, text {X2.shape}, {num_labels} classes; "
          f"fit split '{fit_split}' n={len(y_fit)}, test n={len(y_te)} "
          f"(estimator sees {len(idx)})", flush=True)

    grid = sorted(z_te.keys(), key=lambda k: (k if isinstance(k, tuple) else (k,)))
    idxs = (list(range(len(grid))) if a.lambdas == "all"
            else [int(i) for i in a.lambdas.split(",")])
    out = dict(meta=dict(enc_dir=a.enc_dir, dataset=a.dataset, film=film,
                         lam_club=meta.get("lam_club"), ckpt_seed=meta.get("seed"),
                         readout="dim", probe_fit=a.probe_fit, task=a.task,
                         max_n=int(len(idx)), num_labels=num_labels,
                         discrim_epochs=a.discrim_epochs, ce_epochs=a.ce_epochs,
                         null_perms=a.null_perms, device=DEVICE),
               grid=[list(g) if isinstance(g, tuple) else g for g in grid], curve={})

    for gi in idxs:
        lam = grid[gi]
        t0 = time.time()
        yhat_all, C = fit_probe(z_fit[lam], np.asarray(y_fit), z_te[lam])
        acc = float((yhat_all == np.asarray(y_te)).mean())
        yhat = yhat_all[idx] if a.target == "yhat" else np.asarray(y_te)[idx]
        r = batch_pid(X1, X2, yhat, num_labels, a.seed, a.discrim_epochs, a.ce_epochs)
        if a.null_perms > 0:
            rng = np.random.default_rng(a.seed)
            nulls = [batch_pid(X1, X2, rng.permutation(yhat), num_labels, a.seed,
                               a.discrim_epochs, a.ce_epochs) for _ in range(a.null_perms)]
            r["null"] = {k: float(np.mean([n[k] for n in nulls])) for k in ("R", "U1", "U2", "S")}
        nl = r.get("null") or dict(R=0, U1=0, U2=0, S=0)
        out["curve"][str(gi)] = dict(lam=list(lam) if isinstance(lam, tuple) else lam,
                                     probe_acc=acc, probe_C=C,
                                     seconds=round(time.time() - t0, 1), **r)
        print(f"  lam={lam} acc={acc:.3f}  R={r['R'] - nl['R']:.3f} "
              f"U_vision={r['U1'] - nl['U1']:.3f} U_text={r['U2'] - nl['U2']:.3f} "
              f"S={r['S'] - nl['S']:.3f}  [{out['curve'][str(gi)]['seconds']}s]", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
