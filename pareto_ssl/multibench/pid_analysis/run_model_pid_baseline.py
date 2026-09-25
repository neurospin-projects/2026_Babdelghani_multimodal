#!/usr/bin/env python3
"""Model PID of a FIXED-representation baseline (GMC / CoMM / FactorCL / CLIP).

The STEER version sweeps 15 preferences and asks whether the decomposition MOVES. A
baseline has one representation, so its PID is a single point, and the comparison the
paper wants is "STEER spans a range, the baselines sit at points".

Input is the `--save_z` dump that probe.py writes in `matched` mode, i.e. the EXACT
representation behind the reported accuracy -- not a re-implementation of each
baseline's read-out (gmc/comm fuse, FactorCL concatenates ten heads, CLIP concatenates
two projections; re-deriving those by hand is how a silent mismatch gets in).

    yhat = probe fitted on z_valid, predicting z_test     (the matched protocol)
    PID  = BATCH decomposition of I(X_vision, X_text ; yhat)

Sources are the RAW modality features of the test split, flattened, in the same order
the dump was written; the label column is checked against the dump before anything is
computed.

Read the output ordinally -- BATCH is a neural estimator and the absolute bits are not
calibrated. What is comparable is the BALANCE across R / U_vision / U_text between
methods measured the same way, and against STEER's range on the same dataset.

Usage:
  python pareto_ssl/multibench/pid_analysis/run_model_pid_baseline.py \
      --z pareto_ssl/multibench/pid_analysis/z/mosei_gmc_s42.npz \
      --dataset mosei --method gmc --out results/baseline_pid/mosei_gmc_s42.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (ROOT, os.path.join(ROOT, "CoMM"), os.path.join(HERE, "estimator")):
    sys.path.insert(0, p)

from run_model_pid_multibench import batch_pid, raw_sources, fit_probe   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", required=True, help="probe.py --save_z dump (.npz)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True, help="label for the output only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-n", type=int, default=2000)
    ap.add_argument("--discrim-epochs", type=int, default=40)
    ap.add_argument("--ce-epochs", type=int, default=10)
    ap.add_argument("--null-perms", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    d = np.load(a.z)
    z_val, y_val = d["z_valid"], d["y_valid"].reshape(-1)
    z_te, y_te = d["z_test"], d["y_test"].reshape(-1)
    print(f"{a.method} / {a.dataset}: z_valid {z_val.shape} z_test {z_te.shape}", flush=True)

    X1, X2, y_raw, idx = raw_sources(a.dataset, "test", a.batch_size, a.max_n, a.seed)
    if not np.array_equal(np.asarray(y_raw).reshape(-1), y_te[idx]):
        raise SystemExit("raw-source labels do not match the dump's test labels -- the "
                         "sources and yhat would be misaligned, aborting")

    num_labels = int(max(y_val.max(), y_te.max())) + 1
    t0 = time.time()
    yhat_all, C = fit_probe(z_val, y_val, z_te)          # matched protocol: fit on valid
    acc = float((yhat_all == y_te).mean())
    yhat = yhat_all[idx]
    r = batch_pid(X1, X2, yhat, num_labels, a.seed, a.discrim_epochs, a.ce_epochs)
    if a.null_perms > 0:
        rng = np.random.default_rng(a.seed)
        nulls = [batch_pid(X1, X2, rng.permutation(yhat), num_labels, a.seed,
                           a.discrim_epochs, a.ce_epochs) for _ in range(a.null_perms)]
        r["null"] = {k: float(np.mean([n[k] for n in nulls])) for k in ("R", "U1", "U2", "S")}
    nl = r.get("null") or dict(R=0, U1=0, U2=0, S=0)
    tot = sum(max(r[k] - nl[k], 0.0) for k in ("R", "U1", "U2", "S")) or 1e-12
    out = dict(meta=dict(method=a.method, dataset=a.dataset, z=os.path.abspath(a.z),
                         probe_fit="valid", num_labels=num_labels, n_eval=int(len(idx)),
                         seconds=round(time.time() - t0, 1)),
               probe_acc=acc, probe_C=C, **r,
               share=({k: (max(r[k] - nl[k], 0.0) / tot) for k in ("R", "U1", "U2", "S")}))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    sh = out["share"]
    print(f"  acc={acc:.3f}  R={r['R'] - nl['R']:.3f} U_vision={r['U1'] - nl['U1']:.3f} "
          f"U_text={r['U2'] - nl['U2']:.3f} S={r['S'] - nl['S']:.3f}   "
          f"balance R/Uv/Ut/S = {100*sh['R']:.0f}/{100*sh['U1']:.0f}/{100*sh['U2']:.0f}/"
          f"{100*sh['S']:.0f}%  [{out['meta']['seconds']}s]", flush=True)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
