#!/usr/bin/env python
"""Pick the consensus lambda from a set of probed seeds and print it machine-readably.

The single-label probe (probe.py --mode validation) scores all 15 preferences on
validation but evaluates TEST only at each seed's own argmax. Reporting those test
numbers mixes five different operating points and is optimistically biased, since
each seed chose its lambda by looking at 15 options. The protocol used everywhere
else is: consensus on validation across seeds -> re-probe every seed at that ONE
lambda -> report. This script is the middle step, so it can be chained rather than
run by hand.

  eps        = mean over lambda of the seed-to-seed sd of validation score
  accept(l)  = #seeds whose regret (own best - score at l) is <= eps
  pick       = max acceptance, ties broken by worst rank then mean rank

stdout is ONLY the lambda as "a,b,c" so it can be substituted directly into
FIXED_LAM. Diagnostics go to stderr.

Usage
  python consensus_pick.py "<glob of run dirs>" [--readout dim] [--sub ""]
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

import numpy as np


def consensus(seed_curves, lams, seeds):
    eps = float(np.mean([st.pstdev([seed_curves[s][l] for s in seeds]) for l in lams]))
    accept = {l: sum(max(seed_curves[s].values()) - seed_curves[s][l] <= eps
                     for s in seeds) for l in lams}
    rank = {s: {l: i + 1 for i, l in enumerate(
        sorted(lams, key=lambda l: -seed_curves[s][l]))} for s in seeds}
    mean_rank = {l: float(np.mean([rank[s][l] for s in seeds])) for l in lams}
    worst = {l: max(rank[s][l] for s in seeds) for l in lams}
    top = max(accept.values())
    tied = [l for l in lams if accept[l] == top]
    pick = min(tied, key=lambda l: (worst[l], mean_rank[l]))
    return pick, eps, accept[pick], worst[pick], mean_rank[pick]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="glob matching the per-seed RUN directories")
    ap.add_argument("--readout", default="dim")
    ap.add_argument("--sub", default="", help="extra protocol subdir, e.g. tunec_widec")
    ap.add_argument("--key", default="val_acc_pct",
                    help="validation curve to select on (val_acc_pct | val_nn_pct)")
    ap.add_argument("--min_seeds", type=int, default=3,
                    help="refuse to emit a lambda from fewer seeds than this: a "
                         "consensus over 2 runs is not a consensus, and on AV-MNIST "
                         "the pick moved between n=3 and n=5.")
    a = ap.parse_args()

    dirs = sorted(d for d in glob.glob(a.pattern) if os.path.isdir(d))
    cur = {}
    for d in dirs:
        parts = [d, f"readout_{a.readout}"] if a.readout != "amp" else [d]
        if a.sub:
            parts.append(a.sub)
        f = os.path.join(*parts, "probe_validation.json")
        if not os.path.exists(f):
            print(f"[skip] no probe at {f}", file=sys.stderr)
            continue
        j = json.load(open(f))
        curve = j.get(a.key)
        if not curve:
            print(f"[skip] {f} has no '{a.key}'", file=sys.stderr)
            continue
        seed = next((p.replace("seed", "") for p in os.path.basename(d).split("_")
                     if p.startswith("seed")), os.path.basename(d))
        cur[seed] = dict(zip([tuple(x) for x in j["lambdas"]], curve))

    if len(cur) < a.min_seeds:
        print(f"ERROR: {len(cur)} seed(s) probed, need >= {a.min_seeds}", file=sys.stderr)
        sys.exit(2)

    seeds = sorted(cur)
    lams = sorted(cur[seeds[0]])
    pick, eps, acc, wr, mr = consensus(cur, lams, seeds)

    print(f"seeds={seeds}  eps={eps:.2f}  accept={acc}/{len(seeds)}  "
          f"worstRank={wr}  meanRank={mr:.1f}", file=sys.stderr)
    for s in seeds:
        own = max(cur[s], key=cur[s].get)
        rk = sorted(lams, key=lambda l: -cur[s][l]).index(pick) + 1
        print(f"  seed{s}: own best {tuple(round(c,2) for c in own)} "
              f"{cur[s][own]:.2f} | at pick {cur[s][pick]:.2f} "
              f"(regret {cur[s][own]-cur[s][pick]:.2f}, rank {rk})", file=sys.stderr)
    if wr > 5:
        print(f"  [warn] worst rank {wr} of {len(lams)}: at least one seed considers "
              f"this preference poor. The fixed lambda is weakly supported.",
              file=sys.stderr)
    print(",".join(f"{c}" for c in pick))          # stdout: machine-readable only


if __name__ == "__main__":
    main()
