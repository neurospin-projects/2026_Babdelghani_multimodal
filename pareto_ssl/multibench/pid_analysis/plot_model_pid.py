#!/usr/bin/env python3
"""Plot R(lambda), U1(lambda), U2(lambda), S(lambda) from run_model_pid_steer.py.

One panel per target (Y = R / U1 / U2). Lambda points are ordered along the simplex
edge that the target's own coordinate walks, so a monotone trend is visible as a slope:
    Y = R   -> ordered by lambda_R      Y = U1 -> by lambda_U1     Y = U2 -> by lambda_U2
The target's own PID component is drawn bold; probe accuracy is overlaid on a twin axis,
because the question this figure has to answer is whether PID says anything ACCURACY
does not. The printed correlation between the two is part of the result, not decoration.

  python pareto_ssl/multibench/pid_analysis/plot_model_pid.py results/steer_s42_lc05.json --out fig.png
"""
import argparse
import json

import numpy as np

# Component keys are always R/U1/U2 internally (first modality, second modality), but
# WHICH factor that is depends on the trained pair: M1 deformation, M2 texture, M3 colour.
# The panel labels follow train_meta so a colour-pair run is never captioned "deformation".
FACTOR = {"U1": "deformation", "U2": "texture", "U3": "colour"}
KEY = {"share": ("R", r"$R$ (shape)"), "unique1": ("U1", r"$U_1$"),
       "unique2": ("U2", r"$U_2$")}


def _labels_for(meta):
    u1, u2 = (meta.get("unique_labels") or ["U1", "U2"])
    k = dict(KEY)
    k["unique1"] = ("U1", rf"${u1[0]}_{{{u1[1:]}}}$ ({FACTOR.get(u1, '?')})")
    k["unique2"] = ("U2", rf"${u2[0]}_{{{u2[1:]}}}$ ({FACTOR.get(u2, '?')})")
    return k
COORD = {"share": 0, "unique1": 1, "unique2": 2}
COLORS = {"R": "#1f4fa3", "U1": "#E69F00", "U2": "#009E73", "S": "#CC79A7"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_files", nargs="+", help="one or more result JSONs (seeds)")
    ap.add_argument("--out", default="model_pid_lambda.png")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                         "axes.labelsize": 11, "savefig.facecolor": "white"})

    # One seed may arrive as several files when the preferences were split across jobs
    # (CHUNKS in the launcher). Merge by checkpoint, or each chunk would be averaged in
    # as if it were an independent seed over the few preferences it happens to hold.
    merged = {}
    for f in a.json_files:
        d = json.load(open(f))
        key = d["meta"]["enc_dir"]
        if key in merged:
            dup = set(merged[key]["curve"]) & set(d["curve"])
            if dup:
                raise SystemExit(f"{f}: preferences {sorted(dup)} already loaded for this "
                                 f"checkpoint -- overlapping chunks, refusing to average")
            merged[key]["curve"].update(d["curve"])
        else:
            merged[key] = d
    runs = list(merged.values())
    print(f"{len(a.json_files)} file(s) -> {len(runs)} checkpoint(s); preferences per "
          f"checkpoint: {[len(r['curve']) for r in runs]}")
    targets = [t for t in ("share", "unique1", "unique2") if t in runs[0]["dataset_pid"]]
    KEYL = _labels_for(runs[0]["meta"])
    fig, axes = plt.subplots(1, len(targets), figsize=(4.4 * len(targets), 3.9), squeeze=False)

    for ax, tgt in zip(axes[0], targets):
        c = COORD[tgt]
        xs, comp, acc = [], {k: [] for k in ("R", "U1", "U2", "S")}, []
        floor = []          # shuffled-target level for this target's own component
        for gi in sorted(runs[0]["curve"], key=int):
            lam = runs[0]["curve"][gi]["lam"]
            vals = [r["curve"][gi][tgt] for r in runs if gi in r["curve"]]
            xs.append(lam[c])
            for k in comp:
                # raw minus the shuffled-target floor stored with each cell
                comp[k].append(np.mean([v[k] - ((v.get("null") or {}).get(k, 0.0))
                                        for v in vals]))
            acc.append(np.mean([v["probe_acc"] for v in vals]))
            floor.append(np.mean([(v.get("null") or {}).get(KEYL[tgt][0], 0.0)
                                  for v in vals]))
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        own = KEYL[tgt][0]
        # what a RANDOM target scores on these alphabets: values are plotted with it
        # already removed, so this band shows how much was taken off, not a threshold
        fl = np.asarray(floor)[order]
        if np.any(fl > 0):
            ax.fill_between(xs, 0, fl, color="0.6", alpha=0.18, lw=0,
                            label="_nolegend_", zorder=1)
            ax.text(xs[0], fl[0], " shuffled-target floor (removed)", fontsize=7,
                    color="0.35", va="bottom", zorder=1)
        for k in ("R", "U1", "U2", "S"):
            y = np.asarray(comp[k])[order]
            ax.plot(xs, y, "o-", ms=4, color=COLORS[k], lw=2.2 if k == own else 1.0,
                    alpha=1.0 if k == own else 0.55, label=k, zorder=3 if k == own else 2)
        ax.set_xlabel(rf"$\lambda_{{{own}}}$")
        ax.set_ylabel("bits")
        ax.set_title(f"Y = {KEYL[tgt][1]}", fontsize=12)
        ax.grid(alpha=0.25, lw=0.5)
        ax2 = ax.twinx()
        acc = np.asarray(acc)[order]
        ax2.plot(xs, acc, "k--", lw=1.0, alpha=0.6)
        ax2.set_ylabel("probe accuracy", fontsize=9)
        ax2.set_ylim(0, 1.02)
        r = np.corrcoef(np.asarray(comp[own])[order], acc)[0, 1] if len(xs) > 2 else np.nan
        ax.text(0.03, 0.96, rf"corr({own}, acc) = {r:+.2f}", transform=ax.transAxes,
                va="top", fontsize=9)
        print(f"{tgt:8s} corr(PID-{own}, probe acc) = {r:+.3f}   "
              f"{own} range {np.min(comp[own]):.3f}..{np.max(comp[own]):.3f} bits   "
              f"acc range {acc.min():.3f}..{acc.max():.3f}")

    axes[0][0].legend(frameon=False, fontsize=9, loc="center left")
    m = runs[0]["meta"]
    _pair = "/".join(m.get("modality_pair") or ["M1", "M2"])
    fig.suptitle(a.title or f"Model PID of STEER across the preference simplex "
                            f"({_pair}, {m.get('method')}, lam_club={m.get('lam_club')}, "
                            f"{len(runs)} seed(s), read-out {m.get('readout')})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    if a.out.endswith(".png"):
        fig.savefig(a.out[:-4] + ".pdf", bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
