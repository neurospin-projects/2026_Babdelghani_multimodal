"""
Pareto frontier plot — all SSL methods + lambda-conditioned curve.

Left   : R (share) vs mean(U1, U2)  — main Pareto scatter + lambda curve
Right  : performance-vs-lambda curves per task (lambda_cond only)
"""

import os
import json
import numpy as np


METHOD_STYLES = {
    "simclr_both":   dict(marker="o",  color="#2196F3", zorder=4, s=100),
    "clip":          dict(marker="s",  color="#FF9800", zorder=4, s=100),
    "gmc":           dict(marker="P",  color="#795548", zorder=5, s=140),
    "comm":          dict(marker="*",  color="#F44336", zorder=5, s=220),
    "factorcl":      dict(marker="D",  color="#4CAF50", zorder=4, s=120),   # concat baseline
    "factorcl_heads":dict(marker="D",  color="#9C27B0", zorder=5, s=140),   # oracle / display-only
}

# lambda sweep methods — each gets its own curve colour
LAMBDA_CURVE_STYLES = {
    "simclr_single_per_batch":    dict(color="#00897B", label="simclr_single_per_batch"),    # teal
    "factorcl_lambda":  dict(color="#E91E63", label="factorcl_lambda"),  # pink
    "factorcl_warmup":  dict(color="#3F51B5", label="factorcl_warmup"),  # indigo
    "hyper_lambda":     dict(color="#FF6F00", label="hyper_lambda"),     # amber
    "simclr_grid_per_batch":     dict(color="#00BCD4", label="simclr_grid_per_batch"),     # cyan
}


def _pareto_front(points):
    pts = np.array(points)
    is_pareto = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        if is_pareto[i]:
            dominated = np.all(pts[is_pareto] >= p, axis=1) & ~np.all(pts[is_pareto] == p, axis=1)
            is_pareto[is_pareto] &= ~dominated
            is_pareto[i] = True
    return np.where(is_pareto)[0]


def plot_pareto(all_results: dict, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Split data into fixed-method results and lambda sweep curves
    fixed = {m: d for m, d in all_results.items() if "probes" in d}
    lambda_curves = {m: d["curve"] for m, d in all_results.items()
                     if "curve" in d and len(d["curve"].get("lambdas", [])) > 0}

    # One-modality data: lambda methods have curves, fixed methods have single points
    one_mod_curves = {m: (d["curve_m1"], d["curve_m2"])
                      for m, d in all_results.items()
                      if "curve_m1" in d and "curve_m2" in d}
    one_mod_fixed  = {m: (d["probes_m1"], d["probes_m2"])
                      for m, d in all_results.items()
                      if "probes_m1" in d and "probes_m2" in d}
    has_one_mod = len(one_mod_curves) > 0 or len(one_mod_fixed) > 0

    has_lc = len(lambda_curves) > 0

    n_cols = 3 if has_one_mod else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(7.5 * n_cols, 6))

    # Left: R vs mean(U1, U2) — Pareto scatter + lambda curve
    ax = axes[0]
    points = []
    for method, data in fixed.items():
        p = data["probes"]
        x = p.get("share",   0)
        y = (p.get("unique1", 0) + p.get("unique2", 0)) / 2
        sty = {k: v for k, v in METHOD_STYLES.get(method, {}).items() if k != "s"}
        s   = METHOD_STYLES.get(method, {}).get("s", 100)
        ax.scatter(x, y, s=s, label=method, **sty)
        ax.annotate(method, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
        points.append([x, y])

    # Lambda sweep curves
    for method, lc_data in lambda_curves.items():
        sty = LAMBDA_CURVE_STYLES.get(method, dict(color="#607D8B", label=method))
        lam_vals = lc_data["lambdas"]
        xs = lc_data["share"]
        ys = [(lc_data["unique1"][i] + lc_data["unique2"][i]) / 2
              for i in range(len(lam_vals))]
        ax.plot(xs, ys, color=sty["color"], linewidth=2, zorder=6,
                label=f"{sty['label']} (curve)", alpha=0.85)
        ax.scatter(xs, ys, color=sty["color"], s=40, zorder=7,
                   edgecolors="white", linewidths=0.5)
        for i, lam in enumerate(lam_vals):
            if lam in (0.0, 0.5, 1.0):
                ax.annotate(f"λ={lam:.1f}", (xs[i], ys[i]),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=7, color=sty["color"])

    ax.axhline(0.1, color="gray", linestyle="--", linewidth=0.8, label="chance (10%)")
    ax.axvline(0.1, color="gray", linestyle="--", linewidth=0.8)

    if len(points) >= 2:
        pf_idx = _pareto_front(points)
        pf = np.array(points)[pf_idx]
        pf = pf[pf[:, 0].argsort()]
        ax.plot(pf[:, 0], pf[:, 1], "r--", linewidth=1.2, label="Pareto frontier", zorder=1)

    ax.set_xlabel("share probe  (shared / redundant info)", fontsize=11)
    ax.set_ylabel("mean(U1, U2) probe  (unique info)", fontsize=11)
    ax.set_title("Pareto frontier — R ↔ U trade-off\n(lambda_cond traces the full curve)", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Right: performance-vs-lambda curves per task
    ax2 = axes[1]
    if has_lc:
        task_colors = {"share": "#FF9800", "unique1": "#2196F3", "unique2": "#9C27B0"}
        task_markers = {"share": "s", "unique1": "o", "unique2": "D"}
        linestyles = list(lambda_curves.keys())

        for m_idx, (method, lc_data) in enumerate(lambda_curves.items()):
            sty = LAMBDA_CURVE_STYLES.get(method, dict(color="#607D8B", label=method))
            lam_vals = lc_data["lambdas"]
            ls = "-" if m_idx == 0 else "--"
            for task in ["share", "unique1", "unique2"]:
                if task not in lc_data:
                    continue
                label = f"{task} [{sty['label']}]" if len(lambda_curves) > 1 else task
                ax2.plot(lam_vals, lc_data[task], color=task_colors[task],
                         linestyle=ls, linewidth=2, label=label,
                         marker=task_markers[task], markersize=5)
                best_i = int(np.argmax(lc_data[task]))
                ax2.annotate(f"λ*={lam_vals[best_i]:.1f}",
                             (lam_vals[best_i], lc_data[task][best_i]),
                             textcoords="offset points", xytext=(4, 5),
                             fontsize=7, color=task_colors[task])

        ax2.axhline(0.1, color="gray", linestyle="--", linewidth=0.8, label="chance")
        ax2.set_xlabel("lambda", fontsize=11)
        ax2.set_ylabel("probe accuracy", fontsize=11)
        ax2.set_title("Performance per task vs lambda", fontsize=12)
        ax2.set_xlim(-0.02, 1.02)
        ax2.set_ylim(-0.02, 1.02)
        ax2.legend(fontsize=7, loc="center right")
        ax2.grid(True, alpha=0.3)
    else:
        # Fallback: U1 vs U2 scatter for fixed methods
        for method, data in fixed.items():
            p = data["probes"]
            x  = p.get("share",   0)
            u1 = p.get("unique1", 0)
            u2 = p.get("unique2", 0)
            sty = {k: v for k, v in METHOD_STYLES.get(method, {}).items()
                   if k not in ("s", "zorder")}
            s   = METHOD_STYLES.get(method, {}).get("s", 100)
            ax2.scatter(x, u1, s=s, zorder=4, label=f"{method} U1", **sty)
            ax2.scatter(x, u2, s=s, zorder=4, facecolors="none",
                        edgecolors=sty.get("color", "gray"),
                        marker=sty.get("marker", "o"), linewidths=1.5)
            ax2.annotate(method, (x, u1), textcoords="offset points",
                         xytext=(4, 2), fontsize=7)
        ax2.axhline(0.1, color="gray", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("share probe", fontsize=11)
        ax2.set_ylabel("unique probe accuracy", fontsize=11)
        ax2.set_title("U1 (filled) and U2 (hollow) separately", fontsize=12)
        ax2.set_xlim(-0.02, 1.02)
        ax2.set_ylim(-0.02, 1.02)
        ax2.grid(True, alpha=0.3)
        ax2.text(0.98, 0.15, "filled = U1 (deformation)\nhollow = U2 (texture)",
                 transform=ax2.transAxes, fontsize=8, ha="right",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

    # Right-most: one-modality performance curves (optional)
    if has_one_mod:
        ax3 = axes[2]
        task_colors  = {"share": "#FF9800", "unique1": "#2196F3", "unique2": "#9C27B0"}
        task_markers = {"share": "s", "unique1": "o", "unique2": "D"}
        mod_ls = {"m1": "-", "m2": "--"}
        mod_lw = {"m1": 2.0, "m2": 1.5}
        plotted_labels = set()

        # Lambda methods: full curves
        for method, (c_m1, c_m2) in one_mod_curves.items():
            sty = LAMBDA_CURVE_STYLES.get(method, dict(color="#607D8B", label=method))
            for mod_key, c in [("m1", c_m1), ("m2", c_m2)]:
                lam_vals = c["lambdas"]
                for task in ["share", "unique1", "unique2"]:
                    if task not in c:
                        continue
                    lab = f"{task} [{sty['label']} / {mod_key}]"
                    ax3.plot(lam_vals, c[task],
                             color=task_colors[task],
                             linestyle=mod_ls[mod_key],
                             linewidth=mod_lw[mod_key],
                             marker=task_markers[task], markersize=4,
                             alpha=0.85,
                             label=lab if lab not in plotted_labels else "_nolegend_")
                    plotted_labels.add(lab)

        # Fixed methods: single scatter points at x=0.5 (no lambda axis)
        x_positions = {"m1": 0.35, "m2": 0.65}
        for method, (p_m1, p_m2) in one_mod_fixed.items():
            sty = METHOD_STYLES.get(method, {})
            color = sty.get("color", "#607D8B")
            marker = sty.get("marker", "o")
            for mod_key, p in [("m1", p_m1), ("m2", p_m2)]:
                for task in ["share", "unique1", "unique2"]:
                    val = p.get(task, 0)
                    lab = f"{task} [{method} / {mod_key}]"
                    ax3.scatter(x_positions[mod_key], val,
                                color=task_colors[task], marker=marker,
                                s=60, zorder=5, alpha=0.8,
                                label=lab if lab not in plotted_labels else "_nolegend_")
                    plotted_labels.add(lab)
                    ax3.annotate(f"{method[:6]}/{mod_key}",
                                 (x_positions[mod_key], val),
                                 textcoords="offset points", xytext=(4, 3), fontsize=6,
                                 color=color)

        ax3.axhline(0.1, color="gray", linestyle="--", linewidth=0.8, label="chance")
        ax3.set_xlabel("lambda  (• = fixed methods at 0.35/0.65)", fontsize=10)
        ax3.set_ylabel("probe accuracy", fontsize=11)
        ax3.set_title("One-modality inference\n(solid/left=M1-only, dashed/right=M2-only)", fontsize=12)
        ax3.set_xlim(-0.02, 1.02)
        ax3.set_ylim(-0.02, 1.02)
        ax3.legend(fontsize=6, loc="center right")
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "pareto_frontier.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPareto plot saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    with open(args.results) as f:
        data = json.load(f)
    plot_pareto(data, args.out_dir)
