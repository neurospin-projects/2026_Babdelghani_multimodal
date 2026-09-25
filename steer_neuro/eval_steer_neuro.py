"""Sanity-check a trained STEER-neuro checkpoint. Computes the `dim`-readout
embedding z(lambda) at every grid point on a held-out subject batch and
reports four things, each answering a different failure mode:
  - per-block mean norm at each lambda (a block should be exactly silent only
    at its own zero-vertex, per the readout rule)
  - endpoint CKA, vertex vs centroid (collapse toward 1.0 means the lambda
    axis isn't doing anything at all -- the failure mode CLUB + dim readout
    were built to avoid)
  - decomp/cka_u_r: is the "unique" head actually unique, or a redundant copy
    of the "shared" head? (diagnostics.py::unique_shared_cka)
  - adapter/endpoint_cka, pairwise among the 3 vertices: does lambda change
    anything between any PAIR of pure preferences, not just vs the centroid?
    (diagnostics.py::endpoint_cka_pairwise)
  - probe_cka_matrix: is the curve learned or merely interpolated? ratio ~ 0.5
    along an edge means the interior point looks like a linear blend of its
    endpoints; deviation means it's a genuinely different, learned subspace
    (diagnostics.py::all_edge_interpolation_ratios)

This is a diagnostic, not a downstream-task probe (no label is used) -- it
answers "did training produce a usable family at all", which the STEER papers
insist on checking before any lambda-selection claim.
"""
import argparse

import torch

from diagnostics import unique_shared_cka, endpoint_cka_pairwise, all_edge_interpolation_ratios
from model import SteerNeuroModel
from paired_dataset import PairedSkeletonDataset, paired_subjects
from simplex import simplex_grid, dim_readout, linear_cka
from wrap_lora import SCOPES


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--limit", type=int, default=64, help="held-out subject count for this diagnostic")
    p.add_argument("--total_dim", type=int, default=None, help="defaults to proj_dim (the 'dim' readout width, see the method description sec:readout)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]
    assert train_args["scope"] in SCOPES

    model = SteerNeuroModel(scope=train_args["scope"], rank=train_args["rank"],
                             rank_champo=train_args.get("rank_champo"), rank_alma=train_args.get("rank_alma"),
                             rank_ratio=train_args.get("rank_ratio"),
                             rank_ratio_champo=train_args.get("rank_ratio_champo"),
                             rank_ratio_alma=train_args.get("rank_ratio_alma"),
                             alpha=train_args["alpha"], alpha_champo=train_args.get("alpha_champo"),
                             alpha_alma=train_args.get("alpha_alma"), proj_dim=train_args["proj_dim"],
                             head_hidden=train_args["head_hidden"],
                             club_hidden=train_args["club_hidden"],
                             beta_club=train_args["beta_club"]).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_dim = args.total_dim or train_args["proj_dim"]  # "dim" readout width is D, not 3*D

    subjects = paired_subjects(limit=args.limit)
    ds = PairedSkeletonDataset(subjects=subjects)
    loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False, num_workers=0)
    champo_a, _, alma_a, _, _ = next(iter(loader))
    champo_a, alma_a = champo_a.to(args.device), alma_a.to(args.device)

    grid = simplex_grid()
    z_by_pref = {}
    print(f"{'lambda':>22s}  {'|R|':>6s} {'|U_champo|':>10s} {'|U_alma|':>8s}")
    for lam in grid:
        r, uc, ua = model.embed_blocks(champo_a, alma_a, *lam)
        z = dim_readout([r, uc, ua], list(lam), total_dim=total_dim)
        z_by_pref[lam] = z
        print(f"{str(tuple(round(x, 2) for x in lam)):>22s}  "
              f"{r.norm(dim=-1).mean().item():6.3f} {uc.norm(dim=-1).mean().item():10.3f} "
              f"{ua.norm(dim=-1).mean().item():8.3f}")

    centroid = min(grid, key=lambda l: max(abs(x - 1 / 3) for x in l))
    vertices = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    print("\nendpoint CKA (vertex vs centroid) -- should be well below ~0.9 for a live axis:")
    for v in vertices:
        cka = linear_cka(z_by_pref[v], z_by_pref[centroid])
        print(f"  CKA(vertex={v}, centroid={centroid}) = {cka:.4f}")

    all_pts = list(z_by_pref.keys())
    ckas = [linear_cka(z_by_pref[all_pts[i]], z_by_pref[all_pts[j]])
            for i in range(len(all_pts)) for j in range(i + 1, len(all_pts))]
    print(f"\nall-pairs CKA: min={min(ckas):.4f} mean={sum(ckas)/len(ckas):.4f} max={max(ckas):.4f}")

    print("\ndecomp/cka_u_r -- is the unique head actually unique? (LOW is good)")
    usc = unique_shared_cka(model, champo_a, alma_a, grid=grid)
    print(f"  cka(u_champo, r_champo) = {usc['cka_u_r_champo']:.4f}")
    print(f"  cka(u_alma,   r_alma)   = {usc['cka_u_r_alma']:.4f}")
    print(f"  mean                    = {usc['cka_u_r_mean']:.4f}")

    print("\nadapter/endpoint_cka -- pairwise among the 3 vertices (~1 = no specialisation between that pair):")
    pw = endpoint_cka_pairwise(model, champo_a, alma_a, total_dim=total_dim)
    for k, v in pw.items():
        print(f"  {k} = {v:.4f}")

    print("\nprobe_cka_matrix -- is the curve learned or merely interpolated? (ratio ~0.5 = linear blend, deviation = real curve)")
    edges = all_edge_interpolation_ratios(model, champo_a, alma_a, total_dim=total_dim)
    for edge, r in edges.items():
        print(f"  edge {edge}: CKA(0,mid)={r['cka_0_mid']:.4f} CKA(0,1)={r['cka_0_1']:.4f} ratio={r['ratio']:.4f}")


if __name__ == "__main__":
    main()
