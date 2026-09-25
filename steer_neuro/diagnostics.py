"""Quality diagnostics for a trained SteerNeuroModel, ported from the four
measures the user specified (originally implemented in pareto_ssl's
benchmark_multibench.py / probe.py for the 1-D-lambda MOSEI work), adapted to
this project's 2-simplex preference space. Each answers a different failure
mode a reviewer would probe for:

  unique_shared_cka   -- decomp/cka_u_r: is the "unique" head actually unique,
                         or just a redundant copy of the "shared" head?
  endpoint_cka_pairwise -- adapter/endpoint_cka generalised to 3 vertices: does
                         lambda change the representation AT ALL between any
                         pair of pure preferences?
  edge_interpolation_ratio -- probe_cka_matrix: is the interior of an edge a
                         genuinely different representation, or just a linear
                         blend of its two endpoints (i.e. "we trained 2 points
                         and interpolated" rather than "we trained many")?

All three take a model + a (champo, alma) probe batch and are read-only
(@torch.no_grad()) -- safe to call periodically during training or once
post-hoc on a saved checkpoint. They are wired into train_steer_neuro.py and
eval_steer_neuro.py.
"""
import torch
import torch.nn.functional as F

from simplex import linear_cka, dim_readout, simplex_grid

AXIS = {"R": 0, "U_champo": 1, "U_alma": 2}


@torch.no_grad()
def unique_shared_cka(model, champo, alma, grid=None):
    """decomp/cka_u_r -- "is the unique head actually unique?"

    Linear CKA between each modality's unique-head output and its own
    shared-head output, at every preference in `grid` (default: the full
    15-point grid), averaged over the grid and over both modalities. LOW is
    good: it means u_m(h_m(lambda)) is not just a copy of r_m(h_m(lambda)).
    """
    grid = grid or simplex_grid()
    vals = []
    for lam in grid:
        w_r, w_uc, w_ua = lam
        h_c = model.encode_champo(champo, w_r, w_uc)
        h_a = model.encode_alma(alma, w_r, w_ua)
        r_c = F.normalize(model.r_champo(h_c), dim=-1)
        u_c = F.normalize(model.u_champo(h_c), dim=-1)
        r_a = F.normalize(model.r_alma(h_a), dim=-1)
        u_a = F.normalize(model.u_alma(h_a), dim=-1)
        vals.append(linear_cka(u_c, r_c))
        vals.append(linear_cka(u_a, r_a))
    mean = sum(vals) / len(vals)
    champo_vals = vals[0::2]
    alma_vals = vals[1::2]
    return dict(
        cka_u_r_mean=mean,
        cka_u_r_champo=sum(champo_vals) / len(champo_vals),
        cka_u_r_alma=sum(alma_vals) / len(alma_vals),
    )


@torch.no_grad()
def endpoint_cka_pairwise(model, champo, alma, total_dim=None):
    """adapter/endpoint_cka, generalised from a single (0,1) pair to the 3
    pure vertices of the 2-simplex -- pairwise CKA among z(1,0,0), z(0,1,0),
    z(0,0,1) (dim readout). ~1 for a pair means those two preferences produce
    the SAME representation -- no specialisation between them. This is a
    stricter, more direct check than model.py's endpoint_cka_diagnostic
    (which compares each vertex to the centroid, not vertices to each other).
    """
    total_dim = total_dim or model.proj_dim  # "dim" readout width is D, not 3*D
    vertices = {"R": (1.0, 0.0, 0.0), "U_champo": (0.0, 1.0, 0.0), "U_alma": (0.0, 0.0, 1.0)}
    z = {}
    for name, lam in vertices.items():
        r, uc, ua = model.embed_blocks(champo, alma, *lam)
        z[name] = dim_readout([r, uc, ua], list(lam), total_dim=total_dim)
    pairs = [("R", "U_champo"), ("R", "U_alma"), ("U_champo", "U_alma")]
    return {f"cka_{a}_{b}": linear_cka(z[a], z[b]) for a, b in pairs}


@torch.no_grad()
def edge_interpolation_ratio(model, champo, alma, edge=("R", "U_champo"), total_dim=None):
    """probe_cka_matrix -- "is the curve learned or merely interpolated?"

    Sweeps t in {0, 0.25, 0.5, 0.75, 1} along one edge of the simplex (the
    other axis held at 0), computing CKA(z_0, z_mid) and CKA(z_0, z_1).
      - ratio ~ 0.5  => z_mid behaves like a linear blend of the two
        endpoints -- consistent with "we trained 2 points and interpolated".
      - ratio far from 0.5 => the interior point occupies its own subspace,
        not predictable from the endpoints alone -- consistent with a
        genuinely learned (not merely interpolated) curve.
    `edge` names two of {"R","U_champo","U_alma"}; the third axis is 0 along
    the whole sweep.
    """
    total_dim = total_dim or model.proj_dim  # "dim" readout width is D, not 3*D
    i0, i1 = AXIS[edge[0]], AXIS[edge[1]]
    ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    zs = []
    for t in ts:
        lam = [0.0, 0.0, 0.0]
        lam[i0] = 1.0 - t
        lam[i1] = t
        r, uc, ua = model.embed_blocks(champo, alma, *lam)
        zs.append(dim_readout([r, uc, ua], lam, total_dim=total_dim))
    cka_0_mid = linear_cka(zs[0], zs[2])
    cka_0_1 = linear_cka(zs[0], zs[4])
    ratio = cka_0_mid / cka_0_1 if abs(cka_0_1) > 1e-8 else float("nan")
    return dict(cka_0_mid=cka_0_mid, cka_0_1=cka_0_1, ratio=ratio)


@torch.no_grad()
def all_edge_interpolation_ratios(model, champo, alma, total_dim=None):
    """edge_interpolation_ratio for all 3 edges of the simplex, keyed by edge name."""
    edges = [("R", "U_champo"), ("R", "U_alma"), ("U_champo", "U_alma")]
    return {f"{a}_{b}": edge_interpolation_ratio(model, champo, alma, edge=(a, b), total_dim=total_dim)
            for a, b in edges}
