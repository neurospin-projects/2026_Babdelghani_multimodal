"""Preference-simplex utilities: the 15-point grid, PaLoRA annealing, and the
consensus-lambda selection rule, as used for the MultiBench experiments.
"""
import itertools
import math

import torch


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Standard linear CKA (sample-centered). Used as the "is the lambda axis
    doing anything" diagnostic throughout the STEER work: adjacent/endpoint CKA
    predicts nothing about
    accuracy or lambda-selectability, but a collapsed (~1.0) endpoint-vs-
    centroid CKA does mean the family isn't separating at all."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic = (X.T @ Y).pow(2).sum()
    norm_x = (X.T @ X).pow(2).sum().sqrt()
    norm_y = (Y.T @ Y).pow(2).sum().sqrt()
    return (hsic / (norm_x * norm_y + 1e-12)).item()


def simplex_grid(spacing: int = 4):
    """15 points on the 2-simplex at spacing 1/`spacing` (default: side-5 grid).
    Returns a list of (lambda_R, lambda_U1, lambda_U2) tuples summing to 1."""
    pts = []
    for i, j in itertools.product(range(spacing + 1), repeat=2):
        k = spacing - i - j
        if k < 0:
            continue
        pts.append((i / spacing, j / spacing, k / spacing))
    return pts


def anneal(lam, tau: float, Q: float = 1.0, eps: float = 1e-12):
    """lambda_b -> lambda_b^gamma / sum(lambda_.^gamma), gamma = tau/Q.
    Vertices (exact 0 components) and edge midpoints (equal nonzero components)
    are fixed points; only the simplex interior moves, from centroid (tau=0) to
    the nominal grid (tau=1)."""
    gamma = tau / Q
    if gamma <= 0:
        n = len(lam)
        return tuple(1.0 / n for _ in lam)
    powered = [max(x, 0.0) ** gamma for x in lam]
    total = sum(powered) + eps
    return tuple(p / total for p in powered)


def sample_preferences(grid, m: int, generator: torch.Generator = None):
    """Sample `m` preferences from `grid` without replacement (or with, if m > len(grid))."""
    n = len(grid)
    if m >= n:
        return list(grid)
    idx = torch.randperm(n, generator=generator)[:m].tolist()
    return [grid[i] for i in idx]


def consensus_lambda(scores_by_seed, grid, tol=None):
    """Consensus operating-point selection.

    scores_by_seed: dict[seed] -> list[float] validation score per grid point
                    (same order as `grid`).
    Returns (lambda_star, info) where info has acceptance_count, worst_rank,
    mean_rank, epsilon (the noise floor used).

    epsilon = mean over grid points of the seed-to-seed std of the score.
    Seed s "accepts" lambda_i if its regret (its own best score minus its score
    at lambda_i) is <= epsilon. Pick the lambda accepted by the most seeds;
    break ties by worst rank across seeds, then by mean rank.
    """
    seeds = list(scores_by_seed)
    n_grid = len(grid)
    scores = torch.tensor([scores_by_seed[s] for s in seeds])  # (n_seeds, n_grid)

    if tol is None:
        per_point_std = scores.std(dim=0, unbiased=True)
        tol = per_point_std.mean().item()

    best_per_seed = scores.max(dim=1).values  # (n_seeds,)
    regret = best_per_seed.unsqueeze(1) - scores  # (n_seeds, n_grid)
    accepted = regret <= tol  # (n_seeds, n_grid)
    acceptance_count = accepted.sum(dim=0)  # (n_grid,)

    # rank of each grid point within each seed (1 = best)
    ranks = torch.zeros_like(scores)
    for s_idx in range(scores.shape[0]):
        order = torch.argsort(scores[s_idx], descending=True)
        r = torch.empty(n_grid, dtype=torch.long)
        r[order] = torch.arange(1, n_grid + 1)
        ranks[s_idx] = r
    worst_rank = ranks.max(dim=0).values
    mean_rank = ranks.mean(dim=0)

    # pick: max acceptance, tie-break by lowest worst_rank, then lowest mean_rank
    key = list(zip((-acceptance_count).tolist(), worst_rank.tolist(), mean_rank.tolist()))
    best_idx = min(range(n_grid), key=lambda i: key[i])

    info = dict(
        epsilon=tol,
        acceptance_count=int(acceptance_count[best_idx]),
        n_seeds=len(seeds),
        worst_rank=float(worst_rank[best_idx]),
        mean_rank=float(mean_rank[best_idx]),
    )
    return grid[best_idx], info


def dim_readout(blocks, weights, total_dim):
    """the "dim" readout: allocate `total_dim` output
    dimensions across `blocks` in proportion to `weights` by largest remainder,
    keep the first d_b coords of each block, concat + renormalise.

    blocks: list of (B, D_b) tensors (already L2-normalised), one per PID block.
    weights: list of floats, same length, summing to 1 (a grid point).
    Returns (B, total_dim) L2-normalised tensor.
    """
    import torch.nn.functional as F

    n = len(blocks)
    raw = [w * total_dim for w in weights]
    base = [int(math.floor(r)) for r in raw]
    remainder = total_dim - sum(base)
    order = sorted(range(n), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:remainder]:
        base[i] += 1

    pieces = []
    B = blocks[0].shape[0]
    device = blocks[0].device
    for blk, d_b in zip(blocks, base):
        if d_b == 0:
            continue
        pieces.append(blk[:, :d_b])
    if not pieces:
        return torch.zeros(B, total_dim, device=device)
    z = torch.cat(pieces, dim=-1)
    if z.shape[1] < total_dim:
        z = F.pad(z, (0, total_dim - z.shape[1]))
    return F.normalize(z, dim=-1)


def fit_block_pca(blocks_full):
    """Fit one PCA per block -- orders each block's own D_b coordinates by
    variance explained instead of leaving them in their arbitrary (but
    deterministic) native order. Feeds dim_readout_pca() below. Fit ONCE on a
    representative population (e.g. the whole extraction cohort at a given
    lambda -- note the block outputs themselves are lambda-dependent, since
    the encoder is preference-conditioned, so a fit is only valid for the
    SAME lambda point it's later applied at), reused across every subject
    afterward -- unlike dim_readout's plain first-d_b-coordinates slice,
    PCA needs a real sample to estimate directions of variance from, and a
    single minibatch (or the tiny --probe_limit CKA batch) is far too small/
    unstable a sample for this, especially for a wide raw block.

    blocks_full: list of (N, D_b) tensors/arrays, N should be >> D_b for a
    stable fit (e.g. the full paired cohort at this lambda, not one batch).
    Returns: list of fitted sklearn.decomposition.PCA objects, one per block.
    """
    from sklearn.decomposition import PCA

    pcas = []
    for blk in blocks_full:
        arr = blk.detach().cpu().numpy() if hasattr(blk, "detach") else blk
        n, d = arr.shape
        if n < d:
            # e.g. a --limit smoke test with fewer subjects than block width --
            # PCA can only ever return min(n, d) components, so cap instead of
            # crashing; dim_readout_pca's own d_b<=n check further down still
            # governs whether the eventual SELECTION is meaningful, this only
            # keeps the FIT itself from raising.
            print(f"[fit_block_pca] WARNING: block has n={n} samples < d={d} dims -- "
                  f"PCA capped to {n} components, fit will be unstable/degenerate. "
                  f"Use a larger population for a real (non-smoke-test) run.")
        pca = PCA(n_components=min(n, d))
        pca.fit(arr)
        pcas.append(pca)
    return pcas


def dim_readout_pca(blocks, weights, total_dim, pcas):
    """Same dimension ALLOCATION as dim_readout() (largest-remainder d_b per
    block), but instead of keeping each block's first d_b RAW coordinates
    (arbitrary -- see dim_readout's own docstring: "coordinates carry no
    privileged order"), projects onto that block's PRE-FITTED PCA basis (see
    fit_block_pca) and keeps the top d_b PRINCIPAL components -- the d_b
    directions carrying the most of that block's own variance, rather than
    an arbitrary axis-aligned slice.

    Only changes MIXED lambda points (any point where a block's own
    d_b < D_b, e.g. lambda=(0.25,0.25,0.50)). At a PURE VERTEX a block
    already gets its full D_b (nothing to select, d_b == D_b), so this is
    numerically identical to dim_readout() there -- confirmed intentional,
    not a bug: it means every pure-vertex score anywhere in this project is
    completely unaffected by switching readouts.

    pcas: list of already-.fit()'d sklearn PCA objects, one per block, from
    fit_block_pca() called on THIS SAME lambda's own block outputs across a
    real population -- passing PCAs fit at a different lambda, or fit on too
    few subjects, silently gives a degenerate/meaningless projection.
    """
    import torch.nn.functional as F

    n = len(blocks)
    raw = [w * total_dim for w in weights]
    base = [int(math.floor(r)) for r in raw]
    remainder = total_dim - sum(base)
    order = sorted(range(n), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:remainder]:
        base[i] += 1

    pieces = []
    B = blocks[0].shape[0]
    device = blocks[0].device
    for blk, d_b, pca in zip(blocks, base, pcas):
        if d_b == 0:
            continue
        if d_b >= blk.shape[1]:
            # PURE VERTEX (or any point where this block gets its full raw
            # width): nothing to select, so skip PCA entirely and keep the
            # raw block -- pca.transform() mean-centers + rotates even when
            # every component is kept, which would silently change values at
            # vertices despite there being no actual truncation. Bypassing
            # here is what makes dim_readout_pca's pure-vertex numbers
            # provably identical to dim_readout()'s, as scoped.
            pieces.append(blk[:, :d_b])
            continue
        arr = blk.detach().cpu().numpy()
        proj = pca.transform(arr)[:, :d_b]
        pieces.append(torch.as_tensor(proj, dtype=blk.dtype, device=device))
    if not pieces:
        return torch.zeros(B, total_dim, device=device)
    z = torch.cat(pieces, dim=-1)
    if z.shape[1] < total_dim:
        z = F.pad(z, (0, total_dim - z.shape[1]))
    return F.normalize(z, dim=-1)
