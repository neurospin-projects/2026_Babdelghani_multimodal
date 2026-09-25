#!/usr/bin/env python3
"""BROJA-2PID (Bertschinger et al., 2014) on a discrete joint p(x1, x2, y), via cvxpy.

Used for the CVX arm of the PID plan: exact on small alphabets, no estimator noise,
unlike the BATCH neural estimator that the continuous datasets need.

    Delta_p = { q >= 0 : sum_x2 q = p(x1,y),  sum_x1 q = p(x2,y) }
    q*      = argmin_{q in Delta_p} I_q(X1,X2 ; Y)

    UI1 = I_q*(X1;Y | X2)      unique to X1
    UI2 = I_q*(X2;Y | X1)      unique to X2
    R   = I_p(X1;Y) - UI1      redundant   (= I_p(X2;Y) - UI2 at the optimum)
    S   = I_p(X1,X2;Y) - I_q*(X1,X2;Y)     synergistic

Every marginal of Y and every pairwise (Xi, Y) marginal is fixed by the constraints, so
I_p(Xi;Y) may be computed under p; only the terms involving the *joint* source differ
between p and q*. Objective: minimising I_q(X1X2;Y) equals minimising -H_q(Y|X1X2)
because H(Y) is fixed, and -H_q(Y|X1X2) = sum rel_entr(q, q_x) is convex in q, which is
what makes this a convex program rather than the non-convex-looking MI minimisation.

Returns bits.
"""
import numpy as np

LOG2 = np.log(2.0)


def _mi(p_xy):
    """I(X;Y) in bits from a 2-D joint."""
    p_xy = np.asarray(p_xy, float)
    px = p_xy.sum(1, keepdims=True)
    py = p_xy.sum(0, keepdims=True)
    nz = p_xy > 0
    return float((p_xy[nz] * np.log(p_xy[nz] / (px @ py)[nz])).sum() / LOG2)


def _cmi_x1_y_given_x2(q):
    """I(X1;Y|X2) in bits from a 3-D joint q[x1,x2,y]."""
    q = np.asarray(q, float)
    q_x2 = q.sum((0, 2))                       # [x2]
    q_x1x2 = q.sum(2)                          # [x1,x2]
    q_x2y = q.sum(0)                           # [x2,y]
    num = q * q_x2[None, :, None]
    den = q_x1x2[:, :, None] * q_x2y[None, :, :]
    nz = (q > 0) & (den > 0)
    return float((q[nz] * np.log(num[nz] / den[nz])).sum() / LOG2)


def broja_pid(p, solver=None, verbose=False):
    """p: 3-D array p[x1, x2, y], sums to 1. Returns dict of R/U1/U2/S in bits."""
    import cvxpy as cp
    p = np.asarray(p, float)
    p = p / p.sum()
    n1, n2, ny = p.shape

    p_x1y = p.sum(1)          # [x1,y]
    p_x2y = p.sum(0)          # [x2,y]

    # cvxpy has no 3-D variables, so q is kept as [(x1 x2), y] and the two source
    # marginals are taken with sparse selector matrices -- dense ones would be
    # n1*n2*n1 floats, i.e. hundreds of MB once an alphabet reaches ~150 symbols.
    import scipy.sparse as sp
    qf = cp.Variable((n1 * n2, ny), nonneg=True)
    rows = n1 * n2
    cols = np.arange(rows)
    A1 = sp.csr_matrix((np.ones(rows), (np.repeat(np.arange(n1), n2), cols)),
                       shape=(n1, rows))       # sum over x2 -> [x1, y]
    A2 = sp.csr_matrix((np.ones(rows), (np.tile(np.arange(n2), n1), cols)),
                       shape=(n2, rows))       # sum over x1 -> [x2, y]
    cons = [A1 @ qf == p_x1y, A2 @ qf == p_x2y]

    q_x = cp.sum(qf, axis=1, keepdims=True)    # [(x1 x2), 1]
    obj = cp.Minimize(cp.sum(cp.rel_entr(qf, q_x @ np.ones((1, ny)))))

    prob = cp.Problem(obj, cons)
    tried, best = [], None
    # "optimal_inaccurate" still returns a point, but on these structured joints it can
    # be far enough off to move a component by more than the effect being measured, so
    # an inaccurate solve is only accepted once every solver has had a turn.
    for s in ([solver] if solver else ["CLARABEL", "SCS", "ECOS"]):
        try:
            prob.solve(solver=s, verbose=verbose)
            tried.append((s, prob.status))
            if qf.value is None:
                continue
            if prob.status == "optimal":
                best = (np.asarray(qf.value), s, prob.status)
                break
            if best is None:
                best = (np.asarray(qf.value), s, prob.status)
        except Exception as e:                  # solver missing or failed
            tried.append((s, f"error: {type(e).__name__}"))
    if best is None:
        raise RuntimeError(f"no solver succeeded: {tried}")
    qval, used_solver, used_status = best

    qv = np.maximum(qval.reshape(n1, n2, ny), 0.0)
    qv /= qv.sum()

    ui1 = _cmi_x1_y_given_x2(qv)
    ui2 = _cmi_x1_y_given_x2(np.transpose(qv, (1, 0, 2)))
    red = _mi(p_x1y) - ui1
    i_joint_p = _mi(p.reshape(n1 * n2, ny))
    syn = i_joint_p - _mi(qv.reshape(n1 * n2, ny))
    return dict(R=red, U1=ui1, U2=ui2, S=syn, I_total=i_joint_p,
                I_x1y=_mi(p_x1y), I_x2y=_mi(p_x2y),
                red_consistency=abs((_mi(p_x2y) - ui2) - red),
                status=used_status, solver=used_solver, attempts=tried)


# canonical gates, for validating the solver
def _gates():
    """Four distributions whose PID is known analytically (bits)."""
    g = {}
    # RDN: X1 = X2 = Y ~ Bern(1/2)          -> R = 1
    p = np.zeros((2, 2, 2)); p[0, 0, 0] = p[1, 1, 1] = 0.5
    g["RDN"] = (p, dict(R=1, U1=0, U2=0, S=0))
    # UNQ: Y = (X1, X2), both uniform bits  -> U1 = U2 = 1
    p = np.zeros((2, 2, 4))
    for a in range(2):
        for b in range(2):
            p[a, b, 2 * a + b] = 0.25
    g["UNQ"] = (p, dict(R=0, U1=1, U2=1, S=0))
    # XOR: Y = X1 xor X2                    -> S = 1
    p = np.zeros((2, 2, 2))
    for a in range(2):
        for b in range(2):
            p[a, b, a ^ b] = 0.25
    g["XOR"] = (p, dict(R=0, U1=0, U2=0, S=1))
    # AND: Y = X1 and X2   -> R=.311 U=0 S=.5 (Bertschinger et al. 2014, Table 1)
    p = np.zeros((2, 2, 2))
    for a in range(2):
        for b in range(2):
            p[a, b, a & b] = 0.25
    g["AND"] = (p, dict(R=0.31127812, U1=0.0, U2=0.0, S=0.5))
    return g


def validate(tol=5e-3, verbose=True):
    """Solve the gates; raise if any component is off by more than tol bits."""
    bad = []
    for name, (p, want) in _gates().items():
        got = broja_pid(p)
        err = {k: abs(got[k] - want[k]) for k in ("R", "U1", "U2", "S")}
        if verbose:
            print(f"  {name:4s} R={got['R']:.4f} U1={got['U1']:.4f} "
                  f"U2={got['U2']:.4f} S={got['S']:.4f}   max err {max(err.values()):.1e}")
        if max(err.values()) > tol:
            bad.append((name, err))
    if bad:
        raise AssertionError(f"BROJA solver failed validation: {bad}")
    return True


if __name__ == "__main__":
    print("BROJA-2PID solver validation (bits):")
    validate()
    print("OK")
