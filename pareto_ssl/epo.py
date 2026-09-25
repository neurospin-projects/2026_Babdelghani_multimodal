"""
EPO (Exact Pareto Optimality) solver for combining two losses.

Source: https://github.com/dbmptr/EPOSearch
Reference: Mahapatra & Rajan (2020) "Multi-Task Learning as Multi-Objective
           Optimization" — adapted for 2-objective Pareto SSL.

Only EPO is implemented here. Linear scalarization (LS) is handled inline
in benchmark.py as the default (no import needed for the fast path).
"""

import warnings
from abc import abstractmethod

import numpy as np
import torch

# cvxopt and cvxpy are imported lazily inside ExactParetoLP.__init__
# so the module loads fine even when those packages are not installed.
# They are only required when pareto_solver="epo" is actually used.


# Abstract base

class Solver:
    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks

    @abstractmethod
    def get_weighted_loss(self, losses, ray, parameters=None, **kwargs):
        pass

    def __call__(self, losses, ray, parameters, **kwargs):
        return self.get_weighted_loss(losses, ray, parameters, **kwargs)


# EPO wrapper

class EPOSolver(Solver):
    """
    Drop-in solver for EPO.

    Usage:
        solver = EPOSolver(n_tasks=2, n_params=n_trainable_params)
        weighted_loss = solver(
            losses = torch.stack([L_R, L_U]),
            ray    = torch.tensor([lam, 1 - lam], device=device),
            parameters = list(model.parameters()),
        )
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

    Notes
    -----
    * EPO calls torch.autograd.grad internally (with retain_graph=True) to
      find the optimal alpha; the returned weighted_loss is still in the
      computation graph, so the subsequent .backward() does the real update.
    * allow_unused=True is set so that heads specialised to one loss
      (proj_r only receives L_R gradient, proj_u only L_U) don't raise errors.
      Missing gradients are replaced with zeros.
    """

    def __init__(self, n_tasks: int, n_params: int):
        super().__init__(n_tasks)
        self.epo = EPO(n_tasks=n_tasks, n_params=n_params)

    def get_weighted_loss(self, losses, ray, parameters=None, **kwargs):
        assert parameters is not None, "EPOSolver requires parameters="
        return self.epo.get_weighted_loss(losses, ray, parameters)


# EPO core

class EPO:
    def __init__(self, n_tasks: int, n_params: int):
        self.n_tasks  = n_tasks
        self.n_params = n_params

    def __call__(self, losses, ray, parameters):
        return self.get_weighted_loss(losses, ray, parameters)

    @staticmethod
    def _flatten(grad, parameters):
        """Flatten per-parameter gradients into one vector; None → zeros."""
        return torch.cat(
            tuple(
                (g if g is not None else torch.zeros_like(p)).reshape(-1)
                for g, p in zip(grad, parameters)
            ),
            dim=0,
        )

    def get_weighted_loss(self, losses, ray, parameters):
        lp = ExactParetoLP(m=self.n_tasks, n=self.n_params, r=ray.cpu().numpy())

        grads = []
        for loss in losses:
            g = torch.autograd.grad(
                loss, parameters,
                retain_graph=True,
                allow_unused=True,   # heads specialized to one loss have None grads
            )
            grads.append(self._flatten(g, parameters).data)

        G     = torch.stack(grads)           # (n_tasks, n_params)
        GG_T  = (G @ G.T).detach().cpu().numpy()
        l_np  = losses.detach().cpu().numpy()

        try:
            alpha = lp.get_alpha(l_np, G=GG_T, C=True)
        except Exception as exc:
            print(f"[EPO] LP failed ({exc}), falling back to ray weights")
            alpha = None

        if alpha is None:
            alpha = (ray / ray.sum()).cpu().numpy()

        alpha  = torch.from_numpy(alpha * self.n_tasks).to(losses.device)
        return torch.sum(losses * alpha)


# LP problem

class ExactParetoLP:
    """
    Solves two GLPK LPs (balance / dominance) to find the Pareto-optimal
    gradient combination alpha.  Adapted from https://github.com/dbmptr/EPOSearch.
    """

    def __init__(self, m: int, n: int, r, eps: float = 1e-4):
        try:
            import cvxopt
            import cvxpy as cp
        except ImportError as e:
            raise ImportError(
                "EPO requires cvxopt and cvxpy. "
                "Install them with:  pip install cvxopt cvxpy\n"
                f"Original error: {e}"
            ) from e

        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self._GLPK = cp.GLPK   # store solver constant for use in get_alpha
        self.m, self.n, self.r, self.eps = m, n, r, eps
        self.last_move = None

        # CVXPY parameters (filled each call to get_alpha)
        self.a   = cp.Parameter(m)
        self.C   = cp.Parameter((m, m))
        self.Ca  = cp.Parameter(m)
        self.rhs = cp.Parameter(m)

        self.alpha = cp.Variable(m)

        # Balance LP
        self.prob_bal = cp.Problem(
            cp.Maximize(self.alpha @ self.Ca),
            [self.alpha >= 0, cp.sum(self.alpha) == 1, self.C @ self.alpha >= self.rhs],
        )

        # Dominance LPs
        obj_dom = cp.Maximize(cp.sum(self.alpha @ self.C))
        self.prob_dom = cp.Problem(
            obj_dom,
            [
                self.alpha >= 0,
                cp.sum(self.alpha) == 1,
                self.alpha @ self.Ca >= -cp.neg(cp.max(self.Ca)),
                self.C @ self.alpha >= 0,
            ],
        )
        self.prob_rel = cp.Problem(
            obj_dom,
            [self.alpha >= 0, cp.sum(self.alpha) == 1, self.C @ self.alpha >= 0],
        )

        self.gamma  = 0.0
        self.mu_rl  = 0.0

    def get_alpha(self, l, G, r=None, C: bool = False, relax: bool = False):
        r = self.r if r is None else r
        assert len(l) == len(G) == len(r) == self.m

        rl, self.mu_rl, self.a.value = adjustments(l, r)
        self.C.value  = G if C else G @ G.T
        self.Ca.value = self.C.value @ self.a.value

        if self.mu_rl > self.eps:
            J = self.Ca.value > 0
            if np.where(J)[0].size > 0:
                J_star = np.where(rl == np.max(rl))[0]
                self.rhs.value            = self.Ca.value.copy()
                self.rhs.value[J]         = -np.inf
                self.rhs.value[J_star]    = 0
            else:
                self.rhs.value = np.zeros_like(self.Ca.value)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*DPP.*", category=UserWarning)
                self.prob_bal.solve(solver=self._GLPK, verbose=False)
            self.last_move = "bal"
        else:
            prob = self.prob_rel if relax else self.prob_dom
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*DPP.*", category=UserWarning)
                prob.solve(solver=self._GLPK, verbose=False)
            self.last_move = "dom"

        return self.alpha.value


# Helpers

def mu(rl, normed: bool = False) -> float:
    if np.any(rl < 0):
        raise ValueError(f"rl contains negatives: {rl}")
    m     = len(rl)
    l_hat = rl if normed else rl / rl.sum()
    eps   = np.finfo(rl.dtype).eps
    l_hat = l_hat[l_hat > eps]
    return float(np.sum(l_hat * np.log(l_hat * m)))


def adjustments(l, r=1):
    # Clip r away from 0 to avoid log(0)=−∞ when λ∈{0,1}.
    # 1e-9 is negligible relative to any real loss value, so semantics are preserved.
    r      = np.clip(r, 1e-9, np.inf)
    rl     = r * l
    l_hat  = rl / rl.sum()
    mu_rl  = mu(l_hat, normed=True)
    a      = r * (np.log(l_hat * len(l)) - mu_rl)
    return rl, mu_rl, a
