"""Constrained mixed (continuous + 2 discrete) GP-BO toolkit for task 005.

All points produced here are snapped (x9, x10 at exact levels) and satisfy the
six linear constraints with a safety margin.
"""
import math
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.optimize import minimize
from scipy.stats import qmc, spearmanr, skew, norm, rankdata

torch.set_default_dtype(torch.double)

D = 11
CONT = list(range(9))
A = np.zeros((6, 11))
A[0, [1, 4]] = [-0.80, 0.86]
A[1, [0, 3]] = [-0.54, 0.83]
A[2, [0, 4]] = [0.54, -0.86]
A[3, [0, 4]] = [-0.54, 0.86]
A[4, [0, 1, 2]] = [0.54, -0.80, 0.50]
A[5, [3, 4, 5]] = [0.83, -0.86, 0.96]
b = np.array([0.63, 0.62, 0.37, 0.54, 0.50, 0.81])
MARGIN = 1e-6
L9 = np.array([0.0, 0.166667, 1.0])
L10 = np.array([0, 0.075697, 0.167331, 0.187251, 0.219124, 0.227092, 0.239044,
                0.286853, 0.298805, 0.418327, 0.657371, 0.697211, 0.984064, 1.0])
COMBOS = np.array([(a, c) for a in L9 for c in L10])  # 42 x 2


def _nearest(v, levels):
    d = np.abs(levels - v)
    m = d.min()
    return levels[np.flatnonzero(d <= m + 1e-15)[0]]  # ties -> lower


def snap(x):
    x = np.clip(np.asarray(x, float).copy(), 0.0, 1.0)
    x[9] = _nearest(x[9], L9)
    x[10] = _nearest(x[10], L10)
    return x


def violation(x):
    return np.maximum(A @ np.asarray(x) - b, 0).sum()


def allowed(x):
    return violation(x) <= 1e-10


def allowed_strict(X):
    X = np.atleast_2d(X)
    return np.all(X @ A.T <= b - MARGIN / 2, axis=1) & np.all((X >= 0) & (X <= 1), axis=1)


def project(x):
    """Minimum-distance projection onto {A x <= b - MARGIN} within the box."""
    x0 = np.asarray(x, float)
    cons = [{"type": "ineq", "fun": lambda z, i=i: (b[i] - MARGIN) - A[i] @ z,
             "jac": lambda z, i=i: -A[i]} for i in range(6)]
    res = minimize(lambda z: 0.5 * np.sum((z - x0) ** 2), x0, jac=lambda z: z - x0,
                   bounds=[(0, 1)] * D, constraints=cons, method="SLSQP",
                   options={"ftol": 1e-14, "maxiter": 500})
    z = res.x.copy()
    z[9], z[10] = x0[9], x0[10]
    return z


def sample_allowed(n, rng, lo=None, hi=None):
    """Uniform allowed points (continuous dims), random discrete levels."""
    lo = np.zeros(D) if lo is None else lo
    hi = np.ones(D) if hi is None else hi
    out = []
    tries = 0
    while sum(len(o) for o in out) < n and tries < 200:
        Z = lo + (hi - lo) * rng.random((max(4 * n, 256), D))
        Z[:, 9] = rng.choice(L9, len(Z))
        Z[:, 10] = rng.choice(L10, len(Z))
        out.append(Z[allowed_strict(Z)])
        tries += 1
    Z = np.vstack(out)[:n] if out else np.empty((0, D))
    return Z


def initial_design(n, seed):
    rng = np.random.default_rng(seed)
    sob = qmc.Sobol(d=9, scramble=True, seed=seed).random(4096)
    P = np.zeros((len(sob), D))
    P[:, :9] = sob
    P = P[allowed_strict(P)]
    # greedy maximin on continuous dims
    idx = [int(rng.integers(len(P)))]
    dmin = np.linalg.norm(P - P[idx[0]], axis=1)
    while len(idx) < n:
        j = int(np.argmax(dmin))
        idx.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(P - P[j], axis=1))
    X = P[idx].copy()
    # stratified discrete levels (random permutations)
    X[:, 9] = rng.permutation(np.resize(L9, n))
    X[:, 10] = rng.permutation(np.resize(L10, n))
    return X


# ---------------------------------------------------------------- modeling

def transform_y(y, mode):
    y = np.asarray(y, float)
    if mode == "none":
        return y.copy()
    if mode == "log":
        iqr = np.subtract(*np.percentile(y, [75, 25]))
        delta = max(0.1 * iqr, 1e-12)
        return np.log(y - y.min() + delta)
    if mode == "rank":
        r = rankdata(y)
        return norm.ppf((r - 0.5) / len(y))
    raise ValueError(mode)


def heavy_tail_check(y):
    y = np.asarray(y, float)
    med = np.median(y)
    ratio = (y.max() - med) / max(med - y.min(), 1e-12)
    return {"tail_ratio": float(ratio), "skew": float(skew(y)),
            "heavy": bool(ratio > 10 or skew(y) > 2)}


def fit_gp(X, yt):
    X_t = torch.tensor(np.asarray(X), dtype=torch.double)
    Y_t = torch.tensor(np.asarray(yt), dtype=torch.double).unsqueeze(-1)
    model = SingleTaskGP(X_t, Y_t, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return model


def loo_spearman(model):
    with torch.no_grad():
        Xtr = model.train_inputs[0]
        ytr = model.train_targets
        K = model.covar_module(Xtr).to_dense()
        noise = model.likelihood.noise.squeeze()
        K = K + noise * torch.eye(len(ytr))
        c = model.mean_module.constant
        Kinv = torch.linalg.inv(K)
        alpha = Kinv @ (ytr - c)
        loo = ytr - alpha / torch.diag(Kinv)
    return float(spearmanr(loo.numpy(), ytr.numpy()).statistic)


def lengthscales(model):
    return model.covar_module.lengthscale.detach().numpy().ravel()


from botorch.acquisition.analytic import AnalyticAcquisitionFunction, _log_ei_helper
from botorch.utils.transforms import t_batch_mode_transform


class LogEIObs(AnalyticAcquisitionFunction):
    """Log-EI (minimization) on the predictive distribution of observations,
    i.e. latent posterior plus the fitted nugget variance."""

    def __init__(self, model, best_f):
        super().__init__(model=model)
        self.best_f = best_f

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        post = self.model.posterior(X, observation_noise=True)
        mean = post.mean.squeeze(-1).squeeze(-1)
        sigma = post.variance.clamp_min(1e-18).sqrt().squeeze(-1).squeeze(-1)
        u = (self.best_f - mean) / sigma
        return _log_ei_helper(u) + sigma.log()


def _ineq(margin=MARGIN):
    cons = []
    for i in range(6):
        idx = np.flatnonzero(A[i])
        cons.append((torch.tensor(idx, dtype=torch.long),
                     torch.tensor(-A[i, idx], dtype=torch.double),
                     float(-(b[i] - margin))))
    return cons


def _acq_vals(acqf, Z, bs=2048):
    out = []
    with torch.no_grad():
        for s in range(0, len(Z), bs):
            out.append(acqf(torch.tensor(Z[s:s + bs]).unsqueeze(1)).numpy())
    return np.concatenate(out)


def _enumerate_combos(acqf, x):
    Z = np.repeat(x[None], len(COMBOS), 0)
    Z[:, 9:] = COMBOS
    v = _acq_vals(acqf, Z)
    j = int(np.argmax(v))
    return Z[j], float(v[j])


def _refine(acqf, starts, lo, hi):
    """Gradient refinement of continuous dims, discrete combo fixed per group."""
    bounds = torch.tensor(np.vstack([lo, hi]), dtype=torch.double)
    results = []
    keys = {}
    for s in starts:
        keys.setdefault((s[9], s[10]), []).append(s)
    for (v9, v10), S in keys.items():
        S = np.array(S)
        bnd = bounds.clone()
        bnd[0, 9] = bnd[1, 9] = v9
        bnd[0, 10] = bnd[1, 10] = v10
        ic = torch.tensor(S).unsqueeze(1)
        try:
            cand, val = optimize_acqf(
                acqf, bounds=bnd, q=1, num_restarts=len(S),
                batch_initial_conditions=ic,
                inequality_constraints=_ineq(), fixed_features={9: v9, 10: v10},
                return_best_only=False, options={"maxiter": 200})
            C = cand.detach().numpy().reshape(-1, D)
        except Exception as e:  # fall back to the starts themselves
            print("refine failed:", repr(e)[:200])
            C = S
        for c in C:
            c = c.copy()
            c[9], c[10] = v9, v10
            c = np.clip(c, lo, hi)
            if not allowed_strict(c)[0]:
                c = project(c)
                c = np.clip(c, lo, hi)
            if allowed_strict(c)[0]:
                results.append(c)
    return results


def propose(X, yt, seed, center=None, halfwidth=None, n_rand=6000, n_loc=3000,
            n_starts=10, min_dist=1e-3, verbose=True, model=None, acq="logei",
            min_dist_cont=0.0):
    """Maximize LogEI over the allowed set (optionally inside a TR box)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    X = np.asarray(X)
    yt = np.asarray(yt)
    if model is None:
        model = fit_gp(X, yt)
    if acq == "logei_obs":
        acqf = LogEIObs(model, best_f=float(yt.min()))
    else:
        acqf = LogExpectedImprovement(model, best_f=float(yt.min()), maximize=False)
    lo, hi = np.zeros(D), np.ones(D)
    if center is not None:
        ls = lengthscales(model)[:9]
        w = ls / np.exp(np.mean(np.log(ls)))
        w = np.clip(w, 0.25, 4.0)
        lo[:9] = np.clip(center[:9] - halfwidth * w, 0, 1)
        hi[:9] = np.clip(center[:9] + halfwidth * w, 0, 1)
    # candidate pool
    pool = [sample_allowed(n_rand, rng, lo, hi)]
    order = np.argsort(yt)
    anchors = [center] if center is not None else [X[i] for i in order[:5]]
    for a in anchors:
        k = n_loc // len(anchors)
        for sd in (0.02, 0.05, 0.15):
            Z = np.repeat(a[None], k // 3, 0)
            mask = rng.random((len(Z), 9)) < 0.4
            Z[:, :9] += mask * rng.normal(0, sd, (len(Z), 9))
            Z[:, :9] = np.clip(Z[:, :9], lo[:9], hi[:9])
            sw = rng.random(len(Z)) < 0.3
            Z[sw, 9] = rng.choice(L9, sw.sum())
            sw = rng.random(len(Z)) < 0.3
            Z[sw, 10] = rng.choice(L10, sw.sum())
            pool.append(Z[allowed_strict(Z)])
    P = np.vstack(pool)
    v = _acq_vals(acqf, P)
    # diversified top starts
    starts = []
    for j in np.argsort(-v):
        if all(np.linalg.norm(P[j] - s) > 0.05 for s in starts):
            starts.append(P[j])
        if len(starts) >= n_starts:
            break
    refined = _refine(acqf, starts, lo, hi)
    # discrete re-enumeration then one more refinement round
    re = [_enumerate_combos(acqf, r)[0] for r in refined]
    refined2 = _refine(acqf, re, lo, hi) if re else []
    C = np.array(list(starts) + refined + refined2)
    C = C[allowed_strict(C)]
    cv = _acq_vals(acqf, C)
    dists = np.min(np.linalg.norm(C[:, None, :] - X[None, :, :], axis=2), axis=1)
    dcont = np.min(np.linalg.norm(C[:, None, :9] - X[None, :, :9], axis=2), axis=1)
    ok = (dists > min_dist) & (dcont > min_dist_cont)
    if not ok.any():
        ok = dists > 0.05 * 0  # degenerate; take anything
    cv_ok = np.where(ok, cv, -np.inf)
    j = int(np.argmax(cv_ok))
    x = snap(C[j])
    assert allowed_strict(x)[0], "proposal not allowed"
    with torch.no_grad():
        post = model.posterior(torch.tensor(x[None]))
        mu = float(post.mean.squeeze())
        sd = float(post.variance.sqrt().squeeze())
    info = {"logei": float(cv[j]), "pred_mu_t": mu, "pred_sd_t": sd,
            "min_dist": float(dists[j]), "min_dist_cont": float(dcont[j]), "pool_max_logei": float(v.max()),
            "n_pool": len(P)}
    if verbose:
        print("propose:", {k: round(v_, 4) if isinstance(v_, float) else v_
                           for k, v_ in info.items()})
    return x, info, model
