"""Landscape probes P1-P5 (design §4). Deferred to v2.1: P6 separability, P7 noise.

Every probe evaluation goes through the controller's evaluator, so it lands
in the archive and is never wasted. Features are raw numbers; how to read
them is the playbook's job (playbook/skills/probing), not this module's.

Budget split for a probe budget B (d = dim):
    P1  space-filling Sobol sample      min(10d, 0.45 B)
    P2  fitness-distance correlation    0 (uses P1)
    P3  linear / quadratic fit R^2      0 (uses P1 + archive)
    P5  finite-difference gradients     2 points x (2d + 1), if B allows
    P4  multi-start short local runs    the rest, split over 3 starts
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew, spearmanr
from scipy.stats.qmc import Sobol

from src.optimizers import OptimizerExhausted

ALL_PROBES = ["P1", "P2", "P3", "P4", "P5"]


def _r(v, n=4):
    return None if v is None or not np.isfinite(v) else float(f"{v:.{n}g}")


def run_probes(ctl, probe_ids=None, budget=None) -> dict:
    ids = set(probe_ids or ALL_PROBES)
    unknown = ids - set(ALL_PROBES)
    if unknown:
        raise ValueError(f"unknown probes {sorted(unknown)}; available {ALL_PROBES}")
    d = ctl.dim
    B = int(min(budget or ctl.intake.probe_budget(), ctl.ev.left))
    lo, hi = ctl.archive.lower, ctl.archive.upper
    w = hi - lo
    feats: dict = {"probe_budget_used": 0}
    n0 = ctl.ev.used

    # ---------------------------------------------------------------- P1
    n1 = 0
    if ids & {"P1", "P2", "P3"}:
        n1 = int(max(d + 2, min(10 * d, 0.45 * B)))
        n1 = min(n1, B)
        pts = Sobol(d, scramble=True, seed=int(ctl.rng.integers(1 << 30))).random(n1)
        X1 = lo + pts * w
        y1 = np.array([ctl._eval(x, "probe:P1") for x in X1])
        ly = y1 - y1.min()
        feats["P1"] = {
            "n": n1,
            "y_min": _r(float(y1.min()), 8),
            "y_median": _r(float(np.median(y1)), 6),
            "log10_range": _r(np.log10(max(np.ptp(y1), 1e-300))),
            "skew": _r(float(skew(y1))),
            "kurtosis": _r(float(kurtosis(y1))),
            # fraction of samples above median + 3*IQR: heavy upper tail
            "frac_extreme": _r(float(np.mean(
                y1 > np.median(y1) + 3 * (np.quantile(y1, .75) - np.quantile(y1, .25))))),
            "skew_log": _r(float(skew(np.log10(ly + 1e-12 * max(1, np.ptp(y1)) + 1e-300)))),
        }

        # ------------------------------------------------------------ P2
        if "P2" in ids:
            Xn = (X1 - lo) / w
            ib = int(np.argmin(y1))
            dist = np.linalg.norm(Xn - Xn[ib], axis=1)
            fdc = spearmanr(dist, y1).correlation
            top = np.argsort(y1)[: max(3, n1 // 10)]
            T = Xn[top]
            pd = np.linalg.norm(T[:, None] - T[None], axis=2)
            spread = pd[np.triu_indices(len(T), 1)].mean() / np.sqrt(d)
            # mean pairwise distance of uniform points in the unit cube ~ 0.41*sqrt(d)
            feats["P2"] = {
                "fdc_spearman": _r(fdc),
                "top10_spread_over_diag": _r(spread),
                "top10_spread_vs_uniform": _r(spread / 0.408),
            }

        # ------------------------------------------------------------ P3
        if "P3" in ids:
            feats["P3"] = _fit_r2(ctl.archive.X, ctl.archive.y, lo, w)

    # ---------------------------------------------------------------- P5
    left = B - (ctl.ev.used - n0)
    if "P5" in ids and left >= 2 * (2 * d + 1) + 3 * (d + 2):
        feats["P5"] = _grad_consistency(ctl, w, n_points=2)

    # ---------------------------------------------------------------- P4
    left = B - (ctl.ev.used - n0)
    if "P4" in ids and left >= 2 * (d + 3):
        feats["P4"] = _multistart(ctl, left)

    feats["probe_budget_used"] = int(ctl.ev.used - n0)
    return feats


def _fit_r2(X, y, lo, w):
    """In-sample R^2 of least-squares fits on box-normalized inputs:
    linear, separable quadratic (no interactions) and full quadratic (only
    if there are >= 1.5x as many points as parameters). Also on rank(y), to
    be robust to heavy tails."""
    Z = (X - lo) / w * 2 - 1
    n, d = Z.shape
    out = {"n": int(n)}

    def r2(F, t):
        coef, *_ = np.linalg.lstsq(F, t, rcond=None)
        res = t - F @ coef
        return 1 - res.var() / max(t.var(), 1e-300)

    ones = np.ones((n, 1))
    F_lin = np.hstack([ones, Z])
    F_sep = np.hstack([F_lin, Z ** 2])
    iu = np.triu_indices(d, 1)
    F_full = np.hstack([F_sep, (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]])
    ranks = np.argsort(np.argsort(y)).astype(float)
    for name, F in [("linear", F_lin), ("quad_sep", F_sep), ("quad_full", F_full)]:
        if n >= 1.5 * F.shape[1]:
            out[f"r2_{name}"] = _r(r2(F, y))
            out[f"r2_{name}_rank"] = _r(r2(F, ranks))
        else:
            out[f"r2_{name}"] = None
            out[f"r2_{name}_rank"] = None
    return out


def _grad_consistency(ctl, w, n_points=2):
    """Forward-difference gradients at step h1 = 1e-3 * width and
    h2 = 1e-5 * width at the incumbent and another good point. Cosine
    similarity close to 1 => gradients are reliable (smooth, noise-free at
    these scales); low => rough or noisy, avoid finite-difference methods."""
    d = ctl.dim
    pts = [ctl.archive.incumbent()[0]]
    if n_points > 1 and len(ctl.archive) > 3:
        pts.append(ctl.archive.diverse_starts(2)[-1])
    cos, norms = [], []
    for x in pts:
        f0 = ctl._eval(x, "probe:P5")
        gs = []
        for h in (1e-3, 1e-5):
            g = np.zeros(d)
            for i in range(d):
                e = np.zeros(d)
                step = h * w[i]
                # step inwards if at the upper bound
                if x[i] + step > ctl.archive.upper[i]:
                    step = -step
                e[i] = step
                g[i] = (ctl._eval(x + e, "probe:P5") - f0) / step
            gs.append(g)
        a, b = gs
        c = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))
        cos.append(c)
        norms.append(float(np.linalg.norm(b * w)))
    return {"cosine_h1e-3_vs_h1e-5": [_r(c) for c in cos],
            "cosine_min": _r(min(cos)),
            "grad_norm_scaled": [_r(v) for v in norms]}


def _multistart(ctl, budget):
    """Short L-BFGS-B runs from 3 different good points. If they all improve
    strongly and their endpoints move toward each other (contraction < 1)
    and end at similar values, the landscape looks like one smooth basin.
    Endpoints at clearly different values => several basins."""
    d = ctl.dim
    m = 3
    per = budget // m
    starts = ctl.archive.diverse_starts(m)
    method = "lbfgsb" if "lbfgsb" in ctl.allowed else "nelder_mead"
    ends, y_start, y_end = [], [], []
    for s in starts:
        opt = _local_from(method, ctl, s)
        y0 = None
        for _ in range(per):
            try:
                X = opt.ask(1)
            except OptimizerExhausted:
                break
            y = ctl._eval(X[0], f"probe:P4")
            if y0 is None:
                y0 = y
            opt.tell(X, [y])
        opt.close()
        xb, yb = opt.best()
        ends.append((xb - ctl.archive.lower) / ctl.archive.width)
        y_start.append(y0)
        y_end.append(float(yb))
    S = (starts - ctl.archive.lower) / ctl.archive.width
    E = np.array(ends)

    def mpd(A):
        D = np.linalg.norm(A[:, None] - A[None], axis=2)
        return D[np.triu_indices(len(A), 1)].mean()

    y_start, y_end = np.array(y_start), np.array(y_end)
    gain = (y_start - y_end) / np.maximum(np.abs(y_start - y_end.min()), 1e-300)
    spread_end = (y_end.max() - y_end.min()) / max(np.median(y_start) - y_end.min(), 1e-300)
    return {
        "method": method,
        "evals_per_start": int(per),
        "y_start": [_r(v, 6) for v in y_start],
        "y_end": [_r(v, 6) for v in y_end],
        "rel_gain_median": _r(float(np.median(gain))),
        "endpoint_contraction": _r(mpd(E) / max(mpd(S), 1e-300)),
        "endpoint_y_spread": _r(spread_end),
    }


def _local_from(method, ctl, x0):
    from src.optimizers.lbfgsb import LBFGSB
    from src.optimizers.nelder_mead import NelderMead
    cls = LBFGSB if method == "lbfgsb" else NelderMead
    return cls(ctl.bounds, x0=x0)
