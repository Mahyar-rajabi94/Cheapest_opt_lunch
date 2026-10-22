"""Build any optimizer warm-started from the archive (design §6.2).

    opt = make_optimizer("cma_es", bounds, archive, start="incumbent", seed=0)

`start`:
    "incumbent" (default) start from the archive's best point
    "diverse"   start from a good point far from the incumbent (pass
                `start_index` to pick among archive.diverse_starts())
    "center"    box midpoint (the original, cold-start behavior)
    "random"    uniform random point
`scale` optionally overrides the per-dimension search scale (default:
archive.local_scale()).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.rule_based_policy.optimizers.bobyqa import BOBYQA
from src.rule_based_policy.optimizers.cma_es import CMAES
from src.rule_based_policy.optimizers.differential_evolution import DifferentialEvolution
from src.rule_based_policy.optimizers.gp_bo import GPBO
from src.rule_based_policy.optimizers.lbfgsb import LBFGSB
from src.rule_based_policy.optimizers.nelder_mead import NelderMead
from src.rule_based_policy.optimizers.random_search import RandomSearch
from src.rule_based_policy.optimizers.sobol_search import SobolSearch
from src.rule_based_policy.optimizers.turbo import TuRBO

from src.rule_based_policy.archive import Archive

METHODS = {
    "random_search": RandomSearch,
    "sobol": SobolSearch,
    "cma_es": CMAES,
    "de": DifferentialEvolution,
    "gp_bo": GPBO,
    "turbo": TuRBO,
    "bobyqa": BOBYQA,
    "nelder_mead": NelderMead,
    "lbfgsb": LBFGSB,
}

# Local solvers only ask one point at a time and stop by themselves.
LOCAL_METHODS = {"bobyqa", "nelder_mead", "lbfgsb"}
# GP-BO refits an exact GP every step (cubic cost): cap warm-start data.
GPBO_MAX_TRAIN = 150


def _start_point(archive: Archive, start: str, rng, start_index: int = 1):
    lo, hi = archive.lower, archive.upper
    if start == "center" or len(archive) == 0:
        return (lo + hi) / 2
    if start == "random":
        return rng.uniform(lo, hi)
    if start == "incumbent":
        return archive.incumbent()[0]
    if start == "diverse":
        S = archive.diverse_starts(max(2, start_index + 1))
        return S[min(start_index, len(S) - 1)]
    raise ValueError(f"unknown start {start!r}")


def make_optimizer(
    method: str,
    bounds,
    archive: Archive,
    *,
    start: str = "incumbent",
    scale=None,
    seed: Optional[int] = None,
    start_index: int = 1,
    x0=None,
    **overrides,
):
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    width = hi - lo
    rng = np.random.default_rng(seed)
    cold = start == "center" or len(archive) == 0

    if x0 is not None:
        # Explicit start point (e.g. the best point of the active method):
        # scale from the spread of the archive points nearest to it.
        x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)
        if scale is None:
            Xn, _ = archive.nearest(x0, max(2 * len(x0), 10))
            scale = np.clip(Xn.std(axis=0), 1e-6 * width, 0.3 * width)
        cold = False
    else:
        x0 = _start_point(archive, start, rng, start_index)
    if scale is None:
        scale = archive.local_scale() if not cold and start == "incumbent" else 0.25 * width
    scale = np.broadcast_to(np.asarray(scale, dtype=float), x0.shape)
    rel = float(np.median(scale / width))  # scale as a fraction of the box

    if method in ("random_search", "sobol"):
        return METHODS[method](bounds, seed=seed, **overrides)

    if method == "cma_es":
        kw = dict(x0=x0, sigma0=float(np.median(scale)))
        kw.update(overrides)
        return CMAES(bounds, seed=seed, **kw)

    if method == "lbfgsb":
        return LBFGSB(bounds, seed=seed, x0=x0, **overrides)

    if method == "nelder_mead":
        # Simplex: x0 plus one vertex per axis, stepped by the local scale
        # (towards the interior if the step would leave the box).
        simplex = [x0.copy()]
        for i in range(len(x0)):
            v = x0.copy()
            step = scale[i]
            v[i] = v[i] + step if v[i] + step <= hi[i] else v[i] - step
            simplex.append(v)
        kw = dict(initial_simplex=np.array(simplex))
        kw.update(overrides)
        return NelderMead(bounds, seed=seed, x0=x0, **kw)

    if method == "bobyqa":
        # scaling_within_bounds=True: BOBYQA works in the unit box, so
        # rhobeg is the relative scale.
        kw = dict(rhobeg=float(np.clip(rel, 1e-4, 0.25)))
        kw.update(overrides)
        return BOBYQA(bounds, seed=seed, x0=x0, **kw)

    if method == "de":
        if cold:
            return DifferentialEvolution(bounds, seed=seed, **overrides)
        # Seed ~20% of the population with the best archive points; the
        # rest stays uniform random so DE keeps its global exploration.
        pop = 10 * len(x0)
        Xk, _ = archive.top_k(max(1, pop // 5))
        return DifferentialEvolution(bounds, seed=seed, init_population=Xk, **overrides)

    if method == "gp_bo":
        if cold:
            return GPBO(bounds, seed=seed, **overrides)
        Xn, yn = archive.nearest(x0, GPBO_MAX_TRAIN - 20)
        Xt, yt = archive.top_k(20)
        X = np.vstack([Xn, Xt]); y = np.concatenate([yn, yt])
        _, uniq = np.unique(np.round(X, 12), axis=0, return_index=True)
        return GPBO(bounds, seed=seed, init_X=X[uniq], init_y=y[uniq], **overrides)

    if method == "turbo":
        if cold:
            return TuRBO(bounds, seed=seed, **overrides)
        n = 4 * len(x0)
        Xn, yn = archive.nearest(x0, n)
        # Side length (unit cube, before lengthscale weighting): about 4x
        # the relative local scale, within TuRBO's own limits.
        return TuRBO(
            bounds, seed=seed, init_X=Xn, init_y=yn,
            length_init=float(np.clip(4 * rel, 0.5**7, 0.8)), **overrides,
        )

    raise AssertionError("unreachable")
