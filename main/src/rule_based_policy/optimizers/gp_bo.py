"""GP-BO with the analytic LogEI acquisition function (Ament et al., 2023).

Wraps BoTorch's SingleTaskGP and LogExpectedImprovement behind this
project's standard ask/tell interface.

Standard, no-tuning defaults:
- GP: SingleTaskGP with its own default kernel and priors (BoTorch's
  current default lengthscale prior is dimension-scaled, following
  Hvarfner et al., ICML 2024), input normalization to the given box, and
  output standardization.
- Acquisition: analytic LogExpectedImprovement (single-candidate,
  noiseless).
- Acquisition optimization: multi-start L-BFGS via BoTorch's
  optimize_acqf, with num_restarts=10 and raw_samples=512 -- the pair
  used throughout BoTorch's own official tutorials.
- Initial design: 2 * (dim + 1) Sobol points (a standard rule of thumb for
  how many points are needed before a GP fit is meaningful), using this
  project's own SobolSearch, before the first acquisition-driven point is
  proposed.

Exact GP fitting costs grow roughly cubically with the amount of data, so
this optimizer is only practical at small budgets -- consistent with its
documented regime in docs/claude_deep_serach/optimizer_landscape.md
(expensive, continuous problems, budgets of order 10 x dim, not 1000+ x
dim).

By convention, this optimizer MINIMIZES the objective (matching the rest
of this package).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch

# Harmless warnings triggered by botorch's own import chain / internals,
# not by anything specific to this problem.
warnings.filterwarnings("ignore", message=".*torch.jit.script.*")

from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from src.rule_based_policy.optimizers.sobol_search import SobolSearch


class GPBO:
    """Sequential GP-BO with the analytic LogEI acquisition function.

    This optimizer only supports asking for one point at a time (n=1):
    each `ask` fits a fresh GP on all data told so far and optimizes LogEI
    for the single next point; batch (q > 1) BO would need a different
    acquisition function (e.g. q-LogNEI) and is not implemented here.

    Parameters
    ----------
    bounds : array-like of shape (d, 2)
        Lower and upper bound for each of the d dimensions.
    seed : int, optional
        Seed for the initial Sobol design and the acquisition optimizer's
        random restarts.

    Example
    -------
        opt = GPBO(bounds=[(-5, 5)] * 10, seed=0)
        X = opt.ask(n=1)
        y = [f(x) for x in X]
        opt.tell(X, y)
        x_best, y_best = opt.best()
    """

    def __init__(self, bounds, seed: Optional[int] = None, init_X=None, init_y=None):
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError(
                "bounds must have shape (d, 2): one (low, high) pair per dimension"
            )
        self.dim = self.bounds.shape[0]

        self.n_init = 2 * (self.dim + 1)
        self.num_restarts = 10
        self.raw_samples = 512

        self._sobol = SobolSearch(bounds=bounds, seed=seed)
        self._torch_bounds = torch.tensor(self.bounds.T, dtype=torch.double)
        if seed is not None:
            torch.manual_seed(seed)

        self.X_ = np.empty((0, self.dim))
        self.y_ = np.empty((0,))

        # Warm start: already-evaluated points (e.g. from the agent's
        # archive) are used as GP training data and count toward the
        # initial design, so no evaluations are spent re-sampling. They are
        # kept separate from X_/y_ so best()/history only reflect points this
        # optimizer proposed itself.
        self._prior_X = np.empty((0, self.dim))
        self._prior_y = np.empty((0,))
        if init_X is not None:
            self._prior_X = np.atleast_2d(np.asarray(init_X, float))
            self._prior_y = np.atleast_1d(np.asarray(init_y, float))

    def ask(self, n: int = 1) -> np.ndarray:
        """Propose 1 new candidate point, shape (1, dim)."""
        if n != 1:
            raise NotImplementedError(
                "GPBO only supports asking for one point at a time (n=1); "
                "batch BO would need a q>1 acquisition function (e.g. q-LogNEI)."
            )

        X_all = np.vstack([self._prior_X, self.X_])
        y_all = np.concatenate([self._prior_y, self.y_])

        if len(y_all) < self.n_init:
            return self._sobol.ask(n=1)

        X_t = torch.tensor(X_all, dtype=torch.double)
        Y_t = torch.tensor(y_all, dtype=torch.double).unsqueeze(-1)

        model = SingleTaskGP(
            X_t,
            Y_t,
            input_transform=Normalize(d=self.dim, bounds=self._torch_bounds),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        best_f = float(y_all.min())
        acqf = LogExpectedImprovement(model=model, best_f=best_f, maximize=False)

        candidate, _ = optimize_acqf(
            acqf,
            bounds=self._torch_bounds,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        return candidate.detach().numpy()

    def tell(self, X, y) -> None:
        """Record evaluated points `X` (n, dim) and their values `y` (n,)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        self.X_ = np.vstack([self.X_, X])
        self.y_ = np.concatenate([self.y_, y])

    def best(self):
        """Return (x_best, y_best) among all points told so far (minimum y)."""
        if len(self.y_) == 0:
            raise RuntimeError("No evaluations recorded yet; call tell() first.")
        i = np.argmin(self.y_)
        return self.X_[i], self.y_[i]

    def state(self) -> dict:
        """Internal state for the agent's StateReport."""
        return {
            "n_told": int(len(self.y_)),
            "n_train": int(len(self.y_) + len(self._prior_y)),
            "exhausted": False,
        }

    def close(self) -> None:
        pass
