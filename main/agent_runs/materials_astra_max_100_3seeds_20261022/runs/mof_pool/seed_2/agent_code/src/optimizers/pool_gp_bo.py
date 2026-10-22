"""Gaussian-process Bayesian optimization with analytic LogExpectedImprovement.

For finite pools, provide candidates and direction="max", then tell the ten
initial observations sampled by the harness. ask_pool(available_indices) fits
SingleTaskGP on all observed raw y and scores every remaining actual row in
bounded chunks. No unobserved labels enter the model. Constant feature columns
are supported. Each proposal requires feedback before another proposal.

The continuous ask/tell interface remains available with minimization as its
default, a Sobol initial design, and continuous acquisition optimization.
"""


from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
from torch.quasirandom import SobolEngine

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

from src.optimizers import OptimizerExhausted


class GPBO:
    """Sequential GP-BO with the analytic LogEI acquisition function.

    This optimizer only supports asking for one point at a time (n=1):
    each `ask` fits a fresh GP on all data told so far and optimizes LogEI
    for the single next point; batch (q > 1) BO would need a different
    acquisition function (e.g. q-LogNEI) and is not implemented here.

    Parameters
    ----------
    bounds : array-like of shape (d, 2), optional for a pool
        Lower and upper bound for each dimension; inferred from candidates
        when omitted in pool mode.
    seed : int, optional
        Seed for the initial Sobol design and the acquisition optimizer's
        random restarts.
    candidates : array-like of shape (pool_size, d), optional
        Public pool features, without labels. Enables ask_pool(), with ten
        external initial observations and one proposal per feedback update.
    n_init : int, optional
        Required observations before fitting; pool default 10, configurable per block.
    direction : {"min", "max"}
        Objective direction. tell() always accepts raw observed labels.

    Example
    -------
        opt = GPBO(bounds=[(-5, 5)] * 10, seed=0)
        X = opt.ask(n=1)
        y = [f(x) for x in X]
        opt.tell(X, y)
        x_best, y_best = opt.best()
    """

    def __init__(self, bounds=None, seed: Optional[int] = None, *,
                 candidates=None, direction: str = "min", n_init: int | None = None):
        if direction not in {"min", "max"}:
            raise ValueError("direction must be 'min' or 'max'")
        self.direction = direction
        self.candidates = None
        if candidates is not None:
            self.candidates = np.asarray(candidates, dtype=float).copy()
            if (self.candidates.ndim != 2 or not self.candidates.size
                    or not np.isfinite(self.candidates).all()):
                raise ValueError("candidates must be a nonempty finite feature matrix")
            if bounds is None:
                bounds = np.column_stack((self.candidates.min(0), self.candidates.max(0)))
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError(
                "bounds must have shape (d, 2): one (low, high) pair per dimension"
            )
        self.dim = self.bounds.shape[0]
        if (not np.isfinite(self.bounds).all()
                or np.any(self.bounds[:, 1] < self.bounds[:, 0])):
            raise ValueError("bounds must be finite and have low <= high")
        if self.candidates is not None:
            if (self.candidates.shape[1] != self.dim
                    or np.any(self.candidates < self.bounds[:, 0])
                    or np.any(self.candidates > self.bounds[:, 1])):
                raise ValueError("candidate features must lie within bounds")

        self.n_init = 10 if self.candidates is not None else 2 * (self.dim + 1)
        if n_init is not None:
            if isinstance(n_init, bool) or not isinstance(n_init, int) or n_init < 2:
                raise ValueError("n_init must be an integer of at least 2")
            self.n_init = n_init
        self.num_restarts = 10
        self.raw_samples = 512
        self.pool_chunk_size = 512
        self._pending_pool_index = None

        self._sobol = (SobolEngine(dimension=self.dim, scramble=True, seed=seed)
                       if self.candidates is None else None)
        # A constant feature stays at zero after normalization. Supplying a
        # unit span to Normalize avoids division by zero without changing rows.
        model_bounds = self.bounds.copy()
        constant = model_bounds[:, 0] == model_bounds[:, 1]
        model_bounds[constant, 1] = model_bounds[constant, 0] + 1.0
        self._torch_bounds = torch.tensor(model_bounds.T, dtype=torch.double)
        if seed is not None:
            torch.manual_seed(seed)

        self.X_ = np.empty((0, self.dim))
        self.y_ = np.empty((0,))

    def ask(self, n: int = 1) -> np.ndarray:
        """Propose 1 new candidate point, shape (1, dim)."""
        if self.candidates is not None:
            raise RuntimeError("Use ask_pool(available_indices) for a finite pool")
        if n != 1:
            raise NotImplementedError(
                "GPBO only supports asking for one point at a time (n=1); "
                "batch BO would need a q>1 acquisition function (e.g. q-LogNEI)."
            )

        if len(self.y_) < self.n_init:
            unit = self._sobol.draw(1, dtype=torch.double).numpy()
            return self.bounds[:, 0] + unit * (self.bounds[:, 1] - self.bounds[:, 0])

        model = self._fit_model()
        best_f = float(self.y_.max() if self.direction == "max" else self.y_.min())
        acqf = LogExpectedImprovement(model=model, best_f=best_f,
                                     maximize=self.direction == "max")

        candidate, _ = optimize_acqf(
            acqf,
            bounds=self._torch_bounds,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        return candidate.detach().numpy()

    def _fit_model(self):
        """Fit only labels observed through tell(), in their raw direction."""
        X_t = torch.tensor(self.X_, dtype=torch.double)
        Y_t = torch.tensor(self.y_, dtype=torch.double).unsqueeze(-1)

        model = SingleTaskGP(
            X_t,
            Y_t,
            input_transform=Normalize(d=self.dim, bounds=self._torch_bounds),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        return model

    def ask_pool(self, available_indices) -> int:
        """Choose the highest-LogEI remaining row after ten initial observations.

        ``available_indices`` contains row IDs, not labels. The caller removes
        evaluated IDs and tells the selected row's raw y before asking again.
        Scoring in chunks bounds memory while still examining every remaining
        row with the same freshly fitted GP.
        """
        indices = self._pool_indices(available_indices)
        model = self._fit_model()
        best_f = float(self.y_.max() if self.direction == "max" else self.y_.min())
        acqf = LogExpectedImprovement(model=model, best_f=best_f,
                                     maximize=self.direction == "max")
        best_index, best_score = None, -np.inf
        with torch.no_grad():
            for start in range(0, len(indices), self.pool_chunk_size):
                chunk = indices[start:start + self.pool_chunk_size]
                X = torch.tensor(self.candidates[chunk], dtype=torch.double).unsqueeze(1)
                scores = acqf(X).reshape(-1)
                position = int(torch.argmax(scores))
                value = float(scores[position])
                if best_index is None or value > best_score:
                    best_index, best_score = int(chunk[position]), value
        self._pending_pool_index = best_index
        return best_index

    def _pool_indices(self, available_indices):
        if self.candidates is None:
            raise RuntimeError("ask_pool requires candidates at construction")
        indices = np.asarray(available_indices)
        if indices.ndim != 1:
            raise ValueError("available_indices must be a one-dimensional row ID array")
        if not len(indices):
            raise OptimizerExhausted("No unevaluated candidate rows remain")
        if (not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= len(self.candidates))
                or len(np.unique(indices)) != len(indices)):
            raise ValueError("available_indices must contain unique valid integer row IDs")
        if self._pending_pool_index is not None:
            raise RuntimeError("Call tell() with the last proposed row before asking again")
        if len(self.y_) < self.n_init:
            raise RuntimeError("Supply at least n_init pool observations through tell() first")
        return indices

    def tell(self, X, y) -> None:
        """Record evaluated points `X` (n, dim) and their values `y` (n,)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float)).reshape(-1)
        if (X.shape != (len(y), self.dim) or not len(y)
                or not np.isfinite(X).all() or not np.isfinite(y).all()):
            raise ValueError("tell requires matching finite X (n, dim) and y (n,)")
        if (self._pending_pool_index is not None
                and not np.any(np.all(X == self.candidates[self._pending_pool_index], axis=1))):
            raise ValueError("tell must include the last proposed pool row")
        self.X_ = np.vstack([self.X_, X])
        self.y_ = np.concatenate([self.y_, y])
        self._pending_pool_index = None

    def best(self):
        """Return the best observed point and raw y in the requested direction."""
        if len(self.y_) == 0:
            raise RuntimeError("No evaluations recorded yet; call tell() first.")
        i = np.argmax(self.y_) if self.direction == "max" else np.argmin(self.y_)
        return self.X_[i], self.y_[i]
