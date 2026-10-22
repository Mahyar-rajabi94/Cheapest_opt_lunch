"""Trust-region Bayesian optimization adapted to a finite candidate pool.

Pool mode receives ten initial observations from the harness and then selects
one actual unused row per step. A fresh SingleTaskGP uses all shared observed
features and raw labels. Its anisotropic trust region centers on the best
observed row; an empty region expands deterministically until feasible rows
are available. Thompson sampling considers at most 512 such rows to bound
joint posterior memory. Success/failure feedback adapts the region length.
Region restarts preserve observed history and require no further random design.

For direction="max", posterior sampling maximizes raw y. The continuous
ask/tell interface is retained with minimization as its default, batched Sobol
candidates, and local-data restarts.
"""


from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
import torch
from torch.quasirandom import SobolEngine

warnings.filterwarnings("ignore", message=".*torch.jit.script.*")

from botorch.fit import fit_gpytorch_mll
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from src.optimizers import OptimizerExhausted


class TuRBO:
    """TuRBO-1: a single trust region, restarted when it shrinks too far.

    Internally proposes and updates a whole batch at a time (batch_size=4
    by default); this wrapper hides that bookkeeping so it can still be
    driven exactly like the other optimizers in this package: ask for any
    number of points (including one at a time) and tell them back in the
    same order.

    For the continuous API, `tell` must receive points most recently returned
    by `ask`, in the same order. The pool API also accepts observations chosen
    by another optimizer, keeping every optimizer's history synchronized.

    Parameters
    ----------
    bounds : array-like of shape (d, 2), optional for a pool
        Lower and upper bound for each dimension; inferred from candidates
        when omitted in pool mode.
    seed : int, optional
        Seed for the random number generator.
    candidates : array-like of shape (pool_size, d), optional
        Public pool features, without labels. Enables the sequential
        ask_pool() API after ten external initial observations.
    n_init : int, optional
        Required observations before fitting; pool default 10, configurable per block.
    direction : {"min", "max"}
        Objective direction. tell() always accepts raw observed labels.

    Example
    -------
        opt = TuRBO(bounds=[(-5, 5)] * 10, seed=0)
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
        self.rng = np.random.default_rng(seed)

        self.batch_size = 1 if self.candidates is not None else 4
        self.n_init = 10 if self.candidates is not None else 2 * self.dim
        if n_init is not None:
            if isinstance(n_init, bool) or not isinstance(n_init, int) or n_init < 2:
                raise ValueError("n_init must be an integer of at least 2")
            self.n_init = n_init
        self.n_candidates = min(100 * self.dim, 512 if self.candidates is not None else 5000)
        self._pending_pool_index = None

        self._length_min = 0.5 ** 7
        self._length_max = 1.6
        self._length_init = 0.8
        self._success_tolerance = 10
        self._failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, self.dim / self.batch_size)
        )

        self._lower = self.bounds[:, 0]
        self._upper = self.bounds[:, 1]
        self._scale = np.where(self._upper > self._lower, self._upper - self._lower, 1.0)
        self._pool_unit = (self._to_unit(self.candidates)
                           if self.candidates is not None else None)
        self._unit_bounds = torch.stack(
            [torch.zeros(self.dim), torch.ones(self.dim)]
        ).double()

        self._reset_trust_region()
        self._to_ask: list = []  # points generated, not yet asked out

        self.X_ = np.empty((0, self.dim))
        self.y_ = np.empty((0,))

    def _to_unit(self, x_raw):
        return (x_raw - self._lower) / self._scale

    def _from_unit(self, x_unit):
        return self._lower + x_unit * (self._upper - self._lower)

    def _reset_trust_region(self) -> None:
        """Start a fresh trust region: reset its length and counters, and
        its own local data. The global history (X_, y_, used by best())
        is untouched."""
        self._length = self._length_init
        self._success_counter = 0
        self._failure_counter = 0
        self._local_X_unit = np.empty((0, self.dim))
        self._local_y = np.empty((0,))
        self._local_best_y = np.inf
        self._evolve_evals_since_update = 0
        self._phase = "init"

    def ask(self, n: int = 1) -> np.ndarray:
        """Propose `n` new candidate points, shape (n, dim)."""
        if self.candidates is not None:
            raise RuntimeError("Use ask_pool(available_indices) for a finite pool")
        out = []
        for _ in range(n):
            if not self._to_ask:
                self._generate_next_batch()
            out.append(self._to_ask.pop(0))
        return np.array(out)

    def _generate_next_batch(self) -> None:
        if self._phase == "init":
            sobol = SobolEngine(
                dimension=self.dim,
                scramble=True,
                seed=int(self.rng.integers(1 << 30)),
            )
            unit_points = sobol.draw(self.n_init).numpy()
            self._to_ask = list(self._from_unit(unit_points))
            return

        # Fit a GP on this trust region's own local data so far (not the
        # whole run's history -- that's reset on restart).
        # BoTorch's MaxPosteriorSampling (used below) picks the candidates
        # with the *highest* sampled values, i.e. it assumes maximization.
        # This optimizer minimizes, so the GP is fit on -y: the maximum of
        # -y is the minimum of y. (Fitting on raw y made TuRBO chase the
        # worst regions of the trust region.)
        X_t = torch.tensor(self._local_X_unit, dtype=torch.double)
        utility = self._local_y if self.direction == "max" else -self._local_y
        Y_t = torch.tensor(utility, dtype=torch.double).unsqueeze(-1)

        model = SingleTaskGP(
            X_t,
            Y_t,
            input_transform=Normalize(d=self.dim, bounds=self._unit_bounds),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        weights = model.covar_module.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

        best_idx = int(np.argmax(self._local_y) if self.direction == "max"
                       else np.argmin(self._local_y))
        x_center = torch.tensor(self._local_X_unit[best_idx], dtype=torch.double)

        tr_lb = torch.clamp(x_center - weights * self._length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * self._length / 2.0, 0.0, 1.0)

        sobol = SobolEngine(
            dimension=self.dim, scramble=True, seed=int(self.rng.integers(1 << 30))
        )
        pert = tr_lb + (tr_ub - tr_lb) * sobol.draw(self.n_candidates).double()

        prob_perturb = min(20.0 / self.dim, 1.0)
        mask = torch.rand(self.n_candidates, self.dim) <= prob_perturb
        no_pert_rows = ~mask.any(dim=-1)
        if no_pert_rows.any():
            idx = no_pert_rows.nonzero(as_tuple=True)[0]
            cols = torch.randint(0, self.dim, (len(idx),))
            mask[idx, cols] = True

        X_cand = x_center.expand(self.n_candidates, self.dim).clone()
        X_cand[mask] = pert[mask]

        with torch.no_grad():
            thompson_sampler = MaxPosteriorSampling(model=model, replacement=False)
            X_next_unit = thompson_sampler(X_cand, num_samples=self.batch_size)

        self._to_ask = list(self._from_unit(X_next_unit.numpy()))

    def _fit_pool_model(self):
        """Fit all shared observations; the GP's utility always maximizes."""
        X_t = torch.tensor(self._to_unit(self.X_), dtype=torch.double)
        utility = self.y_ if self.direction == "max" else -self.y_
        Y_t = torch.tensor(utility, dtype=torch.double).unsqueeze(-1)
        model = SingleTaskGP(
            X_t, Y_t,
            input_transform=Normalize(d=self.dim, bounds=self._unit_bounds),
            outcome_transform=Standardize(m=1),
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        return model

    def ask_pool(self, available_indices) -> int:
        """Thompson-sample one feasible row from an anisotropic trust region.

        Empty regions expand geometrically for this proposal until they contain
        an available row. This deterministic feasibility adjustment is necessary
        for sparse/high-dimensional pools; it does not reset the learned region
        state. At most ``n_candidates`` actual rows enter joint posterior
        sampling, avoiding a covariance matrix over the whole data pool.
        """
        indices = self._pool_indices(available_indices)
        model = self._fit_pool_model()
        weights = model.covar_module.lengthscale.detach().reshape(-1).double()
        weights = weights.clamp_min(1e-12)
        weights = weights / torch.exp(torch.log(weights).mean())
        weights = weights.numpy()
        best_idx = int(np.argmax(self.y_) if self.direction == "max"
                       else np.argmin(self.y_))
        center = self._to_unit(self.X_[best_idx])
        # A row belongs to the clipped anisotropic box precisely when this
        # required side length is no greater than the effective region length.
        required = np.max(2.0 * np.abs(self._pool_unit[indices] - center) / weights, axis=1)
        effective_length = self._length
        while required.min() > effective_length:
            effective_length *= 2.0
        inside = indices[required <= effective_length]
        self._last_pool_effective_length = effective_length
        self._last_pool_region_indices = inside.copy()
        if len(inside) > self.n_candidates:
            inside = np.sort(self.rng.choice(inside, size=self.n_candidates, replace=False))
        X_candidates = torch.tensor(self._pool_unit[inside], dtype=torch.double)
        # Keep the optimizer's seeded draws independent of other optimizers.
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(self.rng.integers(1 << 30)))
            sampler = MaxPosteriorSampling(model=model, replacement=False)
            selected = sampler(X_candidates, num_samples=1)[0]
        position = int(torch.nonzero(torch.all(X_candidates == selected, dim=1))[0, 0])
        self._pending_pool_index = int(inside[position])
        return self._pending_pool_index

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
        """Record evaluated points `X` (n, dim) and their values `y` (n,).

        Continuous calls follow ask() order. Pool calls can include shared
        observations; an outstanding ask_pool() proposal must be included.
        """
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

        for xi, yi in zip(X, y):
            self._local_X_unit = np.vstack([self._local_X_unit, self._to_unit(xi)])
            self._local_y = np.append(self._local_y, yi)

            if self._phase == "init":
                if len(self._local_y) >= self.n_init:
                    self._local_best_y = float(self._local_y.max() if self.direction == "max"
                                               else self._local_y.min())
                    self._phase = "evolve"
                    self._evolve_evals_since_update = 0
            else:
                self._evolve_evals_since_update += 1
                if self._evolve_evals_since_update >= self.batch_size:
                    if self.candidates is not None:
                        self._update_pool_trust_region(float(yi))
                    else:
                        self._update_trust_region()
                    self._evolve_evals_since_update = 0

    def _update_pool_trust_region(self, value):
        sign = 1.0 if self.direction == "max" else -1.0
        improved = sign * value > sign * self._local_best_y + 1e-3 * abs(self._local_best_y)
        if improved:
            self._success_counter += 1
            self._failure_counter = 0
        else:
            self._success_counter = 0
            self._failure_counter += 1
        self._local_best_y = (max(self._local_best_y, value) if self.direction == "max"
                              else min(self._local_best_y, value))
        if self._success_counter >= self._success_tolerance:
            self._length = min(2.0 * self._length, self._length_max)
            self._success_counter = 0
        elif self._failure_counter >= self._failure_tolerance:
            self._length /= 2.0
            self._failure_counter = 0
        if self._length < self._length_min:
            # Pool runs seed once. Restart the region, preserving all shared
            # observations and staying immediately ready for the next GP fit.
            self._length = self._length_init
            self._success_counter = self._failure_counter = 0

    def _update_trust_region(self) -> None:
        recent_y = self._local_y[-self.batch_size:]
        best_recent = float(recent_y.max() if self.direction == "max" else recent_y.min())
        sign = 1.0 if self.direction == "max" else -1.0
        improved = sign * best_recent > sign * self._local_best_y + 1e-3 * abs(self._local_best_y)

        if improved:
            self._success_counter += 1
            self._failure_counter = 0
        else:
            self._success_counter = 0
            self._failure_counter += 1

        self._local_best_y = (max(self._local_best_y, best_recent) if self.direction == "max"
                              else min(self._local_best_y, best_recent))

        if self._success_counter == self._success_tolerance:
            self._length = min(2.0 * self._length, self._length_max)
            self._success_counter = 0
        elif self._failure_counter == self._failure_tolerance:
            self._length /= 2.0
            self._failure_counter = 0

        if self._length < self._length_min:
            self._reset_trust_region()

    def best(self):
        """Return the best observed point and raw y in the requested direction."""
        if len(self.y_) == 0:
            raise RuntimeError("No evaluations recorded yet; call tell() first.")
        i = np.argmax(self.y_) if self.direction == "max" else np.argmin(self.y_)
        return self.X_[i], self.y_[i]
