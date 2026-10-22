"""TuRBO-1 (Trust Region Bayesian Optimization), Eriksson et al. (NeurIPS 2019).

No packaged, ask/tell-capable implementation of TuRBO exists: the original
authors' code (uber-research/TuRBO) is a monolithic run-to-completion
script on an older GP stack, not a pip package with an ask/tell API. The
commonly used modern reference is BoTorch's own official tutorial, which
is example code, not a library class. This module follows that tutorial's
recipe directly, reusing this project's existing BoTorch-based GP setup
(as in gp_bo.py).

Standard, no-tuning defaults, all from the TuRBO paper / BoTorch's own
tutorial:
- Local modeling: SingleTaskGP (BoTorch's own default kernel and priors),
  fit on the data collected since the current trust region's last restart
  (not the whole history -- see "restarts" below).
- Candidate generation: a perturbed Sobol candidate set inside the trust
  region, from which a batch is selected by Thompson sampling
  (botorch.generation.sampling.MaxPosteriorSampling).
- Trust region: side length starts at 0.8 (in the normalized [0, 1]^d
  cube), doubling after success_tolerance=10 consecutive improving
  batches (up to length_max=1.6), and halving after failure_tolerance
  consecutive non-improving batches (down to length_min=0.5**7, which
  triggers a restart). failure_tolerance = ceil(max(4/batch_size,
  dim/batch_size)), as in the tutorial. The box is centered on the current
  local best point and scaled by the GP's own per-dimension lengthscales.
- Batch size: 4 (the fixed value used throughout the tutorial).
- Initial design, and each restart's new design: 2 * dim random points.
- Restarts: when the trust region shrinks below length_min, its local
  data is discarded and a fresh random design starts a new trust region.
  The externally visible history (X_, y_, best()) is never discarded --
  only the data used to fit the *current* trust region's GP is local.

By convention, this optimizer MINIMIZES the objective (matching the rest
of this package). Internally the GP is fit on -y, because BoTorch's
Thompson-sampling selector (MaxPosteriorSampling) maximizes.
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


class TuRBO:
    """TuRBO-1: a single trust region, restarted when it shrinks too far.

    Internally proposes and updates a whole batch at a time (batch_size=4
    by default); this wrapper hides that bookkeeping so it can still be
    driven exactly like the other optimizers in this package: ask for any
    number of points (including one at a time) and tell them back in the
    same order.

    IMPORTANT: as with CMAES/DifferentialEvolution, `tell` must be called
    with exactly the points most recently returned by `ask`, in the same
    order.

    Parameters
    ----------
    bounds : array-like of shape (d, 2)
        Lower and upper bound for each of the d dimensions.
    seed : int, optional
        Seed for the random number generator.

    Example
    -------
        opt = TuRBO(bounds=[(-5, 5)] * 10, seed=0)
        X = opt.ask(n=1)
        y = [f(x) for x in X]
        opt.tell(X, y)
        x_best, y_best = opt.best()
    """

    def __init__(self, bounds, seed: Optional[int] = None):
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError(
                "bounds must have shape (d, 2): one (low, high) pair per dimension"
            )
        self.dim = self.bounds.shape[0]
        self.rng = np.random.default_rng(seed)

        self.batch_size = 4
        self.n_init = 2 * self.dim
        self.n_candidates = min(100 * self.dim, 5000)

        self._length_min = 0.5 ** 7
        self._length_max = 1.6
        self._length_init = 0.8
        self._success_tolerance = 10
        self._failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, self.dim / self.batch_size)
        )

        self._lower = self.bounds[:, 0]
        self._upper = self.bounds[:, 1]
        self._unit_bounds = torch.stack(
            [torch.zeros(self.dim), torch.ones(self.dim)]
        ).double()

        self._reset_trust_region()
        self._to_ask: list = []  # points generated, not yet asked out

        self.X_ = np.empty((0, self.dim))
        self.y_ = np.empty((0,))

    def _to_unit(self, x_raw):
        return (x_raw - self._lower) / (self._upper - self._lower)

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
        Y_t = torch.tensor(-self._local_y, dtype=torch.double).unsqueeze(-1)

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

        best_idx = int(np.argmin(self._local_y))
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

    def tell(self, X, y) -> None:
        """Record evaluated points `X` (n, dim) and their values `y` (n,).

        Must be called with exactly the points most recently returned by
        `ask`, in the same order (see class docstring).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))

        self.X_ = np.vstack([self.X_, X])
        self.y_ = np.concatenate([self.y_, y])

        for xi, yi in zip(X, y):
            self._local_X_unit = np.vstack([self._local_X_unit, self._to_unit(xi)])
            self._local_y = np.append(self._local_y, yi)

            if self._phase == "init":
                if len(self._local_y) >= self.n_init:
                    self._local_best_y = float(self._local_y.min())
                    self._phase = "evolve"
                    self._evolve_evals_since_update = 0
            else:
                self._evolve_evals_since_update += 1
                if self._evolve_evals_since_update >= self.batch_size:
                    self._update_trust_region()
                    self._evolve_evals_since_update = 0

    def _update_trust_region(self) -> None:
        recent_y = self._local_y[-self.batch_size:]
        improved = recent_y.min() < self._local_best_y - 1e-3 * abs(self._local_best_y)

        if improved:
            self._success_counter += 1
            self._failure_counter = 0
        else:
            self._success_counter = 0
            self._failure_counter += 1

        self._local_best_y = min(self._local_best_y, float(recent_y.min()))

        if self._success_counter == self._success_tolerance:
            self._length = min(2.0 * self._length, self._length_max)
            self._success_counter = 0
        elif self._failure_counter == self._failure_tolerance:
            self._length /= 2.0
            self._failure_counter = 0

        if self._length < self._length_min:
            self._reset_trust_region()

    def best(self):
        """Return (x_best, y_best) among all points told so far (minimum y)."""
        if len(self.y_) == 0:
            raise RuntimeError("No evaluations recorded yet; call tell() first.")
        i = np.argmin(self.y_)
        return self.X_[i], self.y_[i]
