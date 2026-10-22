"""Shared archive of every evaluation in a run (design §6.1).

The archive is the single source of truth for what has been evaluated. Every
optimizer, probe and race reads from and writes to it, so switching methods
never loses progress: a new method is warm-started from `incumbent()`,
`top_k()` and `local_scale()` (see agent/warmstart.py).
"""

from __future__ import annotations

import numpy as np


class Archive:
    def __init__(self, bounds):
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = self.bounds.shape[0]
        self.lower, self.upper = self.bounds[:, 0], self.bounds[:, 1]
        self.width = self.upper - self.lower
        self.X = np.empty((0, self.dim))
        self.y = np.empty((0,))
        self.tags: list[str] = []

    # ------------------------------------------------------------------ write
    def add(self, x, y: float, tag: str) -> None:
        x = np.asarray(x, dtype=float).reshape(1, -1)
        self.X = np.vstack([self.X, x])
        self.y = np.append(self.y, float(y))
        self.tags.append(str(tag))

    # ------------------------------------------------------------------- read
    def __len__(self) -> int:
        return len(self.y)

    def incumbent(self) -> tuple[np.ndarray, float]:
        if len(self.y) == 0:
            raise RuntimeError("Archive is empty.")
        i = int(np.argmin(self.y))
        return self.X[i].copy(), float(self.y[i])

    def incumbent_index(self) -> int:
        return int(np.argmin(self.y))

    def top_k(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        idx = np.argsort(self.y)[: max(1, int(k))]
        return self.X[idx].copy(), self.y[idx].copy()

    def nearest(self, x, n: int) -> tuple[np.ndarray, np.ndarray]:
        """The n archive points closest to x (box-normalized distance)."""
        d = np.linalg.norm((self.X - np.asarray(x)) / self.width, axis=1)
        idx = np.argsort(d)[: max(1, int(n))]
        return self.X[idx].copy(), self.y[idx].copy()

    def local_scale(self, k: int | None = None) -> np.ndarray:
        """Per-dimension spread of the k best points: an estimate of how far
        from the incumbent it is still worth searching. Used for CMA-ES
        sigma0, BOBYQA rhobeg, Nelder-Mead simplex size and the TuRBO trust
        region. Clipped to [1e-6, 0.3] x box width so it is never zero or
        larger than a global step."""
        if k is None:
            k = max(2 * self.dim, 10)
        if len(self.y) < 2:
            return 0.3 * self.width
        Xk, _ = self.top_k(min(k, len(self.y)))
        s = Xk.std(axis=0)
        return np.clip(s, 1e-6 * self.width, 0.3 * self.width)

    def diverse_starts(self, m: int, pool_frac: float = 0.2) -> np.ndarray:
        """m good AND mutually distant points: greedy max-min selection
        within the best `pool_frac` of the archive, starting from the
        incumbent. Used for restarts in different basins."""
        n_pool = max(m, int(np.ceil(pool_frac * len(self.y))))
        P, _ = self.top_k(min(n_pool, len(self.y)))
        Pn = (P - self.lower) / self.width
        chosen = [0]
        while len(chosen) < min(m, len(P)):
            d = np.min(
                np.linalg.norm(Pn[:, None, :] - Pn[None, chosen, :], axis=2), axis=1
            )
            d[chosen] = -1
            chosen.append(int(np.argmax(d)))
        return P[chosen]

    def best_curve(self) -> np.ndarray:
        return np.minimum.accumulate(self.y) if len(self.y) else self.y

    def evals_since_improvement(self, rel_tol: float = 1e-9) -> int:
        """Evaluations since the best value last improved by more than
        rel_tol (relative to max(1, |best|))."""
        if len(self.y) == 0:
            return 0
        bc = self.best_curve()
        final = bc[-1]
        tol = rel_tol * max(1.0, abs(final))
        # first index at which best was already within tol of the final best
        idx = int(np.argmax(bc <= final + tol))
        return len(bc) - 1 - idx
