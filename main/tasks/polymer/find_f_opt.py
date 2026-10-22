"""Search the OPV emulator for its best feasible PCE: the best-known f_opt of
OPVTask (src/tasks/opv.py). Run from main/tasks/polymer with the project venv:

    ../../../.venv/bin/python find_f_opt.py

Three searches in the task's unit box, all under the six required constraints:
  1. 2^20 scrambled Sobol points.
  2. Coordinate ascent from the best Sobol points. The forest is constant
     between consecutive split thresholds of each input, so each step is an
     exact line search: one candidate per interval within the feasible range.
  3. CMA-ES runs (pycma) on the campaign merit (run_campaign.py's penalty).
Prints the best value and point; nothing is written.
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cma
import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.tasks.base import FEASIBILITY_TOL  # noqa: E402
from src.tasks.opv import OPVTask  # noqa: E402

SEED = 0
SOBOL_LOG2 = 20
CHUNK = 4096
N_STARTS = 64          # coordinate-ascent starts: best distinct Sobol points
N_CMA = 20
CMA_EVALS = 100_000
EDGE = 1e-12           # keep line-search candidates strictly inside the feasible range

task = OPVTask()


def feasible(U):
    G = task.decode(U) @ task._A.T - task._b
    return np.maximum(G, 0).sum(axis=1) <= FEASIBILITY_TOL


def feasible_pce(U):
    """PCE at feasible points, -inf elsewhere."""
    ok = feasible(U)
    pce = np.full(len(U), -np.inf)
    pce[ok] = task.pce(task.decode(U[ok]))
    return pce


def unit_thresholds():
    """Split thresholds of each input, in unit-box coordinates."""
    inner = task._left >= 0
    w = task._upper - task._lower
    return [np.unique((task._threshold[inner & (task._feature == i)] - task._lower[i]) / w[i])
            for i in range(task.dim)]


THRESHOLDS = unit_thresholds()


def feasible_range(u, i):
    """Interval of u[i] (others fixed) inside [0, 1] and every constraint."""
    x = task.decode(u[None])[0]
    w = task._upper[i] - task._lower[i]
    a_i = task._A[:, i] * w                       # constraint slope in u[i]
    slack = task._b - task._A @ x                  # >= 0 at a feasible point
    lo, hi = 0.0, 1.0
    for s, r in zip(a_i, slack):
        if s > 0:
            hi = min(hi, u[i] + r / s)
        elif s < 0:
            lo = max(lo, u[i] + r / s)
    return lo, hi


def line_candidates(u, i):
    if i in task._levels:  # snapped input: one candidate per level
        levels = task._levels[i]
        return np.clip((levels - task._lower[i]) / (task._upper[i] - task._lower[i]), 0.0, 1.0)
    lo, hi = feasible_range(u, i)
    lo, hi = min(lo + EDGE, u[i]), max(hi - EDGE, u[i])
    t = THRESHOLDS[i]
    pts = np.concatenate([[lo], t[(t > lo) & (t < hi)], [hi]])
    return np.concatenate([pts, (pts[:-1] + pts[1:]) / 2])


def coordinate_ascent(args):
    u, seed = args
    rng = np.random.default_rng(seed)
    u = u.copy()
    best = task.pce(task.decode(u[None]))[0]
    improved = True
    while improved:
        improved = False
        for i in rng.permutation(task.dim):
            c = line_candidates(u, i)
            U = np.repeat(u[None], len(c), axis=0)
            U[:, i] = c
            ok = feasible(U)
            pce = np.where(ok, task.pce(task.decode(U)), -np.inf)
            j = int(np.argmax(pce))
            if pce[j] > best + 1e-12:
                best, u, improved = pce[j], U[j], True
    return best, u


def cma_run(seed):
    rng = np.random.default_rng(seed)
    es = cma.CMAEvolutionStrategy(rng.random(task.dim), 0.3,
                                  {"bounds": [0, 1], "seed": seed + 1, "verbose": -9, "maxfevals": CMA_EVALS})
    best, best_u, used = -np.inf, None, 0
    while used < CMA_EVALS:
        if es.stop():  # restart from a new random point until the budget is used
            es = cma.CMAEvolutionStrategy(rng.random(task.dim), 0.3, {
                "bounds": [0, 1], "seed": int(rng.integers(1 << 30)), "verbose": -9, "maxfevals": CMA_EVALS - used})
        U = np.clip(np.array(es.ask()), 0, 1)
        X = task.decode(U)
        viol = np.maximum(X @ task._A.T - task._b, 0).sum(axis=1)
        pce = task.pce(X)
        ok = viol <= FEASIBILITY_TOL
        merit = np.where(ok, -pce, 1e12 + 1e6 * viol + np.tanh(-pce))
        es.tell(list(U), list(merit))
        used += len(U)
        if ok.any() and pce[ok].max() > best:
            k = np.flatnonzero(ok)[np.argmax(pce[ok])]
            best, best_u = pce[k], U[k]
    return best, best_u


def main():
    t0 = time.time()
    with Pool() as pool:
        print(f"1. Sobol: {2 ** SOBOL_LOG2:,} points", flush=True)
        U = qmc.Sobol(task.dim, scramble=True, seed=SEED).random_base2(SOBOL_LOG2)
        pce = np.concatenate(pool.map(feasible_pce, np.split(U, len(U) // CHUNK)))
        order = np.argsort(-pce)
        sobol_best = pce[order[0]]
        print(f"   feasible {np.isfinite(pce).mean():.1%}, best PCE {sobol_best:.4f}% ({time.time() - t0:.0f}s)", flush=True)

        print(f"2. Coordinate ascent from the {N_STARTS} best Sobol points", flush=True)
        starts = [(U[k], SEED + n) for n, k in enumerate(order[:N_STARTS])]
        ascent = []
        for n, r in enumerate(pool.imap_unordered(coordinate_ascent, starts), 1):
            ascent.append(r)
            print(f"   {n}/{N_STARTS}: {r[0]:.4f}%", flush=True)
        ascent_vals = np.array([r[0] for r in ascent])
        print(f"   best {ascent_vals.max():.4f}%, reached by {np.sum(ascent_vals >= ascent_vals.max() - 1e-12)} of "
              f"{N_STARTS} ({time.time() - t0:.0f}s)", flush=True)

        print(f"3. CMA-ES: {N_CMA} runs of {CMA_EVALS:,} evaluations", flush=True)
        cma_res = []
        for n, r in enumerate(pool.imap_unordered(cma_run, range(N_CMA)), 1):
            cma_res.append(r)
            print(f"   {n}/{N_CMA}: {r[0]:.4f}%", flush=True)
        cma_vals = np.array([r[0] for r in cma_res])
        print(f"   best {cma_vals.max():.4f}% ({time.time() - t0:.0f}s)", flush=True)

    best_pce, best_u = max(ascent + cma_res + [(sobol_best, U[order[0]])], key=lambda r: r[0])
    ev = task.evaluate(best_u)
    assert ev.feasible and abs(-ev.y - best_pce) < 1e-12, "best point does not re-evaluate to the same value"
    x = task.decode(best_u[None])[0]
    print(f"\nBest-known f_opt = {ev.y!r} (PCE {-ev.y:.4f}%)")
    print("u =", np.array2string(best_u, precision=17, separator=", ", max_line_width=200))
    print("x =", ", ".join(f"{n} {v:.4g}" for n, v in zip(task.input_names, x)))
    print("g =", ", ".join(f"{n} {v:.3g}" for n, v in zip(task.constraint_names, ev.g)))
    centre = task.evaluate(np.full(task.dim, 0.5))
    print(f"Box centre: PCE {-centre.y:.4f}%, feasible {centre.feasible}")


if __name__ == "__main__":
    main()
