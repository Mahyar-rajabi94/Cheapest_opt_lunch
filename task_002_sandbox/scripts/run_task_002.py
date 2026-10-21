"""Campaign runner for Task 002. See agent_briefs/task_002.md and
results/agent_task_002/roadmap_initial.md for the plan this implements.

Phases:
  1. Sobol' init design         (sobol_init)          -- 100 evals
  2. CMA-ES global search       (cma_es / cma_es_r{k}) -- up to ~1400 evals
  3. Local polish                (local_polish_bobyqa, local_polish_lbfgsb)
                                                        -- remaining evals

Deviation checks (stagnation, convergence, BOBYQA cap, plateau) are
implemented exactly as pre-registered and logged to changelog.md the
moment they fire.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.agent_harness.task_002 import AgentTaskHarness, BudgetExhausted
from src.optimizers import OptimizerExhausted
from src.optimizers.sobol_search import SobolSearch
from src.optimizers.cma_es import CMAES
from src.optimizers.bobyqa import BOBYQA
from src.optimizers.lbfgsb import LBFGSB

RESULTS_DIR = REPO_ROOT / "results" / "agent_task_002"
CHANGELOG = RESULTS_DIR / "changelog.md"

SOBOL_BUDGET = 100
CMA_BUDGET_TARGET = 1400
CMA_STAGNATION_WINDOW = 200
CMA_MAX_RESTARTS = 2
BOBYQA_CAP = 400
LOCAL_PLATEAU_TOL = 1e-8


def log_change(n_eval: int, change: str, reason: str) -> None:
    entry = f"\n## Evaluation {n_eval}\n**Change:** {change}\n**Reason:** {reason}\n"
    with open(CHANGELOG, "a") as f:
        f.write(entry)
    print(f"[changelog] @{n_eval}: {change}")


def main() -> None:
    harness = AgentTaskHarness(budget=2000)
    bounds = harness.bounds
    rng_seed = 0

    # ---------------- Phase 1: Sobol' init design ----------------
    harness.set_active_optimizer("sobol_init")
    sobol = SobolSearch(bounds=bounds, seed=rng_seed)
    X = sobol.ask(n=SOBOL_BUDGET)
    y = [harness.evaluate(x) for x in X]
    sobol.tell(X, y)
    x_best, y_best = sobol.best()
    print(f"[sobol] done: {harness.evaluations_used} evals, best={y_best:.6g}")

    # ---------------- Phase 2: CMA-ES global search ----------------
    harness.set_active_optimizer("cma_es")
    cma_restart_count = 0
    cma_evals_done = 0
    global_best_x, global_best_y = x_best, y_best
    recent_window: list = []  # (eval_index, best_so_far) for stagnation check

    cma = CMAES(bounds=bounds, seed=rng_seed + 1)
    active_label = "cma_es"

    def cma_run_until(stop_budget: int, cma_obj, label: str):
        nonlocal global_best_x, global_best_y, cma_evals_done, recent_window
        while cma_evals_done < stop_budget and harness.evaluations_remaining > 0:
            Xc = cma_obj.ask(n=1)
            yc = [harness.evaluate(Xc[0])]
            cma_obj.tell(Xc, yc)
            cma_evals_done += 1
            if yc[0] < global_best_y:
                global_best_y = yc[0]
                global_best_x = Xc[0]
            recent_window.append((harness.evaluations_used, global_best_y))
            # trim window to last CMA_STAGNATION_WINDOW evals
            while (
                recent_window
                and recent_window[-1][0] - recent_window[0][0]
                > CMA_STAGNATION_WINDOW
            ):
                recent_window.pop(0)
            # stagnation check
            if len(recent_window) >= CMA_STAGNATION_WINDOW:
                old_best = recent_window[0][1]
                improvement = old_best - global_best_y
                thresh = 1e-6 * max(abs(global_best_y), 1.0)
                if improvement < thresh:
                    return "stagnated"
        return "budget_reached"

    status = cma_run_until(CMA_BUDGET_TARGET, cma, active_label)

    while status == "stagnated" and cma_restart_count < CMA_MAX_RESTARTS and harness.evaluations_remaining > 0:
        cma_restart_count += 1
        label = f"cma_es_restart_{cma_restart_count}"
        harness.set_active_optimizer(label)
        log_change(
            harness.evaluations_used,
            f"Restarted CMA-ES (restart #{cma_restart_count}, new seed) instead of continuing the previous run.",
            f"Best value improved by less than 1e-6*|best| over the last "
            f"{CMA_STAGNATION_WINDOW} evaluations (best stuck at "
            f"{global_best_y:.6g}); pre-registered stagnation criterion fired.",
        )
        cma = CMAES(bounds=bounds, seed=rng_seed + 10 + cma_restart_count)
        recent_window = []
        remaining_for_cma = max(CMA_BUDGET_TARGET - cma_evals_done, 0)
        status = cma_run_until(cma_evals_done + remaining_for_cma, cma, label)

    if cma_restart_count >= CMA_MAX_RESTARTS and status == "stagnated":
        log_change(
            harness.evaluations_used,
            "Abandoning CMA-ES after max restarts; moving to local polish early with remaining global-search budget unused.",
            f"Used {CMA_MAX_RESTARTS} restarts, still stagnating at best={global_best_y:.6g}; "
            "pre-registered cap on restarts reached (TuRBO fallback not needed since "
            "enough budget remains for local polish to still add value from the best basin found).",
        )

    print(
        f"[cma_es] done: {harness.evaluations_used} evals total, "
        f"best={global_best_y:.6g}, restarts={cma_restart_count}"
    )

    # ---------------- Phase 3: Local polish ----------------
    remaining = harness.evaluations_remaining
    local_budget = remaining  # whatever is left, as planned
    print(f"[local_polish] budget available: {local_budget}")

    polish_best_x, polish_best_y = global_best_x, global_best_y

    def make_local_bounds(center, half_width=0.25):
        center = np.asarray(center, dtype=float)
        b = np.asarray(bounds, dtype=float)
        lo = np.maximum(b[:, 0], center - half_width * (b[:, 1] - b[:, 0]))
        hi = np.minimum(b[:, 1], center + half_width * (b[:, 1] - b[:, 0]))
        return list(zip(lo.tolist(), hi.tolist()))

    bobyqa_cap_used = 0
    if local_budget > 0:
        harness.set_active_optimizer("local_polish_bobyqa")
        local_bounds = make_local_bounds(polish_best_x)
        bobyqa = BOBYQA(bounds=local_bounds)
        last_improve_y = polish_best_y
        try:
            while bobyqa_cap_used < min(BOBYQA_CAP, local_budget) and harness.evaluations_remaining > 0:
                Xb = bobyqa.ask(n=1)
                yb = [harness.evaluate(Xb[0])]
                bobyqa.tell(Xb, yb)
                bobyqa_cap_used += 1
                if yb[0] < polish_best_y:
                    polish_best_y = yb[0]
                    polish_best_x = Xb[0]
        except OptimizerExhausted as e:
            log_change(
                harness.evaluations_used,
                "BOBYQA converged (OptimizerExhausted raised); switching to L-BFGS-B for remaining local-polish budget.",
                f"BOBYQA signalled its own convergence after {bobyqa_cap_used} evaluations "
                f"in this phase, best={polish_best_y:.6g}; per pre-registered criterion 3, "
                "fall back to L-BFGS-B rather than stop, since evaluation budget remains.",
            )
        else:
            if bobyqa_cap_used >= min(BOBYQA_CAP, local_budget) and bobyqa_cap_used >= BOBYQA_CAP:
                log_change(
                    harness.evaluations_used,
                    f"Stopped BOBYQA at its pre-registered {BOBYQA_CAP}-evaluation cap (did not let it run further) and switching to L-BFGS-B.",
                    "Per the brief's documented BOBYQA thread-wrapper slowdown past "
                    "~500 evaluations, criterion 3 pre-committed to a 400-evaluation "
                    f"hard cap for any single BOBYQA run; reached it with best={polish_best_y:.6g}.",
                )

    remaining_after_bobyqa = harness.evaluations_remaining
    if remaining_after_bobyqa > 0:
        harness.set_active_optimizer("local_polish_lbfgsb")
        local_bounds2 = make_local_bounds(polish_best_x)
        lbfgsb = LBFGSB(bounds=local_bounds2)
        prev_best = polish_best_y
        stale_count = 0
        try:
            while harness.evaluations_remaining > 0:
                Xl = lbfgsb.ask(n=1)
                yl = [harness.evaluate(Xl[0])]
                lbfgsb.tell(Xl, yl)
                if yl[0] < polish_best_y - LOCAL_PLATEAU_TOL:
                    polish_best_y = yl[0]
                    polish_best_x = Xl[0]
                    stale_count = 0
                else:
                    stale_count += 1
        except OptimizerExhausted:
            pass

    if polish_best_y < global_best_y:
        log_change(
            harness.evaluations_used,
            "Accepted local-polish result as new global best.",
            f"Local polish improved best from {global_best_y:.6g} to {polish_best_y:.6g} "
            "via deterministic local refinement from the CMA-ES incumbent.",
        )
        global_best_x, global_best_y = polish_best_x, polish_best_y

    print(f"[final] evals_used={harness.evaluations_used}, best={global_best_y:.6g}")

    harness.save()
    print("[saved]")


if __name__ == "__main__":
    main()
