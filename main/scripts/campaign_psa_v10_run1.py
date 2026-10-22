"""Campaign psa_v10_blackbox_run1: Sobol' -> CMA-ES restarts -> seeded DE -> local search.

Implements the pre-registered plan in agent_runs/psa_v10_blackbox_run1/roadmap_initial.md,
including its deviation criteria 2-6, which append entries to changelog.md when they fire.

    .venv/Scripts/python.exe main/scripts/campaign_psa_v10_run1.py             # real campaign
    .venv/Scripts/python.exe main/scripts/campaign_psa_v10_run1.py --dry-run D # synthetic stand-in, output to D

The real campaign may be run exactly once: a second harness would reset the budget counter.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="Could not import matplotlib.*")
import cma  # noqa: E402

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))
from _psa_v10_legacy_harness import BlackBoxHarness  # noqa: E402
from src.optimizers.sobol_search import SobolSearch  # noqa: E402

EVALUATOR = Path(r"C:\Users\rahim\Desktop\psa_v10_blackbox.py")
OUTDIR = MAIN / "agent_runs" / "psa_v10_blackbox_run1"
DIM, BUDGET = 8, 2000
BOUNDS = [[0.0, 1.0]] * DIM

# Cumulative phase end targets (evaluations used), from the roadmap allocation.
A_END, B_END, C_END = 256, 256 + 800, 256 + 800 + 600
CACHE_HIT_LIMIT = 50
REHIT_WINDOW, REHIT_MIN_NEW, EXPLORE_EVALS = 100, 10, 100


class PhaseEnd(Exception):
    """Raised by Campaign.f when the active phase must stop before evaluating."""


class Campaign:
    def __init__(self, harness, changelog: Path):
        self.h = harness
        self.changelog = changelog
        self.cache = {}
        self.X, self.Y = [], []
        self.seen_values = set()
        self.best_y, self.best_x, self.best_eval = math.inf, None, 0
        self.phase = None

    # -- bookkeeping -------------------------------------------------------
    @property
    def used(self):
        return self.h.evaluations_used

    def log_change(self, change, reason):
        with open(self.changelog, "a", encoding="utf-8") as fh:
            fh.write(f"## Evaluation {self.used}\n**Change:** {change}\n**Reason:** {reason}\n\n")
        print(f"[changelog @ {self.used}] {change}", flush=True)

    def start_phase(self, label, end, check_rehit=True):
        self.phase, self.phase_end, self.check_rehit = label, min(end, BUDGET), check_rehit
        self.phase_start, self.segment_new, self.cache_hits = self.used, [], 0
        self.h.set_active_optimizer(label)
        print(f"-- phase {label}: evals {self.used}..{self.phase_end}, best so far {self.best_y}", flush=True)

    def distinct_best(self, k):
        """The k best points with distinct values (one representative per material value)."""
        order = np.argsort(self.Y, kind="stable")
        picked, vals = [], set()
        for i in order:
            if self.Y[i] not in vals:
                vals.add(self.Y[i])
                picked.append((np.array(self.X[i]), self.Y[i]))
                if len(picked) == k:
                    break
        return picked

    def on_phase_limit(self):
        """Hook for criterion 3; returns True if the phase was extended."""
        return False

    # -- the one evaluation entry point ------------------------------------
    def f(self, x):
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        key = tuple(x.tolist())
        if key in self.cache:
            self.cache_hits += 1
            if self.cache_hits >= CACHE_HIT_LIMIT:
                self.log_change(
                    f"Ended phase `{self.phase}` early (criterion 6).",
                    f"{self.cache_hits} proposals in this phase repeated already-evaluated points exactly.",
                )
                raise PhaseEnd("cache")
            return self.cache[key]
        if self.used >= self.phase_end and not self.on_phase_limit():
            raise PhaseEnd("budget")

        y = self.h.evaluate(x)
        self.cache[key] = y
        self.X.append(x)
        self.Y.append(y)
        is_new = y not in self.seen_values
        self.seen_values.add(y)
        self.segment_new.append(is_new)
        if y < self.best_y:
            self.best_y, self.best_x, self.best_eval = y, x, self.used
            print(f"   eval {self.used:4d} [{self.phase}] new best {y:.6g}", flush=True)

        # Criterion 4: mostly re-hitting known materials.
        if self.check_rehit and len(self.segment_new) >= REHIT_WINDOW:
            n_new = sum(self.segment_new[-REHIT_WINDOW:])
            if n_new < REHIT_MIN_NEW and self.used < BUDGET:
                self.log_change(
                    f"Ended phase `{self.phase}` early; next {EXPLORE_EVALS} evaluations go to continued "
                    "Sobol' exploration (`sobol_explore`), then the plan resumes (criterion 4).",
                    f"Only {n_new} of the last {REHIT_WINDOW} evaluations returned a previously unseen value "
                    f"({len(self.seen_values)} distinct values in {self.used} evaluations; best {self.best_y:.6g}).",
                )
                raise PhaseEnd("rehit")
        return y


def run(objective, outdir: Path, task_id: str):
    outdir.mkdir(parents=True, exist_ok=True)
    harness = BlackBoxHarness(objective=objective, bounds=BOUNDS, budget=BUDGET, outdir=str(outdir), task_id=task_id)
    camp = Campaign(harness, outdir / "changelog.md")
    sobol = SobolSearch(bounds=BOUNDS, seed=0)

    def explore():
        camp.start_phase("sobol_explore", camp.used + EXPLORE_EVALS, check_rehit=False)
        try:
            while True:
                x = sobol.ask(1)[0]
                sobol.tell([x], [camp.f(x)])
        except PhaseEnd:
            pass

    def run_phase(label, end, body, **kw):
        """Run one phase; returns the PhaseEnd reason. Re-hit endings trigger one exploration block."""
        camp.start_phase(label, end, **kw)
        try:
            body()
            reason = "done"
        except PhaseEnd as e:
            reason = str(e)
        if reason == "rehit":
            explore()
        return reason

    # ---- Phase A: Sobol' --------------------------------------------------
    def phase_a():
        while True:
            x = sobol.ask(1)[0]
            sobol.tell([x], [camp.f(x)])

    # ---- Phase B: CMA-ES with restarts ----------------------------------
    state = {"b_extended": False, "c2_checked": False}

    def phase_b():
        k = 0
        while True:
            starts = camp.distinct_best(k + 1)
            x0 = starts[min(k, len(starts) - 1)][0]
            es = cma.CMAEvolutionStrategy(
                x0.tolist(), 0.15, {"bounds": [0.0, 1.0], "verbose": -9, "seed": 1000 + k}
            )
            camp.h.set_active_optimizer(f"cma_es_r{k}")
            run_best, last_imp = math.inf, camp.used
            while not es.stop():
                X = es.ask()
                ys = [camp.f(x) for x in X]
                es.tell(X, ys)
                if min(ys) < run_best:
                    run_best, last_imp = min(ys), camp.used
                # Criterion 2: CMA-ES has not beaten the Sobol' best after 300 evaluations.
                if not state["c2_checked"] and camp.used - camp.phase_start >= 300:
                    state["c2_checked"] = True
                    if camp.best_eval <= A_END:
                        camp.log_change(
                            "Ended Phase B (CMA-ES) early; its remaining allocation moves to Phase C (DE) (criterion 2).",
                            f"After {camp.used - camp.phase_start} CMA-ES evaluations the global best "
                            f"{camp.best_y:.6g} is still the Sobol' best (eval {camp.best_eval}).",
                        )
                        raise PhaseEnd("criterion2")
                if camp.used - last_imp >= 60 or es.sigma < 0.01:
                    break
            k += 1

    def b_limit():
        # Criterion 3: extend Phase B once by 200 if the global best improved in its last 150 evaluations.
        if camp.phase == "cma_es" and not state["b_extended"] and camp.best_eval > camp.phase_end - 150:
            state["b_extended"] = True
            camp.phase_end = min(camp.phase_end + 200, BUDGET)
            camp.log_change(
                "Extended Phase B (CMA-ES) by 200 evaluations, taken from Phase C (criterion 3).",
                f"Global best improved to {camp.best_y:.6g} at eval {camp.best_eval}, within the last 150 "
                "evaluations of Phase B.",
            )
            return True
        return False

    # ---- Phase C: DE/rand-to-best/1/bin seeded from the archive ---------
    def phase_c():
        rng = np.random.default_rng(1)
        npop, F, CR = 40, 0.7, 0.9
        seeds = camp.distinct_best(npop)
        if len(seeds) < npop:  # pad with the best remaining points if few distinct values exist
            for i in np.argsort(camp.Y, kind="stable"):
                if len(seeds) == npop:
                    break
                if not any(np.array_equal(camp.X[i], s[0]) for s in seeds):
                    seeds.append((np.array(camp.X[i]), camp.Y[i]))
        pop = np.array([s[0] for s in seeds])
        fit = np.array([s[1] for s in seeds])
        c_start_best_eval = camp.used
        while True:
            for i in range(len(pop)):
                b = int(np.argmin(fit))
                r1, r2, r3 = rng.choice([j for j in range(len(pop)) if j != i], size=3, replace=False)
                mutant = pop[r1] + F * (pop[b] - pop[r1]) + F * (pop[r2] - pop[r3])
                mask = rng.random(DIM) < CR
                mask[rng.integers(DIM)] = True
                trial = np.clip(np.where(mask, mutant, pop[i]), 0.0, 1.0)
                y = camp.f(trial)
                if y <= fit[i]:
                    pop[i], fit[i] = trial, y
                # Criterion 5: DE plateau.
                if camp.used - max(camp.best_eval, c_start_best_eval) >= 250:
                    camp.log_change(
                        "Ended Phase C (DE) early; moving to Phase D (local search) (criterion 5).",
                        f"No improvement of the global best ({camp.best_y:.6g}, eval {camp.best_eval}) "
                        "during the last 250 DE evaluations.",
                    )
                    raise PhaseEnd("criterion5")

    # ---- Phase D: (1+1) local search around the 5 best materials -------
    def phase_d():
        rng = np.random.default_rng([2, camp.used])  # fresh stream if Phase D is re-entered
        centres = [[x.copy(), y, 0.08, 0] for x, y in camp.distinct_best(5)]
        i = 0
        while True:
            c = centres[i % len(centres)]
            x = np.clip(c[0] + rng.normal(0.0, c[2], DIM), 0.0, 1.0)
            y = camp.f(x)
            if y < c[1]:
                c[0], c[1], c[2], c[3] = x, y, 0.08, 0
            else:
                c[3] += 1
                if c[3] >= 15:
                    c[2], c[3] = max(c[2] / 2, 0.01), 0
            i += 1

    try:
        run_phase("sobol", A_END, phase_a, check_rehit=False)
        camp.on_phase_limit = b_limit  # only acts while camp.phase == "cma_es"
        run_phase("cma_es", B_END, phase_b)
        run_phase("de_seeded", C_END, phase_c)
        while camp.used < BUDGET:  # Phase D runs to the end; criteria 4/6 may interrupt it
            run_phase("local_search", BUDGET, phase_d)
    finally:
        harness.save()

    print(f"\nDONE: {camp.used} evaluations, best {camp.best_y:.10g} at eval {camp.best_eval}")
    print("best x:", np.array2string(camp.best_x, precision=6, separator=", "))
    print(f"distinct values seen: {len(camp.seen_values)}")
    return camp


def load_blackbox():
    spec = importlib.util.spec_from_file_location("psa_v10_blackbox", EVALUATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BlackBox()


def synthetic(seed):
    """Stand-in for the dry run only: a nearest-neighbour lookup over random points."""
    rng = np.random.default_rng(seed)
    P = rng.random((1500, DIM))
    v = np.sum((P - 0.3) ** 2, axis=1) + 0.3 * np.sin(9 * P[:, 0]) + 0.05 * rng.standard_normal(1500)
    return lambda x: float(v[np.argmin(np.sum((P - x) ** 2, axis=1))])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", metavar="OUTDIR", help="use a synthetic objective and write to OUTDIR")
    ap.add_argument("--seed", type=int, default=0, help="synthetic objective seed (dry run only)")
    args = ap.parse_args()
    if args.dry_run:
        out = Path(args.dry_run)
        out.mkdir(parents=True, exist_ok=True)
        (out / "changelog.md").write_text("")
        camp = run(synthetic(args.seed), out, "dry_run")
        assert camp.used == BUDGET, camp.used
    else:
        if (OUTDIR / "trace.csv").exists():
            sys.exit(f"{OUTDIR} already holds a campaign; refusing to run again.")
        f = load_blackbox()
        assert f.dim == DIM and f.direction == "min" and f.bounds == BOUNDS
        run(f, OUTDIR, "psa_v10_blackbox")
