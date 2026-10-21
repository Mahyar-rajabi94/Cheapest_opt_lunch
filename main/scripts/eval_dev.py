"""Dev smoke comparison (NOT official results): scripted policy vs fixed
single methods on the dev tasks. Prints a table of median final regret and
median evals to reach regret thresholds.

    python scripts/eval_dev.py --seeds 5 --jobs 2
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

TASKS = ["task001", "dev_f1_d10", "dev_f10_d10", "dev_f15_d10", "dev_f21_d10"]
POLICIES = ["scripted", "fixed_cma_es", "fixed_cma_es_restarts", "fixed_lbfgsb", "fixed_bobyqa"]
DETERMINISTIC = {"fixed_lbfgsb", "fixed_bobyqa"}


def run_one(args):
    task_name, pol_name, seed, budget = args
    warnings.filterwarnings("ignore")
    from agent.controller import Controller
    from agent.dev_tasks import get_task
    from agent.intake import Intake
    from agent.policies.fixed import FixedPolicy
    from agent.policies.scripted import ScriptedPolicy

    t = get_task(task_name)
    intake = Intake(dim=t.dim, bounds=t.bounds.tolist(), budget_evals=budget,
                    eval_cost="cheap", noise="deterministic")
    if pol_name == "scripted":
        pol = ScriptedPolicy()
    else:
        m = pol_name.removeprefix("fixed_").removesuffix("_restarts")
        pol = FixedPolicy(m, restarts=pol_name.endswith("_restarts"))
    ctl = Controller(t, intake, pol, seed=seed)
    ctl.run()
    bc = ctl.archive.best_curve() - t.f_opt
    hit = lambda th: int((bc <= th).argmax()) + 1 if (bc <= th).any() else None
    return {"task": task_name, "policy": pol_name, "seed": seed,
            "regret": float(bc[-1]), "t1e-2": hit(1e-2), "t1e-6": hit(1e-6),
            "guess": getattr(pol, "landscape", None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default="/tmp/eval_dev.json")
    a = ap.parse_args()
    jobs = [(t, p, s, a.budget) for t in TASKS for p in POLICIES
            for s in (range(1) if p in DETERMINISTIC else range(a.seeds))]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(run_one, jobs))
    json.dump(res, open(a.out, "w"), indent=1)

    def med(v):
        v = [x for x in v if x is not None]
        return None if not v else float(np.median(v))

    print(f"{'task':13s} {'policy':22s} {'n':>2s} {'med regret':>11s} {'med t(1e-2)':>11s} {'med t(1e-6)':>11s} solved")
    for t in TASKS:
        for p in POLICIES:
            rs = [r for r in res if r["task"] == t and r["policy"] == p]
            solved = sum(r["t1e-6"] is not None for r in rs)
            t2 = [r["t1e-2"] for r in rs]; t6 = [r["t1e-6"] for r in rs]
            fmt = lambda v: f"{v:11.0f}" if v is not None else f"{'-':>11s}"
            print(f"{t:13s} {p:22s} {len(rs):2d} {med([r['regret'] for r in rs]):11.3g} "
                  f"{fmt(med(t2) if all(x is not None for x in t2) else None)} "
                  f"{fmt(med(t6) if solved == len(rs) else None)} {solved}/{len(rs)}")
        guesses = {r["guess"] for r in res if r["task"] == t and r["policy"] == "scripted"}
        print(f"{'':13s} scripted landscape guess: {guesses}")


if __name__ == "__main__":
    main()
