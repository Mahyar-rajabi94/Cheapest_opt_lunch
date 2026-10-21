import sys
import numpy as np

sys.path.insert(0, ".")
from src.agent_harness.task_001 import AgentTaskHarness, BudgetExhausted
from src.optimizers.sobol_search import SobolSearch
from src.optimizers.cma_es import CMAES
from src.optimizers.bobyqa import BOBYQA
from src.optimizers import OptimizerExhausted


def run_phase(h, opt, name, count, start_eval, best_value):
    h.set_active_optimizer(name)
    last_improve = start_eval
    for k in range(count):
        X = opt.ask(n=1)
        y = float(h.evaluate(X[0]))
        opt.tell(X, [y])
        if y < best_value:
            best_value = y
            last_improve = start_eval + k + 1
        if (start_eval + k + 1) % 100 == 0:
            print(f"eval={start_eval+k+1} phase={name} best={best_value:.16g} last_improve={last_improve}", flush=True)
    return best_value, last_improve


def main():
    h = AgentTaskHarness(budget=2000)
    bounds = [(-5.0, 5.0)] * 10
    best_value = float("inf")
    eval_no = 0
    phases = [
        (SobolSearch(bounds, seed=1001), "sobol_search", 256),
        (CMAES(bounds, seed=2001), "cma_es", 1244),
        (BOBYQA(bounds), "bobyqa", 500),
    ]
    for opt, name, count in phases:
        try:
            best_value, _ = run_phase(h, opt, name, count, eval_no, best_value)
            eval_no += count
        except (OptimizerExhausted, BudgetExhausted) as exc:
            print(f"stopped at eval={eval_no} phase={name} reason={exc}", flush=True)
            h.save()
            break
    else:
        h.save()
    print(f"finished evals={eval_no} best={best_value:.16g}", flush=True)


if __name__ == "__main__":
    main()
