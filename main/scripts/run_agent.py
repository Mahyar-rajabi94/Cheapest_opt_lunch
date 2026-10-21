"""Run the v2 agent on one task.

    python scripts/run_agent.py --task task001 --policy scripted --seed 0
    python scripts/run_agent.py --task dev_f15_d10 --policy llm --provider anthropic

Writes trace.csv, meta.json, decisions.jsonl, intake.yaml to
results_agent/<task>/<policy>/seed_<seed>/ (or --outdir). The task's
identity and optimum are used only for the final printout and meta.json,
never shown to the policy.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from agent.controller import Controller  # noqa: E402
from agent.dev_tasks import get_task  # noqa: E402
from agent.intake import Intake  # noqa: E402


def make_policy(args):
    if args.policy == "scripted":
        from agent.policies.scripted import ScriptedPolicy
        return ScriptedPolicy(use_probes=not args.no_probes)
    if args.policy == "llm":
        from agent.policies.llm import LLMPolicy
        return LLMPolicy(provider=args.provider, model=args.model,
                         playbook=not args.no_playbook)
    raise SystemExit(f"unknown policy {args.policy}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="task001")
    p.add_argument("--policy", default="scripted", choices=["scripted", "llm"])
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    p.add_argument("--model", default=None)
    p.add_argument("--no-probes", action="store_true")
    p.add_argument("--no-playbook", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--eval-cost", default="cheap")
    p.add_argument("--noise", default="deterministic")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    task = get_task(args.task)
    intake = Intake(dim=task.dim, bounds=task.bounds.tolist(), budget_evals=args.budget,
                    eval_cost=args.eval_cost, noise=args.noise)
    policy = make_policy(args)
    ctl = Controller(task, intake, policy, seed=args.seed)
    res = ctl.run()
    label = args.policy + ("" if args.policy == "scripted" else f"_{args.provider}")
    if args.no_probes:
        label += "_noprobes"
    if args.no_playbook:
        label += "_noplaybook"
    out = Path(args.outdir or f"results_agent/{task.name}/{label}/seed_{args.seed}")
    ctl.save(out, {"task_name": task.name, "task_f_opt": task.f_opt})
    regret = res["best_y"] - task.f_opt
    bc = ctl.archive.best_curve() - task.f_opt
    hits = {t: (int((bc <= t).argmax()) + 1 if (bc <= t).any() else None)
            for t in (1.0, 1e-2, 1e-6)}
    print(json.dumps({"task": task.name, "policy": label, "seed": args.seed,
                      "final_regret": regret, "evals_to_regret": hits,
                      "rounds": res["rounds"], "out": str(out)}, default=str))


if __name__ == "__main__":
    main()
