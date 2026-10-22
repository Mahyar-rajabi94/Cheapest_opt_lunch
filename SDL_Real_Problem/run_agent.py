"""Run the team's v2 agent (branch agent-v2, main/agent/) on one SDL problem.

The agent package lives on the agent-v2 branch, not on master, so point
--agent-main at a checkout of that branch's main/ folder, e.g.
    git worktree add ../agent_v2 origin/agent-v2
    python SDL_Real_Problem/run_agent.py 02_viscometer_calibration --agent-main ../agent_v2/main

The agent only ever sees x -> y through its Evaluator (budget-enforced); the
task's f_opt is written to meta.json for scoring and never passed to the
policy. Only the scripted policy runs without an API key; --policy llm needs
ANTHROPIC_API_KEY, OPENAI_API_KEY or VP_API_KEY (see agent/policies/llm.py).

Writes <problem>/agent_runs/<label>/seed_<n>/{trace.csv,meta.json,decisions.jsonl,...}
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
warnings.filterwarnings("ignore")


def load_task(problem: str):
    """Build the task with master's src/ on the path, then drop that src/ so
    the agent's own src/ (agent-v2 branch) can be imported afterwards."""
    spec = importlib.util.spec_from_file_location(f"sdl_{problem}_task", ROOT / problem / "task.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    task = module.make_task()
    master_main = str(ROOT.parent / "main")
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(master_main).resolve()]
    for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[name]
    return task


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("problem")
    p.add_argument("--agent-main", required=True, help="main/ folder of an agent-v2 checkout")
    p.add_argument("--policy", default="scripted", choices=["scripted", "llm"])
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--budget", type=int, default=2000)
    args = p.parse_args()

    task = load_task(args.problem)
    sys.path.insert(0, str(Path(args.agent_main).resolve()))
    from agent.controller import Controller
    from agent.intake import Intake

    if args.policy == "scripted":
        from agent.policies.scripted import ScriptedPolicy

        make_policy, label = (lambda: ScriptedPolicy()), "v2_scripted"
    else:
        from agent.policies.llm import LLMPolicy

        make_policy, label = (lambda: LLMPolicy(provider=args.provider)), f"v2_llm_{args.provider}"

    for seed in range(args.seeds):
        intake = Intake(dim=task.dim, bounds=task.bounds.tolist(), budget_evals=args.budget,
                        eval_cost="cheap", noise="deterministic")
        ctl = Controller(lambda x: task._f(x), intake, make_policy(), seed=seed)
        res = ctl.run()
        out = ROOT / args.problem / "agent_runs" / label / f"seed_{seed}"
        ctl.save(out, {"task_name": task.name, "task_f_opt": task.f_opt})
        print(json.dumps({"problem": args.problem, "agent": label, "seed": seed,
                          "best_y": res["best_y"], "regret": res["best_y"] - task.f_opt,
                          "rounds": res["rounds"]}, default=str), flush=True)


if __name__ == "__main__":
    main()
