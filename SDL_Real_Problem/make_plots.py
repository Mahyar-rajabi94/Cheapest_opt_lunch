"""Team-format figures for each SDL problem: the agent against every baseline.

Uses main/scripts/make_comparison_plots.py unchanged; this script only
registers each problem's agent run in its AGENT_RUNS table at runtime and
turns the v2 agent's decisions.jsonl into the changelog.md that the
timeline's stars read ("## Evaluation N" per method change).

The plotted agent run is the median-final-value seed of agent_runs/v2_scripted.

Writes, in <problem>/results/<task>/:
    plot1_counterfactual_bar.png, plot2_regret_vs_eval.png, plot3_agent_method_timeline.png

Usage (from the repository root): python SDL_Real_Problem/make_plots.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROBLEMS = ["01_nonnewtonian_blend", "02_viscometer_calibration", "03_robot_cell_layout"]
AGENT = "v2_scripted"

spec = importlib.util.spec_from_file_location("mcp", ROOT.parent / "main" / "scripts" / "make_comparison_plots.py")
mcp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp)


def final_best(run_dir: Path) -> float:
    with open(run_dir / "trace.csv") as f:
        return float(list(csv.DictReader(f))[-1]["y_best_so_far"])


def write_changelog(run_dir: Path) -> None:
    """One '## Evaluation N' entry per decision that changes what the agent runs."""
    lines = ["# Changelog (generated from decisions.jsonl)", ""]
    with open(run_dir / "decisions.jsonl") as f:
        decisions = [json.loads(line) for line in f if line.strip()]
    for d in decisions:
        a = d["action"]
        if d["round"] == 0 or a["kind"] in ("continue", "stop"):
            continue
        lines += [f"## Evaluation {d['evals_used']}", f"{a['kind']} -> {a.get('method') or '-'}: {a.get('rationale', '')}", ""]
    (run_dir / "changelog.md").write_text("\n".join(lines))


def main():
    for problem in PROBLEMS:
        results = ROOT / problem / "results"
        task_dir = next(p for p in results.iterdir() if p.is_dir())
        runs = sorted((ROOT / problem / "agent_runs" / AGENT).glob("seed_*"), key=final_best)
        median_run = runs[len(runs) // 2]
        write_changelog(median_run)
        label = f"Agent (v2 scripted, {median_run.name})"
        mcp.AGENT_RUNS[task_dir.name] = {label: median_run.relative_to(mcp.SANDBOX_ROOT).as_posix()}
        mcp.make_plots(task_dir)


if __name__ == "__main__":
    main()
