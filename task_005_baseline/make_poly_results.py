"""Write this task's baseline and agent plots to poly_results/, named so the
task and the baseline/agent split are clear outside the repo. Uses the
plotting functions of main/scripts/make_comparison_plots.py on the results in
main/results_linux/opv_pce_d11/. Run from task_005_baseline with the project venv:

    ../.venv/bin/python make_poly_results.py

Writes poly_results/task_005_opv_pce_d11/:
    baselines/      the 9 fixed optimizers only
    agent_claude/   the Claude agent run (task_005_sandbox) against them
"""
import shutil
import sys
import tempfile
from pathlib import Path

from matplotlib.axes import Axes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main" / "scripts"))
import make_comparison_plots as plots  # noqa: E402

RESULTS = ROOT / "main" / "results_linux" / "opv_pce_d11"
NAME = "task_005_opv_pce_d11"  # plot-title prefix and file-name prefix
OUT = ROOT / "poly_results" / NAME

# A bar chart without agents is titled "(no agent run yet)" by the shared
# script; here the agent run exists and is plotted separately.
_set_title = Axes.set_title
Axes.set_title = lambda ax, label, *a, **k: _set_title(
    ax, label.replace("(no agent run yet)", "(baselines only)"), *a, **k)

baselines, agents = plots.load_task(RESULTS)
if list(agents) != ["Claude agent"]:
    raise SystemExit(f"Expected one Claude agent run for opv_pce_d11, found {list(agents)}.")

# The shared functions save fixed file names into a folder whose name they put
# in the titles, so render into a temporary folder named NAME, then rename.
renders = {
    "baselines": (
        {},
        {"plot1_counterfactual_bar.png": "baselines_final_regret.png",
         "plot2_regret_vs_eval.png": "baselines_regret_vs_eval.png"},
    ),
    "agent_claude": (
        agents,
        {"plot1_counterfactual_bar.png": "agent_claude_vs_methods_used_final_regret.png",
         "plot2_regret_vs_eval.png": "agent_claude_vs_baselines_regret_vs_eval.png",
         "plot3_agent_method_timeline.png": "agent_claude_method_timeline.png"},
    ),
}
for sub, (run_agents, names) in renders.items():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp) / NAME
        tmp_dir.mkdir()
        plots.plot_counterfactual_bar(tmp_dir, baselines, run_agents)
        plots.plot_regret_vs_eval(tmp_dir, baselines, run_agents)
        if run_agents:
            plots.plot_agent_timeline(tmp_dir, baselines, run_agents)
        (OUT / sub).mkdir(parents=True, exist_ok=True)
        for src, dst in names.items():
            shutil.copy(tmp_dir / src, OUT / sub / f"{NAME}_{dst}")
for path in sorted(OUT.rglob("*.png")):
    print(path.relative_to(ROOT))
