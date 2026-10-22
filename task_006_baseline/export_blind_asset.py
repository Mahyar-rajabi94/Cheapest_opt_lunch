"""Write the asset of the blind harness main/src/agent_harness/task_006.py:
main/src/agent_harness/assets/task_006.npz, OPVPoolTask's candidate points
and objective values as bare arrays (no feature names, no metadata), so a
sandbox needs neither src/tasks/ nor this folder. Run from task_006_baseline
with the project venv:

    ../.venv/bin/python export_blind_asset.py

Then checks that the harness reproduces OPVPoolTask: replaying every
baseline trace in main/results_linux/opv_pool_pce_d11/ through a fresh
harness must return the trace's values exactly.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

MAIN = Path(__file__).resolve().parents[1] / "main"
sys.path.insert(0, str(MAIN))
from src.tasks.opv_pool import OPVPoolTask  # noqa: E402

OUT = MAIN / "src" / "agent_harness" / "assets" / "task_006.npz"
RESULTS = MAIN / "results_linux" / "opv_pool_pce_d11"

task = OPVPoolTask()
OUT.parent.mkdir(exist_ok=True)
np.savez_compressed(OUT, points=task.candidates, values=-task._pce)
print(f"Wrote {OUT.relative_to(MAIN)} ({OUT.stat().st_size / 1e3:.0f} kB)")

from src.agent_harness.task_006 import AgentTaskHarness  # noqa: E402

traces = sorted(RESULTS.glob("*/seed_*/trace.csv"))
if not traces:
    raise SystemExit(f"No baseline traces in {RESULTS}; run the sweep first.")
n_diff = 0
for path in traces:
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    X = np.column_stack([rows[f"x{d}"] for d in range(task.dim)])
    with tempfile.TemporaryDirectory() as tmp:
        harness = AgentTaskHarness(budget=len(X), outdir=tmp)
        y = np.array([harness.evaluate(x) for x in X])
    n_diff += int(np.sum(y != rows["y"]))
print(f"Harness vs OPVPoolTask on {len(traces)} baseline traces: {n_diff} values differ")
if n_diff:
    raise SystemExit("The harness does not reproduce OPVPoolTask.")
