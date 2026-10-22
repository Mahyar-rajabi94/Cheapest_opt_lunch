"""Write the asset of the blind harness src/agent_harness/task_005.py:
src/agent_harness/assets/task_005.npz, OPVTask's forest and decoding as bare
arrays (no input names, no training info), so a sandbox needs neither
src/tasks/ nor this folder. Run from main/tasks/polymer with the project venv:

    ../../../.venv/bin/python export_blind_asset.py

Then checks that the harness returns exactly OPVTask's objective.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

MAIN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MAIN))
from src.tasks.opv import OPVTask  # noqa: E402

OUT = MAIN / "src" / "agent_harness" / "assets" / "task_005.npz"
N_CHECK = 20_000

task = OPVTask()
OUT.parent.mkdir(exist_ok=True)
np.savez_compressed(
    OUT,
    left=task._left, right=task._right, feature=task._feature.astype(np.int8),
    threshold=task._threshold, value=task._value, roots=task._roots,
    lower=task._lower, upper=task._upper,
    snap_dims=np.array(sorted(task._levels)),
    **{f"levels_{i}": levels for i, levels in task._levels.items()},
)
print(f"Wrote {OUT.relative_to(MAIN)} ({OUT.stat().st_size / 1e6:.2f} MB)")

# Check points: uniform, the box's corners and centre, snapped coordinates
# exactly halfway between two levels (ties), and continuous coordinates
# exactly on a split threshold.
rng = np.random.default_rng(0)
U = rng.random((N_CHECK, task.dim))
U = np.vstack([U, np.zeros(task.dim), np.ones(task.dim), np.full(task.dim, 0.5)])
for i, levels in task._levels.items():
    mids = ((levels[:-1] + levels[1:]) / 2 - task._lower[i]) / (task._upper[i] - task._lower[i])
    U[: len(mids), i] = mids
inner = task._left >= 0
for i in range(task.dim):
    if i not in task._levels:
        t = (task._threshold[inner & (task._feature == i)] - task._lower[i]) / (task._upper[i] - task._lower[i])
        rows = rng.choice(N_CHECK, 500, replace=False)
        U[rows, i] = np.clip(rng.choice(t, 500), 0.0, 1.0)

from src.agent_harness.task_005 import AgentTaskHarness  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    harness = AgentTaskHarness(budget=len(U), outdir=tmp)
    y_harness = np.array([harness.evaluate(u) for u in U])
y_task = np.array([task.evaluate(u).y for u in U])
n_diff = int(np.sum(y_harness != y_task))
print(f"Harness vs OPVTask on {len(U):,} points: {n_diff} differ")
if n_diff:
    raise SystemExit("The harness does not reproduce OPVTask.")
