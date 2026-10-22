"""Robot cell layout task: place the UR3e and the CNC so the mixer-loading cycle
is fast and collision-free (SDL5 formulation cell; see PROBLEM.md).

The objective is model.objective (pure numpy, a few ms per call). `f_opt` is
the best-known value from build_data.py's multi-start search
(data/reference_optimum.json), not a proven optimum.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "main"))
sys.path.insert(0, str(HERE))

import model  # noqa: E402
from src.tasks.base import Task  # noqa: E402


class RobotCellLayoutTask(Task):
    """15-D mixed-integer layout of the SDL5 formulation cell.

    x = [base_x, base_y, base_yaw, cnc_x, cnc_y, lid_open, tilt_x, tilt_y,
         pick_azimuth, shoulder_A, elbow_A, wrist_A, shoulder_B, elbow_B, wrist_B];
    the last six are IK branch flags in [0, 1], read as >= 0.5. Minimize the
    penalized cycle time in seconds (model.evaluate gives every term).
    """

    def __init__(self):
        self.name = "robot_cell_layout_d15"
        self.dim = len(model.NAMES)
        self.bounds = model.BOUNDS.copy()
        self.f_opt = float(json.loads((HERE / "data" / "reference_optimum.json").read_text())["f_opt"])

    def _f(self, x: np.ndarray) -> float:
        return model.objective(x)


def make_task() -> RobotCellLayoutTask:
    return RobotCellLayoutTask()
