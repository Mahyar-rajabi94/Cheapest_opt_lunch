"""Objective definition for this optimization run.

This file is intentionally self-contained and does not expose the true
optimum anywhere. Do not inspect this file -- see Agent.md and this run's
prompt for the rules of this exercise.

This task is a REAL engineering problem (not a synthetic test function):
laying out a robot cell (robot base pose, a CNC machine's position, and
an arm's inverse-kinematics branch choices) to minimize a real
mixer-loading cycle time, computed from a calibrated kinematic model with
collision checking. The last 6 of the 15 coordinates are IK branch flags
in [0, 1] (the model reads them as a boolean via a >= 0.5 threshold); the
rest are continuous geometric/kinematic parameters. The landscape has
real discontinuities where an IK branch switch occurs, and heavily
penalized regions where the layout is infeasible/colliding.
"""

import sys
from pathlib import Path

_MODEL_DIR = Path(
    "/Users/mahi/Desktop/Active_projects/Agentoptimizer_2/SDL_Real_Problem/03_robot_cell_layout"
)
sys.path.insert(0, str(_MODEL_DIR))
import model as _model  # noqa: E402

# Number of real-valued input dimensions (9 continuous + 6 IK-branch flags).
DIM = len(_model.NAMES)

# Per-coordinate bounds, length == DIM. Each entry is [lower, upper].
BOUNDS = _model.BOUNDS.tolist()

# "min" or "max".
DIRECTION = "min"

# Set True if calling fn(x) twice with the same x can return different
# values (observation noise). False means deterministic.
NOISY = False


def fn(x):
    """Evaluate the objective at one point.

    x : length-DIM sequence of floats.
    Returns a single float: the penalized cycle time in seconds (lower
    is better; large values indicate an infeasible/colliding layout).
    """
    return float(_model.objective(x))
