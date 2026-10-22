"""Objective definition for this optimization run.

This file is intentionally self-contained and does not expose the true
optimum anywhere. Do not inspect this file -- see Agent.md and this run's
prompt for the rules of this exercise.

This task is MIXED-INTEGER: some coordinates must only ever take integer
values. See BOUNDS and INTEGER_COORDS below -- any x you propose should
already respect this, but the evaluator also rounds integer coordinates
internally, so the objective is piecewise constant along those coordinates.
"""

from src.tasks.coco_mixint import COCOMixintTask

_problem = COCOMixintTask(17, dim=10, instance=1)

# Number of real-valued input dimensions.
DIM = _problem.dim

# Per-coordinate bounds, length == DIM. Each entry is [lower, upper].
BOUNDS = _problem.bounds.tolist()

# 0-indexed coordinates that must take integer values only.
INTEGER_COORDS = list(range(int(_problem._problem.number_of_integer_variables)))

# "min" or "max".
DIRECTION = "min"

# Set True if calling fn(x) twice with the same x can return different
# values (observation noise). False means deterministic.
NOISY = False


def fn(x):
    """Evaluate the objective at one point.

    x : length-DIM sequence of floats. Coordinates listed in
        INTEGER_COORDS are rounded internally before evaluation.
    Returns a single float.
    """
    return float(_problem.evaluate(x).y)
