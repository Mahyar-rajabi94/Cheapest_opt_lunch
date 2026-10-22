"""Objective definition for this optimization run.

This file is intentionally self-contained and does not expose the true
optimum anywhere. Do not inspect this file -- see Agent.md and this run's
prompt for the rules of this exercise.

This task has inequality constraints. Rather than exposing them separately,
fn(x) returns a single penalized merit value: y + PENALTY_LAMBDA * total
constraint violation (violation = sum of positive constraint excesses; 0 if
feasible). Lower merit is better, exactly like every other task here, and
the minimum of the merit coincides with the true constrained optimum
(violation = 0 there). A merit value far outside the typical range you've
seen usually means a point was heavily infeasible, not that the objective
itself is actually that extreme.
"""

from src.tasks.coco_constrained import COCOConstrainedTask

_problem = COCOConstrainedTask(38, dim=10, instance=1)
PENALTY_LAMBDA = 1.0

# Number of real-valued input dimensions.
DIM = _problem.dim

# Per-coordinate bounds, length == DIM. Each entry is [lower, upper].
BOUNDS = _problem.bounds.tolist()

# "min" or "max".
DIRECTION = "min"

# Set True if calling fn(x) twice with the same x can return different
# values (observation noise). False means deterministic.
NOISY = False


def fn(x):
    """Evaluate the objective at one point.

    x : length-DIM sequence of floats.
    Returns a single float (penalized merit; see module docstring).
    """
    ev = _problem.evaluate(x)
    return float(ev.y + PENALTY_LAMBDA * ev.violation)
