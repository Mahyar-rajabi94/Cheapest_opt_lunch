"""Objective definition for this optimization run.

This file is intentionally self-contained and does not expose the true
optimum anywhere. Do not inspect this file -- see Agent.md and this run's
prompt for the rules of this exercise.
"""

from src.tasks.yahpo import YAHPOTask

_problem = YAHPOTask("rbv2_xgboost", instance="40981")

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
    Returns a single float.
    """
    return float(_problem.evaluate(x).y)
