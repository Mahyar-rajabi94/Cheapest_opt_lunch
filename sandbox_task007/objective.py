"""Objective definition for this optimization run.

This file is intentionally self-contained and does not expose the true
optimum anywhere. Do not inspect this file -- see Agent.md and this run's
prompt for the rules of this exercise.
"""

import ioh
import numpy as np

_problem = ioh.get_problem(7, instance=1, dimension=10, problem_class=ioh.ProblemClass.BBOB)

# Number of real-valued input dimensions.
DIM = _problem.meta_data.n_variables

# Per-coordinate bounds, length == DIM. Each entry is [lower, upper].
BOUNDS = np.column_stack([_problem.bounds.lb, _problem.bounds.ub]).tolist()

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
    return float(_problem(np.asarray(x, dtype=float)))
