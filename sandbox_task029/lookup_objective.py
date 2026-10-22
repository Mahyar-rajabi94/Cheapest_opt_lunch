"""Objective definition for this optimization run.

This file is intentionally self-contained and does not expose the true
optimum anywhere. Do not inspect this file -- see Agent.md and this run's
prompt for the rules of this exercise.

This is a LOOKUP-TABLE task: a fixed, finite pool of real molecules (given
as SMILES strings). There is no continuous vector to propose -- you choose
one candidate index at a time from the pool, and each candidate may only be
evaluated once (no re-measuring the same molecule).
"""

import csv
from pathlib import Path

_DATA_PATH = Path(
    "/Users/mahi/Desktop/Active_projects/Agentoptimizer_2/main/data/dionysus/freesolv/candidates.csv"
)

with open(_DATA_PATH) as _f:
    _rows = list(csv.DictReader(_f))

# Public: the pool of candidates, as SMILES strings. CANDIDATES[i] is the
# molecule you are choosing when you call fn(i).
CANDIDATES = [r["smiles"] for r in _rows]

# "min" or "max".
DIRECTION = "min"

# Set True if calling fn(i) twice with the same i can return different
# values (observation noise). False means deterministic.
NOISY = False

_Y = [float(r["y"]) for r in _rows]


def fn(idx):
    """Evaluate candidate `idx` (0 <= idx < len(CANDIDATES)).

    Returns a single float: the measured property for that molecule.
    """
    return _Y[int(idx)]
