"""Task registry: every task key accepted by scripts/run_campaign.py.

Factories are lazy, so listing keys never constructs a backend. Add a task by
adding one entry here; nothing else needs to change.
"""

from __future__ import annotations

from src.tasks.bbob import BBOBTask
from src.tasks.cec2017 import CEC2017Task

TASKS = {
    # f8: ill-conditioned, axis-aligned curved valley; unimodal.
    "rosenbrock10": lambda: BBOBTask("rosenbrock", 8, dim=10),
    "rosenbrock20": lambda: BBOBTask("rosenbrock", 8, dim=20),
    "rosenbrock40": lambda: BBOBTask("rosenbrock", 8, dim=40),
    # f15: rotated, highly multimodal with a regular global structure.
    "rastrigin10": lambda: BBOBTask("rastrigin", 15, dim=10),
    # f12: one very steep direction (condition 1e6); unimodal.
    "bent_cigar10": lambda: BBOBTask("bent_cigar", 12, dim=10),
    # f1: separable, convex, well-conditioned; a sanity check.
    "sphere10": lambda: BBOBTask("sphere", 1, dim=10),
    # f7: plateaus and discontinuous steps; zero gradient almost everywhere.
    "step_ellipsoid10": lambda: BBOBTask("step_ellipsoid", 7, dim=10),
    # f13: nondifferentiable ridge.
    "sharp_ridge10": lambda: BBOBTask("sharp_ridge", 13, dim=10),
    # f16: rugged, highly multimodal, repetitive.
    "weierstrass10": lambda: BBOBTask("weierstrass", 16, dim=10),
    # CEC-2017 F6 (D04/D05): expanded Schaffer F6; multimodal, nonseparable, 30 and 50 dims.
    "schaffer30": lambda: CEC2017Task(6, dim=30),
    "schaffer50": lambda: CEC2017Task(6, dim=50),
    # CEC-2017 F5 (S04): shifted/rotated Rastrigin on a wide box [-100, 100].
    "cec_rastrigin30": lambda: CEC2017Task(5, dim=30),
    # CEC-2017 F8 (S05): Rastrigin with rounded steps; plateaus on top of multimodality.
    "noncont_rastrigin30": lambda: CEC2017Task(8, dim=30),
}
