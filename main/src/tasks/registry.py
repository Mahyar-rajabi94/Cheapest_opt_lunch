"""Task registry: every task key accepted by scripts/run_campaign.py.

Factories are lazy, so listing keys never constructs a backend. Add a task by
adding one entry here; nothing else needs to change.
"""

from __future__ import annotations

from src.tasks.bbob import BBOBTask
from src.tasks.cec2017 import CEC2017Task
from src.tasks.coco_constrained import COCOConstrainedTask
from src.tasks.coco_mixint import COCOMixintTask
from src.tasks.coco_noisy import COCONoisyTask
from src.tasks.nevergrad_noisy import NevergradNoisyTask
from src.tasks.opv import OPVTask
from src.tasks.pymoo_g import PymooGTask
from src.tasks.yahpo import YAHPOTask

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
    # f3: separable Rastrigin; rewards coordinate-wise search over rotation-invariant methods.
    # Prefix "sep_rastrigin" keeps its folder apart from f15's "rastrigin_d10_i1".
    "sep_rastrigin10": lambda: BBOBTask("sep_rastrigin", 3, dim=10),
    # f20: Schwefel x*sin(x); deceptive, best region near the box boundary.
    "schwefel10": lambda: BBOBTask("schwefel", 20, dim=10),
    # f21/f22: Gallagher 101 / 21 random Gaussian peaks; weak global structure.
    "gallagher101_10": lambda: BBOBTask("gallagher101", 21, dim=10),
    "gallagher21_10": lambda: BBOBTask("gallagher21", 22, dim=10),
    # CEC-2017 F6 (D04/D05): expanded Schaffer F6; multimodal, nonseparable, 30 and 50 dims.
    # Box [-5, 5], not CEC's [-100, 100], where no method beats random search.
    "schaffer30": lambda: CEC2017Task(6, dim=30, half_width=5.0),
    "schaffer50": lambda: CEC2017Task(6, dim=50, half_width=5.0),
    # CEC-2017 F5 (S04): shifted/rotated Rastrigin on a wide box [-100, 100].
    "cec_rastrigin30": lambda: CEC2017Task(5, dim=30),
    # CEC-2017 F8 (S05): Rastrigin with rounded steps; plateaus on top of multimodality.
    "noncont_rastrigin30": lambda: CEC2017Task(8, dim=30),
    # G04 (C04): 5 variables, quadratic objective, 6 nonlinear constraints, 2 active at the optimum.
    "pymoo_g04": lambda: PymooGTask("g4"),
    # G07 (C05): 10 variables, quadratic objective, 8 constraints, 6 active at the optimum.
    "pymoo_g07": lambda: PymooGTask("g7"),
    # bbob-constrained f20 (C01): rotated ellipsoid, 3 linear constraints, ill-conditioned.
    "constrained_ellipsoid10": lambda: COCOConstrainedTask(20, dim=10),
    # bbob-constrained f38 (C02): different powers, 3 linear constraints; flat near the optimum.
    "constrained_diffpowers10": lambda: COCOConstrainedTask(38, dim=10),
    # bbob-constrained f44 (C03): Rastrigin, 3 linear constraints; multimodal.
    "constrained_rastrigin10": lambda: COCOConstrainedTask(44, dim=10),
    # bbob-noisy f110: Rosenbrock, moderate Gaussian noise that vanishes at the optimum.
    "noisy_rosenbrock_gauss5": lambda: COCONoisyTask(110, dim=5),
    # bbob-noisy f111: Rosenbrock, severe uniform noise, mostly upward.
    "noisy_rosenbrock_unif5": lambda: COCONoisyTask(111, dim=5),
    # bbob-noisy f112: Rosenbrock, Cauchy noise: a constant offset plus rare heavy-tailed outliers.
    "noisy_rosenbrock_cauchy5": lambda: COCONoisyTask(112, dim=5),
    # nevergrad noisy Rosenbrock, 20 dims: curved valley under noise that grows with local slope.
    "ng_noisy_rosenbrock20": lambda: NevergradNoisyTask("rosenbrock", dim=20),
    # nevergrad noisy cigar, 20 dims: ill-conditioned (one easy, 19 steep directions) under noise.
    "ng_noisy_cigar20": lambda: NevergradNoisyTask("cigar", dim=20),
    # bbob-mixint f8 (V01): Rosenbrock, 8 integer + 2 continuous coordinates.
    "mixint_rosenbrock10": lambda: COCOMixintTask(8, dim=10),
    # bbob-mixint f15 (V02): rotated Rastrigin, 8 integer + 2 continuous coordinates.
    "mixint_rastrigin10": lambda: COCOMixintTask(15, dim=10),
    # bbob-mixint f17 (V03): Schaffer F7, 8 integer + 2 continuous coordinates; multimodal.
    "mixint_schaffer10": lambda: COCOMixintTask(17, dim=10),
    # YAHPO rbv2_svm (V04): SVM logloss on OpenML 40981; 6 dims, 2 categorical, 2 conditional on the kernel.
    "yahpo_svm": lambda: YAHPOTask("rbv2_svm"),
    # YAHPO rbv2_xgboost (V05): XGBoost logloss on OpenML 40981; 14 dims, 2 categorical, 8 conditional on the booster.
    "yahpo_xgboost": lambda: YAHPOTask("rbv2_xgboost"),
    # OPV device PCE from a random-forest emulator of 526 devices; 11 dims, 6 linear constraints, piecewise constant.
    "opv_pce": lambda: OPVTask(),
}
