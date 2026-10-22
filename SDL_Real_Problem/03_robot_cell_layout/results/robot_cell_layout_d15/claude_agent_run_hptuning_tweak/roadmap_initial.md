# Roadmap — Initial Step Only

## Understood problem

- Real engineering objective (not synthetic): layout/kinematics of a robot
  cell (UR3e base pose + CNC position + IK branch choices) to minimize a
  mixer-loading cycle time (seconds), from a calibrated kinematic model with
  collision checking.
- Dimension: 15. Coordinates 0-8 are continuous geometric/kinematic
  parameters (base position/yaw, CNC position, lid-open angle, tilt angles,
  pick azimuth). Coordinates 9-14 are IK branch flags in [0,1], read as
  booleans via a >=0.5 threshold inside the objective (selecting between two
  IK solution branches at two points in the cycle).
- Bounds (from `harness.bounds`, 15 pairs):
  ```
  [-0.316, 0.3566], [-0.7, 0.2651], [-3.1416, 3.1416], [-0.0801, 0.1207],
  [-0.9599, 0.0180], [0.3491, 1.2217], [-0.5236, 0.5236], [-0.5236, 0.5236],
  [0.0, 6.2832], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1]
  ```
- Direction: minimize.
- Noise: deterministic (no observation noise) — so repeated evaluation of
  the same point is never useful, and any apparent "noise" between nearby
  points reflects a real discontinuity (IK branch switch, infeasible/
  collision penalty), not measurement error.
- Budget: 100 `evaluate()` calls total, strictly enforced by the harness.
- The objective is treated as a sealed interface beyond what is stated above:
  I will not inspect `objective.py` or other harness source, and will reason
  only from `(x, y)` pairs I collect myself.

## Key assumption driving the first step

With a small budget (100) relative to the dimension (15, with 6 of those
being branch-like flags whose effect is a step function rather than a
smooth gradient) and known heavily-penalized infeasible/collision regions,
the single biggest early risk is wasting evaluations entirely inside an
infeasible region, or inside one bad combination of IK branch flags, before
any sense of the feasible landscape exists. A pure gradient-style or
single-basin local method (e.g. CMA-ES from the box midpoint, L-BFGS-B)
would commit early to one region/branch-flag setting without evidence that
it's a good one. A space-filling design is the appropriate way to get a
first, unbiased read of where feasible/low-cost regions and favorable
branch-flag settings lie, given no prior information about this specific
calibrated model.

## Step 1 plan

**Method:** Sobol' quasi-random sequence (`src/optimizers/sobol_search.py`,
wraps `scipy.stats.qmc.Sobol`, scrambled). Standard library defaults are used
for the sampler itself (scramble=True, no custom tuning) since Sobol' has no
problem-specific hyperparameters worth adjusting here.

**Domain:** the full 15-dimensional box exactly as given by `harness.bounds`,
including coordinates 9-14 (the IK-branch flags) sampled continuously over
[0,1] along with the rest — this still produces a good mix of values above
and below the 0.5 threshold across the batch without needing a separate
discrete-enumeration mechanism yet. No snapping/rounding is applied before
evaluation; the objective itself thresholds at >=0.5 internally.

**Initialization:** fresh `SobolSearch(bounds=harness.bounds, seed=0)`
instance; `seed=0` only for reproducibility of this run, not a tuned choice.

**Step length:** 16 evaluations (a power of 2, which is when Sobol'
sequences have their best balance/discrepancy properties per the wrapper's
own docstring). This uses 16 of the 100-call budget, leaving 84. 16 is small
enough to preserve the large majority of the budget for whatever
follow-up method the evidence suggests (local refinement around good
points, targeted branch-flag search, or a different global method), while
being large enough (~1 point per dimension) to get a first real read of
feasibility rate and value spread across the box.

**Cumulative count at which I will pause:** 16 evaluations.

## What I want to learn / evidence to inspect at the checkpoint

- The distribution of observed y values: best, worst, median — to gauge how
  much of the box is feasible vs. heavily penalized, and the scale of the
  penalty vs. genuine cycle-time values.
- Whether any points landed in a clearly non-penalized (plausible cycle-time)
  regime, and if so what their coordinates look like (especially the 6
  branch-flag coordinates — do good points share a flag pattern?).
- Whether there's enough signal to justify a local method (e.g. CMA-ES or a
  bounded local optimizer) seeded near the best point(s), versus needing
  more global/space-filling exploration first because feasible region looks
  small or fragmented.

**Conditions for pausing sooner:** none — 16 fast, deterministic evaluations
is already a small, bounded block; there is no reason to stop earlier than
planned within this step.
