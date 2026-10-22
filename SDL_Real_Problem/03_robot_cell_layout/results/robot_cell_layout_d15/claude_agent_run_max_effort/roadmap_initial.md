# Roadmap — Robot Cell Layout Optimization (15D, budget = 100)

## Problem summary

- 15-dimensional real engineering objective: lay out a robot cell (UR3e arm
  base pose, CNC machine position, IK branch choices) to minimize mixer-
  loading cycle time (seconds), from a calibrated kinematic model with
  collision checking. Direction: **minimize**.
- Coordinates 0-8: continuous geometric/kinematic parameters (base
  position/yaw, CNC position, lid-open angle, tilt angles, pick azimuth).
  Bounds read from `harness.bounds` after construction (zero evaluations
  spent): e.g. `[-0.316, 0.357]`, `[-0.7, 0.265]`, `[-pi, pi]`, `[-0.08,
  0.121]`, `[-0.96, 0.018]`, `[0.349, 1.222]`, `[-0.524, 0.524]` (x2),
  `[0, 2*pi]`.
- Coordinates 9-14: 6 IK-branch flags, bounds confirmed exactly `[0, 1]`
  each. The model reads each via a `>= 0.5` threshold — a pure boolean
  switch with **no gradient/variation within either side** of that
  threshold.
- Deterministic (`harness.noisy == False`), no observation noise.
- Budget: 100 `evaluate()` calls total, hard limit enforced by the harness.
- Known landscape hazards (stated in the task): real discontinuities at
  every IK-branch threshold crossing, plus heavily-penalized dead zones
  wherever the layout is geometrically infeasible or the arm/CNC collide.

## Key structural observation driving the plan

Because coordinates 9-14 only ever affect the objective through a `>=0.5`
threshold, the 6 of them jointly define a combinatorial space of exactly
`2^6 = 64` discrete "IK-branch combinations" — the *exact* value within a
side of 0.5 carries no information. Treating all 15 coordinates as jointly
continuous (feeding them straight into CMA-ES or a GP) would waste budget
letting a continuous method "discover" hard step functions it is not built
to model well, and 100 evaluations is already far too few to afford that.
So the plan explicitly splits the problem: (a) a combinatorial search over
the 64 flag-combinations, crossed with (b) continuous optimization over the
9 truly continuous coordinates, conditional on a chosen combination.

100 evaluations for a 15-dim space with real collision/feasibility cliffs
is a very tight budget: the standard GP-BO rule of thumb wants roughly
10x dim = 150; classic DE's own standard population size alone is
10 x dim = 150 (already over budget, even before any generations run);
TuRBO's initial design alone is 2 x dim = 30. The plan below is
deliberately front-loaded toward a cheap global scan and a sample-efficient
population method operating on a *reduced* 9-dim subspace, rather than
relying on any single method to explore all 15 dims from a cold start.

## Planned methods and budget split (100 total)

**Phase 1 — `combo_scan_sobol` (32 evaluations).**
Stratified global scan: draw 32 distinct IK-branch combinations uniformly
at random without replacement out of the 64 possible (fixed seed), paired
one-to-one with 32 quasi-random Sobol' points (via this project's
`SobolSearch`) over the 9 continuous coordinates. Evaluate all 32 pairs.
32 is used (a power of two) for Sobol' balance, and covers half of the
64-combination space. Purpose: a first, cheap read on (i) which
flag-combinations tend to be more/less favorable, and (ii) roughly where
the feasible/good continuous region sits — before committing the bulk of
the budget to any one combination.

**Phase 2 — `cma_es_comboA` / `cma_es_comboB` (48 evaluations).**
Take the best-performing combination from Phase 1 ("combo A") and the best
*distinct* combination among the rest ("combo B"). Fix the 6 flag
coordinates at combo A and run CMA-ES (`src/optimizers/cma_es.py`, library
defaults throughout — only `bounds` is supplied) on the remaining 9
continuous coordinates, warm-started by centering CMA-ES's own standard
box-midpoint/quarter-width convention on a *shrunk* box: half-width = 0.25
x (each coordinate's full range), clipped to the original bounds, centered
on combo A's best Phase-1 point. This keeps CMA-ES's "no hand-tuned
parameters" convention intact (it still only ever uses box-midpoint and
box-width/4) while placing that box where Phase-1 evidence says is
promising, rather than the arbitrary global box-midpoint (which could
easily sit in a collision/infeasible region). Run an initial chunk of 3
generations (30 evaluations — CMA-ES's own default population size for
9D), then check the deviation criteria below before committing the
remaining ~18 evaluations of this phase.

**Phase 3 — local polish (remaining ~20 evaluations).**
Take the single best (x, y, combo) found anywhere so far across every
phase. Fix the flags at that combo and run a local refiner on the 9
continuous coordinates, box again shrunk (half-width = 0.12 x full range,
clipped) and centered on that best point. Use Nelder-Mead
(`src/optimizers/nelder_mead.py`) if the evaluations remaining at that
point are < 40, otherwise BOBYQA (`src/optimizers/bobyqa.py`) — see
criterion D4 below for why. This squeezes a final local improvement out of
whichever region Phase 2 converged to.

All three phases funnel through one running "global best" tracker (updated
after every single real evaluation, regardless of phase), so Phase 3
always polishes the true best point seen anywhere in the campaign, not
just Phase 2's own local notion of best.

## Why these methods (reasoning from problem properties alone, no data yet)

- **Sobol' stratified scan first**: the cheapest way to get unbiased,
  space-filling coverage of both the discrete flag space and the
  continuous space before committing budget; a low-discrepancy sequence is
  preferred over plain random for better coverage per sample, matching
  this project's own `SobolSearch`.
- **CMA-ES for the main continuous refinement**: robust to non-convex,
  non-smooth, multi-modal landscapes (expected here, given the stated
  collision/infeasibility cliffs) without needing gradients; population-
  based so it can escape small local penalty basins; well suited to a
  ~10-50 evaluation regime once reduced to 9 dimensions. GP-BO and TuRBO
  were considered but rejected for this phase: both need a substantial
  Sobol' initial design *inside whatever region/combo is fixed* (
  2*(9+1)=20 for GP-BO, 2*9=18 for TuRBO) before their models are
  meaningful at all, which would consume most of Phase 2's budget on
  initialization alone, leaving little for actual acquisition-driven
  refinement. CMA-ES has no such up-front tax (its population does double
  duty as exploration and optimization from generation 1). Classic DE is
  ruled out entirely: its standard population size (10 x dim = 90, even
  for the reduced 9D subspace) alone exceeds this phase's whole budget.
- **A late local-refiner polish pass (Nelder-Mead / BOBYQA)**: both are
  classical, sample-efficient local-convergence tools for a smooth-ish
  basin once a good region has already been found — exactly the "exploit
  the last mile" step a global/population method like CMA-ES is not
  designed to finish cheaply.
- **Fixing flags combinatorially rather than optimizing them as continuous
  inputs**: justified above (zero within-side gradient); this also means
  every evaluation's information content toward the continuous optimizers
  is not contaminated by irrelevant jitter in 6 of the 15 coordinates.

## Explicit, quantitative deviation criteria

- **D1 (combo switch):** After Phase 2's first 30-evaluation chunk on
  combo A, if the best y found under combo A has improved on combo A's
  Phase-1 incumbent value by less than 1% (relative), spend the remainder
  of Phase 2's budget on combo B instead (fresh CMA-ES run, same
  warm-start convention) rather than continuing combo A.
- **D2 (sigma collapse):** If CMA-ES's step-size sigma falls below 1% of
  its own initial sigma0 before a Phase-2 chunk's allotted evaluations are
  used up (i.e. it has effectively converged), stop that CMA-ES run
  immediately and carry the unused evaluations forward into Phase 3 early.
- **D3 (saturated/infeasible combo):** If 3 consecutive evaluations under
  the currently-fixed combo all land within `max(1e-6, 1e-3 * |worst_y|)`
  of the single worst y observed anywhere so far in the campaign (strong
  evidence that this combo/region is a saturated infeasibility/collision
  penalty), abandon it immediately: fall back to the best not-yet-tried
  Phase-1 combo for the rest of that phase's evaluations.
- **D4 (polish method choice):** Phase 3 uses Nelder-Mead if the
  evaluations remaining when it starts are < 40 (BOBYQA's own
  interpolation-model build cost of `2*9+1=19` points would otherwise
  consume the bulk of a small remaining budget before taking a single real
  trust-region step); otherwise BOBYQA.
- **D5 (local solver exhausted early):** If Phase 3's local solver raises
  `OptimizerExhausted` (converged) while evaluations still remain, restart
  it once more from the current global-best point jittered by independent
  uniform noise of +/-5% of each coordinate's range, consuming the rest of
  the budget there.

Any time one of these actually fires, it will be logged as a
`changelog.md` entry citing the real observed numbers, per Agent.md
section (c).
