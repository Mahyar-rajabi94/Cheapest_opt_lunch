# Roadmap — Initial Step

## Problem understanding
- Dimension: 30. Bounds: [-5, 5] per coordinate (box-constrained).
- Direction: minimize a sealed black-box objective (`objective.py`,
  intentionally not inspected beyond its documented `DIM`/`BOUNDS`/
  `DIRECTION`/`fn` interface, accessed only through
  `AgentTaskHarness`).
- Noise: deterministic (no observation noise) — repeated evaluation of the
  same point would return the same value, so there is no need for
  replication or a noise-robust acquisition/selection rule.
- Total evaluation budget: 100 `harness.evaluate()` calls, hard-enforced by
  the harness (`BudgetExhausted` once used up).
- Key framing fact: 100 evaluations for a 30-dimensional problem is a very
  small budget relative to dimension (~3.3x dim), far below the ~2000-eval
  convention used elsewhere in this project. Methods that need a large
  fraction of the budget just to build an initial model (e.g. classic
  Differential Evolution with pop_size = 10*dim = 300, or a full
  quasi-Newton/BOBYQA-style quadratic interpolation model needing on the
  order of 2*dim+1 ≈ 61 points before any adaptation starts) are a poor
  match here, since they would leave little or no budget for genuine
  refinement. No assumption is made about the function's identity,
  modality or smoothness; the plan is to let early evidence (progress
  across generations) inform whether to continue, adjust, or switch.

## Method chosen for this first step: CMA-ES (plain, no restarts)
- Implementation: `src/optimizers/cma_es.py`, wrapping `cma.CMAEvolutionStrategy`
  from `pycma` version 4.5.0 (checked in the run environment).
- Rationale: CMA-ES is a population-based, derivative-free method whose
  default population size scales sensibly with dimension
  (pycma's own default: popsize = 4 + floor(3 * ln(dim))), giving a
  reasonable number of generations within a 100-eval budget at dim=30,
  while still adapting a full covariance structure (not just a fixed-shape
  local step) from the very first generations. Unlike a single-point local
  solver (BOBYQA, Nelder-Mead) started once from the box center, CMA-ES
  retains some stochastic/global character every generation, which matters
  because nothing is assumed yet about whether this black box is unimodal.
  Unlike classic DE (pop_size = 10*dim = 300 here) it does not require more
  evaluations than the entire budget just to initialize.
- Hyperparameters for this step (all inherited library/wrapper defaults —
  no tuning applied yet, since there are no observations to tune against):
  - `bounds`: the task's box, [-5, 5]^30, passed straight to pycma's own
    `bounds` option (enforced internally throughout the run).
  - `x0` (initial mean): the midpoint of the box, i.e. the zero vector —
    the wrapper's standard convention (`(lower+upper)/2`).
  - `sigma0` (initial step size): one quarter of the mean box width =
    10/4 = 2.5 — the wrapper's standard convention, not hand-picked for
    this task.
  - `popsize` (lambda): pycma's own internal default for dim=30, which
    pycma computes as `4 + floor(3 * ln(30))` = 4 + 10 = 14. Not
    overridden.
  - All other pycma internal hyperparameters (learning rates for the
    covariance/step-size adaptation, recombination weights, etc.): pycma
    4.5.0's own defaults, left untouched, per the wrapper's documented
    "no tuning" convention.
  - `seed`: 0, for reproducibility.

## Step length for this first step
- Plan: run 2 full generations of CMA-ES, i.e. 2 * popsize = 28
  `evaluate()` calls. Pause at cumulative evaluation 28 (28 used, 72
  remaining).
- Why this length: a single generation gives only one noisy snapshot of
  the local neighborhood around the box center and is not enough to see
  whether CMA-ES's step-size/covariance adaptation is actually improving
  things generation-over-generation. Two full generations is the smallest
  block that shows a trend (best-so-far and per-generation mean fitness
  across at least one adaptation update) while still committing well under
  a third of the total budget, leaving ample budget to react to what is
  observed (continue for more generations, shift to a different method, or
  hybridize with a local refiner once a promising basin is found).

## What I want to learn / review criteria at the checkpoint
- Whether the best-observed value is improving from generation 1 to
  generation 2 (evidence that CMA-ES is making progress, not just sampling
  noise around the initial point).
- The spread of objective values within and across generations, as a rough
  signal of how rugged/noisy-looking the landscape is at this scale (this
  is a deterministic function, so spread reflects landscape structure, not
  measurement noise).
- Whether 100 - 28 = 72 remaining evaluations look sufficient to let
  CMA-ES converge further, or whether the step-size/population scale
  suggests switching to a local refiner (e.g. BOBYQA or Nelder-Mead from
  the current best point) would use the remaining budget more efficiently.
- No early-pause condition is set for this very first step: with only two
  generations planned (28 evaluations) and zero evaluations spent so far,
  there is no prior evidence yet that would justify stopping before this
  small block completes.
