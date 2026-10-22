# Roadmap — sandbox_task028

## Problem summary
- Dimension: 14, all coordinates in [0, 1] (pre-normalized box; any
  categorical/conditional structure already encoded continuously).
- Direction: minimize.
- Noise: deterministic.
- Budget: 100 `evaluate()` calls total.
- Objective: sealed black box (described as a surrogate of an ML
  hyperparameter-tuning loss landscape, but treated as fully opaque — no
  attempt will be made to identify it).

## Method plan

**Primary method: TuRBO-1** (`src/optimizers/turbo.py`), run for the full
budget (100 evaluations), used as a single continuous campaign (no fixed
hand-off to a second method unless a deviation trigger below fires).

Why TuRBO, chosen from the problem statement alone (no data yet):

- Budget/dimension ratio is very tight: 100 evals for 14 dimensions is
  roughly 7x dim. Purely global methods that need many evaluations to
  converge (CMA-ES, standard DE) are a poor fit at this ratio — population
  sizes for CMA-ES/DE in 14D (typically 4+3*ln(14) ≈ 12 per generation)
  would only afford ~8 generations, not enough for self-adaptation to pay
  off.
- TuRBO's own initial design costs exactly `2*dim = 28` Sobol points,
  leaving `72` evaluations (18 batches of 4) for GP-surrogate-guided local
  trust-region search — a workable split between global coverage and
  exploitation at this budget.
- The function is deterministic, so a GP surrogate (no explicit noise
  model needed) and trust-region Thompson sampling should model local
  structure efficiently without wasting evals on replication.
- It is described as a surrogate of a real hyperparameter-loss landscape:
  such landscapes are typically continuous but can have ridges/local
  structure that benefit from a local trust region adapting its shape via
  GP lengthscales, rather than a fixed-shape global sampler (Sobol/random)
  or a derivative-free local solver with a single fixed starting point
  (BOBYQA/Nelder-Mead both start at the box midpoint by convention in this
  package, with no built-in global exploration first).
- Compared to full GP-BO (non-trust-region), TuRBO avoids the
  over-exploration/boundary-chasing failure mode that vanilla GP-EI can
  have in >10 dimensions, which is one of the documented motivations for
  TuRBO in the literature.

No fixed secondary phase is pre-committed; the deviation criteria below
define when and how the plan would change based on observed behavior.

## Explicit deviation criteria

1. **Stagnation during the evolve phase:** if the running best value has
   not improved for 30 consecutive evaluations after TuRBO's init phase
   (i.e. after eval 28) AND at least 20 evaluations of budget remain,
   stop driving TuRBO and spend the remaining budget on a local polish:
   evaluate small coordinate-wise perturbations / a manual
   Nelder-Mead-style refinement seeded at the current best point (not the
   box midpoint), to squeeze out local gains near the best point found.
2. **Repeated trust-region collapse:** if TuRBO's internal trust region
   restarts (shrinks below `length_min` and resets) more than twice
   before evaluation 80, treat this as evidence the landscape is more
   multimodal/rugged than a single local trust region can track well; in
   that case, switch the remaining budget to plain Sobol exploration
   (`SobolSearch`) to diversify coverage, then spend any leftover budget
   (if >=10 evals) refining near the best point found overall.
3. **Numerical/fitting failures:** if BoTorch's GP fit
   (`fit_gpytorch_mll`) raises or produces degenerate (NaN/inf)
   lengthscales repeatedly (>3 times), abandon TuRBO and fall back to
   `DifferentialEvolution` or `CMAES` for the remaining budget, since
   those don't depend on a GP fit succeeding.
4. Any such deviation will be logged in `changelog.md` with the
   evaluation count at which it happened and the evidence observed.

## Budget accounting
- TuRBO init: 28 evals (Sobol, 2*dim).
- TuRBO evolve: up to 72 evals (18 batches of 4), unless a deviation
  trigger above redirects part of this budget.
- Every evaluation, regardless of method, is tagged via
  `harness.set_active_optimizer(...)`.
