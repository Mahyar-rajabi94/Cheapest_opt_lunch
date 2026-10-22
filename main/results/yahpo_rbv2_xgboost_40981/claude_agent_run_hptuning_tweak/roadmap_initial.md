# Roadmap — Initial Step

## Problem understood

- Input: real-valued vector `x` of dimension 14, each coordinate bounded in
  `[0, 1]` (confirmed via `harness.bounds` after constructing the harness —
  a clean, pre-normalized unit box; any categorical/conditional structure
  in the underlying problem has already been encoded into this continuous
  representation).
- Direction: minimize.
- Noise: deterministic (no observation noise), per the task statement.
- Objective: a sealed black-box surrogate of a real-world ML
  hyperparameter-tuning loss landscape. Its identity is not to be
  investigated; `objective.py` and other task source files are not to be
  opened. All decisions below come only from values returned by
  `harness.evaluate()`.
- Budget: 100 `evaluate()` calls total, hard-enforced by the harness.

## Assumptions

- A 14-dimensional, deterministic, expensive-feeling (budget ≈ 7x
  dimension) black box with no gradient access is the classic regime for
  sample-efficient Bayesian optimization rather than population-based
  metaheuristics (which typically want hundreds-to-thousands of
  evaluations per dimension to pay off) or deterministic local solvers
  (which risk wasting the whole budget in one basin of a landscape that,
  being an ML-tuning surrogate, is plausibly multimodal/non-convex).
- No prior structural knowledge (e.g. additive decomposition, known
  smoothness) is assumed beyond what the harness states.

## Method for this first step

**GP-BO** (`src/optimizers/gp_bo.py`, BoTorch `SingleTaskGP` +
analytic LogEI), used via its ask/tell interface. This wrapper's own
documented regime is "expensive, continuous problems, budgets of order 10x
dim, not 1000+ x dim" — a close match to this task (100 evals, dim 14,
~7x). Exact GP inference is practical at this scale and LogEI gives
principled explore/exploit balance one point at a time, which suits a
tight, deterministic budget better than a method that needs large
populations or many restarts to be effective.

This first step only covers GP-BO's **initial space-filling design**: a
scrambled Sobol' sequence, scaled to the unit box, generated internally by
the wrapper. Its size is an inherited default: `n_init = 2 * (dim + 1) =
30` for dim=14 — a standard rule of thumb for how many points are needed
before a GP fit is meaningful (the wrapper's own documented default; not
changed for this run). Seed: 0, for reproducibility, applied to both the
Sobol sequence and (later) the acquisition optimizer's random restarts.
No GP fitting or acquisition optimization happens yet during this step —
those are irrelevant until the design is complete, since `GPBO.ask(n=1)`
returns Sobol points until `len(y_) >= n_init`.

Points will be tagged `sobol_init` via `harness.set_active_optimizer`
(distinct from the `gp_bo` tag used once acquisition-driven points begin),
since this portion is mechanistically plain Sobol sampling, not yet
model-guided.

## Step length

**30 evaluations** (cumulative evaluations 1–30), i.e. exactly the
initial design. This is the natural unit for this step: it is self-
contained (a fixed-size design, not an arbitrarily chosen batch), and GP-
BO's acquisition machinery has nothing to act on until it completes. No
observations exist yet to justify a shorter or longer design.

Pause at **cumulative evaluation 30**.

## What I want to learn / what I'll inspect at the checkpoint

- The range and distribution of observed `y` across the design (spread,
  presence of outliers, whether most mass is flat vs. wide-ranging) —
  informs whether a single global GP is likely to fit reasonably or
  whether the landscape looks highly rugged/discontinuous.
- The current best `y` and where it sits relative to the rest of the
  sample.
- Rough signal of multimodality (clusters of good points vs. a single
  isolated good point) to sanity-check that GP-BO (as opposed to a more
  exploratory/global method) is still the right choice for continuing.
- Any sign the 30-point budget was poorly spent (e.g. evaluation errors,
  degenerate/constant outputs) that would call for a different approach
  before committing more budget to GP-driven acquisition.

No early-pause condition applies to this step: a fixed-size Sobol design
is generated as a whole and evaluating it is cheap relative to the total
budget, so there is nothing informative to re-decide mid-design.
