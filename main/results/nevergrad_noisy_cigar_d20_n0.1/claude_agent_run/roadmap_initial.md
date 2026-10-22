# Roadmap — initial plan (pre-registered, written before any evaluate() call)

## Problem as given
- Dimension: 20, bounds [-5, 5] per coordinate, minimize.
- Noise: present but mild (small, roughly multiplicative-scale; not heavy-tailed).
- Budget: 100 evaluate() calls total — a very small budget relative to the
  dimension (5x dim, well below the ~10x-dim rule of thumb this codebase's
  own GP-BO docstring cites as its practical regime, and far below what
  population-based methods like CMA-ES or TuRBO need to adapt in 20-D).
- Sealed black box: no attempt will be made to identify or exploit the
  function's identity; the plan below is derived only from dimension,
  bounds, noise characteristics, direction, and budget.

## Method sequence and budget split

**Phase 1 — Sobol initial design (evals 1–42).**
Use the project's `GPBO` optimizer, whose built-in initial design is
`2*(dim+1) = 42` scrambled Sobol points. This buys broad, low-discrepancy
coverage of the 20-D box before committing to any model-driven search,
which matters at this budget because a 20-D space is sparse and a GP (or
any model) needs a reasonably spread-out design before its fit is
meaningful.

**Phase 2 — Sequential GP-BO / LogEI (evals 43–~95).**
Continue driving the same `GPBO` instance one point at a time
(`ask(n=1)` → `evaluate` → `tell`). This is the single method in this
toolbox explicitly designed for very small, expensive-feeling budgets
(its own docs cite ~10x-dim as the intended regime; ~5x-dim here is below
that but still far better suited than a population method that needs many
generations to adapt a 20-D covariance/trust region). `SingleTaskGP`
learns an observation-noise term from the data, which gives it some
inherent robustness to the stated mild noise without any extra plumbing.

**Phase 3 — Local exploitation (evals ~96–100, last ~5 calls).**
Spend the final handful of evaluations resampling small perturbations
(Gaussian, sigma ≈ 2% of box width per coordinate, clipped to bounds)
around whatever point is currently best. Rationale: this implementation's
LogEI uses the raw observed minimum (`y_.min()`) as `best_f`, not a
denoised/model-based estimate — under noise that raw minimum can itself be
a lucky low draw rather than a truly good point. A short local-exploitation
burst spends the last few calls checking/refining around the incumbent
rather than trusting a single noisy observation as "the" answer.

## Explicit deviation criteria (quantitative, decided now)

1. **GP/acquisition failure fallback.** If any `GPBO.ask()` call in Phase 2
   raises an exception (numerical failure in GP fitting or acquisition
   optimization), stop using GP-BO immediately and fill all remaining
   budget with `SobolSearch` points instead, to avoid wasting evaluations
   on a broken model.
2. **Stagnation → early exit to Phase 3.** If `y_best_so_far` has not
   improved for 20 consecutive evaluations during Phase 2, end Phase 2
   early and move directly into Phase 3 (local exploitation around the
   incumbent best), rather than continuing undirected global BO that has
   stopped paying off.
3. **Budget accounting.** If Phase 1's Sobol design would overshoot the
   100-call budget (it won't: 42 < 100), truncate it and skip straight to
   whatever budget remains for Phase 2/3.

Any time one of these triggers fires, or any other unplanned change is
made, an entry will be appended to `changelog.md` (append-only, never
edited) citing the evaluation index and the observed evidence.
