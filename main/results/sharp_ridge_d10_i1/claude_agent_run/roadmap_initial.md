# Roadmap — sandbox_task008

## Problem as given
- Dimension: 10, bounds [-5, 5]^10, minimize, deterministic (no noise).
- Budget: 100 `evaluate()` calls total — only 10x dimension, a small budget
  (explicitly not the 2000-budget convention used elsewhere in this
  project).

## Planned method(s) and budget split

**Primary method: GP-BO (`src/optimizers/gp_bo.py`), for the full 100-eval
budget, run as a single continuous campaign (label: `gp_bo`).**

- GP-BO's own initial design is `2 * (dim + 1) = 22` Sobol points, drawn
  internally by its wrapped `SobolSearch`, before the first
  acquisition-driven (LogEI) proposal. So within the single `gp_bo` label,
  evaluations 1–22 are effectively a Sobol space-filling phase and
  23–100 (78 evals) are sequential GP + analytic LogEI acquisition,
  one point at a time, fit on all data told so far.
- No hyperparameter tuning: using the library's own stated defaults
  (SingleTaskGP default kernel/priors, Normalize/Standardize transforms,
  num_restarts=10, raw_samples=512 for acquisition optimization).

**Reasoning:**
- At `budget = 10 * dim`, this is precisely the regime GP-BO's own
  module docstring calls out as its intended use case ("expensive,
  continuous problems, budgets of order 10 x dim, not 1000+ x dim"),
  while explicitly noting GP fitting cost grows roughly cubically with
  data, making GP-BO impractical at larger budgets. Every other
  population-based or local method in this package (CMA-ES, DE, TuRBO,
  BOBYQA, Nelder-Mead, L-BFGS-B) is designed around budgets that are
  multiples of hundreds-to-thousands of evaluations to make real
  progress (e.g. BOBYQA's own docstring reports needing ~1187 evals on a
  comparable project task; CMA-ES/DE need enough generations at
  population size ~10-20 to adapt; TuRBO's trust-region adaptation and
  restart logic assumes room for many batches). With only 100 evals
  total, none of those have enough evaluations to meaningfully outperform
  a method that builds a global surrogate from very few points and
  chooses each subsequent point by expected improvement — which is
  exactly what GP-BO is designed to do with a few dozen points in 10
  dimensions.
- Deterministic objective means no replication or noise-robust
  acquisition is needed; plain analytic LogEI (noiseless) is the
  appropriate acquisition choice, which is what this wrapper already
  implements.
- A single continuous `gp_bo` campaign (rather than, say, Sobol as a
  separately-labeled explicit phase before switching) is chosen because
  the built-in init/acquisition split is already a principled use of the
  budget, and running one coherent sequential BO loop lets every
  acquisition step condition on literally all data collected so far,
  which is the main advantage of BO at this scale.

## Explicit, quantitative deviation criteria

I will deviate from the single-method GP-BO plan, and switch the
*remaining* budget to **CMA-ES** (label: `cmaes_fallback`), if **either**:

1. **Technical failure:** `ask()`/`tell()` on the GP-BO optimizer raises
   an exception (e.g. GP fit failure, acquisition optimization failure)
   on 3 consecutive attempts. GP-BO is the only method in this campaign
   with model-fitting machinery that can numerically fail; the others
   are simple/robust by comparison. CMA-ES is chosen as the fallback
   because, unlike the closed-loop local solvers (BOBYQA, Nelder-Mead,
   L-BFGS-B — all of which restart from the box midpoint with no state
   carried over), CMA-ES is population-based and does not require a GP
   fit, so it is a clean, simple, robust substitute that still explores
   globally rather than just locally polishing one point.
2. **Stagnation:** after at least 50 evaluations have been used (i.e.
   GP-BO has had a fair chance — more than double its own 22-point init
   budget — to make acquisition-driven progress), the running best value
   (`y_best_so_far`) has not improved for 25 consecutive evaluations.
   25 is chosen as roughly a third of the acquisition-phase budget (78
   evals): a plateau that long, this early in such a short budget, is
   strong evidence that LogEI is no longer finding useful improving
   moves in the current GP posterior, and the remaining budget is better
   spent letting a population-based method explore more broadly.

If a deviation triggers, I will record it in `changelog.md` with the
evaluation index at which it happened and the evidence observed
(the stagnation window's actual y_best_so_far trace, or the specific
exception text), before switching `set_active_optimizer` and continuing
with the remaining budget under the new method. If both the full
100-budget is reached and no deviation criterion has fired, the campaign
finishes as pure GP-BO (100 evals, no changelog entries beyond this
roadmap).
