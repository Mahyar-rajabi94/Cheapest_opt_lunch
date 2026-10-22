# Roadmap — sandbox_task016 (dim=5, bounds=[-5,5]^5, minimize, NOISY heavy-tailed, budget=100)

## Problem properties driving the plan
- Dimension is low (5) and budget is extremely small (100 evals, i.e. only 20x
  dim). This rules out population-based global methods that need hundreds of
  evaluations to make progress (CMA-ES, standard DE) and rules out
  finite-difference gradient methods entirely (noise + tiny budget would make
  gradient estimates unusable, and the task explicitly warns about this).
- A sample-efficient surrogate-based method (Bayesian optimization with a GP
  surrogate) is the standard choice for continuous, low-dimensional,
  expensive/limited-budget problems, which is exactly this regime.
- The noise is real observation noise, occasionally heavy-tailed (large
  outliers). A plain GP-BO loop that trusts every single observation (as the
  provided `gp_bo.py` does by default: LogEI with `best_f = min(observed y)`)
  is vulnerable to being misled by one lucky-looking low outlier, both during
  the search (bad surrogate fit) and at reporting time (reporting a fluke as
  "best"). The plan therefore adds two explicit noise-robustness mechanisms
  around the base GP-BO loop: (1) an outlier guard during the search that
  re-evaluates a suspicious reading before trusting it, and (2) a dedicated
  replication phase at the end to confirm which of the top candidates is
  truly best, rather than reporting whichever single raw evaluation happened
  to be lowest.

## Method sequence and budget split (100 total)

1. **`gp_bo` (init sub-phase, ~12 evals, labeled `gp_bo_sobol_init`)**
   The project's `GPBO` wrapper (src/optimizers/gp_bo.py) begins with a Sobol
   space-filling design of `2*(dim+1) = 12` points before it starts fitting a
   GP and proposing via LogEI. This is used as-is: a short, low-cost
   space-filling phase to seed the GP, run through the same ask/tell object
   as phase 2 (so no data/state is discarded between phases).

2. **`gp_bo` (GP-driven sub-phase, up to ~78 evals, labeled `gp_bo`)**
   Continue driving the same `GPBO` instance one point at a time
   (`ask(n=1)` / `tell`), which fits a fresh `SingleTaskGP` + analytic LogEI
   each step. Budget target: leave the last ~10 evaluations for phase 3.
   - **Outlier guard (noise-robustness mechanism #1):** maintain a rolling
     window of the last 10 raw `y` values. Before calling `tell()` on a new
     reading, compute the median and MAD (median absolute deviation) of that
     window. If `|y - median| > 6 * MAD` (and MAD > 0), treat the reading as
     a suspected heavy-tail outlier: spend **one extra evaluation** at the
     *same* `x` and `tell()` the GP the mean of the two readings instead of
     the single suspicious one. This is capped at at most 8 extra
     evaluations across the whole run (outliers should be rare); if the cap
     is hit, stop re-checking and just feed raw readings, logging this in
     the changelog.

3. **`confirm_best` (last ~10 evals)**
   Identify the two distinct x points with the lowest observed y from phases
   1–2. Spend the remaining budget evaluating each of them several more
   times (roughly split evenly, e.g. 5/5, adjusted for whatever budget
   remains) and compare their **sample means**, not their single best raw
   reading. Report the point with the lower sample mean as the final
   recommendation. This directly targets the heavy-tailed-noise risk: a
   single lucky low reading should not be mistaken for a genuinely good
   point.

All evaluations are tagged via `set_active_optimizer` with one of:
`gp_bo_sobol_init`, `gp_bo`, `gp_bo_outlier_recheck`, `confirm_best`.

## Explicit, quantitative deviation criteria

I will deviate from the plan above, and log it in `changelog.md`, if any of:

1. **GP fitting instability:** if `fit_gpytorch_mll` / the GP fit raises an
   exception, or the acquisition optimizer raises an exception, on 2
   consecutive `ask()` calls, I will abandon GP-BO for the remaining budget
   and switch to `src/optimizers/cma_es.py` (or random search if CMA-ES also
   fails), since a broken surrogate is worse than a simple global method.
2. **No improvement plateau:** if the running best (based on raw
   `y_best_so_far`) has not improved for 30 consecutive evaluations during
   phase 2, I will end phase 2 early and move directly into phase 3
   (confirm_best) with whatever budget remains, since further unconfirmed
   GP proposals are unlikely to be worth more than solidifying the
   evidence for the best candidate(s) already found.
3. **Outlier-guard budget cap:** as stated above, outlier re-evaluation is
   capped at 8 extra evaluations total; if hit, I stop re-checking for the
   rest of the run and note it.
4. **Phase-3 budget:** the exact split of the final ~10 evaluations between
   the top two candidates may be adjusted (e.g. 6/4 instead of 5/5) depending
   on exactly how much budget remains when phase 2 ends; this is not a
   deviation worth logging unless the split changes qualitatively (e.g.
   dropping to a single candidate, or expanding past 2 candidates).
