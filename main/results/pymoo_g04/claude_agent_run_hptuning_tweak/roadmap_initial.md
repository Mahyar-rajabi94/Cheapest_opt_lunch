# Roadmap — Initial Step

## Understood problem

- Sealed black-box **merit** function (`merit = y + PENALTY_LAMBDA * total
  constraint violation`), to be **minimized**, over a continuous input
  vector `x`.
- Deterministic (no observation noise) — confirmed both by the task
  description and by `harness.noisy == False`.
- Total evaluation budget: **100** calls to `harness.evaluate()`, enforced
  by the harness itself (`BudgetExhausted` once used up).
- The task prompt stated dimension 10 and deferred bounds to the harness.
  Constructing the harness, however, reports:
  - `harness.dim = 5`
  - `harness.bounds = [[78.0, 102.0], [33.0, 45.0], [27.0, 45.0], [27.0, 45.0], [27.0, 45.0]]`
  - `harness.direction = "min"`
  - `harness.noisy = False`

  **Assumption:** per the harness/task README convention for this sandbox
  (bounds/dim/direction are read from the constructed harness, which is
  the authoritative public interface), I am treating `dim = 5` and the
  asymmetric per-coordinate bounds above as ground truth for this run,
  overriding the dimension figure given in the prose instructions. This is
  exactly the kind of mismatch between stated and actual input domain that
  would normally be worth flagging before proceeding; since this is a
  non-interactive run I am proceeding on the harness's own reported values
  (the more directly verifiable source) and recording this explicitly here
  as a consequential assumption, to be surfaced in the final report. I am
  not using the recognizable shape of these specific bounds to guess the
  function's identity — only its numeric value for sizing the search.
- No other constraints are visible to me beyond the single scalar `merit`
  value returned by `evaluate()`. I cannot separate `y` from the
  constraint-violation term; I only observe the penalized merit. A merit
  value far outside the typical range likely signals infeasibility rather
  than an extreme true objective, so early high-merit points should not be
  over-interpreted as "bad regions of the true objective" without more
  data.

## Step 1 — plan

- **Method:** Sobol' quasi-random space-filling design
  (`src/optimizers/sobol_search.py`, wraps `scipy.stats.qmc.Sobol`,
  `scramble=True`, scipy's own default — no tuning). Chosen because I have
  zero observations yet: a low-discrepancy design gives broad, even
  coverage of the 5-D box, is cheap (no model fitting), and its output
  directly informs whether a model-based method (GP-BO, TuRBO) or a
  simpler evolutionary/local method is better suited to the observed merit
  landscape (scale, smoothness, feasibility fraction).
- **Hyperparameters / initialization:** `SobolSearch(bounds=harness.bounds,
  seed=0)` — inherited defaults from the wrapper (scrambled Sobol,
  scipy's own RNG handling under that seed). No other knobs exposed by
  this wrapper. Seed fixed only for reproducibility of this run, not a
  tuned choice.
- **Active-optimizer tag:** `"sobol_init"`.
- **Step length:** up to **15** evaluations (cumulative evaluation count
  100 → pause at **15**). Rationale: 15 ≈ 3× dim is enough points to get a
  first read on the merit range, variance, and apparent feasibility
  fraction across the box, while committing only 15% of the total budget
  of 100 before any model-based or evolutionary method is chosen — leaving
  ample budget (85 evaluations, well within the "~10–20× dim" regime
  GP-BO/TuRBO are designed for at dim=5) for whichever method this first
  look favors.
- **What I want to learn / evidence to inspect at the checkpoint:**
  - Range and spread of observed merit values (min, max, median) — to
    gauge whether most Sobol points are feasible-ish or whether the box is
    mostly infeasible (large penalty contributions).
  - Whether the best-so-far value is concentrated in a sub-region of the
    box or scattered, as a rough signal of landscape structure
    (unimodal/smooth vs. rugged/discontinuous from the penalty term).
  - Any evaluation errors/NaNs/extreme outliers.
- **Early-pause conditions:** none expected to trigger with only 15 cheap
  Sobol calls, but I will stop the batch early if a non-finite (`NaN`/`inf`)
  merit value appears, since that would call for inspecting the issue
  before spending more budget the same way.
