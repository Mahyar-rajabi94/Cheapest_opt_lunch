# Roadmap — initial step only

## Understood problem

- Black-box, sealed objective (`objective.py` not inspected beyond the
  documented harness interface).
- Dimension: 20. Bounds: [-5, 5] per coordinate (confirmed via
  `harness.dim` / `harness.bounds`).
- Direction: minimize (confirmed via `harness.direction == "min"`).
- Noise: confirmed via `harness.noisy == True`. Per the task description,
  the noise is mild and roughly multiplicative-scale, not heavy-tailed.
  Repeated `evaluate()` calls at the same `x` can differ.
- Total evaluation budget: 100 `evaluate()` calls, hard-enforced by the
  harness (`BudgetExhausted`).
- This is a very small budget relative to dimension (5x dim, not the
  larger 2000-eval convention used elsewhere in this project). Methods
  that need large budgets to show their strength (plain DE at
  pop_size=10*dim=200, for instance) are not viable here at all —
  10*dim alone already exceeds the whole budget. This rules out DE and
  makes any method that burns evaluations on finite-difference gradients
  (L-BFGS-B: ~dim+1=21 evals per internal step) expensive relative to the
  total budget, especially under noise, which corrupts finite-difference
  gradient estimates.

## Assumptions

- No further clarification is needed: the four task parameters (dim,
  bounds, direction, noise) are stated and have been confirmed
  programmatically against the harness before use.
- "Mild, roughly multiplicative-scale noise" is taken to mean the noise
  standard deviation scales with the magnitude of `y`, so distinguishing
  real differences in `y` from noise will matter more in regions of small
  |y| (e.g. near an optimum) than in regions of large |y|.
- Given the tight budget, every evaluation should be treated as valuable:
  this favors a sample-efficient, model-based strategy (e.g. Bayesian
  optimization with a GP, which this project's `gp_bo.py` / `turbo.py`
  both document as suited to budgets of order 10x dim) over large
  population-based or many-restart methods. That choice is not committed
  to yet — it is deferred to the next checkpoint, once there is real data
  — but it is the reason this first step is built to directly produce
  (a) a space-filling sample usable as a GP's initial design, and
  (b) a direct, cheap estimate of the noise scale, both of which are
  needed before deciding how to configure any such model-based method
  (e.g. whether to fix or infer an observation-noise term).

## Step 1 plan

**Method for this step:** two parts, no model fitting yet.

1. **Noise probe (3 evaluations):** evaluate the same point — the midpoint
   of the box, `x = [0]*20` (the project's own standard neutral starting
   point, used by CMA-ES/BOBYQA/L-BFGS-B/Nelder-Mead in this codebase) —
   three times. This gives a direct, cheap, assumption-free estimate of
   the noise scale at one concrete location before any model is fit or
   any noise-handling setting (e.g. a GP's observation-noise term) is
   chosen. Three repeats is the minimum that yields both a mean and a
   non-degenerate sample standard deviation.
2. **Space-filling design (13 evaluations):** a scrambled Sobol' sequence
   (`src/optimizers/sobol_search.SobolSearch`, library default
   `scramble=True`, seed=0 for reproducibility) over the full box,
   13 points. Combined with the 3 repeated center evaluations, this
   brings the step total to 16 — under 1x the problem dimension — which
   is deliberately modest: this project's GP-BO wrapper uses
   2*(dim+1)=42 Sobol points as its own initial-design convention, but
   that number is not adopted here untested; instead this first,
   shorter step is used to check evidence (spread of observed y, rough
   noise-to-signal ratio) before committing to a specific initial-design
   size or to any one model-based method.
   13 (not 16, to account for the 3 already spent on the probe) keeps the
   combined Sobol set from this step at a reasonable fraction of what a
   later GP would want, while leaving flexibility to add more Sobol
   points next if the evidence favors it.

No hyperparameters beyond `seed=0` (for reproducibility) are set for
SobolSearch; its only other option, `scramble=True`, is the library's own
default and is left as-is (no tuning rationale needed for a baseline
space-filling design).

## Step length and pause point

- This step uses at most **16** evaluations (3 probe + 13 Sobol).
- Pause at **cumulative evaluation 16** (of 100 total; 84 remaining
  afterward).
- This length is short relative to the total budget (16%) because its
  purpose is purely diagnostic — to gather the minimum evidence needed to
  make an informed method choice for the (much larger) remaining budget,
  not to make optimization progress by itself.

## What this step should show, and what will be inspected at the checkpoint

- The 3 center-point repeats: mean and sample standard deviation of `y`
  at the same `x`, giving a concrete noise-scale estimate (and a sanity
  check on whether noise looks roughly multiplicative, i.e. small
  relative to |y|, as described).
- The 13 Sobol points: the range and rough shape of `y` across the box
  (e.g. is there a clear gradient toward one region, or is it flat/rugged
  at this resolution), and the best `y` seen so far.
- No early-pause condition applies within this step: 16 evaluations is
  already the minimum needed to get both signals, so there is nothing
  shorter to check against. If, implausibly, a BudgetExhausted or other
  error occurs, that would itself be the trigger to stop and reassess
  immediately.
