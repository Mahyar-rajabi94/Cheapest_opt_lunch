# Roadmap — Initial Step Only

## Problem understood

- Objective: sealed black-box function `fn(x)`, to be **minimized**.
- Input: real vector `x` of dimension **10**, each coordinate bounded in **[-5, 5]**.
- Noise: **deterministic** (confirmed by `harness.noisy == False`); no need for
  replication or noise-robust acquisition.
- Budget: **100** `evaluate()` calls total, hard-enforced by
  `AgentTaskHarness`. No other structure (smoothness, separability,
  modality, conditioning) is known yet — nothing about this will be assumed
  beyond what the harness's public interface states, and the function's
  identity will not be investigated.
- Harness confirmed live via a `ping`: `dim=10`, `bounds=[-5,5]*10`,
  `direction="min"`, `noisy=False`, `n_used=0`. (Mechanical note on
  execution, not part of the optimization plan: because each tool-call
  boundary in this environment loses local Python state, and this
  project's closed-loop optimizer wrappers (BOBYQA/NelderMead/LBFGSB) wrap
  a non-picklable background thread, the single `AgentTaskHarness` instance
  and all optimizer objects live inside one persistent background Python
  process for the entire campaign; I send it short commands and read back
  plain-JSON results between checkpoints. No optimizer object is ever
  pickled, and the harness's budget counter is never reset or duplicated.)

## Step 1 — method, hyperparameters, initialization

With a budget this small (100 calls) relative to the dimension (10), and
zero prior evidence about the landscape (unimodal vs. multimodal, isotropic
vs. ill-conditioned, smooth vs. rugged), committing immediately to a single
local or single global method would be premature. The first step is a
**space-filling design**, used purely to collect calibration evidence
(range/scale of `y`, rough variation across the box) before choosing a
search method for the remaining budget.

- **Method:** Sobol' quasi-random sequence (`src/optimizers/sobol_search.py`,
  wrapping `scipy.stats.qmc.Sobol`, `scramble=True`). Chosen over plain
  random search because, for a fixed small sample count, a scrambled Sobol'
  sequence covers the 10-d box far more evenly (lower discrepancy), giving a
  more representative first look at the landscape per evaluation spent.
- **Hyperparameters / initialization:** `bounds = [[-5, 5]] * 10` (from the
  harness); `seed = 0` for reproducibility. All other behavior is the
  library's own default (`scipy.stats.qmc.Sobol(d=10, scramble=True,
  seed=0)`) — no tuning parameters are exposed or relevant for a
  pure space-filling sampler. Per the wrapper's own note, Sobol' coverage is
  best when the cumulative draw count is a power of two.
- **Step length:** draw and evaluate **16 points** (a power of two, close to
  the common "2×dim" initial-design heuristic used elsewhere in this
  project, e.g. `turbo.py`'s `n_init = 2 * dim = 20`; 16 is chosen instead of
  20 to keep this calibration step cheaper, reserving more of the 100-call
  budget for the adaptive method(s) that follow, while still a large enough
  sample to see scale and rough shape). Pause at **cumulative evaluation 16**
  (16 used / 84 remaining).
- **What I want to learn / evidence at the checkpoint:** the observed
  range and spread of `y` (is it on the order of 1s, 10s, 1e3s — tells me
  about scaling but not identity), whether the best-so-far improves roughly
  evenly across the batch or is dominated by one or two points (hinting at
  a sharp optimum / multimodality vs. a flatter landscape), and whether any
  points reveal a clear direction of improvement across the box. No early
  stop is planned for this short, fixed calibration batch — 16 cheap,
  parallel-in-spirit draws carry negligible risk of wasted budget, so the
  step will run to completion before the first decision.
