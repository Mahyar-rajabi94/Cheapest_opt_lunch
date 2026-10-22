# Roadmap — initial step only

## Understood problem

- 10-dimensional, continuous, box-bounded ([-5, 5] per coordinate) black-box
  function, to be **minimized**.
- Deterministic (no observation noise) per the task statement; `objective.py`'s
  `NOISY` flag is treated as confirming this, not as something to probe.
- Total evaluation budget: **100 `evaluate()` calls**, hard-enforced by the
  harness. This is only 10x the dimension — a small budget by this project's
  usual convention (elsewhere budgets of ~2000 are used), so the whole
  campaign must be planned around sample efficiency, not asymptotic
  convergence behaviour.
- The objective is treated as a sealed black box: no attempt will be made to
  identify it, and no harness/task source beyond the documented interface
  (`dim`, `bounds`, `direction`, `set_active_optimizer`, `evaluate`, `save`)
  will be used to inform decisions.
- Interface: `AgentTaskHarness(budget=100)` with `evaluate(x)` as the only
  source of real function values; `set_active_optimizer(name)` tags each
  call; `save()` persists `trace.csv`/`meta.json` (also triggered
  automatically at budget exhaustion).

## Method choice for this first step: GP-BO (Sobol initial design)

This project's `src/optimizers/gp_bo.py` wraps BoTorch `SingleTaskGP` +
analytic LogEI behind the standard `ask`/`tell`/`best` interface, and its own
docstring states it is intended for budgets "of order 10 x dim, not 1000+ x
dim" — which is exactly this task's regime (100 = 10 x 10). Classic
population-based methods in this package (CMA-ES default popsize ~4+3ln(10)
≈ 10-14, and especially Differential Evolution with pop_size = 10*dim = 100,
which alone would consume the *entire* budget on its first generation and
never evolve) are documented in this project as needing much larger budgets
to show their strength. A GP surrogate that models the whole function from
few, well-placed points is the better fit for a 100-eval budget in 10
dimensions.

GP-BO's own `ask()` logic does not propose an acquisition-driven point until
`n_init = 2 * (dim + 1) = 22` initial points have been told — before that, it
draws from an internal `SobolSearch` (scrambled Sobol', scipy `qmc.Sobol`).
So the natural, non-arbitrary length of this first step is exactly that
initial design: it is a fixed, self-contained phase of the chosen method,
not an arbitrary batch size chosen by me.

**Hyperparameters / initialization for this step** (`GPBO(bounds, seed=0)`,
all inherited library/package defaults — no tuning applied yet, since there
is no data to tune against):
- `n_init = 2 * (dim + 1) = 22` (package convention, a standard rule of
  thumb for how many points are needed before a GP fit is meaningful).
- Initial design method: scrambled Sobol' sequence (`SobolSearch`, scipy
  `qmc.Sobol`, `scramble=True`), seeded with `seed=0` for reproducibility.
- `num_restarts=10`, `raw_samples=512` for the (not-yet-used) acquisition
  optimizer — BoTorch tutorial defaults; irrelevant until step 2.
- GP model/kernel/priors and acquisition function (`SingleTaskGP` defaults,
  analytic `LogExpectedImprovement`) are likewise not exercised during the
  initial design, only from step 2 onward.
- `harness.set_active_optimizer("gp_bo_sobol_init")` will tag these 22 calls
  distinctly from later acquisition-driven GP-BO calls, so the trace can
  separate "exploration design" from "model-driven exploitation" later.

## Step length

**22 evaluations** (the full Sobol initial design), bringing the cumulative
count to **22 / 100**, leaving 78.

This length is dictated by the method itself (GP-BO will not use the GP/
acquisition machinery until all 22 initial points are told), so stopping
partway through the design would leave an unfinished, arbitrary GP input set
and gain nothing — there is no meaningful intermediate checkpoint inside a
Sobol design. 22 evaluations is also a small enough fraction of the 100-call
budget (22%) that most of the budget (78 calls) remains for
acquisition-driven refinement and any later method reconsideration.

## What I want to learn / review at the checkpoint

- The spread and best value among the 22 initial points: rough shape of the
  landscape (e.g. large dynamic range suggesting multimodality vs. a narrow
  range suggesting near-flatness), and whether the best initial point is
  near the box centre or near an edge/corner (informs whether local
  refinement near the current best, vs. continued global exploration, is
  more promising).
- Whether 22 points in 10 dimensions give a GP fit that looks informative
  enough to trust LogEI's point selection (no hard pre-committed metric —
  judged qualitatively from the value spread and evaluation cost/time; if
  the BoTorch GP fit raises warnings or fails to converge I will reconsider
  before trusting its proposals).
- No condition for pausing the Sobol design sooner: a partial Sobol batch
  does not feed GP-BO's acquisition step at all, so there is no useful
  intermediate stopping point before all 22 are collected.
