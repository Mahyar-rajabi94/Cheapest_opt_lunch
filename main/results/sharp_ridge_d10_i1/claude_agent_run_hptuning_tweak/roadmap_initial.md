# Roadmap — initial step

## Understood problem

- Sealed black-box objective, `f: R^10 -> R`, to **minimize**.
- Bounds: `[-5, 5]` per coordinate (10-dimensional box).
- Deterministic (no observation noise) — confirmed by the task statement,
  so no replication is needed to average out noise.
- Total evaluation budget: **100** `evaluate()` calls, hard-enforced by
  `AgentTaskHarness`. This is only 10x the dimension, much smaller than the
  ~2000-evaluation convention used elsewhere in this project. I must treat
  every evaluation as expensive and avoid any method whose own
  documentation says it needs large budgets to be useful (e.g. this
  project's `DifferentialEvolution` wrapper uses a population of
  `10*dim = 100` — it would consume the *entire* budget just to fill its
  first generation with random points and never run real
  mutation/crossover; similarly, `BOBYQA` is documented as typically
  needing ~1,000+ evaluations to converge on this project's other tasks).
- Per the public interface, I may inspect wrapper docstrings/signatures for
  the optimizer library already provided in this sandbox, but must not
  inspect `objective.py` or try to infer the function's identity. I have
  only looked at `src/agent_harness/harness.py` (public interface) and the
  docstrings/signatures of `src/optimizers/*.py` (method descriptions and
  their own documented hyperparameter rationale) — not at the objective
  itself.

## Method chosen for this step: GP-BO (`src/optimizers/gp_bo.py`)

This project's own `gp_bo.py` docstring states explicitly that exact-GP
Bayesian optimization "is only practical at small budgets... budgets of
order 10 x dim, not 1000+ x dim" — which is exactly this task's regime
(budget = 100 = 10 x dim = 10). Among the optimizers available in this
sandbox, GP-BO is the one whose documented operating regime matches this
budget most directly (TuRBO's own tutorial-derived defaults and DE assume
much larger budgets; BOBYQA/Nelder-Mead/L-BFGS-B are local solvers whose
convergence in this project's experience takes hundreds to low-thousands
of evaluations; CMA-ES is a reasonable alternative but has no specific
documented rationale for *this* budget regime the way GP-BO does). Given a
deterministic, moderately low-dimensional (d=10), expensive-in-budget-terms
problem, a surrogate-based approach that uses every evaluation's
information maximally (via a fitted GP + expected-improvement acquisition)
is the principled first choice.

### Hyperparameters / initialization for this step (all inherited defaults
from `GPBO.__init__`, library: BoTorch, versions as pinned in this venv)

- `bounds = harness.bounds` (the task's own [-5, 5]^10 box) — not a tuned
  choice, just the problem's own domain.
- `seed = 0` — chosen by me for reproducibility of the Sobol initial design
  and the acquisition optimizer's random restarts; GPBO does not set a
  default seed itself (`Optional[int] = None`), so this is my explicit
  choice, not an inherited default.
- `n_init = 2 * (dim + 1) = 22` — GPBO's own inherited default, documented
  in its docstring as "a standard rule of thumb for how many points are
  needed before a GP fit is meaningful." I am not overriding this for the
  first step: it is the method's own prerequisite before any
  acquisition-driven point can be proposed at all, so there is nothing to
  tune yet — no GP has been fit, and no evidence exists yet to justify a
  smaller or larger design.
- Within this initial phase, `GPBO.ask()` internally delegates to its own
  `SobolSearch(bounds, seed=0)` instance (scrambled Sobol', scipy's own
  default `scramble=True`) — this is GP-BO's documented initial-design
  mechanism, not a separate method choice by me.
- `num_restarts=10`, `raw_samples=512` for acquisition-function
  optimization are GPBO's own inherited defaults (not used in this first
  step, since no acquisition-driven point is proposed until after the 22
  initial points, but recorded here as part of this step's configuration
  since they are fixed at construction time).

## Step length

This step consists of exactly the **22** Sobol initial-design evaluations
that GP-BO requires before it can fit a first GP and propose an
acquisition-driven point (`n_init = 2*(dim+1) = 22`). I will call
`harness.evaluate()` 22 times (one GP-BO `ask()`/`tell()` pair per call),
pausing at **cumulative evaluation 22** (22% of the total budget).

This length is appropriate because: (a) it is not an arbitrary choice but
the exact, documented prerequisite for GP-BO's surrogate model to become
meaningful — stopping earlier would mean fitting a GP on too few points to
trust at all, and (b) it is a natural, information-bearing checkpoint: it
is the first point at which I will have any data at all about this
objective's scale, spread, and rough smoothness, which should inform
whether to continue with GP-BO's acquisition phase, adjust it, or switch
methods for the remaining 78 evaluations.

## What I want to learn at the checkpoint

- The range and spread of observed `y` values (is the function's output on
  a scale that suggests a well-behaved, unimodal-ish landscape, or high
  variance suggestive of multimodality / sharp features?).
- Whether any Sobol point already stands out as a clear improvement over
  the rest (informs how much signal the GP has to exploit).
- Whether to proceed into GP-BO's acquisition-driven phase as planned, or
  reconsider (e.g. if the 22 evaluations reveal pathological behavior that
  would make a surrogate model unreliable — e.g. extreme value ranges that
  a GP's output standardization may not handle gracefully). I do not have
  a pre-registered numeric trigger for switching away at this point, since
  I have zero data yet; I will reason qualitatively from the trace once it
  exists, as the protocol requires (no evidence yet to quantify).
- No early-pause condition applies to this step: all 22 initial points are
  unconditionally needed before a GP can be fit, so there is no
  intermediate point at which pausing sooner would add information GP-BO
  doesn't already need.
