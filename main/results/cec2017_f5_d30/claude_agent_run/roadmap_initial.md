# Roadmap — task010 (dim=30, bounds=[-100,100]^30, minimize, deterministic, budget=100)

## Problem read

- Dimension 30, box bounds [-100, 100] per coordinate (huge volume: 200^30).
- Deterministic (no noise) — no need to replicate points or use noise-robust
  acquisition.
- Budget of 100 evaluate() calls is **very small relative to dimension**
  (~3.3x dim). This is well below the ~10x-dim regime this project's own
  GP-BO module documents as its practical floor, and far below what
  population-based global methods (CMA-ES, DE) or finite-difference
  gradient methods (L-BFGS-B, which burns ~dim+1 evals per step) need to do
  anything meaningful in 30D:
  - L-BFGS-B: one finite-difference gradient estimate alone costs dim+1 = 31
    evaluations. At most 3 such steps fit in the whole budget. Not chosen
    as the primary method, but considered if a late-stage local polish is
    affordable.
  - CMA-ES: standard population size for dim=30 is 4 + floor(3*ln(30)) ≈ 14.
    100 evals is only ~7 generations — barely enough for the covariance
    matrix to adapt at all, and no margin for restarts.
  - DE: default population 10*dim = 300 is already 3x the entire budget;
    even a hand-shrunk population would get only a handful of generations.
  - Plain/Sobol random search: fine for coverage but wastes the budget's
    sample-efficiency potential entirely.
  - GP-BO (LogEI): this project's own module is explicitly documented as
    "only practical at small budgets... budgets of order 10x dim" — the
    closest fit of the available tools to this regime, even though our
    budget (3.3x dim) is tighter than its documented sweet spot. Its
    per-point sample efficiency (model-guided acquisition rather than
    population adaptation) is the best lever available when so few
    evaluations are affordable.
  - TuRBO: also GP-based, but its standard initial design alone is 2*dim=60
    points, leaving only 40 for trust-region-guided search, and it carries
    more machinery (trust region sizing, restarts) that needs more data to
    behave sensibly than plain GP-BO does.

## Plan

1. **Phase 1 — Sobol space-filling design, 20 evaluations (evals 1-20).**
   Use SobolSearch over the full box to get a low-discrepancy initial
   sample. 20 points (~0.67 per dimension) is a deliberately reduced
   initial design compared to GP-BO's own built-in default of
   2*(dim+1)=62 — with only 100 evals total, spending 62 of them on a
   space-filling design before any model-guided step would leave too few
   (38) for exploitation. 20 is enough to fit a first (rough) GP without
   consuming most of the budget.
   Label: `sobol_init`.

2. **Phase 2 — GP-BO with LogEI acquisition, remaining ~80 evaluations
   (evals 21-100).** Feed the 20 Sobol points into `GPBO` as its initial
   design (by setting its `n_init` to 20 instead of the module default of
   62, and pre-loading the Sobol points/values via `tell`), then run its
   standard sequential ask/tell loop (fit SingleTaskGP, optimize analytic
   LogEI via multi-start L-BFGS) for the rest of the budget. This is the
   primary exploitation phase: every remaining evaluation is spent where
   the model's acquisition function says is most promising, which is the
   best use of a very small budget in a high (30-d) search space.
   Label: `gp_bo_logei`.

No third phase is pre-planned; see deviation criteria below for what could
trigger one (e.g. a local-polish step with Nelder-Mead/L-BFGS-B near the
incumbent if GP-BO clearly plateaus with evaluations to spare).

## Deviation criteria (quantitative, checked during Phase 2)

- **GP/acquisition failure:** if `GPBO.ask()`/`tell()` raises an exception
  (e.g. GP fitting fails to converge, Cholesky errors from degenerate
  data) on 2 consecutive calls, abandon GP-BO for the remainder of the
  budget and switch to Sobol continuation (`sobol_fallback`) for all
  remaining evaluations, so the budget is never wasted on a broken loop.
- **Stagnation:** if `y_best_so_far` has not improved over the most recent
  30 evaluations (a meaningful fraction of this 100-eval budget) while
  still in Phase 2, treat this as evidence the GP-guided search has
  converged to a local region, and switch to CMA-ES (`cma_es_explore`,
  started fresh from the current incumbent as its mean, default step size
  scaled to ~10% of the box) for the remaining evaluations to inject
  broader exploration while there is still budget left.
- **End-game polish:** if there are 10 or fewer evaluations remaining and
  no stagnation/failure switch has already happened, optionally spend them
  on a short local refinement (`local_polish`, e.g. a few Nelder-Mead or
  coordinate-wise steps from the incumbent) only if that can fit in the
  remaining calls; otherwise just keep running the active method to the
  budget limit.
- Any such change will be logged in `changelog.md` with the evaluation
  index at which it was triggered and the observed evidence.
