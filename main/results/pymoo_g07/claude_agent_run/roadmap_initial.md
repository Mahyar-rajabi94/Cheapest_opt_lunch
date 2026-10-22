# Roadmap — task020 (dim=10, budget=100, minimize, deterministic, hidden-constraint penalized merit)

## Problem properties known before any evaluation
- dim = 10, asymmetric per-coordinate box bounds (to be read from `harness.bounds`), minimize.
- Deterministic (no observation noise) — no need for replication/noise-robust acquisition.
- Budget = 100 total evaluate() calls — small relative to dim (10x dim), which this
  project's own optimizer docstrings flag as exactly the regime where exact-GP
  Bayesian optimization (GP-BO / TuRBO) is the appropriate tool, and where
  population-based methods (CMA-ES, DE) that want hundreds of evaluations just to
  run a handful of generations are not a good fit.
- The returned scalar is a *penalized merit* (true_objective + lambda * violation),
  not a clean objective: points that are heavily infeasible can show large, locally
  discontinuous jumps relative to nearby feasible points, which is different from
  genuine smooth objective structure. The underlying constrained optimum is feasible
  (violation=0 there), so the merit surface is smooth and informative right around
  the optimum — it is only potentially jagged/spiky in infeasible regions far from it.

## Method plan and budget split
1. **GP-BO (LogEI), ~75 evaluations** (`set_active_optimizer("gp_bo")`).
   - Uses this project's `src/optimizers/gp_bo.py`, which internally starts with a
     2*(dim+1) = 22-point Sobol design before switching to acquisition-driven
     proposals. This gives broad initial coverage of the box (helpful for finding
     which regions are feasible/cheap under the penalty) before committing
     evaluations to local exploitation.
   - Rationale for GP-BO as the primary method: budget ≈ 10x dim is squarely the
     regime this project's own docs recommend GP-BO for; a GP surrogate fit on the
     merit values directly treats "optimize the merit" as the stated task, and
     Standardize()/Normalize() transforms plus LogEI (which handles heavy-tailed /
     skewed improvement distributions better than plain EI) give it some natural
     robustness to a handful of outlier-high merit values from infeasible penalty
     regions, without needing to hand-engineer constraint-awareness.
2. **BOBYQA local refinement, ~25 evaluations** (`set_active_optimizer("bobyqa_refine")`).
   - After GP-BO has spent most of the budget building a global picture and
     converging toward a promising basin, switch to a model-based local trust-region
     solver (Py-BOBYQA, already in this project's optimizer package) started from the
     best point GP-BO found, to polish the final answer within the remaining budget.
   - Rationale: BOBYQA is cheap per step (no GP refit) and is a strong closed-loop
     local finisher once the global search has located the right basin — standard
     global-then-local hybridization.

Planned split: Sobol-seeded GP-BO for evaluations 1–75, BOBYQA local refinement for
evaluations 76–100 (approximate; exact boundary depends on where GP-BO is when it
plateaus, see deviation criteria below).

## Explicit deviation criteria
- If GP-BO's best-so-far merit has not improved over a window of 20 consecutive
  GP-BO evaluations (after its initial 22-point Sobol design is complete), stop
  GP-BO early and move to BOBYQA refinement sooner than evaluation 75, using the
  remaining budget for local polishing instead.
- If any single GP fit raises an exception or repeatedly (3+ times) proposes a
  point effectively identical to one already evaluated (degenerate acquisition),
  switch early to BOBYQA refinement from the current best point.
- If BOBYQA converges (raises `OptimizerExhausted`) before the budget is used,
  restart it from the second-best distinct point found so far (a simple multi-start
  local-refinement strategy) rather than leaving budget unused; if it converges
  again, switch to Nelder-Mead from a different remaining promising point for any
  leftover budget.
- If a merit value is observed that is far outside (e.g. >10x) the range seen from
  all nearby/previous points, treat it only as a signal to avoid that region (likely
  a heavily infeasible point under the penalty) — it will not be chased or treated as
  real objective structure, consistent with the task instructions.
- Any such change will be logged in `changelog.md` with the evaluation index at
  which it occurred and the evidence observed.
