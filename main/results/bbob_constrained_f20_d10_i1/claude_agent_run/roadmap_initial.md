# Roadmap — task021 (dim=10, bounds=[-5,5]^10, minimize, deterministic, budget=100)

## Known problem properties
- 10-dimensional, box-bounded continuous minimization.
- Deterministic (no observation noise).
- Objective is a sealed black box exposed only as a single penalized merit
  value: `true_objective + penalty_lambda * violation`. The optimum is
  feasible (violation=0 there), so the merit's global minimum coincides
  with the true constrained optimum. I cannot separate objective from
  penalty from a single scalar, so the plan treats the merit itself as
  the one scalar to minimize, and treats any unusually large jump in merit
  between nearby points as a signal of a heavily-penalized (infeasible)
  region to move away from, not as real objective curvature to chase.
- Hard total budget: 100 `evaluate()` calls.
- Only 100 evaluations are available for a 10-D problem, so the campaign
  must stay cheap-per-evaluation (no finite-difference gradient methods,
  which would burn ~dim+1 evals per step; no exact-GP method requiring a
  large initial design relative to budget).

## Method sequence and budget split

1. **Sobol' quasi-random search — 20 evaluations** (`optimizer="sobol"`).
   Low-discrepancy space-filling design over the full box. Purpose:
   (a) get an even first look at the merit landscape in all 10 dimensions
   with no assumptions about where the feasible/good region is, since
   the hidden constraints could put the optimum anywhere in the box; and
   (b) gather enough spread-out data to judge, from real evaluations,
   whether the landscape is smooth-ish between big penalty jumps or
   highly rugged, which will inform anything I do differently in phase 2.
   20 = roughly 2x(dim+1), a standard minimal space-filling design size,
   while leaving the bulk of the budget for local/adaptive search.

2. **CMA-ES — remaining 80 evaluations** (`optimizer="cma_es"`).
   Population size at dim=10 is pycma's own default,
   popsize = 4 + floor(3*ln(10)) = 10, so this is ~8 generations.
   CMA-ES is chosen as the main workhorse because:
   - It is population-based and derivative-free, so it tolerates the
     non-smooth, possibly discontinuous-looking merit surface created by
     the hidden penalty (no finite-difference gradients to be corrupted
     by a penalty cliff).
   - It adapts its search covariance/step size using the whole population
     each generation, so it can both explore broadly early and contract
     onto a promising basin later within a single run — a reasonable
     match for a tight 100-eval budget where there isn't room for a
     separate "global phase" + "local polish phase" with two independently
     re-initialized solvers.
   - Its own box-bounds handling keeps every proposed point inside
     [-5,5]^10, so it can't waste evaluations far outside the feasible
     domain even before considering the hidden constraint.
   The wrapper re-initializes CMA-ES's mean at the box midpoint (its own
   standard convention, not something this plan can override), so phase 1
   does not literally seed phase 2's starting point; its value is the
   independent read on the landscape described above, plus a fallback
   data source if CMA-ES itself stalls (see deviation criteria).

This two-method sequence (space-filling design -> adaptive population
search) is a standard, low-risk default for a black-box, deterministic,
budget-constrained problem of this size, and the explicit method switch
(Sobol -> CMA-ES) is itself the kind of hybridization the protocol
encourages, not something to avoid.

## Explicit deviation criteria

I will deviate from the plan above, and log it in changelog.md, if any of
the following are observed from real evaluations:

- **CMA-ES stalling**: if the running best merit has not improved for
  3 consecutive CMA-ES generations (30 evaluations) after phase 2 starts,
  I will switch the remaining budget to Differential Evolution
  (`optimizer="de"`), a different population-based method with a
  different exploration mechanism (mutation/crossover vs.
  covariance adaptation), to escape whatever basin CMA-ES is stuck in.
- **Severe, persistent penalty contamination**: if more than half of any
  batch of 10 consecutive evaluations show merit values more than, say,
  100x the median of the best 10 points seen so far (a strong sign most
  of the search is landing in heavily infeasible territory), I will
  narrow the search box for the remaining budget to the smallest
  axis-aligned region containing the best 25% of points evaluated so far
  (padded by 10% of the original per-axis range), and restart whichever
  method is active within that narrowed region, to concentrate the
  remaining budget on the apparently-feasible neighborhood.
- **Budget running out before convergence**: if fewer than 15 evaluations
  remain and CMA-ES has not visibly converged, I will stop feeding new
  CMA-ES generations mid-way if needed and simply report the best point
  found, since partial generations still both use real evaluate() calls
  and are reported correctly by the harness.
- If BOBYQA or another closed-loop local solver is substituted in at any
  point and raises `OptimizerExhausted` before the budget assigned to it
  is used, I will immediately redirect the unused evaluations to whichever
  method is next in the (possibly revised) sequence rather than leaving
  budget unused.

Any of the above, or any other actual change made mid-run, will be logged
as a changelog.md entry citing the specific observed evidence.
