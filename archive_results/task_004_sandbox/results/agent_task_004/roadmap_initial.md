# Roadmap — Task 004 (pre-registered, written before any `evaluate()` call)

## 1. What I know

A deterministic, box-constrained (`[-5, 5]^10`), real-valued, scalar
black-box objective to minimize, with a budget of 2000 evaluations. No
gradient is provided and nothing is known about modality, separability,
conditioning, or noise (there is none, since it's deterministic). That is
the entirety of the information this plan is allowed to use.

## 2. Methods, sequence, and budget split

**Phase 1 — Sobol' space-filling exploration: 150 evaluations (`sobol_explore`)**

A deterministic black box with totally unknown structure could be
unimodal-and-smooth, deceptive, or highly multimodal; nothing in the
problem statement rules any of these out. Before committing most of the
budget to one adaptive method, I want a cheap, unbiased read on the
landscape's rough shape (range of values, whether good points cluster
spatially or are scattered). A low-discrepancy quasi-random design covers
the 10-D box far more evenly per sample than plain random search, so it's
the standard choice for this kind of "look before you leap" step at a
cost most optimizers would also pay informally. 150 = 15×dim is enough
points to compare "top few" clustering without spending a large fraction
of the 2000-evaluation budget.

**Phase 2 — CMA-ES, with restarts as needed: up to 1350 evaluations (`cma_es_run1`, `cma_es_run2`, ...)**

CMA-ES is a strong general-purpose default for continuous, moderate-
dimensional (d=10), black-box, derivative-free optimization: it adapts
both step size and a full covariance model of the search distribution, so
it handles unknown scaling/rotation/conditioning without any tuning from
me, and it is population-based so it tolerates some multimodality better
than a purely local method. This is the single best "first serious guess"
given only Section 1's problem description, which is why it gets the
largest budget share. I'm reserving the ability to restart it (fresh
instance, new seed, from the box midpoint, as the provided wrapper always
starts) if a run stalls — see deviation criteria below — rather than
trusting a single run to use 1350 evaluations well.

**Phase 3 — Local polish from the best point found: remaining budget (~500 evaluations, `local_polish_lbfgsb`, fallback `local_polish_nm`)**

CMA-ES's own internal stopping/step-size behavior is tuned for general
robustness, not for squeezing out the last digits of precision at a
specific point. With evaluations left over and a deterministic objective
(so a derivative-free finite-difference gradient estimate is exact-
function-valued and reusable), a local refiner started *from the best
point found so far* (not the box midpoint, unlike the project's
ready-made wrapper classes) should be able to converge further. I'll
drive this directly with `scipy.optimize.minimize` myself (L-BFGS-B
first, since finite-difference gradients are cheap relative to the
leftover budget and L-BFGS-B typically converges fastest on smooth
local basins) rather than the project's `LBFGSB` wrapper class, purely so
I can set the start point to the incumbent best instead of the box
midpoint. If L-BFGS-B underperforms, I'll fall back to Nelder-Mead
(no gradient estimate needed, so it keeps working even if the local
region is not smooth) from the same incumbent point.

## 3. Explicit, quantitative deviation criteria

1. **Clustering check after Phase 1 (at evaluation 150):** compute the 10
   Sobol points with the lowest y. If their mean pairwise Euclidean
   distance exceeds half the box diagonal (i.e. > 0.5 × sqrt(10 × 10²) ≈
   15.8), I'll treat that as evidence of strong multimodality/no single
   obvious basin, and split Phase 2's budget across more (shorter)
   independent CMA-ES restarts from the start, instead of one long run.

2. **CMA-ES stagnation → restart:** within a CMA-ES run, if the best-so-
   far value improves by less than 0.1% (relative) over any rolling
   window of 15 generations (15 × popsize evaluations) **and** the
   run's internal step size (sigma) has shrunk to below 1e-4 of its
   initial value, I will consider that run converged and start a fresh
   `CMAES` instance (new seed) with whatever Phase-2 budget remains, as
   long as at least 200 evaluations of Phase-2 budget remain. If fewer
   than 200 remain, I'll end Phase 2 early and move to Phase 3 instead
   (see point 4).

3. **Global stagnation safety net:** at any point in the campaign, if the
   running best value has not improved at all over the last 200
   evaluations (absolute, regardless of method), I will end the current
   phase/run early and move to the next planned step, carrying any
   unused budget forward.

4. **Early phase completion reallocation:** if any phase finishes using
   less than its allotted share of the budget (e.g. a CMA-ES restart
   converges tightly, or Phase 1 and 2 trigger an early move via point 3
   above), the leftover evaluations are added to the next phase's budget
   rather than left unused. In particular, if Phase 2 ends early, Phase 3
   (local polish) simply gets a larger share.

5. **Phase 3 fallback:** if L-BFGS-B raises an error, or the best value it
   finds does not improve over any 100 consecutive evaluations, I will
   stop it and switch to Nelder-Mead (from the same incumbent best point)
   for the remaining budget.

6. **Hard floor:** regardless of all of the above, the campaign never
   calls `evaluate()` more than the 2000-evaluation budget allows; any
   phase in progress when the budget is reached simply stops there.
