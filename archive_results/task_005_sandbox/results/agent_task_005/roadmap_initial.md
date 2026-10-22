# Roadmap (pre-registered, written before any evaluate() call) — Task 005

## What I know before spending any budget

- 11-D, box [0,1]^11, minimize, deterministic, hard budget 2000 evaluations.
- Six known linear inequality constraints on x[0..5]. Every call costs budget
  but only allowed points count, so **evaluating an infeasible point is pure
  waste**. Because the constraints are known in closed form, I will enforce
  feasibility *before* every evaluate() call (rejection / projection inside my
  own code), so the planned infeasible-evaluation rate is 0%.
  A Monte-Carlo check (pure geometry, no evaluations) shows ~54% of the box is
  allowed and every coordinate still spans the full [0,1] inside the allowed
  region, so rejection sampling is cheap and only corners of the box are cut.
- x[9] has 3 effective levels and x[10] has 14 effective levels (42 combos).
  I will always submit the exact level values (never in-between values), so no
  evaluation is wasted on duplicates and the recorded x equals the x used.
  In surrogate models they are treated as ordinal numeric inputs at their level
  values; proposal steps snap them to the nearest level and also apply explicit
  categorical mutations so the search does not get stuck on one level.
- 9 effectively continuous coordinates (x0..x8), of which only x0..x5 are
  constrained.
- No noise -> surrogate models can be near-interpolating; local polishing at
  the end can exploit exact function values (no re-evaluation needed).
- 2000 evals in 11-D (~180·d) is moderate: enough for a sample-efficient
  surrogate-based global phase plus a dedicated local-convergence phase, but
  not enough for a pure evolutionary method to both explore and converge
  precisely. Nothing is known about modality, so a single local run is risky.

## Planned methods, sequence and budget split

| Phase | Label (`set_active_optimizer`) | Evals (approx.) | Purpose |
|---|---|---|---|
| A | `sobol_feasible_init` | 110 (= 10·d), evals 1–110 | Space-filling design: scrambled Sobol over x0..x8 rejection-filtered to the allowed polytope; x9/x10 assigned by balanced stratification (each x9 level ≈37×, each x10 level ≈8×), randomly paired. Gives a global picture and GP training data. |
| B | `turbo_feasible` | ~1290, evals 111–~1400 | TuRBO-1 style trust-region BO, my own implementation (the packaged wrapper has no constraint/discrete handling): ARD Matérn-5/2 GP (BoTorch SingleTaskGP), Thompson sampling over a perturbed-Sobol candidate set inside the trust region, candidates **filtered to the allowed polytope**, x9/x10 snapped to levels with a 20% per-candidate categorical-mutation probability; batch 4; standard TuRBO length schedule (init 0.8, min 0.5^7, max 1.6, success tol 3, failure tol ceil(d/4)=3). On collapse: restart with a fresh 22-point feasible Sobol design (global history kept). |
| C1 | `discrete_sweep` | ~15–45 | At the incumbent, one-at-a-time sweep of all other x9 levels (2) and x10 levels (13); if any helps, a small 2-D sweep of the best few combos. |
| C2 | `cma_local` | ~350 | Local CMA-ES on the 9 continuous coords with x9/x10 fixed at the incumbent's levels, x0 = incumbent, sigma0 = 0.05 (or ~ the final TuRBO trust-region length if smaller). Infeasible samples are **projected** onto the allowed polytope (small QP) and the projected point is what gets evaluated; CMA is told the projected point's real value plus a quadratic distance penalty for the original sample. |
| C3 | `lincoa_polish` | remaining (~150–200) | PDFO's LINCOA (model-based DFO that natively handles linear constraints) from the incumbent for final high-precision convergence. Any infeasible point LINCOA asks for is NOT evaluated; it receives a large penalty internally instead (no fabricated value enters the trace). |

Why this order: with no structural knowledge, a sample-efficient global
surrogate method is the best use of the bulk of a 2000-eval budget in 11-D;
TuRBO's local trust regions scale better than a single global GP/EI in 11-D
and its restarts give multi-basin coverage. GP-based methods converge slowly
to high precision, so the last ~25% goes to methods that converge fast locally
on deterministic functions (CMA-ES for robustness to mild ruggedness / bad
scaling, LINCOA for final precision with native linear constraints). The
discrete sweep is placed between them because trust-region BO tends to lock
the categorical coords onto the incumbent's level.

## Bookkeeping

- Every evaluate() call goes through my own wrapper that (1) asserts the
  point is allowed (A x - b <= 1e-10 after clipping to the box), (2) snaps
  x9/x10 to exact levels, (3) records y and my own best-allowed value.
- Best allowed value is tracked independently of the harness's
  `y_best_so_far`.
- Changelog entries cite the evaluation count at which the change happened.

## Pre-registered deviation criteria (quantitative)

1. **Output transform.** After Phase A, if the sample skewness of y is > 2,
   or (max−median) > 20·(median−min), the GP is fit on a log-shift transform
   `log(y − min(y) + 0.1·IQR)` (refreshed every batch) instead of plain
   standardization.
2. **TuRBO stagnation → early switch.** If the global best allowed value has not
   improved by more than 1e-6·max(1,|best|) during the last 300 Phase-B
   evaluations, and at least one TuRBO restart has already occurred, end
   Phase B early and give its remaining budget to Phase C (split 60% CMA-ES,
   40% LINCOA, after the discrete sweep).
3. **TuRBO still productive → extend.** If at eval 1400 the best improved
   within the last 100 evaluations by >1e-3·(observed y range), extend Phase B
   to at most eval 1600.
4. **Trust region collapse** (length < 0.5^7) → restart with a fresh feasible
   design (pre-planned, but each restart is still logged in the changelog).
5. **Compute guard.** If one GP fit + proposal takes > 30 s, cap the GP
   training set to the 500 points nearest the trust-region center.
6. **Discrete sweep.** If any non-incumbent (x9, x10) level beats the incumbent,
   Phase C2/C3 run on the new combination; if two combos are within 1% of the
   observed y-range of each other, split the CMA budget between both.
7. **CMA-ES stagnation.** Stop CMA when sigma < 1e-5 or no improvement greater
   than 1e-9·max(1,|best|) for 150 consecutive evaluations; then go to LINCOA.
8. **LINCOA early termination.** If LINCOA returns (converged) with > 100
   evaluations still left, restart CMA-ES (sigma0 = 0.1) from the best point of
   the second-best distinct basin seen so far (distance > 0.2 from the
   incumbent in x0..x8), then polish the winner.
9. **Feasibility guard.** If, despite the wrapper, more than 1% of evaluations
   turn out to be infeasible, stop and fix the generator before continuing.
