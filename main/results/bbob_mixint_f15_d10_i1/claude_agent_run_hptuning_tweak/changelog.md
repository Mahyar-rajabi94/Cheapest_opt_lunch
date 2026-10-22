## Decision after evaluation 16
**Budget:** 16 used / 100 total; 84 remaining.
**Evidence:** Step 1 (Sobol space-filling, n=16, seed=0) gave y in
[119.43, 181.64], mean 146.39. Best point so far: x = [0.33, 0.77, 1.66,
1.73, 5.93, 0.67, 4.72, 4.45, -4.61, 0.81], y = 119.43. Simple linear
correlations with y across the 16 points (noisy, n=16, but directionally
suggestive): x0 +0.58, x1 -0.61, x5 +0.62, x6 +0.57, x8 +0.39 are the
strongest; x2, x3, x4, x7, x9 are weak (|corr| < 0.13). The best point is
consistent with these signs (low x0, high x1, low x5, low x6, low x8).
This is far too little data to resolve the discrete structure on x0..x7
coordinate-by-coordinate (2*2*4*4*8*8*16*16 = 524,288 combinations), but
it does suggest the objective has some global additive/smooth trend
across both the integer-rounded and continuous coordinates, rather than
being pure noise-like high-frequency structure on the integers. That
favors a population-based global method that adapts a shared
search distribution over the whole vector (letting rounding absorb the
integer coordinates) over a coordinate-by-coordinate discrete local
search at this stage, when only 84 evaluations remain for a 10-d space.
**Decision:** Switch to CMA-ES (`cma.CMAEvolutionStrategy`, pycma, via a
campaign-local adapter rather than the stock `CMAES` wrapper in
`src/optimizers/cma_es.py`). The stock wrapper hardcodes its initial mean
to the box midpoint; here the campaign already has 16 real observations,
so the adapter seeds the initial mean at the current best Sobol point
instead of discarding that information. This is a fresh `cma`
optimizer object (no state to "continue" from Sobol, which has no
internal model to transfer), but it reuses the box bounds and keeps the
same harness/evaluation counter; the 16 Sobol observations remain in the
trace and inform this seeding choice.
**Hyperparameters:** `x0` (initial mean) = best Sobol point
[0.33, 0.77, 1.66, 1.73, 5.93, 0.67, 4.72, 4.45, -4.61, 0.81] (chosen
value, not a library default). `sigma0` = mean(box width)/4 ≈ 1.575,
the same generic box-relative heuristic the stock `CMAES` wrapper uses
(not tuned further) — kept because the seed point is a single
observation, not a trusted local optimum, so a wide-ish initial step
size is appropriate to keep exploring rather than collapsing immediately
onto it. `bounds` = `harness.bounds` (hard box constraints enforced by
pycma's own `bounds` option, same as the wrapper). `seed=1` (pycma RNG
seed, arbitrary fixed value for reproducibility; this is the first use
of CMA-ES in this campaign, so there is no prior CMA-ES seed to keep
distinct from). Population size (`popsize`) and all
other internal strategy parameters: left at pycma's own defaults for
dim=10, which resolve to `popsize=10` (confirmed by instantiation);
identical to what the stock wrapper would use, not tuned.
**Reason:** With 84 evaluations left and clear-but-noisy global trends
across several coordinates (including both integer and continuous ones),
a population-based method that adapts a full covariance over the 10-d
vector is a better use of a tight budget than committing to per-axis
discrete coordinate descent before any local structure is confirmed.
Seeding at the current best point (rather than the box midpoint) uses
the information already paid for instead of re-exploring from scratch.
**Next step limit:** 40 additional objective calls, at most 84; pause at
cumulative evaluation 56.
**Why this length:** popsize=10, so 40 evaluations covers exactly 4
full CMA-ES generations — enough for the step-size/covariance adaptation
to act more than once (so a stall is not just one unlucky generation)
while leaving 44 evaluations (more than half the remaining budget) for
further refinement or a different method afterward.
**Review:** At evaluation 56, inspect the generation-by-generation best
y, whether CMA-ES's internal step size is still shrinking/improving or
has stalled, and whether the implied integer coordinates of the best
point are stable across generations (suggesting convergence on a
specific discrete setting) or still jumping around (suggesting more
exploration is needed). No condition to pause earlier than 40 calls:
CMA-ES's population-level update needs each generation to complete to
be informative, and 4 generations is already a tight lower bound for
seeing a trend.

## Decision after evaluation 56
**Budget:** 56 used / 100 total; 44 remaining.
**Evidence:** The CMA-ES step (seeded at the Sobol best, 4 generations of
popsize 10) produced per-generation bests 114.56, 114.18, 114.56, 109.78
— a net improvement from the seed's y=119.43 down to y=109.78, with the
last generation producing the largest single-generation gain. The
step-size `sigma` barely moved (1.80 -> 1.64), i.e. CMA-ES has not
collapsed into a narrow basin and is still actively exploring the
surrounding region rather than having converged/stalled. Best point now:
x = [0.31, 0.68, 2.38, 2.47, 5.97, 0.42, 5.37, 7.22, -2.26, -0.94]
(integer part rounds to [0,1,2,2,6,0,5,7]) — this has shifted from the
seed's integer pattern [0,1,2,2,6,1,5,4] on x5 and x7 specifically, i.e.
CMA-ES is still revising the discrete coordinates rather than them being
fixed/converged. This is evidence the method is working and has not
plateaued, so there is no reason to switch away from it yet.
**Decision:** Continue CMA-ES, same `es` object (state preserved: mean,
covariance, step size, evolution paths all carried over from the last
told generation) — not a restart. Continuing is the right call because
progress is still being made and sigma has not shrunk to a point of
diminishing returns.
**Hyperparameters:** Unchanged from the previous entry (same `es`
instance: bounds, seed=1, popsize=10, all pycma internal adaptation
parameters at their library defaults). No reconfiguration — CMA-ES's own
internal adaptation of step size and covariance across generations is
the mechanism being relied on, not a manual hyperparameter change.
**Reason:** Evidence shows continued improvement and an unshrunk step
size; there is no indication of stalling that would justify switching
method or restarting. Preserving state lets the adapted covariance
(learned from the first 4 generations) carry into the next ones instead
of discarding that information.
**Next step limit:** 30 additional objective calls, at most 44; pause at
cumulative evaluation 86.
**Why this length:** 30 evaluations = exactly 3 more full generations at
popsize 10, enough to see whether the improvement trend from the last
step continues or saturates, while reserving 14 evaluations (the
remainder of the 44 available now) for a final local-refinement or
confirmation step after this one, rather than committing the entire
remaining budget to a single uninterrupted CMA-ES run.
**Review:** At evaluation 86, inspect generation-by-generation bests
again, current sigma, and how much of the 10-d point (especially the
integer coordinates) has stabilized. If sigma has shrunk substantially
and per-generation bests have flattened, consider a short confirmatory
step (e.g. local exhaustive toggling of the integer coordinates around
the incumbent, or simply stopping) with the remaining budget instead of
continuing CMA-ES further.

## Decision after evaluation 86
**Budget:** 86 used / 100 total; 14 remaining.
**Evidence:** The second CMA-ES block (3 more generations) produced
per-generation bests 111.06, 111.84, 109.87 — none beat the incumbent
best of y=109.78 set two generations earlier (at evaluation ~56-66), and
the values are fluctuating within a narrow band (109.9-111.8) rather
than trending down. `sigma` shrank only modestly, 1.64 -> 1.46, so CMA-ES
has not sharply converged, but its population-level progress has
effectively plateaued over this block. With only 14 evaluations left —
less than one more full CMA-ES generation and a half (popsize=10) — a
fifth generation would barely move the adapted distribution and risks
spending the entire remaining budget without any further checkpoint.
This is a case for switching rather than continuing by default.
**Decision:** Switch to a greedy, budget-capped coordinate-descent local
search directly on the integer coordinates (x0..x7) around the current
incumbent (x = [0.31, 0.68, 2.38, 2.47, 5.97, 0.42, 5.37, 7.22, -2.26,
-0.94] -> rounds to integer part [0,1,2,2,6,0,5,7], y=109.78), keeping
x8, x9 fixed at their CMA-ES-optimized continuous values. CMA-ES
operates on a continuous relaxation and the integer coordinates are only
ever seen through evaluator-side rounding, so it is not guaranteed to
have found the best integer combination near its continuous optimum;
a direct neighbor search on the rounded integers is a cheap, targeted
check that a smooth continuous method cannot perform. This restarts
the search process (no CMA-ES state is reused) but keeps the same
harness/evaluation counter and the incumbent point as the starting
state, so nothing already learned is discarded.
**Hyperparameters:** Not a parametric method in the usual sense; the
only choice is the coordinate visiting order and step size. Visiting
order: x1, x5, x6, x0, x7, x3, x2, x4 — ranked by the magnitude of the
Sobol-stage linear correlation with y from the evaluation-16 checkpoint
(|corr| 0.61, 0.62, 0.57, 0.58, then the weaker 0.10, 0.11, 0.12, -0.12),
on the reasoning that coordinates with the strongest observed
association with y are most likely to still have room for a one-step
local improvement and should be checked first given the tight remaining
budget. Step size: +-1 (the smallest possible move on an integer
coordinate), clipped to that coordinate's bounds. Acceptance: greedy —
try +1 first, then -1 if +1 did not improve; accept the first improving
move found and update the incumbent before moving to the next
coordinate (first-improvement coordinate descent, not exhaustive
best-improvement, to conserve evaluations).
**Reason:** CMA-ES's within-block progress has plateaued and the
remaining budget cannot support a meaningfully sized further CMA-ES
block, so continuing it is unlikely to be worth the delay to the next
checkpoint. A cheap, evidence-ordered discrete neighbor search directly
targets the one thing CMA-ES's continuous relaxation cannot verify (local
optimality on the rounded integer coordinates) and is the best use of a
small remaining budget.
**Next step limit:** 14 additional objective calls, at most 14 (all of
the remaining budget); pause at cumulative evaluation 100 (budget
exhaustion).
**Why this length:** 14 is simply what remains; first-improvement
coordinate descent over at most 8 coordinates costs 1-2 evaluations per
coordinate (1 if the first direction tried improves and is accepted, up
to 2 if neither direction at a coordinate improves), so 14 evaluations
is enough to attempt most or all 8 coordinates once each while leaving a
small margin; if the budget runs out mid-sweep the coordinates are
visited in the priority order above so the most promising ones are
checked first.
**Review:** This is the final step (it exhausts the budget); the
checkpoint afterward records the final outcome rather than a further
decision. No condition to pause earlier than budget exhaustion: each
single-coordinate probe is already minimal (1 evaluation) and cheap
relative to the remaining budget.

## Decision after evaluation 100 (final)
**Budget:** 100 used / 100 total; 0 remaining.
**Evidence:** The greedy integer coordinate descent (evaluations 87-99,
13 calls) tested both in-bounds unit neighbors of every one of the 8
integer coordinates around the incumbent x = [0.31, 0.68, 2.38, 2.47,
5.97, 0.42, 5.37, 7.22, -2.26, -0.94] (y=109.78; integer part
[0,1,2,2,6,0,5,7]). None of the 13 neighbor evaluations improved on
109.78 (values ranged 110.01-122.58, all worse) — the incumbent's
integer coordinates are a local optimum under unit moves on every axis,
within bounds. With the single remaining evaluation (100/100), one more
continuous probe was tried on x8 (shifted by -1, continuing the
direction the Sobol-stage correlation and the incumbent's own negative
x8 suggested was favorable): y=115.78, worse than 109.78. The budget is
now fully exhausted.
**Decision:** Stop. No evaluations remain.
**Hyperparameters:** Not applicable — no further step is being taken.
**Reason:** The evaluation budget stated in the task (100 calls) is
exhausted; this is a hard constraint per the protocol and may not be
exceeded regardless of whether further local improvement might
plausibly exist.
**Next step limit:** 0 (stopping; no budget remains).
**Why this length:** Not applicable.
**Review:** Not applicable — campaign complete.

## Final outcome
- Best objective value found: y = 109.78003932543692 (minimize).
- Best point: x = [0.3113, 0.6828, 2.3828, 2.4714, 5.9707, 0.4181,
  5.3724, 7.2149, -2.2589, -0.9346] (found by CMA-ES at evaluation 66,
  not subsequently improved upon; integer coordinates x0..x7 as rounded
  by the evaluator: [0, 1, 2, 2, 6, 0, 5, 7]).
- Evaluations used: 100 / 100 (budget fully exhausted; harness auto-saved
  `trace.csv` and `meta.json` on the final call, and `harness.save()` was
  also called explicitly after it).
- Method sequence actually used, in order: (1) Sobol space-filling design
  (16 evals) for initial landscape read; (2) CMA-ES seeded at the best
  Sobol point, run to convergence/plateau over two continued blocks
  (70 evals total, same `es` instance carried across both blocks); (3)
  greedy first-improvement coordinate descent on the 8 integer
  coordinates around the CMA-ES incumbent, plus one final continuous
  probe on x8 (14 evals) — confirming the incumbent is a local optimum
  under unit integer moves and small continuous perturbation on the
  budget available.
