# Changelog

## Evaluation 28
**Change:** None (plan check, no deviation) — deciding whether to extend
Phase 1 (`sobol_scan`) by 10 evals per the roadmap's Deviation Criterion 1.
**Reason:** After the planned 28 Sobol-scan evaluations, best_y = 2597.2495
and median_y = 7233.3536 across the batch. Since best_y was clearly better
than the median (not stuck at the edge of a flat sample), the trigger
condition ("best not at least as good as the median") was false, so Phase 1
was **not** extended and the full 50-eval allocation for Phase 2 was kept.

## Evaluation 64
**Change:** Stopped Phase 2 (`cma_es_seeded`) early, after 36 of its planned
50 evaluations, and moved into Phase 3 (`local_polish`) with the recovered
14 evaluations added to Phase 3's budget (36 remaining instead of the
originally planned 22).
**Reason:** Per the roadmap's explicit Deviation Criterion 2 ("if the
running best has not improved for 20 consecutive evaluations, stop Phase 2
early"). Observed data: CMA-ES's best value improved generation-to-generation
(2597.25 -> 1777.11 -> 1367.38) but then held at 1367.379272 through an
entire additional generation, reaching a streak of 22 consecutive
evaluations with no improvement (phase2_used 14 through 36) before the
threshold of 20 was hit — logged at the point the streak counter crossed 20.

## Evaluation 100
**Change:** No further deviation — Phase 3's Powell local search (started
from the best point carried out of Phase 2, with IK-branch flags fixed at
their rounded values [0,0,1,1,1,1]) ran to the full reallocated budget of 36
evaluations without raising an exception or signalling convergence early, so
the planned Nelder-Mead fallback was not needed. Final best value:
y = 1347.431047, found at evaluation 95 and unchanged (within Powell's
internal step tolerance) through evaluation 100, at which point the total
budget of 100 was exhausted and `harness.save()` wrote the final trace.
