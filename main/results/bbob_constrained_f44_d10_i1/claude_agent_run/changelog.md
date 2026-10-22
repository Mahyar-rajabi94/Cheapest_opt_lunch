# Changelog

## Evaluation 25
**Change:** Ended Phase 1 (GP-BO) early and switched to Phase 2 (BOBYQA
local refinement) at evaluation 25, instead of the nominally planned ~85.
**Reason:** Deviation criterion 1 from the roadmap fired: the running best
merit had not improved over the most recent 20 consecutive evaluations
(best stuck at y=-563.2047555402613, first reached at evaluation 6, through
evaluation 25). GP-BO's 22-point Sobol initial design already landed one
point in a good low-merit region (eval 6), but none of the handful of
LogEI-acquisition points that followed (evals 23-25) improved on it, and
several evaluations in this window returned very large merit values
(1e3-1e5 range), consistent with the hidden-constraint penalty hitting
infeasible candidates rather than the acquisition function converging on
better feasible structure. Per the pre-registered plan, this plateau was
treated as the trigger to hand the remaining budget to local refinement
around the current incumbent rather than continuing to spend evaluations
on GP-BO acquisition in a region it was not improving.

## Evaluation 26
**Change:** Started BOBYQA (`bobyqa_local`) with bounds tightened to
`[x_best - 1.0, x_best + 1.0]` per coordinate (clipped to the original
[-5, 5] box), centered on the Phase-1 incumbent `x_best` from evaluation 6,
so that BOBYQA's fixed box-midpoint starting convention effectively starts
it at the incumbent. This follows the roadmap's Phase 2 plan exactly (no
deviation from the pre-registered method, just recording the restart
trigger and parameters used).

## Evaluation 100 (end of run)
**Outcome note (not a deviation):** BOBYQA used the entire remaining
budget (evaluations 26-100, 75 evaluations) without converging or raising
`OptimizerExhausted`, so the planned fallback restart / Nelder-Mead
handoff (deviation criteria 2-3) was never triggered. Best merit improved
monotonically from -620.58 (eval 26) to -1508.01 (eval 100), a ~2.7x
improvement over the Phase-1 incumbent, so no further changes were made.
