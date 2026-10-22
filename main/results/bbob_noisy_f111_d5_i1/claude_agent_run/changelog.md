# Changelog — sandbox_task015

## Evaluation 50
**Change:** Ended Phase 1 (GP-BO) early at evaluation 50 instead of the
planned 80, and moved directly into Phase 2 (noise-averaging exploitation)
with the remaining 50 evaluations instead of the planned 20.
**Reason:** This follows pre-registered deviation criterion #2 in
`roadmap_initial.md` ("if y_best_so_far has not improved over the most
recent 30 consecutive evaluations, end Phase 1 early"). `y_best_so_far`
reached 1120.037 at evaluation 20 and then did not improve at all through
evaluation 50 (30 consecutive non-improving evaluations: idx 21-50), so
Phase 1 was stopped and the remaining budget was reallocated to Phase 2
exactly as pre-committed.

## Evaluation 100 (post-hoc observation, no further evaluations taken)
**Change:** None to the evaluation campaign itself (budget was already
exhausted) — this is a note on how the Phase 2 "pick candidate with lowest
mean of repeats" decision rule behaved in practice, for honesty about its
limits.
**Reason:** The observed y values turned out to be far more volatile than
anticipated when the roadmap was written: at points that produced values
around 1e3-5e3 on one draw, repeat evaluations of the *same* x sometimes
returned values in the 1e5-1e7 range (see `trace.csv`, e.g. eval_idx 52-57
and 58-61 alternate between ~3k-6k and ~3e5-1e6 at the same candidate
point). This is consistent with heavy-tailed/outlier-prone multiplicative
noise rather than the roughly symmetric, bounded-variance noise an
arithmetic-mean-based comparison implicitly assumes. As a result, the
sample mean of 5 repeats per Phase-2 candidate (used only as an internal
aid for picking which candidate to call "most promising") was dominated by
occasional large outliers and did not reliably track the candidates' typical
(e.g. median) behavior; this did not corrupt any recorded evaluation
(every row in trace.csv is a real harness.evaluate() call, untouched), it
only means the Phase-2 "chosen candidate" note in the final report should
be read as a secondary, lower-confidence signal. The harness's own
`y_best_so_far` bookkeeping (minimum single observed y across all 100
real evaluations, = 1120.0367, from evaluation 20) remains the
authoritative, fabrication-free final result and is what is reported as
the campaign's best value.
**Lesson (not an action taken, since budget was exhausted):** under this
kind of heavy-tailed noise, a median (or trimmed-mean) of repeats would
have been a more robust statistic than the arithmetic mean for Phase 2's
candidate comparison.
