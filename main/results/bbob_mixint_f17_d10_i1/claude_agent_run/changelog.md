# Changelog — sandbox_task027

## Evaluation 20
**Change:** Completed Phase A (sobol_scan) as planned for its full 20-eval
allocation; did not shorten it.
**Reason:** The dominance heuristic from the roadmap did not trigger — best
y=-32.06 beat the 2nd-best (y=-31.32) by only 0.75, while the spread across
the rest of the batch was ~1194, so the landscape did not show one point
decisively dominating. Proceeded to Phase B with the full budget as planned.

## Evaluation 31
**Change:** Stopped Phase B (intcoord_descent) after a single full pass over
the 8 integer coordinates (11 evals used, well under the 45-eval cap),
instead of continuing to iterate passes.
**Reason:** Matches the pre-registered deviation criterion: "if a full
coordinate pass yields zero improvement, stop Phase B immediately." The
single pass from the Phase A incumbent (y=-32.06) found no improving
neighbor among x0..x7 (best after the pass was still y=-32.06), so Phase B
was ended early and the remaining ~34 planned-for-B evaluations were
reallocated to Phase C.

## Evaluation 51
**Change:** Phase C (continuous_pattern_search) ran its full 20-eval cap and
was stopped by the cap, not by the step-size threshold (step was still
0.156, above the planned 0.05 floor, when the cap was reached).
**Reason:** Compass search on x8/x9 kept finding small improvements each
round (y went from -32.06 to -69.95), so the step never needed to shrink
below 0.05; the phase simply used its full allocated budget productively.
This is a deviation from the exact stopping rule in the roadmap (cap vs.
step-floor) but not from its intent (give Phase C up to ~20 evals).

## Evaluation 59
**Change:** In Phase D's first short integer-coordinate-descent sub-round
(round2, 8-eval budget), found a further improvement by moving x5 from 0 to
1 (all other coordinates held at the Phase C incumbent), improving y from
-69.95 to -70.84.
**Reason:** This is exactly the interaction Phase D was designed to catch —
a better continuous point (from Phase C) opened up a different integer
neighbor that hadn't been tried from the original Phase B incumbent.

## Evaluation 100
**Change / observed limitation:** From evaluation ~60 onward, the Phase D
alternation loop (short intcoord_descent_round2 / continuous_pattern_search_round2
sub-rounds) repeated the same neighbor evaluations multiple times without
finding further improvement, because each sub-round re-initialized from the
unchanged best_x and the `round_idx > 6` safety break was not reached before
the overall 100-evaluation budget ran out. All of these evaluations are real
`harness.evaluate()` calls (no fabricated data) and used only budget that
was already earmarked for Phase D, so the budget constraint was respected,
but roughly the last 35 evaluations were effectively redundant re-checks of
the same 12 neighbor points rather than new exploration. Noting this
honestly as a design limitation of the Phase D loop's termination condition:
a future run should detect "no improvement across a full round" and break
immediately, rather than relying solely on `round_idx > 6`.
**Reason for keeping the result as-is:** No further evaluations remained
once this was noticed (budget exhausted at eval 100); the final incumbent
(y=-70.8353, x=[1,1,0,2,3,1,15,9,-0.85588,2.81281]) is still the genuine
best point found and is unaffected by the wasted re-checks.
