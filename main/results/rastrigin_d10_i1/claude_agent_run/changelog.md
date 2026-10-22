# Changelog — agent_run

## Evaluation 88
**Change:** Ended the TuRBO phase at evaluation 88 (rather than exactly at
the planned 85-evaluation cap) and switched to BOBYQA for the remaining
12 evaluations (rather than the planned 15-evaluation floor).
**Reason:** TuRBO proposes and evaluates points in fixed batches of 4
(`batch_size=4`), so the "stop once `evaluations_used >= 85`" check is
only tested between batches: at eval 84 the check passed (84 < 85), so
one more full batch of 4 was run, landing on eval 88 before the check
fired again. Across all 85-88 evaluations, the global best-so-far
improved essentially monotonically (1689.0 at eval 1 down to 1113.9 at
eval 78, holding through eval 88 — see `trace.csv`), with no stall of
>=30 evaluations and no trust-region restart observed at any point, so
neither of the roadmap's early-handoff triggers fired; TuRBO simply ran
to (approximately) its allotted cap as the default path. The 3-evaluation
batch-size overrun shrank the BOBYQA polish phase from the planned
15-evaluation floor to 12 evaluations. This is a minor, mechanical
deviation from the stated numeric split (not a change of method or
strategy), noted here because the roadmap gave explicit numeric floors.

## Evaluation 100 (final observation, no further change made)
**Change:** None — reporting an observation, not a deviation in method.
**Reason:** The BOBYQA polish phase (evals 89-100) never improved on
TuRBO's best (y=1113.922083, found at eval 78); BOBYQA's own trajectory
stayed in the 1242-1345 range throughout its 12 evaluations. This
confirms the caveat flagged in `roadmap_initial.md`: this project's
BOBYQA wrapper always starts its internal trust-region search from the
box midpoint (its own fixed convention, not a warm-startable interface),
so it was effectively a fresh, independent local search from the origin
rather than a refinement of TuRBO's best point, and 12 evaluations from
a cold start were not enough for it to reach, let alone beat, a region
TuRBO had already spent 88 evaluations converging toward. The
`OptimizerExhausted`-triggered TuRBO-resumption fallback was not needed
(BOBYQA never exhausted; the budget simply ran out while it was still
proposing points). The final reported best is therefore TuRBO's point,
y=1113.922083, from evaluation 78.
