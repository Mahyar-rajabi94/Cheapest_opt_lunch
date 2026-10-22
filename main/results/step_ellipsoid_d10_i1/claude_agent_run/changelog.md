## Evaluation 52
**Change:** Ended Phase 1 (GP-BO) early, after 52 evaluations instead of the
nominally planned ~80, and moved into Phase 2 (BOBYQA local refinement)
with the remaining 48 evaluations.
**Reason:** This matches the pre-registered deviation criterion in
`roadmap_initial.md` ("if ... the running best value during Phase 1 has not
improved for 15 consecutive GP-BO-proposed evaluations, stop Phase 1 early
and move into Phase 2"). After the 22-point Sobol initial design, GP-BO's
running best value had not improved for 15 consecutive adaptive
(LogEI-guided) evaluations (best stuck at y=193.196 since shortly after
initialization through eval 52), indicating the GP-BO search had stalled in
its current basin. Rather than continue spending budget on an apparently
exhausted global-search phase, the remaining 48 evaluations were
reallocated to BOBYQA's local trust-region refinement around the Phase-1
incumbent, which is the intended complementary use of the saved budget
described in the roadmap. This freed up more evaluations for Phase 2 than
the nominal ~20, which is consistent with the roadmap's note that Phase 2
would receive "more if Phase 1 ends early."

Outcome: BOBYQA's local refinement (starting from a 2.0-wide box per
coordinate centered on the GP-BO incumbent, x_best with y=193.196) improved
the best found value substantially further, down to y=177.445, using all 48
of its remaining evaluations (none of the fallback conditions — GP-BO
exception, or BOBYQA's internal solver thread exhausting via
`OptimizerExhausted` — were triggered, so no further deviation from the
roadmap was needed).
