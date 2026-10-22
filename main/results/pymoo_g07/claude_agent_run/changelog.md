# Changelog

No deviations from the pre-registered roadmap occurred during this run.

- Phase 1 (GP-BO, `gp_bo`) ran its full planned allocation of 75 evaluations
  (including its internal 22-point Sobol initial design) without triggering the
  20-evaluation no-improvement early-stop criterion, and without any GP-fit
  failures or degenerate repeated proposals.
- Phase 2 (BOBYQA local refinement, `bobyqa_refine`) was started from the best
  point found by GP-BO (merit ≈ 115.81) at evaluation 76, exactly as planned, and
  ran to the budget limit (evaluation 100) without converging early (no
  `OptimizerExhausted`), so the restart-from-second-best contingency was never
  needed.
- A few individual BOBYQA trust-region probe points produced merit values far
  above the surrounding local range (e.g. ~694 at one step vs. a local best
  around 85-120), consistent with the task's warning that large jumps usually
  indicate the penalty term from a heavily infeasible probe rather than real
  objective structure. These were treated only as evidence to avoid that
  direction (BOBYQA's own trust-region model handled this automatically via its
  normal trial/accept-reject mechanics) and not investigated further, per the
  task instructions not to try to separate penalty from true objective.

Final result: best merit = 85.05351732152964, found at evaluation 100 (the final
evaluation), using the full 100-evaluation budget.
