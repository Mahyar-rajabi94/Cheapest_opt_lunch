---
name: stall-recovery
description: Use when the best value has not improved for 2+ rounds, or the active method reports converged or exhausted, while budget remains.
---

# Stall recovery

A stall means the active method has finished its job. Decide whether the
current basin is done and the remaining budget should go to finding a better one.

## Definitions
- **Stalled:** `stall_evals` ≥ 2 × `round_size`.
- **Finished:** `last_event` is `exhausted` (local solver converged) or
  `converged` (CMA-ES step size collapsed).

## Rules
1. **Global method stalled or converged, never polished** → `polish` with
   `start="active_best"` using `lbfgsb` (if gradients are reliable) or
   `bobyqa`. A converged population method is usually close to a local
   optimum but not at it. Polishing typically gains several digits in a few
   dozen evaluations.
2. **Local solver exhausted** → this basin is done. With ≥ 3 rounds of
   budget left: `restart` `cma_es` with `start="diverse"` and
   `popsize_factor` 2 to look for a better basin. Each later restart doubles
   the population again (2 → 4 → 8; this is the IPOP strategy).
3. **Restarts keep finding the same value** (two restarts in a row whose
   polished best is within 1e-6 relative of the incumbent) → evidence of a
   single basin, or that the incumbent is the global optimum. With much budget
   left, one `race` of `turbo` vs `de` is a reasonable last check. Otherwise
   `stop`.
4. **Restarts find worse values every time but the landscape has many
   basins** → the incumbent's basin is good. Spend the rest refining
   (`polish` with a different local method) or racing population methods.
5. **Near the end** (< 1 round left) → `polish` from the incumbent, never explore.

## Anti-patterns (seen in earlier agent runs)
- Restarting a local solver from the box center after a global phase. It
  throws away everything the global phase found. Always warm-start.
- Running a population method for 1,000+ evaluations on a landscape the
  probes flagged as a smooth single basin.
- Polishing the same converged incumbent again and again. If a local solver
  just exhausted at the incumbent, polishing it again gains nothing.
