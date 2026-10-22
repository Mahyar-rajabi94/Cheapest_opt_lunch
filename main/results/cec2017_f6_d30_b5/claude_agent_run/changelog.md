## Evaluation 62
**Change:** Soft-restarted CMA-ES (restart #1), re-centered on the current global best point, keeping the same half-width-2.0 local box convention from the roadmap.
**Reason:** No improvement to the running best for 31 consecutive evaluations, exceeding the pre-registered 20-evaluation stall threshold from `roadmap_initial.md`; best-so-far was stuck at y=610.416. Per the pre-registered deviation criterion, a fresh CMA-ES instance was started, re-centered on the current global best, rather than continuing to spend budget on a stalled search.

## Evaluation 90
**Change:** Switched the active optimizer to a local random-perturbation sampler (Gaussian, scale=0.25, clipped to bounds, centered on the current global best) for the final 10 evaluations, instead of continuing with CMA-ES.
**Reason:** Only 10 evaluations remained, fewer than the active CMA-ES instance's population size (14). Per the roadmap's pre-registered fallback, a partial last generation would never be consumed by CMA-ES's `tell()` (which only updates on a full generation), so the remaining budget was spent as small random perturbations around the current best point instead of being wasted on an unconsumed partial generation.
