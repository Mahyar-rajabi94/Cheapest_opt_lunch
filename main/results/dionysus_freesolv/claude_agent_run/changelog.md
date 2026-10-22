# Changelog

(append-only; one entry per deviation from `roadmap_initial.md`)

## Evaluation 125
**Change:** Increased the LCB exploration weight `k` from 1.0 to 2.0 for all
subsequent Phase 2 batches (from evaluation 126 onward).
**Reason:** Plateau criterion #1 from `roadmap_initial.md` triggered: the
running best `y` was unchanged (-23.62, found during Phase 1 diverse-init at
candidate idx 409) across two consecutive Phase 2 batches (checkpoint at
n=75, y_best=-23.62; checkpoint at n=125, y_best=-23.62 — no improvement
over 50 evaluations). As pre-registered, this triggered a switch toward more
exploration before considering the more drastic fallback (pure diversity
sampling). The Random Forest surrogate's OOB R^2 stayed in a healthy range
throughout (0.48-0.50), so the model was not degenerate (deviation criterion
#2 did not apply) — the plateau looks like a genuinely hard pool (the
diverse-init phase may have already found a global or near-global minimum
among these 509 molecules) rather than a broken surrogate.

## Evaluation 275 (final)
**Change:** None — campaign concluded at budget exhaustion (275/275
evaluations used), as planned in Phase 2.
**Reason:** Despite the k=2.0 exploration bump at evaluation 125, the best
value (y=-23.62 at candidate idx 409, SMILES `OC[C@@H](O)[C@@H](O)[C@H](O)
[C@H](O)CO`, found during the very first diverse-init batch) was never
improved upon through evaluation 275, across 150 further surrogate-guided
evaluations with progressively higher exploration weight. Per-batch minimum
y values in Phase 2 ranged from -18.72 down to roughly -6 to -12 in later
batches, all worse than the Phase 1 best, suggesting the diverse initial
sample had already located the (or a) best region of the pool for this
property, and the remaining budget mainly served to confirm this rather than
to find further improvement. No further deviation criteria (degenerate
surrogate, pool exhaustion) were triggered, so the plan concluded normally
at budget exhaustion rather than stopping early.
