## Evaluation 42
**Change:** Ended phase 2 (gp_bo) early and moved directly to phase 3 (confirm_refine),
instead of continuing GP-BO up to the ~70-evaluation allotment planned in
`roadmap_initial.md`.
**Reason:** Best-so-far value (362.380744) had not improved over the last 20
consecutive GP-BO evaluations (evaluations 22-41), meeting the pre-registered
plateau criterion from the roadmap ("if the best-so-far value has not improved
for 20 consecutive GP-BO evaluations, stop phase 2 early and move to phase 3
with whatever budget remains"). This freed 58 evaluations (instead of the
originally planned 18) for phase 3's replication/confirmation and local
refinement, which is a larger confirmation+refinement budget than
pre-registered but follows directly from the roadmap's own stated rule for
reallocating unused phase-2 budget.
