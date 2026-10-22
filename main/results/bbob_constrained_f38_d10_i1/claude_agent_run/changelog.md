## Evaluation 81
**Change:** Replaced the planned Phase 3 method (BOBYQA local polish,
restarted from the best point found by phases 1-2) with scipy's bounded
Powell method, driven closed-loop against `harness.evaluate` directly,
seeded at the same best point.
**Reason:** Before writing any code, inspection of this project's own
`src/optimizers/bobyqa.py` (public optimizer module, not task/objective
source) showed that its `BOBYQA` wrapper hard-codes its starting point
`x0` to the box midpoint inside `__init__` and exposes no parameter to
start from an arbitrary point — the closed-loop solver thread is already
running by the time `ask`/`tell` are available. Restarting BOBYQA from
the best point found by phases 1-2 (rather than the uninformative box
midpoint) was the explicit intent of the roadmap's Phase 3, and the
roadmap itself pre-registered this exact fallback ("drive a local
refiner myself... any such substitution will be logged in
changelog.md") for this situation. Since editing the shared
`bobyqa.py` module to accept a custom x0 was ruled out in the roadmap
(avoid hand-modifying shared project code for a one-off run), Powell
(bounded, `scipy.optimize.minimize(method="Powell")`) was used instead:
it is a classical derivative-free local solver appropriate for final
polish near a basin, accepts an arbitrary starting point and box
bounds natively, and was driven for the remaining 20 evaluations
(evals 81-100) under the `powell_polish` optimizer label.
