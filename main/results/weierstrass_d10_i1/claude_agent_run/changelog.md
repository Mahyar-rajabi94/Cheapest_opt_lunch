## Evaluation 65
**Change:** Switched the active method from GP-BO (LogEI acquisition loop)
to `local_polish`: a coordinate-wise random-step local search around the
current incumbent (Gaussian perturbation, std = 3% of each coordinate's
range, clipped to bounds), for the remaining budget.

**Reason:** This is the pre-registered stagnation trigger from
`roadmap_initial.md` (criterion 2). `y_best_so_far` had not improved since
evaluation 25 (best = 86.410039, reached during the GP-BO acquisition
phase). By evaluation 65 — 40 consecutive evaluations without improvement,
all of them past the 22-point initial Sobol design — the trigger condition
(stale >= 40, and >= 15 evaluations of budget remaining) was met, with 35
evaluations left in the budget. The GP-BO acquisition loop was proposing
points that repeatedly landed in much worse regions of the space (y mostly
in the 100-220 range against a best of 86.41), suggesting the LogEI
acquisition — likely hampered by a GP surrogate fit with relatively little
data (43 points) over a 10-d box — was not successfully exploiting the
neighborhood of the best-known point. A tight local search directly around
the incumbent was chosen as a cheap way to spend the remaining budget on
exploitation rather than continuing an acquisition strategy that had
plateaued.

**Outcome:** The local polish phase (evaluations 66-100) did not improve
on the incumbent best of y = 86.410039 (closest approach: y = 88.940745 at
evaluation 83). The final reported best for the campaign is the GP-BO
incumbent, y = 86.410039, found at evaluation 25.
