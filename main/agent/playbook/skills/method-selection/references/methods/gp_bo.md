# gp_bo — Gaussian-process BO with log-expected improvement
**Good for:** very expensive evaluations and budgets ≤ ~150. The most
sample-efficient option early on.
**Weak at:** more than a few hundred evaluations (exact-GP cost grows cubically per step).
**Note:** disallowed automatically for cheap problems with large budgets.
