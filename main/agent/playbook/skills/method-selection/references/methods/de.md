# de — Differential Evolution (rand/1/bin, population 10·dim)
**Good for:** multimodal problems with weak structure, separable problems,
when CMA-ES restarts keep landing in the same few basins.
**Weak at:** small budgets. One generation costs 10·dim evaluations, so
2,000 evaluations in 10-D is only about 20 generations. Slow final convergence.
**Use as:** a race candidate on weak-structure landscapes, or a late check
for missed basins. Always polish its best point.
