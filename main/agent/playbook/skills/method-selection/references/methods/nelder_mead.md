# nelder_mead — Nelder-Mead simplex (adaptive)
**Good for:** low dimension (≤ 5), non-smooth local refinement.
**Weak at:** dim > 5 (slow, can stall far from the optimum).
**Use as:** last-resort `polish` when lbfgsb and bobyqa both struggled.
