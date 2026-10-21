---
name: method-selection
description: Use when choosing which method to switch to, restart, race or polish with, and with what start.
---

# Method selection

Pick from `problem.allowed_methods`. A card for each method is in
`references/methods/`. In short:

| Situation (evidence) | First choice | Alternatives |
|---|---|---|
| Smooth single basin (probing skill) | `polish` `lbfgsb` from incumbent | `bobyqa` if gradients look unreliable |
| One basin, ill-conditioned or curved | `cma_es` from incumbent, then polish | race `cma_es` vs `bobyqa` |
| Several basins, global structure | `cma_es`, then IPOP restarts (`diverse`, popsize ×2) + polish each | `de` |
| Weak or no global structure | `cma_es` `popsize_factor` 2–4 from `center` | race `cma_es` / `de` / `turbo` |
| Expensive evaluations, budget < 30·dim | `turbo` (or `gp_bo` if budget ≤ 150) | `bobyqa` |
| Rough or noisy (P5 cosine < 0.5) | `cma_es` | `bobyqa`, `de`. Never `lbfgsb` |
| No information at all and no probes | `cma_es` from center | |

## Start points
- `incumbent` (default for switch): continue from the best point found.
- `active_best` (polish): refine the best point of the method that just ran,
  e.g. a restart that found a new basin.
- `diverse` (default for restart): a good point far from the incumbent. Use it to look for other basins.
- `center` / `random`: cold start. Only when you deliberately want to
  ignore what has been found (e.g. weak structure, broad exploration).

## Racing
Use `race` when two or three methods are plausible and the probes don't
separate them. Give it about 10–20% of the budget (it is capped at 30%).
Include at least one method of a different family: local vs population vs
model-based. The winner keeps its state and becomes the active method.

## Local solvers compared
- `lbfgsb`: fastest on smooth problems, about dim+1 evaluations per gradient.
  Breaks on noise or kinks.
- `bobyqa`: builds a quadratic model, needs no gradients. Robust and nearly as fast on
  smooth problems. It needs 2·dim+1 evaluations to start.
- `nelder_mead`: slow in dim > 5. Use it only as a last resort.
