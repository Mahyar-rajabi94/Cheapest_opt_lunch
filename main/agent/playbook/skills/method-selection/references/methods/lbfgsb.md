# lbfgsb — L-BFGS-B with finite-difference gradients
**Good for:** smooth problems; fastest local convergence when gradients are
reliable (P5 cosine > 0.9).
**Weak at:** noise, kinks, plateaus; multimodal problems (finds the nearest
local optimum only). Each gradient costs about dim+1 evaluations.
**Finished:** `last_event = exhausted` (converged). Polishing the same point again is useless.
**Use as:** `polish` from incumbent or active_best.
