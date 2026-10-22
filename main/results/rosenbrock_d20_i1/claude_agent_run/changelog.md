## Evaluation 87
**Change:** Stopped GP-BO, switched to BOBYQA local refinement for the remaining budget.
**Reason:** y_best_so_far has not improved over the last 15 consecutive evaluations while in GP-BO's acquisition-driven phase (n_used=87 >= 70), meeting the pre-registered plateau criterion in roadmap_initial.md. Switching to a cheap local polish around the current best point.

