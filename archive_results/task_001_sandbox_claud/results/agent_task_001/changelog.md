# Changelog — Task 001

Append-only. One entry per actual deviation from `roadmap_initial.md`,
added as it happens during the campaign.

## Evaluation 1946
**Change:** Switched local refinement from lbfgsb_local to the next method in the cascade.
**Reason:** lbfgsb_local raised OptimizerExhausted after 396 evals starting from eval 1550, with best=149.15 (global best=149.15) (roadmap criteria 4/5).
