
## Evaluation 75
**Change:** Switched from Phase 1 (full-box CMA-ES) to Phase 2 (shrunk-box CMA-ES restart around incumbent best).
**Reason:** Best value did not strictly improve over the last 2 full generations (30 evals): best after gen 3 was 617.705974, best after gen 5 is 617.705974. Per roadmap deviation criterion 1, restarting on a shrunk box centered at the incumbent best (half-width 1.5, clipped to [-5,5]) for the remaining 25 evals.

## Evaluation 90
**Change:** Switched to one-point-at-a-time asks for the final partial generation.
**Reason:** Only 10 evals remained, fewer than a full CMA-ES generation (15); per roadmap deviation criterion 2, spending them individually rather than leaving budget unused.
