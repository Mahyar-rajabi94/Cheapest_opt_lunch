# Changelog

No deviations from the pre-registered roadmap occurred. TuRBO-1 ran for
the entire 100-evaluation budget (28 Sobol init evaluations + 72
evolve-phase evaluations in batches of 4) without triggering any of the
three pre-registered deviation criteria:

- Stagnation (>=30 evals without improvement, with >=20 remaining): never
  reached — the longest stagnation streak was well under 30 evals (the
  running best kept ticking down throughout the evolve phase: 0.338 ->
  0.296 -> 0.295 -> 0.289 -> 0.288).
- Repeated trust-region collapse (>2 restarts before eval 80): did not
  occur — the trust region shrank in a controlled way (0.8 -> 0.4 -> 0.2)
  without ever hitting `length_min` and restarting.
- GP-fitting failures: none observed.

This file is otherwise append-only per the protocol; it is being written
once, after the run, to record that no mid-run entries were needed.
