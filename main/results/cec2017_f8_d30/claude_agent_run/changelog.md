# Changelog

No deviations from `roadmap_initial.md` occurred during this run.

The plan (GP-BO with its own standard 2*(dim+1)=62-point Sobol' initial
design, followed by 38 acquisition-driven LogEI evaluations) executed
end-to-end for the full 100-evaluation budget:

- `GPBO.ask()`/`tell()` never raised an exception, so deviation
  criterion 1 (fall back to Sobol' on internal failure) was never
  triggered.
- Best value at the end of the Sobol' init phase (eval 62) was
  1462.077; by eval 77 (15 acquisition-driven evaluations in) the best
  value had already improved to below that level, so deviation
  criterion 2 (switch to BOBYQA local refinement after 15
  non-improving acquisition evals) was never triggered.
- The budget was never exceeded; only one `AgentTaskHarness` instance
  was created.

Final best value after all 100 evaluations: 1150.993407 (see
`trace.csv` / `meta.json`).
