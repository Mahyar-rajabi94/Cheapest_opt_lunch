# Initial roadmap: psa_v10_blackbox_run1

Written 2026-10-22, before the first evaluation. Not to be edited after this point;
deviations go in `changelog.md`.

## Confirmed problem definition

Source: `main/task_definitions/psa_v10_blackbox_run1.md` (Confirmed by user, revision 2).

- Objective: `BlackBox()(x)` from `C:\Users\rahim\Desktop\psa_v10_blackbox.py`, **minimize**.
- Search space: a discrete set of materials, each described by 8 features scaled to
  [0, 1]. Any x in [0, 1]^8 may be evaluated; the evaluator returns the value of the
  material x maps to. No materials list is available to the agent.
- Constraints: box only. Noise: none (deterministic).
- Budget: 2000 evaluations through `BlackBoxHarness` (`src/agent_harness/harness.py`),
  task_id `psa_v10_blackbox`. Every call counts.
- Known structure: piecewise-constant over the box (flat within each material's
  region). Unknown: number and layout of materials, best value, regularity of the
  value-vs-feature relation.
- Prior data: none.

## Method plan and budget allocation

| Phase | Label(s) | Evals (approx.) | Method |
|---|---|---|---|
| A | `sobol` | 256 | Scrambled Sobol' (repo `SobolSearch`, seed 0). 256 = 2^8 for balanced coverage. |
| B | `cma_es_r<k>` | 800 | pycma CMA-ES with restarts. Restart k starts at the k-th best *distinct-valued* point observed so far (a different value means a different material), sigma0 = 0.15, default population (10 in 8-D), box bounds handled by pycma. |
| C | `de_seeded` | 600 | DE/rand-to-best/1/bin (F = 0.7, CR = 0.9, population 40), population seeded from the 40 best distinct observed points using their already-observed values (no re-evaluation). |
| D | `local_search` | remainder (~344) | (1+1) Gaussian local search cycling over the 5 best distinct materials, step 0.08 per coordinate; step halves after 15 consecutive failures at a centre (floor 0.01) and resets to 0.08 on success. |

Throughout: proposals are clipped to [0, 1]. An x already evaluated (exact match) is
not re-evaluated; its recorded value is passed back to the optimizer without
spending budget and without a trace row. At most 50 such cache hits per phase,
after which the phase ends.

## Why these methods

- No gradients exist (flat regions, jumps between them), so L-BFGS-B, BOBYQA and
  Nelder–Mead are excluded: finite differences return zero and simplices collapse.
- GP-BO assumes a smooth surface and its cost grows with 2000 points; not used.
- Sobol' gives an even, model-free first map of which regions hold good materials.
- CMA-ES and DE use only the ranking of values, so flat regions do not mislead them
  as they would a gradient method; a population with step sizes of 0.1–0.2 can
  jump between material regions. Restarting from different good materials covers
  several basins.
- The final local search spends the remainder on the neighbourhoods of the best
  few materials, where nearby materials are most likely to be slightly better.

## Pre-registered deviation criteria

1. **CMA-ES stagnation (restart, within plan):** a CMA-ES run is restarted when its
   own best has not improved in 60 evaluations, sigma < 0.01, or pycma reports a
   stop condition. This is part of the plan, not a deviation.
2. **CMA-ES not helping:** if after 300 Phase-B evaluations the global best is still
   the Sobol' best, move the rest of Phase B's allocation to Phase C (DE).
3. **CMA-ES still improving:** if the global best improved within the last 150
   evaluations of Phase B, extend Phase B by 200 evaluations taken from Phase C.
4. **Re-hitting known materials:** if, in any phase after A, fewer than 10 of the
   last 100 evaluations return a value not seen before, end that phase and spend
   the next 100 evaluations on continued Sobol' exploration (`sobol_explore`), then
   resume the next phase.
5. **DE plateau:** if DE does not improve the global best for 250 evaluations, move
   to Phase D early.
6. **Cache-hit limit:** a phase reaching 50 cache hits ends early (see above).

Any of criteria 2–6 that fires is logged in `changelog.md` with the evaluation count
and evidence.
