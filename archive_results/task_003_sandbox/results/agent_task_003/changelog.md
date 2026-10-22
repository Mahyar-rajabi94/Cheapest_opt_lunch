# Changelog — Task 003

Append-only log of every actual deviation from `roadmap_initial.md`, in the
order it happened. Entries are added live as the campaign runs; nothing
here is edited or removed after the fact.

## Evaluation 128
**Change:** Ended Sobol' scan at 128 evaluations instead of the planned 256; reallocated the saved 128 evaluations to the cma_es phase.
**Reason:** D4 triggered: checked at the 64/128 checkpoint (the earliest point this can be decided without spending the second half first) -- best after 64 points was 3.71156e+07, best after 128 was 3.71156e+07 (relative improvement 0.000e+00 <= 1e-6 threshold), indicating the scan had stopped finding materially better regions.

## Evaluation 128
**Change:** Seeded the main CMA-ES run's mean at the Sobol' scan's best point (not the box midpoint), and drove pycma directly instead of the installed CMAES wrapper for this and all restarts.
**Reason:** Pre-registered in the roadmap ('seed its initial mean using the Sobol scan'), but the installed src.optimizers.cma_es.CMAES wrapper hardcodes x0 to the box midpoint with no override, so a thin direct use of pycma (cma.CMAEvolutionStrategy) was needed to honor this -- same algorithm and same defaults (population size, learning rates, bounds handling) as the wrapper, only x0/sigma0 are parameterized.
