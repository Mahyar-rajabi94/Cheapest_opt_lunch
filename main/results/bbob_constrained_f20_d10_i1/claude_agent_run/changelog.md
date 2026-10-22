# Changelog

## Evaluation 60
**Change:** Deviated from the pre-registered CMA-ES-to-Differential-Evolution
stall fallback. Stopped the CMA-ES phase after 4 generations (evaluations
21-60) and used the remaining 40 evaluations (61-100) as plain uniform
random search (`optimizer="random_search_fallback"`) instead of DE
(`optimizer="de"`), which the roadmap had named as the stall fallback.

**Reason:** The stall criterion from the roadmap fired exactly as planned:
the running best merit was 4357.82 after generation 1, improved to
2535.40 after generation 2, and then stayed at 2535.40 through
generations 3 and 4 (evaluations 31-60) -- 3 consecutive generation-end
readings with no improvement, the pre-registered trigger.

However, before invoking DE, I checked `src/optimizers/differential_evolution.py`'s
own docstring, which states DE's population size for this problem is
`pop_size = 10 * dim = 100` -- i.e. DE's own initial population alone is
as large as the *entire* evaluation budget for this task, and the module
documents explicitly that "the first `pop_size` points asked are the
random initial population itself... real mutation and crossover only
begin once all of those have been told back" and that a budget smaller
than `10 * dim` "will never reach a real generation." With only 40
evaluations left at the stall point, DE could not complete even one real
generation; it would silently run as uniform random sampling while being
labeled "de" in the trace, which would misrepresent what the campaign
actually did. I substituted `RandomSearch` directly for the remaining 40
evaluations -- functionally identical to what DE would have done in this
regime, but labeled honestly.

**Outcome:** The random-search fallback (evaluations 61-100) did not
improve on the best merit found by CMA-ES (2535.40 remained the global
best through evaluation 100). This is itself informative: combined with
the very large spread of merit values seen throughout the run (e.g.
Sobol phase ranged from ~4445 up to ~290,445; CMA-ES generations
repeatedly produced batch maxima in the tens of thousands to
~155,943), it suggests the penalized-merit landscape is dominated by a
narrow, hard-to-locate low-penalty/feasible-ish region inside a much
larger strongly-penalized area of the box, consistent with the stated
hidden-constraint structure -- 100 evaluations in 10 dimensions was not
enough for any of the three methods tried (Sobol, CMA-ES, random search)
to reliably zero in on it further than CMA-ES's generation-2 result.
