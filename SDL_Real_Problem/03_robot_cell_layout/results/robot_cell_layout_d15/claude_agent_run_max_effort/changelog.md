# Changelog

This is an append-only log. Per `Agent.md` section (c), an entry is
required each time the campaign actually deviates from
`roadmap_initial.md`'s pre-registered plan. In this run, the plan's five
quantitative criteria (D1-D5) were evaluated against real data at every
designated checkpoint, and **none of them fired**: the campaign ran
exactly as pre-registered (Phase 1 scan -> continue combo A through both
Phase-2 chunks -> Phase-3 Nelder-Mead polish to completion). The entries
below record those checkpoint evaluations and the evidence behind each
"no deviation" decision, followed by one honest post-hoc observation
about the final result.

## Evaluation 32 (end of Phase 1)
**Change:** none -- checkpoint only.
**Reason:** Phase 1 (32 evaluations, `combo_scan_sobol`) completed as
planned, all 32 drawn flag-combinations distinct. Best combo found: flags
`[1,1,0,0,1,1]` at y=1847.269 ("combo A"); best *distinct* runner-up:
flags `[1,0,0,1,0,1]` at y=2128.604 ("combo B"); worst value observed
anywhere in Phase 1: y=26199.634. Proceeding to Phase 2 on combo A per
the roadmap.

## Evaluation 62 (end of Phase-2 chunk 1: combo A, 30 evaluations)
**Change:** none -- criteria D1/D2/D3 evaluated, none triggered.
**Reason:** Combo A's best y improved from 1847.269 (Phase-1 incumbent)
to 1254.184 over this chunk -- a 32.1% relative improvement, well above
D1's 1% threshold, so no switch to combo B was warranted. CMA-ES's sigma
did not collapse below 1% of sigma0 (D2 not triggered: `stop_reason` =
`chunk_complete`), and no 3 consecutive evaluations landed near the
campaign's worst-observed value (D3 not triggered). Per the
pre-registered plan, continued combo A for the second, 18-evaluation
chunk rather than switching.

## Evaluation 80 (end of Phase-2 chunk 2: combo A, 18 more evaluations)
**Change:** none -- checkpoint only.
**Reason:** Combo A's best y improved further to 1233.350
(`stop_reason` = `chunk_complete`; D2/D3 not triggered). Phase 2
concluded with combo `[1,1,0,0,1,1]` as the winner and y=1233.350 as the
incumbent, handed to Phase 3 (no switch to combo B was ever needed).

## Evaluation 100 (end of Phase 3: local polish)
**Change:** none -- criteria D4/D5 evaluated, D5 not triggered.
**Reason:** 20 evaluations remained when Phase 3 started, below D4's
`nm_threshold=40`, so Nelder-Mead was used (not BOBYQA) exactly as
pre-registered. It ran for the full 20 evaluations without ever raising
`OptimizerExhausted`, so D5's restart-with-jitter logic was never
invoked -- there was no "leftover" budget to redirect. Final best
y=1208.505, same combo `[1,1,0,0,1,1]`. The very last evaluation (#100)
was still a small improvement over #99 (1208.638 -> 1208.505), i.e.
Nelder-Mead had not fully converged when the 100-evaluation budget ran
out; no further evaluations were available to continue it.

## Post-hoc observation (not a deviation -- the campaign had already ended)
All 100 observed y-values across every phase fall in the range
1208.505-26199.634. For a "mixer-loading cycle time in seconds" (the
stated objective), this magnitude is implausible for a genuinely
feasible robot cycle -- it is far more consistent with the task
description's warning about "heavily penalized regions where the layout
is infeasible or the arm/CNC collide." The steady, monotonic improvement
within every phase (1847.269 -> 1254.184 -> 1233.350 -> 1208.505) shows
this was not a flat/saturated penalty plateau (exactly why D3 never
fired) but one graded with configuration, so the optimizers were
correctly descending a real gradient throughout -- evidently just not far
enough, within 100 evaluations total, to fully exit the penalized region
and reach a genuinely feasible, "normal-magnitude" cycle time. This
conclusion is drawn only from the y-values this campaign itself observed
via real `evaluate()` calls (per the pattern above: large, graded,
monotonically-decreasing-but-still-large), not from any inspection of
`objective.py` or other harness source, which was not done.
