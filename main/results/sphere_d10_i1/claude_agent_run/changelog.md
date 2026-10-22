No deviations from the plan in `roadmap_initial.md` occurred during this run.

TuRBO ran its full planned allocation (80 evaluations) without its trust
region collapsing below `length_min` and without a 30-evaluation plateau in
the running best, so neither early-switch criterion was triggered. The
hand-off to BOBYQA at evaluation 80 and the local-box construction around
TuRBO's best point happened exactly as described in the roadmap (the use of
a shrunk local box to emulate "starting BOBYQA at a specific point," since
the BOBYQA wrapper only exposes the midpoint of whatever bounds it is given,
was already anticipated by the roadmap's phase-2 description and is an
implementation detail of the pre-registered plan, not a deviation from it).
BOBYQA used its full remaining allocation (20 evaluations) and did not
exhaust/converge early, so the restart criterion was never exercised either.
