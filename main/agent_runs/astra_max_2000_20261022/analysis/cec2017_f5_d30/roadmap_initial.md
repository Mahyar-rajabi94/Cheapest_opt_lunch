# First optimization block

Minimize the deterministic, sealed scalar black box in 30 real coordinates,
each bounded by [-100, 100], with no other constraints. The hard budget is
2000 calls, including initialization and repeated points. There are no prior
observations. Assume only the supplied determinism; smoothness and suitability
of a local quadratic model are hypotheses to assess, not known properties.

Use Py-BOBYQA 1.5.0 through a campaign-local queue adapter, starting at the
box midpoint x = 0. Select npt = 61 (2d+1 interpolation points),
scaling_within_bounds = True, rhobeg = 0.1 (20 original coordinate units),
rhoend = 1e-8 (2e-6 original units), objfun_has_noise = False,
seek_global_minimum = False, maxfun = 2000, do_logging = False,
print_progress = False. Set user_params['restarts.use_restarts'] = False
so a solver stop returns for an explicit decision. Other trust-region,
geometry and stopping settings inherit Py-BOBYQA 1.5.0 defaults. The initial
radius samples meaningful distances while fitting inside the box. The 61-point
model balances coordinate curvature information against 30-dimensional cost;
a full quadratic initialization would require 496 points. Determinism supports
using each evaluation directly without replication. A local model can exploit
structure efficiently, but there is no claim of global convergence.

Maximum additional objective calls: 181. Pause at cumulative evaluation 181.
This permits a complete 61-point initialization and up to 120 subsequent calls
to observe trust-region progress without assuming the model will be useful.
Pause earlier if the solver terminates or reports a numerical failure. Any
evaluator error or nonfinite objective stops the campaign without retry.

At the checkpoint inspect improvement from the initial value, the sequence of
30-call minima, where the best point lies relative to the bounds, and whether
recent proposals still improve. Check the actual used/remaining counters and
solver termination status. A plateau alone will not establish that the method
is unsuitable. No later method or allocation is committed here.
