## Decision after evaluation 181
**Budget:** 181 used / 2000 total; 1819 remaining.
**Evidence:** The first objective was 1126.0394097190206; the best is
799.6726069430474 at evaluation 181, a reduction of 326.3668027759732.
Running minima at evaluations 90, 120, 150 and 180 were 953.8429114901046,
880.7697390595787, 833.0373534060759 and 800.0517845004377. The best point
has maximum absolute coordinate 22.838, far from the bounds. The solver
has not terminated. Determinism removes measurement-noise ambiguity, but
these observations do not establish global quality or smoothness everywhere.
**Decision:** Continue BOBYQA, preserving its interpolation set, trust radius,
and all solver state in the live thread, with the same harness and observations.
**Hyperparameters:** Unchanged from roadmap_initial.md (Py-BOBYQA 1.5.0,
61 interpolation points, initial scaled radius 0.1, final radius 1e-8,
restarts disabled). Internal radius and model adaptation continue normally.
**Reason:** Consistent improvement across successive windows justifies
letting the learned local model continue. There is no observed boundary issue
or numerical failure motivating a parameter change or restart.
**Next step limit:** 240 additional objective calls; pause at cumulative 421.
**Why this length:** Initialization is complete. Eight 30-call windows allow
more model refinement and a meaningful progress trend without committing
a large fraction of the remaining 1819 calls.
**Review:** Inspect best-value reduction and its distribution across windows,
step distances, boundary proximity and any solver termination. Pause sooner
on solver termination or numerical failure; evaluator errors stop without retry.

## Decision after evaluation 421
**Budget:** 421 used / 2000 total; 1579 remaining.
**Evidence:** The 240-call continuation improved 799.6726069430474 to
785.5502036392214. There were 123 improving calls, but the running minima
at 301, 361 and 421 were 785.572911757416, 785.5514049416073 and
785.5502036392214. Recent proposal distances shrank from about 0.036 to
0.002-0.004 original units. The best point remains interior (maximum absolute
coordinate 23.659). This is evidence of local settling, not proof of global
optimality or of a poor local method.
**Decision:** Switch temporarily to a multiscale directional exploration block
centered on the observed incumbent. Keep the BOBYQA thread and model paused;
retain all observations and the same harness.
**Hyperparameters:** Campaign-local multiscale_probe, NumPy 2.4.6 default_rng
seed 271828. QR-orthonormalize eight Gaussian directions in 30 dimensions;
evaluate both signs at Euclidean radii 2, 10, 40 and 100 in original units.
Use the incumbent fixed at evaluation 421. If needed, shorten a ray to 99%
of its feasible distance to respect bounds. No synthetic values are evaluated
or recorded; differences use the previously measured incumbent. No inherited
optimizer defaults apply.
**Reason:** Continued tiny local steps have sharply diminishing returns.
Controlled multiscale moves can reveal basin scale and possible improvements
at modest cost, supplying evidence for subsequent scale and method choices.
**Next step limit:** 64 additional objective calls; pause at cumulative 485.
**Why this length:** Eight orthogonal directions, two signs and four radii
provide matched directional comparisons for exactly 64 calls (4.1% of the
remaining budget). This is a sparse diagnostic, not a global coverage claim.
**Review:** Compare actual objective changes, directional asymmetry and best
values at each radius. Check whether any larger move improves the incumbent.
Stop immediately without retry on an evaluator error or nonfinite objective.

## Decision after evaluation 485
**Budget:** 485 used / 2000 total; 1515 remaining.
**Evidence:** None of 64 multiscale probes improved the incumbent
785.5502036392214. Radius-2 values ranged 788.5429374597297 to
790.4865540262789 with nearly symmetric opposite-direction changes.
Minimum/median values were 853.5146790958247/862.5555658755201 at radius 10,
1025.7571862798213/1094.5544272704105 at radius 40, and
1048.681947895287/1105.6032635391273 at radius 100. Large-displacement
costs are not simply a continuation of the small-radius quadratic trend.
These sparse directions support local stability but do not rule out better
basins elsewhere. The best radius-100 point is evaluation 477.
**Decision:** Restart BOBYQA at the best observed radius-100 point, evaluation
477. Keep the original BOBYQA state parked and retain all observations and
the single harness. Build a fresh interpolation model and trust-region state
for the new region; no attempt is made to transplant an invalid local model.
**Hyperparameters:** Same Py-BOBYQA 1.5.0 settings as the initial roadmap:
npt=61, scaling_within_bounds=True, rhobeg=0.1 (20 original units),
rhoend=1e-8, no internal restarts, maxfun=2000 and deterministic mode.
The previous radius worked for finding a local basin, and the distant probes
show ample room for a fresh search. Initialization changes to the selected
observed point. Its repeat evaluation counts normally.
**Reason:** Further polishing has small observed benefit. A distant local
start can test another attraction region; its immediate objective need not
beat a polished incumbent. The best outer start has lower value than the
original center (1048.68 versus 1126.04), but its eventual quality is unknown.
**Next step limit:** 301 additional objective calls; pause at cumulative 786.
**Why this length:** 61 initialization points plus up to 240 model-driven calls
allow assessment of a new local trajectory. The first local run made most
of its improvement within this many calls, though this restart may differ.
**Review:** Compare the new region's best value and improvement trend with
the incumbent, and measure distance between the resulting points. Pause
sooner on solver termination or numerical failure; evaluator errors stop
without retry. No later restart or method is committed.

## Decision after evaluation 786
**Budget:** 786 used / 2000 total; 1214 remaining.
**Evidence:** The restart reduced 1048.681947895287 to 747.7553690766579,
improving the previous campaign best by 37.7948345625635. Its best point is
107.6695400989791 units from the first local solution. Recent minima were
748.4543776140281 at 695, 748.0419400163807 at 725, and
747.8076337033334 at 755, so refinement continues with diminishing gain.
The best unrefined radius-40 point, evaluation 463, has measured value
1025.7571862798213, distance 40 from the first local solution and
117.07836684673589 from the current incumbent. Different attained local
values support further exploration, though candidate value does not guarantee
its eventual local minimum.
**Decision:** Restart BOBYQA at evaluation 463. Preserve both prior solver
states and the campaign incumbent; reset only the new solver's local model
and trust region. Keep the same harness and all observations.
**Hyperparameters:** Unchanged Py-BOBYQA 1.5.0 configuration from the initial
roadmap and decision 485, including 61 interpolation points and initial
radius 20 original units. The two previous trajectories both made substantial
progress with this setting, so there is no observed need to change it.
**Reason:** A second local region improved substantially, while its late
refinement was small. The lowest-valued unrefined medium/large probe offers
an evidence-based new start and is separated from both refined regions.
It may still return to an existing basin; distances alone cannot rule that out.
**Next step limit:** 241 additional objective calls; pause at cumulative 1027.
**Why this length:** Allow the 61-point initialization and 180 model-driven
calls. The preceding restart had revealed its quality by about 240 calls;
its last 90 calls improved by less than 0.70. Review the new trajectory earlier
instead of automatically repeating the 301-call allocation.
**Review:** Compare this region's best with 747.7553690766579, its improvement
rate and distance from prior local solutions. Pause on solver termination or
numerical failure; evaluator errors stop without retry.

## Decision after evaluation 1027
**Budget:** 1027 used / 2000 total; 973 remaining.
**Evidence:** The third local search reduced its start 1025.7571862798213
to 782.0109392644467 at evaluation 1026, still 34.2555701877888 above the
incumbent 747.7553690766579. Its last 30-call window improved by about 1.01,
after preceding window reductions of 11.78 and 4.71. Its best is 47.5994
units from the first solution and 118.3894 from the incumbent. Thus the
best initial probe value did not yield the best refined result; starting
value is only a heuristic. Unused outer probe evaluation 484 has value
1061.9275206066086 and is 144.8743 units from the incumbent.
**Decision:** Park the third solver and restart BOBYQA at evaluation 484,
the best unused radius-100 probe. Retain all three prior states, all observations
and the incumbent under the same harness. Initialize a fresh local model.
**Hyperparameters:** Retain the Py-BOBYQA 1.5.0 settings in the initial roadmap
(npt=61, scaled rhobeg=0.1, rhoend=1e-8, deterministic mode, restarts disabled).
All starts have produced large local reductions with those settings; the
current issue is basin quality, not an observed modeling failure.
**Reason:** The current trajectory is well behind and slowing. An already
measured distant start tests a distinct region without another screening block.
There is no guarantee that a new region beats the incumbent.
**Next step limit:** 211 additional objective calls; pause at cumulative 1238.
**Why this length:** 61 initialization plus 150 subsequent calls tests a basin
at lower cost as the remaining budget shrinks. Previous trajectories already
showed their approximate competitive position after about 210 calls. Continued
refinement remains a fresh decision based on the actual result.
**Review:** Assess region minimum, remaining gap to 747.7553690766579,
recent improvements, and distance to prior solutions. Pause sooner on solver
termination or numerical failure; evaluator errors stop without retry.

## Decision after evaluation 1238
**Budget:** 1238 used / 2000 total; 762 remaining.
**Evidence:** The fourth local search improved its start 1061.9275206066086
to 778.6281058091176. Its last window minima were 811.3414744146863,
790.7108016754443 and 779.8327511144257, followed by another improvement
to 778.6281058091176 on the final call. The latest 31 calls gained 12.0827,
so this search is still descending appreciably even though it trails the
incumbent 747.7553690766579 by 30.8727. Its solution is 144.8895 units
from the incumbent; it is not merely reproducing the same location.
**Decision:** Continue the fourth BOBYQA solver with its model, trust-region
state and pending proposal intact. Preserve all campaign observations.
**Hyperparameters:** Unchanged configuration from the initial roadmap and
decision 1027, Py-BOBYQA 1.5.0. Keep the learned model and ordinary internal
radius adaptation; no restart or explicit radius reset is justified during
substantial descent.
**Reason:** A 211-call checkpoint intentionally occurred before convergence.
The recent improvement rate supports another bounded continuation; treating
a still-moving trajectory as a failed basin would be premature.
**Next step limit:** 90 additional objective calls; pause at cumulative 1328.
**Why this length:** Three 30-call windows distinguish continued descent
from local settling at a modest cost relative to 762 remaining calls. There
is no new model initialization to pay for.
**Review:** Compare gain with the 30.8727 gap to the incumbent and inspect
whether window improvements decay. Pause sooner on solver termination or
numerical failure; evaluator errors stop without retry.

## Decision after evaluation 1328
**Budget:** 1328 used / 2000 total; 672 remaining.
**Evidence:** The 90-call continuation improved the fourth region from
778.6281058091176 to 772.6916198433446. Its last three window minima were
774.3024232781464, 772.9119430143703 and 772.6916198433446. Improvement
has slowed to 0.2203 in the last 30 calls while the gap to the incumbent
747.7553690766579 remains 24.9363. The explored local starts have all been
at the center or on radius-40/100 probes about the first solution; much of
the allowed box has not been sampled. The different local values demonstrate
that a local stationary point does not establish global optimality.
**Decision:** Switch to one broad scrambled Sobol exploration block over
the full bounds. Park the fourth BOBYQA solver and preserve every prior
solver, the incumbent, observations, and the same harness.
**Hyperparameters:** scipy.stats.qmc.Sobol in SciPy 1.17.1, d=30,
scramble=True, bits=30, seed=20261022, optimization=None; random_base2(m=5)
produces 32 points. Affinely map the unit cube to the supplied [-100,100]
bounds. No objective-dependent surrogate or synthetic observations are used.
**Reason:** Refining the current inferior basin offers small observed return.
A low-cost full-box sample adds geographic diversity beyond the earlier
orthogonal rays and measures objective levels in much more distant regions.
It cannot establish global coverage in 30 dimensions.
**Next step limit:** 32 additional objective calls; pause at cumulative 1360.
**Why this length:** A power-of-two Sobol block preserves its balance; 32 calls
cost less than 5% of the remaining budget and are sufficient to inspect an
initial distribution of values and candidate regions. No later block is
committed before seeing those measurements.
**Review:** Compare the best and quantiles with unrefined previous starts,
inspect distances and boundary proximity, and check whether any new point
improves the campaign incumbent. Evaluator errors stop without retry.

## Decision after evaluation 1360
**Budget:** 1360 used / 2000 total; 640 remaining.
**Evidence:** The 32 full-box Sobol measurements ranged from
1155.9095326033366 to 1923.494463719897, with median 1553.2345652608901.
None improved 747.7553690766579. The best Sobol point, evaluation 1349,
has norm 285.3840, distance 270.5192 from the incumbent, and minimum bound
gap 0.07577. These values are higher than the earlier radius-40/100 probe
values; this sparse sample suggests a cost to very large excursions but
does not rule out good distant minima. Prior radius-100 local optimization
successfully found a better region, whereas the last two starts did not.
**Decision:** Restart BOBYQA at the midpoint between the incumbent (evaluation
786) and best Sobol point (evaluation 1349). Its distance from the incumbent
is 135.2596; its objective has not been observed and will be evaluated normally.
This uses a new, moderately distant direction while avoiding the nearly-boundary
Sobol start. Preserve all previous solvers, observations and the single harness;
initialize a new interpolation model and trust region.
**Hyperparameters:** Py-BOBYQA 1.5.0, same configuration as the initial roadmap
except rhobeg=0.075 in scaled coordinates (15 original units). The midpoint
has at least 16.5223 units of boundary margin because the incumbent's minimum
margin is 33.0447, so this radius fits without shifting the starting point.
Keep npt=61, rhoend=1e-8, deterministic mode and disabled automatic restarts.
This modest radius reduction accommodates the new geometry; it is not a claim
that the previous radius was generally inferior.
**Reason:** The measured full-box points are costly, but the best supplies an
unexplored direction. A midpoint balances regional diversity with the earlier
success of moderate-distance starts. Its basin quality remains unknown.
**Next step limit:** 211 additional objective calls; pause at cumulative 1571.
**Why this length:** Initialize 61 points and permit 150 subsequent evaluations,
as in the preceding early basin assessment. With 640 calls left, inspect this
new region before allocating a longer refinement; prior trajectories differ
in how quickly they settle.
**Review:** Inspect actual initial value, attained minimum, improvement trend,
and competitiveness with the incumbent. Pause sooner on solver termination
or numerical failure; evaluator errors stop without retry.

## Decision after evaluation 1571
**Budget:** 1571 used / 2000 total; 429 remaining.
**Evidence:** The midpoint's measured initial value was 1048.3785481184373.
The new local search attained 705.1400095792432 at evaluation 1570, improving
the previous incumbent by 42.6153594974147. Window minima at 1510, 1540 and
1570 were 740.3428391055317, 713.8043060326975 and 705.1400095792432.
The best point is 143.0401 units from the former incumbent. The midpoint's
actual minimum bound margin was 34.3132, exceeding the conservative bound
used when selecting radius 15. The latest 30-call gain of 8.6643 shows active
descent; there is no evidence yet that this best region is locally settled.
**Decision:** Continue the fifth BOBYQA solver unchanged, preserving its full
interpolation and trust-region state, all observations and the same harness.
**Hyperparameters:** Retain the configuration in decision 1360 (Py-BOBYQA
1.5.0, npt=61, initial scaled rhobeg=0.075, rhoend=1e-8 and no automatic
restarts). The radius is now internally adapted; do not reset it.
**Reason:** This is both the best observed region and a still-improving
trajectory. Continuing an already learned model is supported more directly
than paying for another new model at this checkpoint.
**Next step limit:** 90 additional objective calls; pause at cumulative 1661.
**Why this length:** Three 30-call windows allow the current descent to develop
and reveal whether the 8.6643 recent gain tapers. With only 429 calls left,
reassess before committing a larger fraction to small refinements.
**Review:** Inspect new best and per-window gain, proposal distances and any
solver stopping condition. Pause sooner on solver termination or numerical
failure; evaluator errors stop without retry.

## Decision after evaluation 1661
**Budget:** 1661 used / 2000 total; 339 remaining.
**Evidence:** Continuing the fifth region improved 705.1400095792432 to
701.0456700265452 at evaluation 1659. Window minima were 702.0518460350642,
701.2746461837196 and 701.0456700265452; corresponding gains were about
3.0882, 0.7772 and 0.2290. Sampled recent proposal distances decreased from
0.309 to 0.133 original units. The local search is settling. The first midpoint
start improved the prior incumbent by 46.7097 after refinement, whereas the
raw distant Sobol endpoint remains costly (1155.9095).
**Decision:** Park the fifth solver and restart BOBYQA at the midpoint between
the current incumbent (evaluation 1659) and that same best Sobol endpoint
(evaluation 1349). This new start is 67.69625061 units from the incumbent
and has minimum bound margin 17.05081728. Its objective is unknown and
will be evaluated normally. Preserve all prior solver states, the incumbent,
all observations and the same harness; reset the new local model only.
**Hyperparameters:** Retain Py-BOBYQA 1.5.0 configuration from decision 1360:
npt=61, scaling_within_bounds=True, rhobeg=0.075 (15 original units),
rhoend=1e-8, deterministic mode, no internal restarts. The new midpoint has
sufficient boundary margin for that radius, and the preceding run supports
its practical effectiveness without proving it optimal.
**Reason:** Small local refinements now offer limited observed gain. The
successful preceding midpoint supplies evidence for another moderately
distant start in this direction. This is a new basin test, not an assumption
of monotonic improvement along the segment or of a good Sobol endpoint.
**Next step limit:** 211 additional objective calls; pause at cumulative 1872.
**Why this length:** The 61-point model initialization plus 150 subsequent
calls is enough to reveal the early trajectory, as in the previous successful
restart. The remaining 339 calls do not justify an unreviewed long search.
**Review:** Compare the new region's value and descent rate with 701.0456700265452,
check its displacement, and reconsider based on actual remaining budget.
Pause sooner on solver termination or numerical failure; evaluator errors
stop without retry.

## Decision after evaluation 1872
**Budget:** 1872 used / 2000 total; 128 remaining.
**Evidence:** The sixth start reduced 1061.957710058823 to 743.1370584758236,
but remains 42.0913884492784 above the incumbent 701.0456700265452.
Its last window minima were 749.8348825488799, 745.3056800676532 and
743.1433360667174, followed by 743.1370584758236. The solution lies
76.4992 units from the incumbent. The new region therefore did not extend
the previous midpoint success. The fifth solver was paused while still
improving, with a 0.2290 gain in its last 30 calls and proposal distances
around 0.13-0.31 units.
**Decision:** Park the sixth solver and resume the fifth BOBYQA solver, which
produced the incumbent, using exactly its preserved interpolation model,
trust-region state and pending proposal. No new initialization evaluations
are needed. Retain all observations and the single harness.
**Hyperparameters:** Unchanged from decision 1360 and its continuation at
1571, Py-BOBYQA 1.5.0. No radius reset; resume its adapted radius. Automatic
restarts remain disabled.
**Reason:** The latest region is far behind and slowing. With 128 calls left,
another 61-point initialization would leave few local-search calls, whereas
the best region has a learned model and demonstrated remaining refinement.
**Next step limit:** 90 additional objective calls; pause at cumulative 1962.
**Why this length:** Three 30-call windows can quantify the remaining local
gain and assess numerical settling while preserving a checkpoint before
spending all remaining evaluations. No later block is committed here.
**Review:** Inspect best-value gain and proposal distances; check for solver
termination. Pause sooner on termination or numerical failure; evaluator
errors stop without retry.

## Decision after evaluation 1962
**Budget:** 1962 used / 2000 total; 38 remaining.
**Evidence:** Resuming the best region improved 701.0456700265452 to
700.9812830402875 at evaluation 1962. The last three window minima were
700.9951175822307, 700.9826641493887 and 700.9812830402875. Thus gains
are small and decreasing but remain positive for this deterministic objective.
Sampled recent proposal distances are about 0.008-0.027 original units.
The solver has not terminated or reported a numerical failure.
**Decision:** Continue the same fifth BOBYQA solver and its adapted state,
with label bobyqa_polish. Preserve all observations and the single harness.
**Hyperparameters:** Unchanged from decision 1872, referencing the Py-BOBYQA
1.5.0 configuration in decision 1360. No restart or radius reset.
**Reason:** There is observed remaining local improvement. A new 61-point
model cannot be initialized within 38 calls; the incumbent's live model can
use all of them for refinement. Other explored regions remain worse.
**Next step limit:** 38 additional objective calls; pause at cumulative 2000,
or sooner if the solver terminates. This is exactly the remaining hard budget.
**Why this length:** The remaining amount is shorter than the preceding
90-call refinement and only slightly longer than one observed 30-call window.
Spending it on the already converging best model is supported by the last
positive gain; this is a deliberate final refinement, not an automatic
allocation of a large unexplored budget.
**Review:** Inspect final best value, actual evaluation count and any early
solver stop. Stop on evaluator error without retry. At the checkpoint record
the stop decision, save through the harness, and verify the four artifacts.

## Decision after evaluation 2000
**Budget:** 2000 used / 2000 total; 0 remaining.
**Evidence:** The final 38-call refinement improved 700.9812830402875 to
700.9807061897204 at evaluation 1998. The final evaluated value was
700.9807151293337; the incumbent remains the actually observed evaluation
1998. The initial center measured 1126.0394097190206. Six local starts,
64 multiscale directional probes and 32 full-box Sobol points were evaluated
within the one campaign counter. No evaluator errors occurred. Best values
are observed results; no global-optimality certificate is available.
**Decision:** Stop because the hard evaluation budget is exhausted. Preserve
the best observed point in the harness-written trace.csv and meta.json.
Call harness.save() and audit the artifacts without further evaluations.
**Hyperparameters:** Not applicable: no further optimization block.
**Reason:** All 2000 authorized objective evaluations have been used.
**Next step limit:** 0 additional objective calls; stop at cumulative 2000.
**Why this length:** No evaluation budget remains.
**Review:** Not applicable to another optimization block. Final validation
checks exactly four campaign files, 2000 recorded real observations, finite
values and in-bounds vectors, correct running minima, matching best point
and metadata, and unchanged initial decision text.
