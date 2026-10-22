## Evaluation 29
**Change:** Switched active optimizer from GP-BO (LogEI) to CMA-ES, started
fresh at the box midpoint with sigma0 = box_width/4 (standard CMA-ES
box-constrained defaults), per the pre-registered plateau trigger.
**Reason:** GP-BO's acquisition phase began at evaluation 15 (after its
internal 14-point Sobol initial design) and produced no improvement to the
running best (0.2885332703590393) over the following 15 consecutive
evaluations (16-29), exactly meeting the pre-registered trigger ("no
improvement in best-so-far for 15 consecutive GP-BO evaluations ->
switch to CMA-ES"). This indicated LogEI's acquisition optimizer was
repeatedly re-exploiting the same local basin rather than finding new
improving points.

## Evaluation 49
**Change:** Switched active optimizer from CMA-ES to a BOBYQA local polish,
restarted on a narrowed box (+-0.05 around the current best point,
clipped to the original [0,1]^6 bounds) so that BOBYQA's fixed
midpoint-start convention coincides with the current best point.
**Reason:** CMA-ES (evaluations 30-49) produced no improvement to the
running best (still 0.2885332703590393) over 20 consecutive evaluations,
exactly meeting the pre-registered trigger ("no improvement in best-so-far
for 20 consecutive CMA-ES evaluations, with >=10 evaluations remaining ->
switch to BOBYQA local polish"); 51 evaluations of budget remained at the
time of the switch. The polish phase then ran for the remaining budget
(evaluations 50-100) and improved the running best from 0.2885332703590393
to 0.2607913017272949, confirming the local-refinement choice was
productive.
