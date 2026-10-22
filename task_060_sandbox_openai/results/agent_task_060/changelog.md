# Task 060 campaign changelog

Entries below are append-only. Evaluation numbers count completed real calls.

## Evaluation 0
**Change:** Switch to sobol_search.
**Reason:** Begin the preregistered 64-point scrambled Sobol design, seed 60; no objective data have previously been collected.

## Evaluation 64
**Change:** Switch to lbfgsb.
**Reason:** The preregistered Sobol phase used 64 real evaluations: observed range [-4.413784503936768, 1.4169338941574097], median -1.2563010454177856. Start the planned normalized L-BFGS-B trial at the best observed point, eps=1e-5, with 483/805-call caps.

## Evaluation 225
**Change:** Switch to cma_es.
**Reason:** L-BFGS-B returned after its first 161-call gradient site with zero projected gradient (nit=0), no improvement, and best still -4.413784503936768. Following the preregistered early-return rule, transfer its unused allocation to a single CMA-ES run at the incumbent, normalized sigma 0.25 and seed 60; check the final-refinement criterion after evaluation 1500.

## Evaluation 2000
**Change:** Finish the campaign at the exact 2,000-evaluation cap and save the harness outputs.
**Reason:** At the completed CMA-ES generation ending at evaluation 1,517, the best had improved by 1.1763553619384766 over the preceding 380 evaluations, exceeding the registered plateau threshold of 0.0005830718398094178. The plan therefore continued the same CMA-ES state for the remaining 483 calls, with no BOBYQA switch. The final best is -10.906188011169434, first observed at evaluation 1,181. Actual allocations were SobolSearch 64, L-BFGS-B 161, and CMA-ES 1,775. Every value came from the original single harness; its save() produced trace.csv and meta.json. The initial roadmap remains unchanged.
