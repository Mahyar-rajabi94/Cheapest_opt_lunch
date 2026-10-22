# Campaign changelog — Task 062

Append-only record of observed campaign changes. No objective evaluations had
been made when this file and roadmap_initial.md were created.

## Evaluation 500
**Change:** End BOBYQA's 500-call allocation and switch to the listed L-BFGS-B optimizer, starting from the best evaluated point. Retain the nominal remaining 1500-call allocation, with a monitoring checkpoint after approximately 500 L-BFGS-B calls.
**Reason:** BOBYQA improved the first value -9.375982284545898 to -11.258078575134277, but b(340) = b(400) = b(500) = -11.258078575134277. Its last 160 calls produced zero improvement, failing both pre-registered extension thresholds. Use scipy's standard L-BFGS-B finite differences and convergence settings on the normalized free-coordinate box. This is the planned method handoff; no new optimizer is introduced.

## Evaluation 661
**Change:** Switch from L-BFGS-B to the listed CMAES wrapper, seed 62 and otherwise default settings, on the normalized full free box. Allocate all remaining 1339 evaluations to CMA-ES, with a monitoring checkpoint at evaluation 1300. Preserve the best real point across methods.
**Reason:** L-BFGS-B returned after 161 evaluations and zero iterations with status "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL". All 161 real values, including the base point and 160 coordinate perturbations, were identical at -11.258078575134277. Earlier BOBYQA probes and steps produced different values and an improvement of 1.882096290588379, so this termination provides no evidence that wider moves cannot improve. Invoke the pre-registered early-termination fallback and use CMA-ES's population-based moves for the remaining budget. No default finite-difference parameter is changed or custom optimizer introduced.

## Evaluation 2000
**Change:** Finish the campaign at the exact 2000-evaluation budget and save through harness.save(). The evaluation-1300 checkpoint resumed the same CMA-ES state and previously committed remaining allocation; it introduced no restart, method change, or hyperparameter change.
**Reason:** The budget is exhausted. The best real value is -40.22267532348633, first obtained at evaluation 1669, compared with the initial -9.375982284545898. Final allocation: BOBYQA 500, L-BFGS-B 161, CMA-ES 1339. Read-only validation confirmed 2000 consecutively indexed trace rows, 188 finite coordinates per row, all 28 fixed coordinates preserved, the correct running minimum, matching metadata with stopped_early=false, the unchanged initial roadmap checksum, and exactly the four required result files. trace.csv and meta.json were generated exclusively by the harness and were not hand-edited.
