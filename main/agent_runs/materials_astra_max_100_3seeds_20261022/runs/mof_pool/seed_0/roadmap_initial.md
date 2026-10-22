# Initial decision: balanced finite-pool design

Minimize the deterministic scalar objective over exactly 1,000 public candidate rows, each with 193 real-valued features. Only unused row indices are legal purchases. There are no additional constraints and no prior objective observations. The hard total budget is 100 evaluations. Campaign seed is 0. The bounding box is descriptive; no generated continuous points will be evaluated. The objective remains sealed, and only returned objective values will be treated as observations.

Public feature inspection finds 165 nonconstant columns, 1,000 distinct rows, and finite values throughout. After range normalization, the first 10 and 20 principal components explain about 84.66% and 95.43% of feature variance. These are feature-only facts, not evidence of objective smoothness or relevance.

## First step only

Use `balanced_pool_design`, a campaign-local NumPy adapter in `work/`. Normalize each nonconstant column by its public bound range; omit the 28 constant columns from distance calculations. Use NumPy 2.4.6 `default_rng(0)` (PCG64) to sample six distinct indices uniformly without replacement. Then add six rows sequentially by maximum minimum squared Euclidean distance to the rows already selected. Break exact distance ties by the smallest row index. Every selected row is evaluated using `h.evaluate_index(index)` and labelled with the current method. There are no learned hyperparameters in this first step.

The mixture includes representative random rows and broader coverage without letting a purely farthest-point design devote all its calls to feature outliers. Normalized Euclidean distance assumes comparable feature ranges for coverage only; it does not assert that all features influence the target equally. No model is fit or selected from nonexistent labels.

**Step limit:** at most 12 additional objective calls; pause at cumulative evaluation 12. Twelve labels provide an initial view of objective variation while leaving 88 evaluations for decisions based on evidence. This choice is not a commitment to any later method or block length.

**Checkpoint review:** inspect all returned values, the best row, distribution and spread, feature distances among better rows, and whether the observations support any learnable feature-to-objective structure. Assess the suitability of a surrogate and its uncertainty only after seeing these observations. Stop earlier on any evaluator error or nonfinite return; do not retry an evaluation of unknown status. Save the harness output at the checkpoint and return control for a new decision.
