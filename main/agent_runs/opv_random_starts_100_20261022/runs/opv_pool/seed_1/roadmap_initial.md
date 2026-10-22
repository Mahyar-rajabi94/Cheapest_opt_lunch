# Initial decision: coverage and history-dependence check

The public task minimizes a scalar objective on [0, 1]^11, with no additional constraints and a hard total budget of 100 calls. It is noise-free but explicitly history-dependent: repeated x values need not return the same y. No prior observations exist. We seek the best actually observed value, without assuming its x will reproduce that value later. All evaluations will use one persistent AgentTaskHarness instance. The objective and evaluator implementation remain sealed.

## First block only
Use `sobol_anchors`: 16 scrambled Sobol points from the provided `src.optimizers.sobol_search.SobolSearch` wrapper (SciPy 1.17.1 `qmc.Sobol`, scramble=True, seed=1, inherited bits=30 and optimization=None). NumPy is 2.4.6. Draw the 16-point design in one ask to retain its power-of-two balance; evaluate it in its generated order. Evaluate the center x=(0.5,...,0.5) at calls 1, 10, and 19, interleaving eight design points between the center observations. Tell the wrapper the actual outcomes and retain all observations in order. No model fit, objective-scale assumption, or prior measurements are used.

This is an exploratory design rather than a static-function fit: uniform domain coverage is useful before choosing a local region, while repeated centers directly expose change over evaluation history. Three repeats can show an initial trend but cannot establish that all history dependence is an additive drift or depends only on call number.

**Step limit:** at most 19 additional objective calls; pause at cumulative evaluation 19. This obtains 16 distinct space-filling points in 11 dimensions plus three history checks, while leaving 81 calls uncommitted. Every repeat counts against both limits. Stop immediately on any evaluator error, invalid or nonfinite result, and do not retry an unknown evaluation.

**Checkpoint evidence:** best point/value, distribution of the 16 design outcomes, the center values and their change relative to between-point variation, any evident temporal trend, and distances of good points from the center and boundaries. These observations will determine the next decision; no subsequent method or budget allocation is selected now.
