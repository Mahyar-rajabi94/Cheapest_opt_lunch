# Initial decision: broad, reproducible sampling

**Problem:** Minimize a deterministic scalar objective in 10 real coordinates, each in [-5, 5], with no additional constraints. There are no prior observations. The total hard budget is 2,000 objective calls, including repeated points. The objective is sealed; no assumptions about its identity, optimum, smoothness, or separability will be made.

**First step:** Evaluate the box center once, then 64 scrambled Sobol points across the full box. Use the supplied `src.optimizers.sobol_search.SobolSearch` wrapper over SciPy 1.17.1 `scipy.stats.qmc.Sobol`, with dimension 10, scramble=True, seed=20261022, and affine scaling to the bounds. Inherited SciPy defaults include bits=None (30 bits) and optimization=None. There is no observation noise, so this block contains no intentional replications. The center is an explicit reference point, not a claim about where the optimum lies. NumPy version is 2.4.6.

**Why this method and configuration:** With no observations, a low-discrepancy design provides broad spatial coverage without requiring derivatives or a smooth response model. A power-of-two Sobol sample preserves its balance properties. The fixed scrambling seed makes the sampling reproducible.

**Step limit:** At most 65 additional objective calls; pause at cumulative evaluation 65. This is 3.25% of the total budget and provides a baseline and useful value distribution without delaying adaptation for a large fraction of the campaign.

**Checkpoint review:** Inspect the center value, best observed point and value, response quantiles, variation among nearby samples, and whether better points cluster spatially. These are sparse observations in 10 dimensions and cannot establish global optimality or smoothness. Decide the next bounded step only after seeing this evidence. Pause immediately on an evaluator error or nonfinite response; do not retry an unknown evaluation.

**Execution integrity:** Use one live harness and persistent optimizer state, with a checked objective-call limit for every block. Every response is obtained from `h.evaluate`; call `h.save()` at checkpoints. Scripts live in `work/`; this file will remain unchanged and subsequent decisions will be appended to `changelog.md` before further evaluation.
