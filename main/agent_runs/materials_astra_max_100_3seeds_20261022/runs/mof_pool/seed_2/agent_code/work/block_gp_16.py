from src.optimizers.pool_gp_bo import GPBO
import torch
from scipy.stats import spearmanr

torch.set_num_threads(1)
Path('results/agent_run/changelog.md').open('a').write('''## Decision after evaluation 16
**Budget:** 16 used / 100 total; 84 remaining.
**Evidence:** All 16 representative rows were evaluated successfully. Best is row 442 at -5.6182484507; next best are 720 at -5.1944973338 and 816 at -4.8459473318. Responses span -5.6182484507 to -0.687448968, median -2.8743864405. Cluster sizes range from 8 to 145. One label per cluster is insufficient to establish local smoothness or a reliable feature-response relationship. The objective is deterministic; predictive uncertainty is model uncertainty, not measurement noise.
**Decision:** Switch from the initial design to sequential pool GP Bayesian optimization (`pool_gp_ei`). Initialize the supplied GPBO with all 16 purchased raw rows and labels, retaining the same harness and observations. Each selection scores only available rows and receives feedback before the next.
**Hyperparameters:** src.optimizers.pool_gp_bo.GPBO with candidates=h.candidates, bounds=h.bounds, direction=min, seed=2, n_init=16, pool_chunk_size=512. Use its BoTorch 0.18.1 SingleTaskGP defaults: bound-normalized features, standardized response, default dimension-scaled RBF ARD prior and Gaussian likelihood, and exact marginal likelihood fitting via fit_gpytorch_mll (GPyTorch 1.15.2, Torch 2.14.1). Analytic LogExpectedImprovement has no added exploration offset. Continuous acquisition restarts/raw samples do not apply to pool enumeration. The likelihood's fitted nugget can absorb model mismatch even though evaluations are deterministic. One Torch CPU thread is a runtime setting, not a model change.
**Reason:** There is substantial response variation across the initial coverage. GP expected improvement offers a data-informed balance between low predicted response and uncertainty while examining the entire unused pool. Sixteen labels do not justify an aggressive local restriction or fitting an unconstrained separate model for each feature.
**Next step limit:** 10 additional objective calls; pause at cumulative evaluation 26.
**Why this length:** Ten sequential feedback updates can reveal whether the surrogate identifies improvements, while preserving 74 evaluations for decisions informed by those results.
**Review:** Assess best-value improvement, selected-value distribution, cluster coverage and nearest-observation distances. Inspect the final GP's length scales, fitted nugget, and leave-one-out prediction diagnostics, recognizing small-sample and adaptive-selection uncertainty. Pause early on an evaluator error, invalid index, nonfinite result or optimizer failure; never retry an unknown evaluation. No next method is preselected.

''')
gp = GPBO(candidates=X, bounds=h.bounds, seed=h.seed, direction='min', n_init=16)
gp.tell(X[observed_indices], observed_y)
start_block('pool_gp_ei', 10)
block_start = len(observed_y)
for _ in range(10):
    index = gp.ask_pool(h.available_indices)
    distance = float(np.min(np.linalg.norm(Z[observed_indices] - Z[index], axis=1)))
    y = buy(index, {'cluster': int(km.labels_[index]), 'nearest_observed_distance': distance})
    gp.tell(X[[index]], [y])
checkpoint()
print(json.dumps({'block_values': observed_y[block_start:]}), flush=True)
