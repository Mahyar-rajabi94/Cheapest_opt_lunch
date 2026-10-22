with Path('results/agent_run/changelog.md').open('a') as log:
    log.write('''## Decision after evaluation 26
**Budget:** 26 used / 100 total; 74 remaining.
**Evidence:** The 10 GP selections improved the incumbent from -5.6182484507 to -8.8133131168 (row 399, evaluation 24), an absolute reduction of 3.1950646661. Row 917 subsequently returned -6.2893123035; both belong to initial feature cluster 11. Other selected responses ranged from -3.1201543795 to -0.1106313867, so success is uneven. A diagnostic GP refit gives approximate leave-one-out RMSE 1.75077 versus 2.19575 for a constant predictor, Spearman 0.4523, and 92.3% of standardized residuals within two predicted standard deviations. Hyperparameters used all observed labels, making this diagnostic optimistic. Standardized fitted noise variance is 0.24926 and length-scale range 0.2303 to 3.5032; the noise is model mismatch, not objective randomness. The best row's approximate LOO prediction is only -4.93853, showing the low tail is underpredicted.
**Decision:** Continue `pool_gp_ei` using the existing GPBO instance, all 26 observations, and the same harness. No restart or discarded observations.
**Hyperparameters:** Unchanged from the decision after evaluation 16: n_init=16, full public-feature bound normalization, BoTorch 0.18.1/GPyTorch 1.15.2 default GP model and fitted likelihood, LogExpectedImprovement over all available rows in chunks of 512, seed=2. Retain fitted-nugget behavior because the diagnostics show model mismatch; do not force exact interpolation of a poorly resolved response surface. Ordinary per-query model refitting remains the wrapper's behavior.
**Reason:** A sizeable improvement, a second good response in the same feature cluster, and positive predictive diagnostics support continuing. Ten variable outcomes are not evidence that GPBO is unsuitable. The low-tail underprediction and sparse data argue for retaining uncertainty-driven acquisition rather than imposing a narrow local region.
**Next step limit:** 12 additional objective calls; pause at cumulative evaluation 38.
**Why this length:** Twelve more sequential observations provide a substantial update to a 26-row model and enough opportunities to test the newly discovered region, without deferring review for most of the remaining budget.
**Review:** Inspect incumbent improvement, repeated success around the best rows, selection distances and fitted GP diagnostics. Pause early on evaluator error, nonfinite output, invalid row or optimizer failure. Future methods remain undecided.

''')
start_block('pool_gp_ei', 12)
block_start = len(observed_y)
for _ in range(12):
    index = gp.ask_pool(h.available_indices)
    distance = float(np.min(np.linalg.norm(Z[observed_indices] - Z[index], axis=1)))
    y = buy(index, {'cluster': int(km.labels_[index]), 'nearest_observed_distance': distance})
    gp.tell(X[[index]], [y])
checkpoint()
print(json.dumps({'block_values': observed_y[block_start:]}), flush=True)
exec(compile(Path('work/diagnose_gp.py').read_text(), 'work/diagnose_gp.py', 'exec'), globals())
