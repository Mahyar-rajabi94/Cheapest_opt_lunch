with Path('results/agent_run/changelog.md').open('a') as log:
    log.write('''## Decision after evaluation 38
**Budget:** 38 used / 100 total; 62 remaining.
**Evidence:** The last 12 GP-EI calls did not improve -8.8133131168. Their best value was -4.3388968697 and six were in feature cluster 11, so this was not solely a failure to revisit the promising cluster. Approximate GP LOO RMSE improved to 1.56530 (constant baseline 2.04839), Spearman 0.58048, and fitted standardized noise variance is 0.22351. However, the best observed response is still underpredicted in LOO (-5.14433). Several unused rows have model means around -5.94 to -5.96 with latent SD 0.74 to 0.83; EI against the unusually low observed incumbent has repeatedly preferred more uncertain alternatives. These are predictions, not objective observations. LOO uses globally fitted hyperparameters and remains optimistic.
**Decision:** Adjust acquisition to `pool_gp_lcb`, minimizing mean minus one latent posterior SD over all unused rows. Keep the same response model and all 38 observations. A campaign-local adapter uses supported BoTorch APIs to expose this acquisition; GP fitting remains fresh per selection as in the original wrapper. The existing GPBO observation store remains synchronized for diagnostics. No harness reset and no data discarded.
**Hyperparameters:** kappa=1.0 in LCB=mean-kappa*SD. All model settings are unchanged from evaluation 16: full 193-column input with constant-column-safe bound normalization; BoTorch 0.18.1 SingleTaskGP defaults, Standardize(m=1), fit_gpytorch_mll, learned nugget, double precision, posterior chunks of 512. Candidate enumeration and fitting are deterministic for this block; retain Torch/NumPy campaign seed state from seed 2 without reseeding. work/gp_lcb.py implements the adapter. Predictive residual diagnostics include the fitted nugget, while acquisition uses latent uncertainty.
**Reason:** The GP has measurable predictive signal but EI against the current extreme incumbent gives little attention to its lowest predicted means. LCB with moderate kappa tests whether stronger emphasis on predicted value is more productive. This adjustment follows the acquisition pattern and diagnostics, not the plateau alone.
**Next step limit:** 8 additional objective calls; pause at cumulative evaluation 46.
**Why this length:** Eight purchases are enough to test the low-mean group and obtain prospective prediction errors without committing much of the 62-call remainder to the changed acquisition.
**Review:** Inspect actual improvement and pre-purchase predictive residuals. Pause sooner if two consecutive absolute residuals exceed three predictive SDs, or on an evaluator error, invalid row, nonfinite result or fitting failure. An early pause triggers reconsideration only.

''')
exec(compile(Path('work/gp_lcb.py').read_text(), 'work/gp_lcb.py', 'exec'), globals())
start_block('pool_gp_lcb', 8)
lcb_records = []
severe_residuals = 0
for _ in range(8):
    index, details = propose_gp_lcb(1.0)
    y = buy(index, details)
    gp.tell(X[[index]], [y])
    residual = (y-details['model_mean'])/details['model_predictive_sd']
    lcb_records.append({'index': index, 'y': y, **details, 'standardized_residual': residual})
    severe_residuals = severe_residuals+1 if abs(residual) > 3 else 0
    if severe_residuals >= 2:
        print('EARLY_CHECKPOINT: two consecutive residuals exceed three predictive SDs', flush=True)
        break
checkpoint()
print(json.dumps({'lcb_records': lcb_records}), flush=True)
exec(compile(Path('work/diagnose_gp.py').read_text(), 'work/diagnose_gp.py', 'exec'), globals())
