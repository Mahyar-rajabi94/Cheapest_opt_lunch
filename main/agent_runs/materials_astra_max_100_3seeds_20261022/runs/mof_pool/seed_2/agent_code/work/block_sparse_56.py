with Path('results/agent_run/changelog.md').open('a') as log:
    log.write('''## Decision after evaluation 56
**Budget:** 56 used / 100 total; 44 remaining.
**Evidence:** Forest-LCB found row 792 at -6.648329128 but did not improve row 399 at -8.8133131168. Its other nine values ranged from -3.1419339268 to -0.0602296135; predictions were generally too low. With only 56 labels for 165 varying features, feature relevance remains weakly resolved. An objective-free diagnostic fitted MAP-SAAS sparse GPs through BoTorch's public API. Five-fold refitting on the same 56 rows (KFold shuffle=True, seed=2) gave raw-target sparse GP with tau=0.1 RMSE 2.28512, Spearman 0.27006, low-10 recall 0.5; ridge 2.20988/0.22789/0.3; random forest 1.92486/0.20478/0.4. Thus sparse GP is not the best overall predictor but has slightly better rank/tail recovery. Additional asinh(y) sparse variants at tau=0.1 and 0.01 gave RMSE 2.47268 and 2.34212, Spearman 0.18004 and 0.14880; these do not support transforming the target. All comparisons use only purchased labels and identical folds. Multiple model checks and adaptive sampling make these estimates tentative. The full raw sparse model puts most length scales very large (median 250.27), with minimum 0.2322, and proposes a different low-mean candidate set; these unpurchased predictions are not observed outcomes.
**Decision:** Switch to a bounded `sparse_gp_lcb` trial. Use MAP-SAAS with raw labels and all 56 observations; retain the same harness and full observed history. The prior GP and forest have no fitted optimizer state needed by this different model. Fresh MAP fitting per proposal is part of the chosen method. Keep the original GPBO observation store synchronized.
**Hyperparameters:** BoTorch 0.18.1 get_fitted_map_saas_model with tau=0.1, train_Yvar=None (learned nugget), Standardize(m=1), and optimizer_kwargs options maxiter=500. Use all 165 varying columns scaled by public bound ranges; no PCA truncation or target transformation. Installed default MAP-SAAS scaled Matern-5/2 kernel, mean, priors, parameter bounds and fitting behavior are inherited from BoTorch 0.18.1/GPyTorch 1.15.2. Fixed global sparsity tau=0.1 favors dependence on relatively few features while permitting data-adapted ARD length scales. Select the minimum mean minus 1.0 latent posterior SD over all unused rows in chunks of 512. Double precision, one Torch thread, and unchanged campaign seed state from 2. No new stochastic seed or dependency. Adapter: work/sparse_lcb.py.
**Reason:** The forest trial did not deliver a new incumbent and its pre-purchase means were repeatedly optimistic. A sparse GP directly addresses the many-features/few-labels imbalance and offers a different candidate ranking with slightly stronger held-out low-tail signal. This is a limited hypothesis test, not a claim that the model is validated.
**Next step limit:** 8 additional objective calls; pause at cumulative evaluation 64.
**Why this length:** Eight sequential predictions can establish whether the new sparse metric identifies a productive region while retaining 36 calls for subsequent evidence-based choices.
**Review:** Inspect improvements, prospective residuals and nugget stability. Pause earlier after two consecutive absolute residuals greater than three predictive SDs, or on evaluator error, invalid selection, nonfinite output or optimizer failure. No next method is selected in advance.

''')
exec(compile(Path('work/sparse_lcb.py').read_text(), 'work/sparse_lcb.py','exec'),globals())
start_block('sparse_gp_lcb',8)
sparse_records = []
severe_residuals=0
for _ in range(8):
    index, details = propose_sparse_lcb(1.0,0.1)
    y = buy(index,details)
    gp.tell(X[[index]],[y])
    residual=(y-details['model_mean'])/details['model_predictive_sd']
    sparse_records.append({'index':index,'y':y,**details,'standardized_residual':residual})
    severe_residuals=severe_residuals+1 if abs(residual)>3 else 0
    if severe_residuals>=2:
        print('EARLY_CHECKPOINT: two consecutive residuals exceed three predictive SDs',flush=True)
        break
checkpoint()
print(json.dumps({'sparse_block':sparse_records}),flush=True)
