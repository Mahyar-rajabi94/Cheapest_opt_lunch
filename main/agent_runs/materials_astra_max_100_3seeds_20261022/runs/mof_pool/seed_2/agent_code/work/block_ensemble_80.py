with Path('results/agent_run/changelog.md').open('a') as log:
    log.write('''## Decision after evaluation 80
**Budget:** 80 used / 100 total; 20 remaining.
**Evidence:** All six nearest-incumbent purchases were weak relative to the incumbent: -3.5720311694, -2.4998430194, -0.3447695799, -0.8421513303, -1.5838467004 and -0.8514313095. This fails the specific hypothesis that the untested immediate neighbors are productive under the equal-weight feature metric. Best remains row 399 at -8.8133131168. An objective-free diagnostic compared MAP-SAAS models at fixed tau values 0.01, 0.1 and 1.0 using all 80 labels. Their candidate predictions differ; model-specific highest-EI choices were 764, 898 and 898. The mixture's highest-EI row is 898, with means -1.7043/-1.5613/-1.0564 and latent SDs 1.8567/2.2202/2.5125. Mixture log EI there is -7.9558. Thus these acquisitions favor uncertain regions, not a prediction of a near-certain improvement. Given previous calibration failures, the low predicted improvement cannot certify exhaustion.
**Decision:** Switch to `sparse_ensemble_ei`, an equal mixture of three sparse GP models with broader and narrower sparsity assumptions. Refit all components sequentially after each real observation, retaining all 80 observations and the same harness. No source identity, unobserved label or row-order response signal is used. Continue synchronizing the original GPBO store.
**Hyperparameters:** Three BoTorch 0.18.1 get_fitted_map_saas_model components with fixed taus [0.01, 0.1, 1.0], equal mixture weights 1/3. The log-spaced sparsity levels bracket the previous 0.1 setting to represent uncertainty about how many features matter. Each component retains the raw-target Standardize(m=1), 165 varying bound-scaled inputs, inferred nugget, maxiter=500, double precision and installed model defaults specified after evaluation 56. Score all available rows by logsumexp(component analytic LogExpectedImprovement against the actual best observed raw y)-log(3); maximize=False, posterior chunks=512. No extra exploration offset. This averages EI, not log EI. Work adapter: work/sparse_ensemble.py. No random hyperparameter samples or new seeds; retain seed state from 2.
**Reason:** Immediate-neighbor testing and recent moderate-value exploitation have not improved the extreme incumbent. A broader acquisition is now justified. Averaging over several sparsity assumptions addresses model uncertainty more directly than continuing one unreliable fitted metric. The decision does not assume that all remaining rows are poor simply because their fitted means are unpromising.
**Next step limit:** 8 additional objective calls; pause no later than cumulative evaluation 88.
**Why this length:** Eight proposals provide a bounded test of the changed exploratory acquisition and leave 12 calls for a decision informed by its outcomes. The remaining budget is not allocated beyond this block.
**Review:** Inspect actual response distribution, component disagreement and any new best. Pause immediately after a strict incumbent improvement, or on evaluator error, invalid selection, nonfinite result or optimizer failure. Future method and block length remain undecided.

''')
start_block('sparse_ensemble_ei',8)
ensemble_records=[]
old_best=min(observed_y)
for _ in range(8):
    index,details=propose_sparse_ensemble()
    y=buy(index,details)
    gp.tell(X[[index]],[y])
    ensemble_records.append({'index':index,'y':y,**details})
    if y<old_best:
        print('EARLY_CHECKPOINT: new incumbent discovered',flush=True)
        break
checkpoint()
print(json.dumps({'ensemble_block':ensemble_records}),flush=True)
