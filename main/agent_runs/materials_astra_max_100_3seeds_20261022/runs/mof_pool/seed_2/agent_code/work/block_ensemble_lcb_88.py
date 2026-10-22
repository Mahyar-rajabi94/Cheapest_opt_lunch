with Path('results/agent_run/changelog.md').open('a') as log:
    log.write('''## Decision after evaluation 88
**Budget:** 88 used / 100 total; 12 remaining.
**Evidence:** Ensemble EI's eight responses ranged from -5.486125218 (row 569) to -0.4366338857; row 837 returned -4.5607927591. None improved the incumbent -8.8133131168. The initially uncertain row 898 returned -1.3443331932, consistent with its weak predicted means. Further selected rows 883, 101 and 108 returned -0.48674, -0.96946 and -0.43663. Broader exploration obtained new labels, but no evidence of an extreme better region. Model means and uncertainties disagree materially for several selected rows, so a single component remains an uncertain guide.
**Decision:** Adjust acquisition to `sparse_ensemble_lcb`, preserving all 88 observations and the same three-model family and harness. Refit components as before, then select by an equal-weight mixture lower confidence bound with reduced uncertainty weight. No model observation history is reset.
**Hyperparameters:** Keep all MAP-SAAS component settings and taus [0.01, 0.1, 1.0] from evaluation 80. Replace mixture EI with score = mixture_mean - 0.5 * mixture_SD. The equal-weight mixture variance is mean(component_variance + component_mean^2) - mixture_mean^2, thus accounting for both each component's latent uncertainty and differences between component means. Kappa=0.5 favors low consensus means while preserving some allowance for model uncertainty. Do not interpret this mixture as a validated posterior. Other fitting, scaling, chunks and seed settings are unchanged. Adapter: work/sparse_ensemble_lcb.py uses the same supported BoTorch APIs.
**Reason:** Exploration selected mostly mediocre responses. With 12 evaluations left, the potential value of collecting labels purely to improve a model is smaller; directly testing lower predicted responses is now more useful. Retaining a mixture avoids committing to a sparsity level that has not been reliably validated.
**Next step limit:** 6 additional objective calls; pause no later than cumulative evaluation 94.
**Why this length:** Six sequential choices allow the changed acquisition to be judged on actual outcomes before deciding how to use the remaining budget. This is not an automatic allocation of the last 12 calls.
**Review:** Assess actual value versus mixture mean, any incumbent improvement, and whether the predicted low-value region is productive. Pause immediately on a strict incumbent improvement, or on evaluator error, invalid selection, nonfinite result or fitting failure. No subsequent block is preselected.

''')
exec(compile(Path('work/sparse_ensemble_lcb.py').read_text(),'work/sparse_ensemble_lcb.py','exec'),globals())
start_block('sparse_ensemble_lcb',6)
ensemble_lcb_records=[]
old_best=min(observed_y)
for _ in range(6):
    index,details=propose_sparse_ensemble_lcb(0.5)
    y=buy(index,details)
    gp.tell(X[[index]],[y])
    ensemble_lcb_records.append({'index':index,'y':y,**details})
    if y<old_best:
        print('EARLY_CHECKPOINT: new incumbent discovered',flush=True)
        break
checkpoint()
print(json.dumps({'ensemble_lcb_block':ensemble_lcb_records}),flush=True)
