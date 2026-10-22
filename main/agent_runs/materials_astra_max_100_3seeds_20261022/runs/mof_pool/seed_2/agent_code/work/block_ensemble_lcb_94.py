with Path('results/agent_run/changelog.md').open('a') as log:
    log.write('''## Decision after evaluation 94
**Budget:** 94 used / 100 total; 6 remaining.
**Evidence:** The six ensemble-LCB purchases produced -5.415052452, -3.954432795, -5.7640891541, -2.0871434822, -2.0137359208 and -6.7094193046. Three were below -5; block median was -4.6847426235. Row 475 is now fourth best observed, though row 399 at -8.8133131168 remains the incumbent. This response yield is stronger than the preceding ensemble-EI block (one of eight below -5). Predictions still miss in both directions: row 528 was much worse than its roughly -5.09 mixture mean, and row 475 was better than its -4.55 mixture mean. There is no evidence of calibrated mixture intervals or global optimality.
**Decision:** Continue `sparse_ensemble_lcb` with all 94 observations, the existing acquisition implementation, and the same harness. Keep ordinary component refits; do not reset data or stochastic state.
**Hyperparameters:** Unchanged from evaluation 88: equal mixture of MAP-SAAS taus [0.01, 0.1, 1.0], mixture LCB kappa=0.5, raw-label standardization, 165 bound-scaled varying features, learned nugget, maxiter=500, chunks=512 and installed BoTorch/GPyTorch defaults from evaluation 80. Retain moderate uncertainty weight because errors occur in both directions; the small new sample does not justify another sparsity search or confidence increase.
**Reason:** Three strong responses, including a new fourth-best row at the latest call, support continuing the current selection rule. No tested alternative has established a better route to the extreme incumbent. Six remaining real purchases can still test the updated model; surrogate confidence is not sufficient grounds to stop early.
**Next step limit:** 6 additional objective calls; pause no later than cumulative evaluation 100.
**Why this length:** The full remainder is only six calls, the same short experimental window that just yielded useful responses. Spending these six is specifically justified by that evidence, with an earlier checkpoint on improvement; it is not an automatic decision to consume a large remainder without review.
**Review:** Pause immediately after a strict incumbent improvement for reconsideration, or on evaluator error, invalid selection, nonfinite result or fitting failure. Otherwise inspect the final six outcomes at budget exhaustion, record the stopping decision, save via the harness and verify all four required artifacts.

''')
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
