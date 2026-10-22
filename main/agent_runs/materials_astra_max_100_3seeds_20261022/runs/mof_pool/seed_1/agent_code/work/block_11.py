entry = '''
## Decision after evaluation 94
**Budget:** 94 used / 100 total; 6 remaining.
**Evidence:** The six hybrid selections had mean -3.2755893656, minimum -5.091969491 at row 637, and prospective RMSE 1.21059. Rows 813 and 694 returned about -3.89 and -3.90, better than their initial forest-only predictions; row 637 was also better than either model predicted. Two later choices returned about -1.8 to -1.9. These are useful but mixed results; none improves the incumbent -9.2928301557. The blend has not established superiority from six adaptively selected rows, but its predictions and several observed low values support continued use more than an untested restart.
**Decision:** Continue the existing GP-forest committee, preserving all 94 observations and sequential feedback state. Keep the same harness and enforce the remaining six-call limit.
**Hyperparameters:** Unchanged from evaluation 88: forest/GP weights 0.75/0.25, beta=0.5, mixture variance including between-model disagreement, raw labels, regularized 512-tree ExtraTrees, and supplied GP model defaults. The small trial gives no clear evidence for changing weights or exploration strength again.
**Reason:** The current committee has meaningful prospective calibration and has identified useful candidates from the remaining pool. Six remaining evaluations are better supported by the fitted models and all accumulated data than by an unsupported new model initialization. Further improvement is uncertain; no available evidence certifies optimality.
**Next step limit:** 6 additional objective calls; pause at cumulative evaluation 100 or earlier on failure.
**Why this length:** Six calls are both a small block, equal to the completed prospective trial, and the entire remaining hard allowance. This length is chosen because the existing sequential method can use every result immediately and there is no evidence requiring an intermediate review after only one or two calls. It is not an earlier budget allocation or an automatic rule to consume all remaining calls.
**Review:** At the final checkpoint record the best actually purchased value and row, confirm exactly 100 total purchases if the block succeeds, append the stop decision, save with the harness, and verify the four required files. Pause immediately on model failure, a nonfinite result, or evaluator error; do not retry unknown evaluations.
'''
with Path('results/agent_run/changelog.md').open('a') as f:
    f.write(entry)
committee_block(6,beta=.5,gp_weight=.25)
