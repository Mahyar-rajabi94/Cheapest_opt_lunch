import csv
from collections import Counter
assert h.evaluations_used==100 and h.evaluations_remaining==0
assert len(observed_indices)==len(set(observed_indices))==100
with Path('results/agent_run/changelog.md').open('a') as log:
    log.write('''## Decision after evaluation 100
**Budget:** 100 used / 100 total; 0 remaining.
**Evidence:** The final six responses were -4.3919040902, -4.1401607986, -1.5502042981, -3.699232456, -1.0733935841 and -1.776972194. No new incumbent was found. Final best observed value is -8.8133131168 at candidate index 399, first purchased on evaluation 24 using pool_gp_ei. The initial 16-point design's best was -5.6182484507, so the campaign reduced the best value by 3.1950646661. Next-best observed rows are 251 at -7.0400773211 and 388 at -6.9514815426. All 100 purchases used distinct valid row indices through the one persistent harness; no evaluator error occurred. Surrogate predictions and diagnostics are not evaluations and did not enter the harness trace. The 900 unpurchased labels remain unknown, so this result does not certify a global optimum.
**Decision:** Stop because the hard objective-evaluation budget is exhausted. Preserve the full observation history and save the trace and metadata through h.save(). Verify that the results directory contains exactly roadmap_initial.md, changelog.md, trace.csv and meta.json, and that saved counts and the incumbent match the actual purchased observations.
**Hyperparameters:** Not applicable; no further optimization block or objective calls.
**Reason:** The campaign has used all 100 authorized evaluations. Further evaluation would violate the hard budget. The finite-pool optimum is private; stopping is based on the budget, not a convergence claim.
**Next step limit:** 0 additional objective calls; final cumulative count 100.
**Why this length:** No budget remains.
**Review:** Not applicable to optimization; perform read-only artifact integrity checks after saving. The initial roadmap remains unchanged and this decision history remains append-only.
''')
h.save()
result_dir=Path('results/agent_run')
expected={'roadmap_initial.md','changelog.md','trace.csv','meta.json'}
assert {p.name for p in result_dir.iterdir()}==expected
meta=json.loads((result_dir/'meta.json').read_text())
with (result_dir/'trace.csv').open(newline='') as handle:
    reader=csv.DictReader(handle)
    trace_rows=list(reader)
    trace_fields=reader.fieldnames
print(json.dumps({'trace_fields':trace_fields,'trace_rows':len(trace_rows),
    'files':{p.name:p.stat().st_size for p in result_dir.iterdir()},
    'saved_evaluations_used':meta['evaluations_used'],
    'saved_best_y':meta['best_y'],'saved_best_candidate_index':meta['best_candidate_index']}),flush=True)
assert len(trace_rows)==100
assert meta['evaluations_used']==100 and meta['evaluations_remaining']==0
assert len(meta['available_indices'])==900
assert meta['best_y']==min(observed_y)
assert meta['best_candidate_index']==observed_indices[int(np.argmin(observed_y))]
assert np.array_equal(np.asarray(meta['best_x']),X[meta['best_candidate_index']])
assert np.array_equal(np.array([float(row['y']) for row in trace_rows]),np.asarray(observed_y))
assert [int(row['candidate_index']) for row in trace_rows]==observed_indices
print(json.dumps({'integrity_checks':'passed','best_index':meta['best_candidate_index'],
    'best_y':meta['best_y'],'evaluations':100,'unique_purchases':100,
    'method_counts':dict(Counter(row['optimizer'] for row in trace_rows))}),flush=True)
