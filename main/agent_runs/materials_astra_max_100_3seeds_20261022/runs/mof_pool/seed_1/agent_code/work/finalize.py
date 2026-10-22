import csv
import re
assert h.evaluations_used==100 and h.evaluations_remaining==0
assert len(observed_indices)==100 and len(set(observed_indices))==100
assert len(h.available_indices)==900
assert Path('results/agent_run/roadmap_initial.md').read_text()==initial_note
entry = '''
## Decision after evaluation 100
**Budget:** 100 used / 100 total; 0 remaining.
**Evidence:** The final six committee selections had mean -3.5649329912, minimum -4.7918143565, and prospective RMSE 1.36490. No final-block observation improved the incumbent. Across all 100 real purchases, the best observed objective is -9.2928301557 at candidate index 561, first purchased at evaluation 54 under raw-target ExtraTrees-LCB with beta=1.0 and min_samples_leaf=2. The next-best observed values are -9.0999790722 at index 86 and -8.5532416635 at index 899. The first block's best was -6.5720049767, so the campaign improved it by 2.7208251790. All 100 purchases used distinct valid candidate indices; 900 candidate rows remain unobserved. No global-optimum claim is supported. There were no evaluator errors or duplicate purchase attempts. A pre-evaluation exclusive file-create attempt encountered the supplied empty roadmap placeholder; it performed no evaluation and the empty placeholder was then populated once before evaluation 1.
**Decision:** Stop because the hard evaluation budget is exhausted. Preserve the results and save through the same harness. No additional objective call is authorized or attempted.
**Hyperparameters:** Not applicable; no next optimization block.
**Reason:** All 100 allowed evaluations have been used. The best observed row is the campaign result, not an inferred or simulated observation.
**Next step limit:** 0 additional objective calls; final cumulative evaluation count 100.
**Why this length:** No evaluation allowance remains.
**Review:** Final verification only: save with the harness; verify exactly roadmap_initial.md, changelog.md, trace.csv, and meta.json in the results directory, 100 trace rows, unchanged initial record, chronological decision checkpoints, and consistency with the live counter. No further evaluations.
'''
with Path('results/agent_run/changelog.md').open('a') as f:
    f.write(entry)
h.save()
result_dir=Path('results/agent_run')
expected={'roadmap_initial.md','changelog.md','trace.csv','meta.json'}
assert {p.name for p in result_dir.iterdir()}==expected
with (result_dir/'trace.csv').open(newline='') as f:
    reader=csv.DictReader(f)
    trace=list(reader)
    columns=reader.fieldnames
assert len(trace)==100
assert all((result_dir/name).stat().st_size>0 for name in expected)
json.loads((result_dir/'meta.json').read_text())
checkpoints=[int(n) for n in re.findall(r'^## Decision after evaluation (\d+)',(result_dir/'changelog.md').read_text(),flags=re.M)]
assert checkpoints==[16,26,34,46,56,64,72,80,88,94,100]
print('FINAL_VERIFICATION '+json.dumps({'evaluations':h.evaluations_used,'unique_indices':len(set(observed_indices)),
      'best_index':observed_indices[int(np.argmin(observed_y))],'best_y':min(observed_y),
      'trace_rows':len(trace),'trace_columns':columns,'files':sorted(expected),
      'initial_record_unchanged':True,'decision_checkpoints':checkpoints}),flush=True)
