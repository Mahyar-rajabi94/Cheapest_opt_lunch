initial_note = '''# First optimization step

The objective is deterministic scalar minimization over exactly 1,000 public candidate rows with 193 features. Only unused row indices are valid; this is not a continuous box search. The hard campaign budget is 100 objective calls. There are no observations and no additional constraints. Use the single live AgentTaskHarness and campaign seed 1 throughout. The objective and its implementation remain sealed.

The public feature matrix has 165 nonconstant columns and 1,000 distinct rows. After range scaling, 14, 20, and 42 principal components explain 90%, 95%, and 99% of feature variance. These are unlabeled geometry statistics, not evidence about the objective. Similar features may have similar values, but this is only a working assumption to assess once labels exist.

## Current step only

Use a seeded diversity design, tagged `diverse_initial`: remove constant columns, scale each remaining coordinate by its public bound span, select one uniformly random row, then draw subsequent rows with probability proportional to squared Euclidean distance to the nearest already selected row (k-means++ seeding without Lloyd iterations). Use NumPy 2.4.6 default_rng(1), its default PCG64 generator, and 16 distinct rows. No objective model or inherited optimizer defaults are used. Range scaling prevents large-unit features from dominating; distance weighting supplies spread without deterministically buying only extreme rows. Collinear features can still influence this geometry.

Purchase at most 16 rows, pausing at cumulative evaluation 16. This supplies an initial range and a modest diversity sample within a 100-call budget, enough to inspect whether a feature-based model has useful signal while keeping reconsideration early. Initialization is this entire current step; no subsequent method or budget allocation is committed here.

At the checkpoint inspect every observed value, their distribution and best row, feature proximity of promising observations, and preliminary predictive evidence using only these purchased labels. Review feature/model scaling before choosing any further calls. Stop sooner on an evaluator error or nonfinite return; do not retry an unknown evaluation. Per-block and campaign counters are checked before every purchase. Save the harness trace at the checkpoint. No repetitions are planned because the task is deterministic and each row can be purchased once.
'''
results = Path('results/agent_run')
results.mkdir(parents=True, exist_ok=True)
assert (results / 'roadmap_initial.md').read_text() == ''
assert (results / 'changelog.md').read_text() == ''
assert h.evaluations_used == 0
with (results / 'roadmap_initial.md').open('w') as f:
    f.write(initial_note)
# The supplied changelog placeholder is already empty; preserve it.
initial_indices = [int(rng.integers(len(C)))]
d2 = np.sum((U - U[initial_indices[0]])**2, axis=1)
for _ in range(15):
    d2[initial_indices] = 0.0
    index = int(rng.choice(len(C), p=d2 / d2.sum()))
    initial_indices.append(index)
    d2 = np.minimum(d2, np.sum((U - U[index])**2, axis=1))
print('INITIAL_INDICES ' + json.dumps(initial_indices), flush=True)
begin_block(16, 'diverse_initial')
for index in initial_indices:
    value = purchase(index)
    if not np.isfinite(value):
        raise RuntimeError('Nonfinite objective: reconsider before any more calls')
report()
