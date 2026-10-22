unique_elites = []
for record in sorted(observations, key=lambda r: r['y']):
    if not any(np.array_equal(record['x'], elite) for elite in unique_elites):
        unique_elites.append(np.array(record['x']))
    if len(unique_elites) == 2:
        break
assert len(unique_elites) == 2
elite_rng = np.random.default_rng(95)
block_start_best = min(r['y'] for r in observations)
early_gain = False
method('elite_revisits')
for _ in range(4):
    for which in elite_rng.permutation(2):
        evaluate(unique_elites[which])
        if min(r['y'] for r in observations) <= block_start_best-2.:
            early_gain = True
            break
    if early_gain:
        break
print('elite_block', json.dumps(dict(early_gain=early_gain)), flush=True)
