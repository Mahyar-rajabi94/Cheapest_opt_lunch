assert np.all(np.array(min(observations, key=lambda r: r['y'])['x']) >= local_bounds[:,0])
assert np.all(np.array(min(observations, key=lambda r: r['y'])['x']) <= local_bounds[:,1])
local_candidates = local_sobol.ask(n=4)
local_cursor = 0
block_start_best = min(r['y'] for r in observations)
early_gain = False
for local_candidate in local_candidates:
    method('gp_history_exploit')
    for _ in range(2):
        gp_step(beta=0.25)
        if min(r['y'] for r in observations) <= block_start_best-2.:
            early_gain = True
            break
    if early_gain:
        break
    method('local_sobol')
    local_y = evaluate(local_candidate)
    local_sobol.tell([local_candidate], [local_y])
    local_cursor += 1
    if min(r['y'] for r in observations) <= block_start_best-2.:
        early_gain = True
        break
print('local_block', json.dumps(dict(early_gain=early_gain, local_used=local_cursor,
                                   local_remaining=4-local_cursor)), flush=True)
