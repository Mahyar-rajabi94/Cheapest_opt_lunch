revisit_reference = np.array(min(observations, key=lambda r: r['y'])['x'])
block_start_best = min(r['y'] for r in observations)
early_gain = False
for _ in range(4):
    method('incumbent_revisit')
    evaluate(revisit_reference)
    if min(r['y'] for r in observations) <= block_start_best-2.:
        early_gain = True
        break
    method('gp_history_exploit')
    gp_step(beta=0.25)
    if min(r['y'] for r in observations) <= block_start_best-2.:
        early_gain = True
        break
print('revisit_block', json.dumps(dict(early_gain=early_gain)), flush=True)
