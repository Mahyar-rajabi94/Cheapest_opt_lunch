face_reference = np.array(min(observations, key=lambda r: r['y'])['x'])
face_active = np.array([1, 2, 3, 6, 7, 10])
face_bounds = np.column_stack([np.maximum(0.,face_reference[face_active]-0.1),
                               np.minimum(1.,face_reference[face_active]+0.1)])
face_sobol = SobolSearch(bounds=face_bounds, seed=86)
face_candidates = face_sobol.ask(n=4)
pair_rng = np.random.default_rng(87)
block_start_best = min(r['y'] for r in observations)
early_gain = False
face_evaluated = 0
for local_subpoint in face_candidates:
    local_point = face_reference.copy()
    local_point[face_active] = local_subpoint
    for kind in pair_rng.permutation(2):
        if kind == 0:
            method('incumbent_revisit')
            evaluate(face_reference)
        else:
            method('face_sobol')
            face_y = evaluate(local_point)
            face_sobol.tell([local_subpoint], [face_y])
            face_evaluated += 1
        if min(r['y'] for r in observations) <= block_start_best-2.:
            early_gain = True
            break
    if early_gain:
        break
print('face_block', json.dumps(dict(early_gain=early_gain, face_evaluated=face_evaluated)), flush=True)
