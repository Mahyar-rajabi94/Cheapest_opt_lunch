from scipy.special import ndtr, ndtri
anchor_idx = int(np.argmin(Y))
anchor = X[anchor_idx].copy()
anchor_old = Y[anchor_idx]
anchor_now = evaluate(anchor)
if anchor_now - anchor_old > 4.0:
    print(json.dumps({'early_checkpoint': 'reference degraded by more than 4',
                      'anchor_old': anchor_old, 'anchor_now': anchor_now}), flush=True)
else:
    local_sigma = 0.08
    uniforms = sobol.ask(8)
    lo_cdf = ndtr(-anchor / local_sigma)
    hi_cdf = ndtr((1.0 - anchor) / local_sigma)
    local_points = anchor + local_sigma * ndtri(lo_cdf + uniforms * (hi_cdf - lo_cdf))
    local_values = [evaluate(point) for point in local_points]
    print(json.dumps({'anchor_old': anchor_old, 'anchor_now': anchor_now,
                      'local_best': min(local_values), 'local_median': float(np.median(local_values)),
                      'local_worst': max(local_values),
                      'local_distances': np.linalg.norm(local_points-anchor, axis=1).tolist()}), flush=True)
