from scipy.stats import qmc
local_center = X[int(np.argmin(Y))].copy()
local_sampler = qmc.Sobol(d=h.dim, scramble=True, seed=2)
local_unit = local_sampler.random_base2(m=3)
local_raw = local_center + 0.12 * (2.0 * local_unit - 1.0)
local_design = 1.0 - np.abs(np.mod(local_raw, 2.0) - 1.0)
begin_block('local_sobol_recheck', 10)
local_anchor_before = evaluate(local_center, 'local_anchor_before')
local_values = [evaluate(x, 'local_sobol') for x in local_design]
local_anchor_after = evaluate(local_center, 'local_anchor_after')
checkpoint()
print(json.dumps({'local_repeat_values': [local_anchor_before, local_anchor_after],
                  'local_min': min(local_values), 'local_median': float(np.median(local_values)),
                  'below_minus10': int(np.sum(np.asarray(local_values) < -10)),
                  'below_minus11': int(np.sum(np.asarray(local_values) < -11))}), flush=True)
