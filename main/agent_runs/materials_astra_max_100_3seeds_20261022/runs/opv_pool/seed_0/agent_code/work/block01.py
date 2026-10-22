from src.optimizers.sobol_search import SobolSearch
sobol = SobolSearch(bounds=h.bounds, seed=0)
center = np.full(h.dim, 0.5)
start_reference = evaluate(center)
initial_points = sobol.ask(16)
for point in initial_points:
    y = evaluate(point)
    sobol.tell([point], [y])
end_reference = evaluate(center)
print(json.dumps({'reference_start': start_reference, 'reference_end': end_reference,
                  'reference_change': end_reference - start_reference,
                  'sample_quantiles': np.quantile(Y[1:-1], [0, .25, .5, .75, 1]).tolist()}), flush=True)
