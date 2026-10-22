population = es.ask()
values = [evaluate(point) for point in population]
es.tell(population, values)
print(json.dumps({'generation': es.countiter, 'generation_best': min(values),
                  'generation_median': float(np.median(values)), 'generation_worst': max(values),
                  'sigma': float(es.sigma), 'mean': es.mean.tolist(),
                  'stds': np.asarray(es.stds).tolist(),
                  'fraction_coords_within_005_boundary': float(np.mean((np.asarray(population)<.05)|(np.asarray(population)>.95))),
                  'stop': str(es.stop())}), flush=True)
