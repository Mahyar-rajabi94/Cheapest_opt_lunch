import cma
cma_start = np.mean(np.asarray(X)[np.argsort(Y)[:4]], axis=0)
es = cma.CMAEvolutionStrategy(cma_start, 0.22, {'bounds': [0.0, 1.0], 'popsize': 12,
                                               'seed': 4294967295, 'verbose': -9, 'verb_log': 0})
population = es.ask()
values = [evaluate(point) for point in population]
es.tell(population, values)
print(json.dumps({'generation': es.countiter, 'generation_best': min(values),
                  'generation_median': float(np.median(values)), 'generation_worst': max(values),
                  'sigma': float(es.sigma), 'mean': es.mean.tolist(),
                  'stds': np.asarray(es.stds).tolist(), 'stop': str(es.stop())}), flush=True)
