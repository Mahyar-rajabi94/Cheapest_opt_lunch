for _ in range(2):
    population = es.ask()
    values = [evaluate(point) for point in population]
    es.tell(population, values)
    print(json.dumps({'generation': es.countiter, 'generation_best': min(values),
                      'generation_median': float(np.median(values)), 'generation_worst': max(values),
                      'sigma': float(es.sigma), 'mean': es.mean.tolist(),
                      'stds': np.asarray(es.stds).tolist(), 'stop': str(es.stop())}), flush=True)
    if es.stop():
        break
