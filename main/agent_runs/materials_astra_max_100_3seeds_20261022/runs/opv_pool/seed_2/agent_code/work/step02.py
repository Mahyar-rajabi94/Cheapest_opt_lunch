method('paired_history')
probe_a = np.array(min(observations, key=lambda r: r['y'])['x'])
probe_b = np.full(11, 0.5)
for probe_point in [probe_a, probe_a, probe_b, probe_b, probe_a, probe_b]:
    evaluate(probe_point)
