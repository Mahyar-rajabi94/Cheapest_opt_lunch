from scipy.stats import spearmanr
X_all = np.array([r['x'] for r in observations])
y_all = np.array([r['y'] for r in observations])
print('coordinate_spearman', [round(float(spearmanr(initial_design[:, j], initial_values).statistic), 3) for j in range(11)])
print('order_spearman', float(spearmanr(np.arange(len(y_all)), y_all).statistic))
for name, location in [('A', probe_a), ('B', probe_b)]:
    vals = y_all[np.all(X_all == location, axis=1)]
    print('repeat', name, dict(values=vals.tolist(), mean=float(vals.mean()), sd=float(vals.std(ddof=1))))
# Leave each distinct location out, including all its repeats. Fit spatial
# ridge models only as a diagnostic; their predictions are not evaluations.
unique, groups = np.unique(X_all, axis=0, return_inverse=True)
for alpha in [0.1, 1., 10., 100.]:
    pred = np.empty(len(y_all)); baseline = np.empty(len(y_all))
    for group in range(len(unique)):
        test = groups == group; train = ~test
        train_mean = y_all[train].mean()
        xc = X_all[train].mean(axis=0)
        z = X_all[train] - xc
        coef = np.linalg.solve(z.T @ z + alpha*np.eye(11), z.T @ (y_all[train]-train_mean))
        pred[test] = train_mean + (X_all[test]-xc) @ coef
        baseline[test] = train_mean
    print('spatial_validation', dict(alpha=alpha, rmse=float(np.sqrt(np.mean((y_all-pred)**2))),
          constant_rmse=float(np.sqrt(np.mean((y_all-baseline)**2)))))
