def propose_forest_lcb(kappa):
    forest = RandomForestRegressor(n_estimators=512, max_features=0.5,
        min_samples_leaf=1, min_samples_split=2, max_depth=None,
        bootstrap=True, max_samples=None, criterion='squared_error',
        random_state=2, n_jobs=1)
    forest.fit(Z[observed_indices], observed_y)
    available = np.asarray(h.available_indices, dtype=int)
    tree_predictions = np.asarray([tree.predict(Z[available]) for tree in forest.estimators_])
    mean = tree_predictions.mean(axis=0)
    spread = tree_predictions.std(axis=0, ddof=1)
    score = mean-kappa*spread
    selected = int(np.argmin(score))
    return int(available[selected]), {
        'model_mean':float(mean[selected]), 'tree_sd':float(spread[selected]),
        'lcb':float(score[selected]), 'kappa':kappa}
