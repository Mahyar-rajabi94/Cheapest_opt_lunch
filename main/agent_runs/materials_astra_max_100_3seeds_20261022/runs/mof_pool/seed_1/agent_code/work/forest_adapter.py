from sklearn.ensemble import ExtraTreesRegressor

forest_predictions = globals().get('forest_predictions', [])
def forest_block(k, beta, min_leaf=2, max_features=.5):
    begin_block(k, 'extra_trees_lcb')
    for _ in range(k):
        forest = ExtraTreesRegressor(n_estimators=512, max_features=max_features,
                    min_samples_leaf=min_leaf, bootstrap=False, random_state=1, n_jobs=1)
        forest.fit(U[observed_indices], np.asarray(observed_y))
        av = np.asarray(h.available_indices)
        tree_predictions = np.asarray([tree.predict(U[av]) for tree in forest.estimators_])
        means = tree_predictions.mean(0)
        sds = tree_predictions.std(0, ddof=1)
        scores = means - beta * sds
        pos = int(np.argmin(scores))
        index = int(av[pos])
        if not np.isfinite([means[pos], sds[pos], scores[pos]]).all():
            raise RuntimeError('Nonfinite forest prediction')
        print('FOREST_PREDICTION '+json.dumps({'index':index,'mean':float(means[pos]),
               'tree_sd':float(sds[pos]),'beta':beta,'score':float(scores[pos])}), flush=True)
        value = purchase(index)
        if not np.isfinite(value):
            raise RuntimeError('Nonfinite objective')
        forest_predictions.append([index, float(means[pos]), float(sds[pos]), value, beta])
    report()
    recent = forest_predictions[-k:]
    print('FOREST_BLOCK_DIAGNOSTICS '+json.dumps({'rmse':float(np.sqrt(np.mean([(p[1]-p[3])**2 for p in recent]))),
           'mean_y':float(np.mean([p[3] for p in recent])), 'block_min_y':float(min(p[3] for p in recent))}), flush=True)
