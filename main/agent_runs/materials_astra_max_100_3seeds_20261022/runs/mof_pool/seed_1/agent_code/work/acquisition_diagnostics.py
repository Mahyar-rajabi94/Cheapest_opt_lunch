forest = ExtraTreesRegressor(n_estimators=512, max_features=.5, min_samples_leaf=2, bootstrap=False, random_state=1, n_jobs=1)
forest.fit(U[observed_indices], observed_y)
av=np.asarray(h.available_indices)
tp=np.asarray([tree.predict(U[av]) for tree in forest.estimators_])
means=tp.mean(0)
sds=tp.std(0,ddof=1)
for beta in [0., 1., 1.5, 2.]:
    order=np.argsort(means-beta*sds)[:8]
    print('ACQUISITION_DIAGNOSTIC '+json.dumps({'beta':beta,'leaders':[{'index':int(av[i]),'mean':float(means[i]),'tree_sd':float(sds[i])} for i in order]}), flush=True)
