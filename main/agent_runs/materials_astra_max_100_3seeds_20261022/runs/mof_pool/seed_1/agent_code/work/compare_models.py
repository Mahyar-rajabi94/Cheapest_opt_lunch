from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from scipy.spatial.distance import pdist

Z = (U - U.mean(0)) / np.maximum(U.std(0), 1e-6)
median_distance = float(np.median(pdist(U)))
models = {
    'extra_leaf1': lambda: ExtraTreesRegressor(n_estimators=256, max_features=1.0, min_samples_leaf=1, random_state=1, n_jobs=1),
    'extra_leaf2': lambda: ExtraTreesRegressor(n_estimators=256, max_features=.5, min_samples_leaf=2, random_state=1, n_jobs=1),
    'rf_leaf1': lambda: RandomForestRegressor(n_estimators=256, max_features=.5, min_samples_leaf=1, random_state=1, n_jobs=1),
    'ridge_1': lambda: Ridge(alpha=1.),
    'ridge_10': lambda: Ridge(alpha=10.),
    'ridge_100': lambda: Ridge(alpha=100.),
    'knn3': lambda: KNeighborsRegressor(n_neighbors=3, weights='distance'),
    'knn5': lambda: KNeighborsRegressor(n_neighbors=5, weights='distance'),
    'rbf_ridge01': lambda: KernelRidge(alpha=.1, kernel='rbf', gamma=1 / median_distance**2),
}
y = np.asarray(observed_y)
X = U[observed_indices]
cv = list(KFold(n_splits=5, shuffle=True, random_state=1).split(X))
comparison = []
for name, factory in models.items():
    pred = np.zeros(len(y))
    for tr, te in cv:
        estimator = factory()
        ym = y[tr].mean()
        estimator.fit(X[tr], y[tr] - ym)
        pred[te] = estimator.predict(X[te]) + ym
    comparison.append({'model':name,'rmse':float(np.sqrt(np.mean((pred-y)**2))),
                       'spearman':float(spearmanr(y,pred).statistic),
                       'best_row_prediction':float(pred[np.argmin(y)])})
print('MODEL_CV '+json.dumps(sorted(comparison, key=lambda r:r['rmse'])), flush=True)
for name in ['extra_leaf1','extra_leaf2','rf_leaf1','ridge_10']:
    estimator = models[name]()
    estimator.fit(X, y)
    av=np.array(h.available_indices)
    means=estimator.predict(U[av])
    print('MODEL_LEADERS '+json.dumps({'model':name,'candidates':[{'index':int(av[i]),'mean':float(means[i])} for i in np.argsort(means)[:8]]}), flush=True)

D = np.sqrt(np.sum((U[observed_indices][:, None, :] - U[observed_indices][None, :, :])**2, axis=2))
np.fill_diagonal(D, np.inf)
print('OBSERVED_NEAREST '+json.dumps([{'index':int(observed_indices[i]),'y':float(y[i]),'nearest':int(observed_indices[np.argmin(D[i])]),'neighbor_y':float(y[np.argmin(D[i])]),'distance':float(D[i].min())} for i in np.argsort(y)[:5]]), flush=True)
