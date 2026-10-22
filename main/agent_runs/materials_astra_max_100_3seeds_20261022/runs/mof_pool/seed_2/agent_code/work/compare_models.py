from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

obs = np.asarray(observed_indices, dtype=int)
yobs = np.asarray(observed_y)
folds = list(KFold(n_splits=5, shuffle=True, random_state=2).split(obs))
cv_predictions = {}
comparison_models = {
    'ridge': lambda: RidgeCV(alphas=np.logspace(-4, 4, 17)),
    'extra_trees': lambda: ExtraTreesRegressor(n_estimators=512, max_features=1.0, min_samples_leaf=1, bootstrap=False, random_state=2, n_jobs=1),
    'random_forest': lambda: RandomForestRegressor(n_estimators=512, max_features=0.5, min_samples_leaf=1, bootstrap=True, random_state=2, n_jobs=1),
}
for name, factory in comparison_models.items():
    pred = np.empty_like(yobs)
    for train, test in folds:
        m = factory().fit(Z[obs[train]], yobs[train])
        pred[test] = m.predict(Z[obs[test]])
    cv_predictions[name] = pred
    print(json.dumps({'model_cv_completed': name}), flush=True)
model_bounds = np.asarray(h.bounds, dtype=float).copy()
model_bounds[model_bounds[:,0] == model_bounds[:,1],1] += 1
pred = np.empty_like(yobs)
for train, test in folds:
    fitted = SingleTaskGP(
        torch.tensor(X[obs[train]], dtype=torch.double),
        torch.tensor(yobs[train], dtype=torch.double).unsqueeze(-1),
        input_transform=Normalize(d=X.shape[1], bounds=torch.tensor(model_bounds.T, dtype=torch.double)),
        outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(fitted.likelihood, fitted))
    with torch.no_grad():
        pred[test] = fitted.posterior(torch.tensor(X[obs[test]], dtype=torch.double)).mean.reshape(-1).numpy()
cv_predictions['gp'] = pred
cv_summary = {}
for name, pred in cv_predictions.items():
    top = np.argsort(yobs)[:10]
    predicted_top = np.argsort(pred)[:10]
    cv_summary[name] = {'rmse': float(np.sqrt(mean_squared_error(yobs,pred))),
                        'spearman': float(spearmanr(yobs,pred).statistic),
                        'top10_recall': len(set(top)&set(predicted_top))/10,
                        'top10_rmse': float(np.sqrt(np.mean((yobs[top]-pred[top])**2))),
                        'incumbent_prediction': float(pred[np.argmin(yobs)])}
    if name in comparison_models:
        full = comparison_models[name]().fit(Z[obs],yobs)
        avail = np.asarray(h.available_indices,dtype=int)
        allpred = full.predict(Z[avail])
        cv_summary[name]['lowest_predicted_indices'] = avail[np.argsort(allpred)[:10]].tolist()
print(json.dumps({'comparison_at_evaluation': h.evaluations_used, 'five_fold_seed':2,
                  'diagnostics_only':True, 'models':cv_summary}),flush=True)
