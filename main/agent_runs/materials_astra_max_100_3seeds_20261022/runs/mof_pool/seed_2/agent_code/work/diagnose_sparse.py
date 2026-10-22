from botorch.fit import get_fitted_map_saas_model
obs = np.asarray(observed_indices, dtype=int)
yobs = np.asarray(observed_y)
folds = list(KFold(n_splits=5, shuffle=True, random_state=2).split(obs))
sparse_pred = np.empty_like(yobs)
start_time = time.monotonic()
for fold, (train, test) in enumerate(folds):
    sparse = get_fitted_map_saas_model(
        train_X=torch.tensor(Z[obs[train]], dtype=torch.double),
        train_Y=torch.tensor(yobs[train], dtype=torch.double).unsqueeze(-1),
        tau=0.1, outcome_transform=Standardize(m=1),
        optimizer_kwargs={'options':{'maxiter':500}})
    with torch.no_grad():
        sparse_pred[test] = sparse.posterior(torch.tensor(Z[obs[test]], dtype=torch.double)).mean.reshape(-1).numpy()
    print(json.dumps({'sparse_cv_fold':fold+1,'elapsed':time.monotonic()-start_time}),flush=True)
sparse_cv = {'rmse':float(np.sqrt(np.mean((yobs-sparse_pred)**2))),
             'spearman':float(spearmanr(yobs,sparse_pred).statistic),
             'top10_recall':len(set(np.argsort(yobs)[:10])&set(np.argsort(sparse_pred)[:10]))/10,
             'incumbent_prediction':float(sparse_pred[np.argmin(yobs)])}
sparse_model = get_fitted_map_saas_model(
    train_X=torch.tensor(Z[obs], dtype=torch.double),
    train_Y=torch.tensor(yobs, dtype=torch.double).unsqueeze(-1),
    tau=0.1, outcome_transform=Standardize(m=1),
    optimizer_kwargs={'options':{'maxiter':500}})
with torch.no_grad():
    avail = np.asarray(h.available_indices, dtype=int)
    p = sparse_model.posterior(torch.tensor(Z[avail], dtype=torch.double))
    pm = p.mean.reshape(-1).numpy()
    ps = p.variance.reshape(-1).sqrt().numpy()
    ls = sparse_model.covar_module.base_kernel.lengthscale.reshape(-1).numpy()
    candidate_rank = np.argsort(pm-ps)[:12]
    sparse_summary = {'five_fold':sparse_cv, 'noise':sparse_model.likelihood.noise.reshape(-1).tolist(),
                      'lengthscale_quantiles':np.quantile(ls,[0,.1,.25,.5,.75,1]).tolist(),
                      'shortest_feature_indices':np.flatnonzero(active)[np.argsort(ls)[:10]].tolist(),
                      'lowest_lcb_predictions':[{'index':int(avail[i]),'model_mean':float(pm[i]),'model_sd':float(ps[i])} for i in candidate_rank]}
print(json.dumps({'sparse_diagnostic_at_evaluation':h.evaluations_used,**sparse_summary}),flush=True)
