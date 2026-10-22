comparison_56 = {'sparse_raw':sparse_cv}
for name in ['ridge','random_forest']:
    pred = np.empty_like(yobs)
    for train,test in folds:
        m = comparison_models[name]().fit(Z[obs[train]],yobs[train])
        pred[test] = m.predict(Z[obs[test]])
    comparison_56[name] = {'rmse':float(np.sqrt(np.mean((yobs-pred)**2))),
        'spearman':float(spearmanr(yobs,pred).statistic),
        'top10_recall':len(set(np.argsort(yobs)[:10])&set(np.argsort(pred)[:10]))/10,
        'incumbent_prediction':float(pred[np.argmin(yobs)])}
for name,tau in [('sparse_asinh_tau01',0.1),('sparse_asinh_tau001',0.01)]:
    pred = np.empty_like(yobs)
    for train,test in folds:
        fitted = get_fitted_map_saas_model(
            train_X=torch.tensor(Z[obs[train]],dtype=torch.double),
            train_Y=torch.tensor(np.arcsinh(yobs[train]),dtype=torch.double).unsqueeze(-1),
            tau=tau, outcome_transform=Standardize(m=1),
            optimizer_kwargs={'options':{'maxiter':500}})
        with torch.no_grad():
            pred[test] = np.sinh(fitted.posterior(torch.tensor(Z[obs[test]],dtype=torch.double)).mean.reshape(-1).numpy())
    comparison_56[name] = {'rmse':float(np.sqrt(np.mean((yobs-pred)**2))),
        'spearman':float(spearmanr(yobs,pred).statistic),
        'top10_recall':len(set(np.argsort(yobs)[:10])&set(np.argsort(pred)[:10]))/10,
        'incumbent_prediction':float(pred[np.argmin(yobs)])}
    print(json.dumps({'completed_diagnostic':name}),flush=True)
print(json.dumps({'comparison_at_evaluation':h.evaluations_used,'all_same_folds':True,'seed':2,'models':comparison_56}),flush=True)
