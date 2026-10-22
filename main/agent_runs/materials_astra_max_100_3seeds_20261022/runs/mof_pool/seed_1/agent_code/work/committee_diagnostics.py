for weight in [.25,.5,.75]:
    pred=(1-weight)*pred_et+weight*pred_gp
    print('COMMITTEE_CV '+json.dumps({'gp_weight':weight,'rmse':float(np.sqrt(np.mean((pred-y)**2))),
      'spearman':float(spearmanr(pred,y).statistic),'tail_rmse':float(np.sqrt(np.mean((pred[tail]-y[tail])**2)))}),flush=True)
forest=ExtraTreesRegressor(n_estimators=512,max_features=.5,min_samples_leaf=2,random_state=1,n_jobs=1)
forest.fit(U[observed_indices],observed_y)
av=np.asarray(h.available_indices)
tp=np.asarray([t.predict(U[av]) for t in forest.estimators_])
fmean=tp.mean(0); fvar=tp.var(0,ddof=1)
with torch.no_grad():
    p=gp.last_model.posterior(torch.tensor(C[av],dtype=torch.double))
    gmean=p.mean[:,0].numpy(); gvar=p.variance[:,0].numpy()
weight=.25
mean=(1-weight)*fmean+weight*gmean
var=(1-weight)*(fvar+(fmean-mean)**2)+weight*(gvar+(gmean-mean)**2)
for beta in [.25,.5,1.]:
    order=np.argsort(mean-beta*np.sqrt(var))[:6]
    print('COMMITTEE_LEADERS '+json.dumps({'beta':beta,'rows':[{'index':int(av[j]),'mean':float(mean[j]),'sd':float(np.sqrt(var[j])),'forest_mean':float(fmean[j]),'gp_mean':float(gmean[j])} for j in order]}),flush=True)
