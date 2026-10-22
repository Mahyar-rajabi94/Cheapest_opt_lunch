from botorch.acquisition import LogExpectedImprovement
y=np.asarray(observed_y)
idx=np.asarray(observed_indices)
folds=list(KFold(n_splits=5,shuffle=True,random_state=1).split(y))
pred_gp=np.zeros(len(y)); pred_et=np.zeros(len(y)); sd_gp=np.zeros(len(y))
for tr,te in folds:
    cv_gp=GPBO(candidates=C,bounds=bounds,seed=1,direction='min',n_init=16)
    cv_gp.tell(C[idx[tr]],y[tr])
    m=cv_gp._fit_model()
    with torch.no_grad():
        post=m.posterior(torch.tensor(C[idx[te]],dtype=torch.double))
        pred_gp[te]=post.mean[:,0].numpy()
        sd_gp[te]=post.variance[:,0].sqrt().numpy()
    cv_et=ExtraTreesRegressor(n_estimators=384,max_features=.5,min_samples_leaf=2,random_state=1,n_jobs=1)
    cv_et.fit(U[idx[tr]],y[tr]); pred_et[te]=cv_et.predict(U[idx[te]])
tail=np.argsort(y)[:len(y)//5]
print('UPDATED_GP_CV '+json.dumps({'n':len(y),
    'gp_rmse':float(np.sqrt(np.mean((pred_gp-y)**2))),
    'gp_spearman':float(spearmanr(y,pred_gp).statistic),
    'gp_tail_rmse':float(np.sqrt(np.mean((pred_gp[tail]-y[tail])**2))),
    'gp_coverage_two_latent_sd':float(np.mean(np.abs(pred_gp-y)<=2*sd_gp)),
    'et_rmse':float(np.sqrt(np.mean((pred_et-y)**2))),
    'et_spearman':float(spearmanr(y,pred_et).statistic),
    'et_tail_rmse':float(np.sqrt(np.mean((pred_et[tail]-y[tail])**2)))}),flush=True)
prior_n=len(gp.y_)
gp.tell(C[idx[prior_n:]],y[prior_n:])
model=gp._fit_model()
av=np.asarray(h.available_indices)
with torch.no_grad():
    tx=torch.tensor(C[av],dtype=torch.double)
    post=model.posterior(tx)
    means=post.mean[:,0].numpy(); sds=post.variance[:,0].sqrt().numpy()
    logei=LogExpectedImprovement(model=model,best_f=float(y.min()),maximize=False)
    scores=logei(tx.unsqueeze(1)).numpy()
    order=np.argsort(scores)[::-1][:8]
    print('UPDATED_GP_LEADERS '+json.dumps({'noise':float(model.likelihood.noise.item()),
       'rows':[{'index':int(av[j]),'mean':float(means[j]),'sd':float(sds[j]),'logei':float(scores[j])} for j in order]}),flush=True)
