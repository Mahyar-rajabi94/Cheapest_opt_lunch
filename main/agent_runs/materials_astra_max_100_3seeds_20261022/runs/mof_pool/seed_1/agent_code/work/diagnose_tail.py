y=np.asarray(observed_y)
indices=np.asarray(observed_indices)
folds=list(KFold(n_splits=5,shuffle=True,random_state=1).split(y))
tail_cv=[]
for power,leaf in [(1,1),(1,2),(2,2)]:
    pred=np.zeros(len(y))
    t=np.sign(y)*np.abs(y)**power
    for tr,te in folds:
        est=ExtraTreesRegressor(n_estimators=384,max_features=.5,min_samples_leaf=leaf,random_state=1,n_jobs=1)
        est.fit(U[indices[tr]],t[tr])
        p=est.predict(U[indices[te]])
        pred[te]=np.sign(p)*np.abs(p)**(1/power)
    tail=np.argsort(y)[:max(5,len(y)//5)]
    tail_cv.append({'power':power,'leaf':leaf,'rmse':float(np.sqrt(np.mean((pred-y)**2))),
                    'tail_rmse':float(np.sqrt(np.mean((pred[tail]-y[tail])**2))),
                    'spearman':float(spearmanr(y,pred).statistic)})
    est=ExtraTreesRegressor(n_estimators=512,max_features=.5,min_samples_leaf=leaf,random_state=1,n_jobs=1)
    est.fit(U[indices],t)
    av=np.asarray(h.available_indices)
    tp=np.asarray([tr.predict(U[av]) for tr in est.estimators_])
    means=tp.mean(0); sds=tp.std(0,ddof=1)
    best=np.argsort(means-sds)[:8]
    print('TAIL_MODEL_LEADERS '+json.dumps({'power':power,'leaf':leaf,'leaders':[{'index':int(av[j]),'inverse_mean':float(np.sign(means[j])*np.abs(means[j])**(1/power)),'score_transformed':float(means[j]-sds[j])} for j in best]}),flush=True)
print('TAIL_CV '+json.dumps(tail_cv),flush=True)
for idx in np.asarray(observed_indices)[np.argsort(y)[:3]]:
    dist=np.sqrt(np.sum((U[av]-U[idx])**2,axis=1))
    print('NEAREST_UNUSED '+json.dumps({'center':int(idx),'neighbors':[{'index':int(av[j]),'distance':float(dist[j])} for j in np.argsort(dist)[:6]]}),flush=True)
