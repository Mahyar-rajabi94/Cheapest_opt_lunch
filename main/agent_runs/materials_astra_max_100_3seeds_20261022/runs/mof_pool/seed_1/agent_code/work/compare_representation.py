from scipy.stats import rankdata
from sklearn.ensemble import GradientBoostingRegressor
R = np.column_stack([rankdata(U[:, j], method='average') for j in range(U.shape[1])]) / len(U)
representations = {'range': U, 'rank': R}
configs = {
    'et2': lambda: ExtraTreesRegressor(n_estimators=384, max_features=.5, min_samples_leaf=2, random_state=1, n_jobs=1),
    'et1': lambda: ExtraTreesRegressor(n_estimators=384, max_features=.5, min_samples_leaf=1, random_state=1, n_jobs=1),
    'rf2': lambda: RandomForestRegressor(n_estimators=384, max_features=.5, min_samples_leaf=2, random_state=1, n_jobs=1),
    'boost2': lambda: GradientBoostingRegressor(n_estimators=160, learning_rate=.03, max_depth=2, min_samples_leaf=3, max_features=.5, loss='squared_error', random_state=1),
}
y=np.asarray(observed_y)
cv=list(KFold(n_splits=5, shuffle=True, random_state=1).split(y))
rep_cv=[]
rep_predictions={}
for rep, features in representations.items():
    for name, factory in configs.items():
        if rep=='rank' and name in ['rf2','boost2']:
            continue
        pred=np.zeros(len(y))
        for tr,te in cv:
            est=factory()
            est.fit(features[np.asarray(observed_indices)[tr]], y[tr])
            pred[te]=est.predict(features[np.asarray(observed_indices)[te]])
        rep_predictions[rep+'_'+name]=pred
        top=np.argsort(y)[:max(5,len(y)//5)]
        rep_cv.append({'model':rep+'_'+name,'rmse':float(np.sqrt(np.mean((pred-y)**2))),
                       'spearman':float(spearmanr(y,pred).statistic),
                       'low_tail_rmse':float(np.sqrt(np.mean((pred[top]-y[top])**2)))})
print('REPRESENTATION_CV '+json.dumps(sorted(rep_cv,key=lambda x:x['rmse'])),flush=True)
av=np.asarray(h.available_indices)
for rep,name in [('range','et2'),('rank','et2'),('rank','et1'),('range','boost2')]:
    forest=configs[name]()
    features=representations[rep]
    forest.fit(features[observed_indices],y)
    means=forest.predict(features[av])
    print('REPRESENTATION_LEADERS '+json.dumps({'model':rep+'_'+name,
           'leaders':[{'index':int(av[i]),'mean':float(means[i])} for i in np.argsort(means)[:6]]}),flush=True)
