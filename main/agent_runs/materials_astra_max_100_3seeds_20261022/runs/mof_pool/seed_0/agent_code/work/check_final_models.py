import sys,json,warnings
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np, torch
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor
from scipy.stats import spearmanr
from work.pool_gp import PoolGP
from work.regularized_gp import RegularizedPoolGP
warnings.filterwarnings('ignore',category=UserWarning)
d=json.loads(Path('work/observations.json').read_text());ids=np.array(d['indices']);y=np.array(d['values'])
X=np.load('candidates.npy');mask=np.ptp(X,axis=0)>0;Z=(X[:,mask]-X[:,mask].min(0))/np.ptp(X[:,mask],axis=0)
folds=list(KFold(5,shuffle=True,random_state=0).split(ids))
constructors={
 'extra_current':lambda:ExtraTreesRegressor(n_estimators=512,max_features=.5,min_samples_leaf=2,random_state=0,n_jobs=1),
 'extra_leaf1':lambda:ExtraTreesRegressor(n_estimators=512,max_features=.5,min_samples_leaf=1,random_state=0,n_jobs=1),
 'extra_features1':lambda:ExtraTreesRegressor(n_estimators=512,max_features=1.,min_samples_leaf=2,random_state=0,n_jobs=1),
 'rf_regularized':lambda:RandomForestRegressor(n_estimators=512,max_features=.5,min_samples_leaf=2,random_state=0,n_jobs=1),
 'boosted_depth2':lambda:GradientBoostingRegressor(n_estimators=200,max_depth=2,min_samples_leaf=4,learning_rate=.03,max_features=.7,random_state=0,loss='squared_error')}
results=[]
for name in list(constructors)+['gp_rbf','gp_matern']:
 p=np.empty(len(y))
 for tr,te in folds:
  if name in constructors:
   model=constructors[name]().fit(Z[ids[tr]],y[tr]);p[te]=model.predict(Z[ids[te]])
  else:
   gp=PoolGP(Z,seed=0,maxiter=200) if name=='gp_rbf' else RegularizedPoolGP(Z,variant='matern32_iso',seed=0,maxiter=150)
   for j in tr:gp.tell(ids[j],y[j])
   model=gp.fit()
   with torch.no_grad():p[te]=model.posterior(torch.tensor(Z[ids[te]],dtype=torch.double)).mean.reshape(-1).numpy()
 true_top=set(map(int,np.argsort(y)[:10]));pred_top=set(map(int,np.argsort(p)[:10]))
 ranks=np.argsort(np.argsort(p))/(len(y)-1)
 result={'model':name,'rmse':float(np.sqrt(np.mean((y-p)**2))),'spearman':float(spearmanr(y,p).statistic),'top10_overlap':len(true_top&pred_top),'mean_pred_percentile_best10':float(ranks[np.argsort(y)[:10]].mean()),'predictions_best3':p[np.argsort(y)[:3]].tolist()}
 results.append(result);print(json.dumps(result),flush=True)
Path('work/model_check75.json').write_text(json.dumps(results,indent=2))
