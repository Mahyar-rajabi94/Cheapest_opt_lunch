import json, warnings, importlib.metadata as md
import numpy as np
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import sys
sys.path.insert(0,str(Path.cwd()))
from work.pool_gp import PoolGP
warnings.filterwarnings('ignore',category=UserWarning)
d=json.loads(Path('work/observations.json').read_text())
ids=np.array(d['indices']); y=np.array(d['values']); X=np.load('candidates.npy')
active=np.ptp(X,axis=0)>0
Z=(X[:,active]-X[:,active].min(0))/np.ptp(X[:,active],axis=0)
folds=list(KFold(n_splits=5,shuffle=True,random_state=0).split(ids))
models={'extra_trees':lambda:ExtraTreesRegressor(n_estimators=256,max_features=0.7,min_samples_leaf=1,random_state=0,n_jobs=1),
        'random_forest':lambda:RandomForestRegressor(n_estimators=256,max_features=0.7,min_samples_leaf=1,random_state=0,n_jobs=1),
        'ridge_1':lambda:Ridge(alpha=1.),'ridge_10':lambda:Ridge(alpha=10.)}
for name,constructor in models.items():
    p=np.empty(len(y))
    for tr,te in folds:
        model=constructor().fit(Z[ids[tr]],y[tr]); p[te]=model.predict(Z[ids[te]])
    print(json.dumps({'model':name,'rmse':float(np.sqrt(np.mean((p-y)**2))), 'spearman':float(spearmanr(p,y).statistic),'predictions_for_best3':p[np.argsort(y)[:3]].tolist()}),flush=True)
p=np.empty(len(y)); s=np.empty(len(y))
for tr,te in folds:
    gp=PoolGP(Z,seed=0,maxiter=150)
    for j in tr:gp.tell(ids[j],y[j])
    m=gp.fit()
    import torch
    with torch.no_grad():
        post=m.posterior(torch.tensor(Z[ids[te]],dtype=torch.double))
        p[te]=post.mean.reshape(-1).numpy();s[te]=post.variance.reshape(-1).sqrt().numpy()
print(json.dumps({'model':'gp_rbf_ard','rmse':float(np.sqrt(np.mean((p-y)**2))),'spearman':float(spearmanr(p,y).statistic),'within2sd':int(np.sum(abs(y-p)<2*s)),'predictions_for_best3':p[np.argsort(y)[:3]].tolist()}),flush=True)
print(json.dumps({'sklearn_version':md.version('scikit-learn'),'n':len(y),'cv':'5-fold shuffled seed 0; exploratory, adaptively selected observations'}))
