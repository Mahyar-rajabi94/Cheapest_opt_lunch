import sys, json, warnings
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np, torch
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from scipy.stats import spearmanr
from work.pool_gp import PoolGP
from work.regularized_gp import RegularizedPoolGP
warnings.filterwarnings('ignore',category=UserWarning)
d=json.loads(Path('work/observations.json').read_text());ids=np.array(d['indices']);y=np.array(d['values'])
X=np.load('candidates.npy'); mask=np.ptp(X,axis=0)>0;Z=(X[:,mask]-X[:,mask].min(0))/np.ptp(X[:,mask],axis=0)
u,s,v=np.linalg.svd(Z-Z.mean(0),full_matrices=False);pc=u[:,:10]*s[:10];pc=(pc-pc.min(0))/np.ptp(pc,axis=0)
folds=list(KFold(5,shuffle=True,random_state=0).split(ids))
for name,feat in [('rbf_default',Z),('rbf_regularized',Z),('matern32_iso',Z),('pca10_default',pc),('extra_trees',Z)]:
    p=np.empty(len(y));sd=np.empty(len(y))
    for tr,te in folds:
        if name=='extra_trees':
            model=ExtraTreesRegressor(n_estimators=256,max_features=.7,min_samples_leaf=1,random_state=0,n_jobs=1).fit(feat[ids[tr]],y[tr])
            draws=np.array([tree.predict(feat[ids[te]]) for tree in model.estimators_]);p[te]=draws.mean(0);sd[te]=draws.std(0)
        else:
            gp=RegularizedPoolGP(feat,variant=name,seed=0,maxiter=150) if name in ('rbf_regularized','matern32_iso') else PoolGP(feat,seed=0,maxiter=150)
            for j in tr:gp.tell(ids[j],y[j])
            model=gp.fit()
            with torch.no_grad():
                post=model.posterior(torch.tensor(feat[ids[te]],dtype=torch.double));p[te]=post.mean.reshape(-1).numpy();sd[te]=post.variance.reshape(-1).sqrt().numpy()
    print(json.dumps({'model':name,'rmse':float(np.sqrt(np.mean((y-p)**2))),'spearman':float(spearmanr(y,p).statistic),'within2sd':int(np.sum(abs(y-p)<=2*sd)),'rms_standardized_error':float(np.sqrt(np.mean(((y-p)/sd)**2))),'best3_predictions':p[np.argsort(y)[:3]].tolist()}),flush=True)
