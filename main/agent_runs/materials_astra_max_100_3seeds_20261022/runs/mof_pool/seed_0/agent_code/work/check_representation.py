import sys,json,warnings
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np,torch
from scipy.stats import rankdata,spearmanr
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from work.pool_gp import PoolGP
from work.regularized_gp import RegularizedPoolGP
warnings.filterwarnings('ignore',category=UserWarning)
d=json.loads(Path('work/observations.json').read_text());ids=np.array(d['indices']);y=np.array(d['values'])
X=np.load('candidates.npy');mask=np.ptp(X,axis=0)>0; Z=(X[:,mask]-X[:,mask].min(0))/np.ptp(X[:,mask],axis=0)
Q=(rankdata(Z,axis=0)-1)/(len(Z)-1)
folds=list(KFold(5,shuffle=True,random_state=0).split(ids))
for name,feat in [('raw_matern',Z),('rank_matern',Q),('rank_rbf',Q),('raw_extra',Z),('rank_extra',Q),('raw_rf',Z),('pls2',Z),('pls4',Z)]:
    p=np.empty(len(y)); sd=np.full(len(y),np.nan)
    for tr,te in folds:
        if name.endswith('extra') or name.endswith('rf'):
            cls=RandomForestRegressor if name.endswith('rf') else ExtraTreesRegressor
            model=cls(n_estimators=512,max_features=.5,min_samples_leaf=2,random_state=0,n_jobs=1).fit(feat[ids[tr]],y[tr]);p[te]=model.predict(feat[ids[te]])
        elif name.startswith('pls'):
            model=PLSRegression(n_components=int(name[-1]),scale=True).fit(feat[ids[tr]],y[tr]);p[te]=model.predict(feat[ids[te]]).reshape(-1)
        else:
            gp=RegularizedPoolGP(feat,variant='matern32_iso',seed=0,maxiter=150) if name.endswith('matern') else PoolGP(feat,seed=0,maxiter=150)
            for j in tr:gp.tell(ids[j],y[j])
            model=gp.fit()
            with torch.no_grad():
                post=model.posterior(torch.tensor(feat[ids[te]],dtype=torch.double));p[te]=post.mean.reshape(-1).numpy();sd[te]=post.variance.reshape(-1).sqrt().numpy()
    result={'model':name,'rmse':float(np.sqrt(np.mean((y-p)**2))),'spearman':float(spearmanr(y,p).statistic),'best3_predictions':p[np.argsort(y)[:3]].tolist()}
    if np.isfinite(sd).all():result['within2sd']=int(np.sum(abs(y-p)<2*sd))
    print(json.dumps(result),flush=True)
print(json.dumps({'rank_representation':'average ties; (rank-1)/(1000-1); public features only','n':len(y)}))
