"""Equal-weight GP/tree predictive mixture for candidate acquisition only."""
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

def choose_ensemble(gp,trees,available,beta=1.5):
    assert gp.ids==trees.ids and gp.y==trees.y
    available=np.asarray(available,dtype=int)
    gp_model=gp.fit()
    mu_gp=[];var_gp=[]
    with torch.no_grad():
        for start in range(0,len(available),gp.chunk_size):
            chunk=available[start:start+gp.chunk_size]
            post=gp_model.posterior(torch.tensor(gp.features[chunk],dtype=torch.double))
            mu_gp.extend(post.mean.reshape(-1).tolist());var_gp.extend(post.variance.reshape(-1).tolist())
    mu_gp=np.asarray(mu_gp);var_gp=np.asarray(var_gp)
    trees.model=ExtraTreesRegressor(n_estimators=512,max_features=.5,min_samples_leaf=2,
                                   random_state=trees.seed,n_jobs=1,bootstrap=False,max_depth=None)
    trees.model.fit(trees.features[trees.ids],trees.y)
    draws=np.asarray([t.predict(trees.features[available]) for t in trees.model.estimators_])
    mu_tree=draws.mean(0);var_tree=draws.var(0)
    mu=.5*(mu_gp+mu_tree)
    variance=.5*(var_gp+var_tree)+.25*(mu_gp-mu_tree)**2
    sd=np.sqrt(np.maximum(variance,0.))
    lcb=mu-beta*sd
    pos=int(np.argmin(lcb));idx=int(available[pos])
    return idx,{'mean':float(mu[pos]),'sd':float(sd[pos]),'lcb':float(lcb[pos]),'beta_sd_multiplier':beta,
                'gp_mean':float(mu_gp[pos]),'gp_sd':float(np.sqrt(var_gp[pos])),
                'tree_mean':float(mu_tree[pos]),'tree_sd':float(np.sqrt(var_tree[pos])),
                'uncertainty':'equal_weight_predictive_mixture_heuristic',
                'nearest_observed_distance':float(np.linalg.norm(gp.features[gp.ids]-gp.features[idx],axis=1).min())}
