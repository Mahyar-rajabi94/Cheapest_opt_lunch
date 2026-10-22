"""Lower-confidence acquisition that reuses the live finite-pool GP data."""
import numpy as np
import torch

def choose_gp_lcb(gp,available,beta=1.0):
    available=np.asarray(available,dtype=int)
    model=gp.fit()
    mus=[];sds=[]
    with torch.no_grad():
        for start in range(0,len(available),gp.chunk_size):
            chunk=available[start:start+gp.chunk_size]
            post=model.posterior(torch.tensor(gp.features[chunk],dtype=torch.double))
            mus.extend(post.mean.reshape(-1).tolist())
            sds.extend(post.variance.reshape(-1).sqrt().tolist())
    mu=np.asarray(mus);sd=np.asarray(sds)
    score=mu-beta*sd
    pos=int(np.argmin(score));idx=int(available[pos])
    scales=model.covar_module.lengthscale.detach().numpy().reshape(-1)
    detail={'mean':float(mu[pos]),'sd':float(sd[pos]),'lcb':float(score[pos]),'beta_sd_multiplier':beta,
            'noise_standardized':float(model.likelihood.noise.detach().reshape(-1)[0]),
            'lengthscale_quantiles':np.quantile(scales,[0,.5,1]).tolist(),
            'nearest_observed_distance':float(np.linalg.norm(gp.features[gp.ids]-gp.features[idx],axis=1).min())}
    return idx,detail
