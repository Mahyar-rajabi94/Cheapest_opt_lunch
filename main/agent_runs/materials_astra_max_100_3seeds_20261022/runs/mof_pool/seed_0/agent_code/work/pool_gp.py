"""Finite-pool GP adapter using BoTorch public model and acquisition APIs."""
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_num_threads(1)

class PoolGP:
    def __init__(self, features, seed=0, maxiter=150, chunk_size=256):
        self.features=np.asarray(features,dtype=float)
        self.seed=seed
        self.maxiter=maxiter
        self.chunk_size=chunk_size
        self.ids=[]
        self.y=[]
        self.model=None
        torch.manual_seed(seed)
    def tell(self,index,y):
        self.ids.append(int(index))
        self.y.append(float(y))
    def fit(self):
        tx=torch.tensor(self.features[self.ids],dtype=torch.double)
        ty=torch.tensor(self.y,dtype=torch.double).unsqueeze(-1)
        self.model=SingleTaskGP(tx,ty,outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(self.model.likelihood,self.model),
                        optimizer_kwargs={'options':{'maxiter':self.maxiter}})
        return self.model
    def choose(self,available):
        available=np.asarray(available,dtype=int)
        model=self.fit()
        acq=LogExpectedImprovement(model,best_f=min(self.y),maximize=False)
        all_scores=[]
        with torch.no_grad():
            for start in range(0,len(available),self.chunk_size):
                batch=available[start:start+self.chunk_size]
                all_scores.extend(acq(torch.tensor(self.features[batch],dtype=torch.double).unsqueeze(1)).reshape(-1).tolist())
            position=int(np.argmax(all_scores))
            index=int(available[position])
            posterior=model.posterior(torch.tensor(self.features[[index]],dtype=torch.double))
            mean=float(posterior.mean.reshape(-1)[0])
            sd=float(posterior.variance.reshape(-1)[0].sqrt())
        covariance=model.covar_module
        base=covariance.base_kernel if hasattr(covariance,"base_kernel") else covariance
        scales=base.lengthscale.detach().cpu().numpy().reshape(-1)
        detail={'mean':mean,'sd':sd,'log_ei':float(all_scores[position]),
                'noise_standardized':float(model.likelihood.noise.detach().reshape(-1)[0]),
                'lengthscale_quantiles':np.quantile(scales,[0,.5,1]).tolist(),
                'nearest_observed_distance':float(np.linalg.norm(self.features[self.ids]-self.features[index],axis=1).min())}
        return index,detail
