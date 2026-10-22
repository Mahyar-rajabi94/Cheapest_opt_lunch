"""Configurable covariance/likelihood regularization for the finite-pool GP."""
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan
from work.pool_gp import PoolGP

class RegularizedPoolGP(PoolGP):
    def __init__(self,features,variant='rbf_regularized',**kwargs):
        super().__init__(features,**kwargs)
        self.variant=variant
    def fit(self):
        tx=torch.tensor(self.features[self.ids],dtype=torch.double)
        ty=torch.tensor(self.y,dtype=torch.double).unsqueeze(-1)
        kwargs={}
        if self.variant=='rbf_regularized':
            likelihood=GaussianLikelihood(noise_prior=LogNormalPrior(-2.0,0.75),noise_constraint=GreaterThan(0.03))
            likelihood.initialize(noise=0.15)
            kwargs['likelihood']=likelihood
        elif self.variant=='matern32_iso':
            likelihood=GaussianLikelihood(noise_prior=LogNormalPrior(-2.0,1.0),noise_constraint=GreaterThan(0.01))
            likelihood.initialize(noise=0.15)
            base=MaternKernel(nu=1.5,ard_num_dims=None,lengthscale_prior=LogNormalPrior(0.0,1.0))
            base.initialize(lengthscale=1.0)
            kernel=ScaleKernel(base,outputscale_prior=LogNormalPrior(0.0,0.75))
            kernel.initialize(outputscale=1.0)
            kwargs.update(likelihood=likelihood,covar_module=kernel)
        self.model=SingleTaskGP(tx,ty,outcome_transform=Standardize(m=1),**kwargs)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(self.model.likelihood,self.model),optimizer_kwargs={'options':{'maxiter':self.maxiter}})
        return self.model
