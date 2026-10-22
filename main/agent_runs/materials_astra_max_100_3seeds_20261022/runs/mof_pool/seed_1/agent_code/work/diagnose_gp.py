import torch
from scipy.stats import spearmanr
from src.optimizers.pool_gp_bo import GPBO

torch.set_num_threads(1)
if 'gp' not in globals():
    gp = GPBO(candidates=C, bounds=bounds, seed=int(h.seed), direction='min', n_init=16)
    gp.tell(C[observed_indices], observed_y)
model = gp._fit_model()
with torch.no_grad():
    train = model.train_inputs[0]
    K = model.covar_module(train).to_dense()
    K += torch.eye(len(train), dtype=torch.double) * model.likelihood.noise.item()
    invK = torch.linalg.inv(K)
    centered = model.train_targets - model.mean_module(train)
    alpha = invK @ centered
    loo_std = model.train_targets - alpha / invK.diag()
    ym = model.outcome_transform.means.item()
    ys = model.outcome_transform.stdvs.item()
    loo = loo_std.numpy() * ys + ym
    y = np.asarray(observed_y)
    metrics = {'n':len(y), 'fixed_hyperparameter_loo_rmse':float(np.sqrt(np.mean((y-loo)**2))),
               'constant_loo_rmse':float(np.sqrt(np.mean((y-(y.sum()-y)/(len(y)-1))**2))),
               'loo_spearman':float(spearmanr(y,loo).statistic),
               'noise_standardized':float(model.likelihood.noise.item()),
               'lengthscale_quantiles':np.quantile(model.covar_module.lengthscale.numpy(),[0,.25,.5,.75,1]).tolist()}
    remaining=np.array(h.available_indices)
    test=torch.tensor(C[remaining], dtype=torch.double)
    post=model.posterior(test)
    means=post.mean[:,0].numpy()
    stds=post.variance[:,0].sqrt().numpy()
    order=np.argsort(means)[:8]
    metrics['lowest_posterior_means']=[{'index':int(remaining[j]),'mean':float(means[j]),'sd':float(stds[j])} for j in order]
    print('GP_DIAGNOSTICS '+json.dumps(metrics), flush=True)
    print('GP_MODEL '+str(model), flush=True)
