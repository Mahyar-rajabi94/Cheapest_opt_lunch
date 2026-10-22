# Diagnostic fitting uses purchased labels only; it makes no evaluator calls.
model = gp._fit_model()
model.eval()
with torch.no_grad():
    tx = model.train_inputs[0]
    ty = model.train_targets
    train_mean = model.mean_module(tx)
    cov = model.covar_module(tx).to_dense()
    noise = model.likelihood.noise.reshape(-1)
    cov = cov + torch.diag(noise.expand(len(ty)))
    inv = torch.linalg.inv(cov)
    alpha = inv @ (ty-train_mean)
    loo_standard = ty-alpha/inv.diag()
    loovar_standard = 1/inv.diag()
    scale_y = float(model.outcome_transform.stdvs.reshape(-1)[0])
    mean_y = float(model.outcome_transform.means.reshape(-1)[0])
    loo = loo_standard.numpy()*scale_y+mean_y
    standardized_resid = ((ty-loo_standard)/loovar_standard.sqrt()).numpy()
    ls = model.covar_module.lengthscale.reshape(-1).numpy()
    avail = np.asarray(h.available_indices, dtype=int)
    post = model.posterior(torch.tensor(X[avail], dtype=torch.double))
    predmean = post.mean.reshape(-1).numpy()
    predstd = post.variance.reshape(-1).sqrt().numpy()
    rank = np.argsort(predmean)[:8]
    summary = {
        'gp_loo_rmse': float(np.sqrt(np.mean((np.asarray(observed_y)-loo)**2))),
        'constant_loo_rmse': float(np.std(observed_y, ddof=0)*len(observed_y)/(len(observed_y)-1)),
        'gp_loo_spearman': float(spearmanr(observed_y, loo).statistic),
        'gp_loo_fraction_abs_z_below_2': float(np.mean(abs(standardized_resid)<2)),
        'noise_variance_standardized': noise.tolist(),
        'lengthscale_quantiles': np.quantile(ls,[0,.25,.5,.75,1]).tolist(),
        'best_response_loo_prediction': float(loo[np.argmin(observed_y)]),
        'lowest_predicted': [{'index':int(avail[j]),'model_mean':float(predmean[j]), 'model_sd':float(predstd[j])} for j in rank],
        'loo_note':'Hyperparameters are fitted on all purchased labels; approximate leave-one-out diagnostics are optimistic, not independent validation.'}
print(json.dumps(summary), flush=True)
