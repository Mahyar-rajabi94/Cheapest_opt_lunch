def propose_sparse_lcb(kappa, tau=0.1):
    fitted = get_fitted_map_saas_model(
        train_X=torch.tensor(Z[observed_indices],dtype=torch.double),
        train_Y=torch.tensor(observed_y,dtype=torch.double).unsqueeze(-1),
        tau=tau, outcome_transform=Standardize(m=1),
        optimizer_kwargs={'options':{'maxiter':500}})
    available = np.asarray(h.available_indices,dtype=int)
    means, sds = [], []
    with torch.no_grad():
        for start in range(0,len(available),512):
            post = fitted.posterior(torch.tensor(Z[available[start:start+512]],dtype=torch.double))
            means.extend(post.mean.reshape(-1).tolist())
            sds.extend(post.variance.reshape(-1).sqrt().tolist())
        mean, sd = np.asarray(means),np.asarray(sds)
        score = mean-kappa*sd
        selected = int(np.argmin(score))
        nugget = float(fitted.likelihood.noise.reshape(-1)[0])
        noise_raw = nugget * float(fitted.outcome_transform.stdvs.reshape(-1)[0])**2
    return int(available[selected]),{
        'model_mean':float(mean[selected]),'model_latent_sd':float(sd[selected]),
        'model_predictive_sd':float(np.sqrt(sd[selected]**2+noise_raw)),
        'lcb':float(score[selected]),'kappa':kappa,'tau':tau,'noise_standardized':nugget}
