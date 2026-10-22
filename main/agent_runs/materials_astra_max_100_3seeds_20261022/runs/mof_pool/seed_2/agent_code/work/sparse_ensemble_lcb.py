def propose_sparse_ensemble_lcb(kappa=0.5,taus=(0.01,0.1,1.0)):
    available=np.asarray(h.available_indices,dtype=int)
    means,sds=[],[]
    for tau in taus:
        fitted=get_fitted_map_saas_model(
            train_X=torch.tensor(Z[observed_indices],dtype=torch.double),
            train_Y=torch.tensor(observed_y,dtype=torch.double).unsqueeze(-1),
            tau=tau,outcome_transform=Standardize(m=1),
            optimizer_kwargs={'options':{'maxiter':500}})
        pm,ps=[],[]
        with torch.no_grad():
            for start in range(0,len(available),512):
                p=fitted.posterior(torch.tensor(Z[available[start:start+512]],dtype=torch.double))
                pm.extend(p.mean.reshape(-1).tolist())
                ps.extend(p.variance.reshape(-1).sqrt().tolist())
        means.append(pm)
        sds.append(ps)
    means,sds=np.asarray(means),np.asarray(sds)
    mean=means.mean(axis=0)
    mixture_sd=np.sqrt(np.maximum(np.mean(sds**2+means**2,axis=0)-mean**2,0))
    score=mean-kappa*mixture_sd
    selected=int(np.argmin(score))
    return int(available[selected]),{
        'taus':list(taus),'component_means':means[:,selected].tolist(),
        'component_sds':sds[:,selected].tolist(),'mixture_mean':float(mean[selected]),
        'mixture_sd':float(mixture_sd[selected]),'kappa':kappa,'lcb':float(score[selected])}
