from botorch.acquisition import LogExpectedImprovement
from scipy.special import logsumexp

def propose_sparse_ensemble(taus=(0.01,0.1,1.0), preview=False):
    available=np.asarray(h.available_indices,dtype=int)
    logeis,means,sds=[],[],[]
    for tau in taus:
        fitted=get_fitted_map_saas_model(
            train_X=torch.tensor(Z[observed_indices],dtype=torch.double),
            train_Y=torch.tensor(observed_y,dtype=torch.double).unsqueeze(-1),
            tau=tau,outcome_transform=Standardize(m=1),
            optimizer_kwargs={'options':{'maxiter':500}})
        acq=LogExpectedImprovement(model=fitted,best_f=min(observed_y),maximize=False)
        logei,pm,ps=[],[],[]
        with torch.no_grad():
            for start in range(0,len(available),512):
                tensor=torch.tensor(Z[available[start:start+512]],dtype=torch.double)
                logei.extend(acq(tensor.unsqueeze(1)).reshape(-1).tolist())
                post=fitted.posterior(tensor)
                pm.extend(post.mean.reshape(-1).tolist())
                ps.extend(post.variance.reshape(-1).sqrt().tolist())
        logeis.append(logei)
        means.append(pm)
        sds.append(ps)
    logeis,means,sds=map(np.asarray,(logeis,means,sds))
    ensemble_logei=logsumexp(logeis,axis=0)-np.log(len(taus))
    selected=int(np.argmax(ensemble_logei))
    details={'taus':list(taus),'component_means':means[:,selected].tolist(),
             'component_sds':sds[:,selected].tolist(),
             'component_logei':logeis[:,selected].tolist(),
             'ensemble_logei':float(ensemble_logei[selected])}
    if preview:
        rank=np.argsort(-ensemble_logei)[:12]
        print(json.dumps({'ensemble_preview_at':h.evaluations_used,
            'predictions_only':[{'index':int(available[j]),'means':means[:,j].tolist(),
              'sds':sds[:,j].tolist(),'ensemble_logei':float(ensemble_logei[j])} for j in rank],
            'each_component_best_index':[int(available[np.argmax(v)]) for v in logeis]}),flush=True)
    return int(available[selected]),details
