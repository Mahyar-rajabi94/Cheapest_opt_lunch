obs = np.asarray(observed_indices,dtype=int)
yobs = np.asarray(observed_y)
avail=np.asarray(h.available_indices,dtype=int)
best_index=int(obs[np.argmin(yobs)])
distance_to_best=np.linalg.norm(Z-Z[best_index],axis=1)
nearest_all=np.argsort(distance_to_best)[:25]
observations=dict(zip(observed_indices,observed_y))
print(json.dumps({'best_neighborhood':[{'index':int(i),'distance':float(distance_to_best[i]),'observed_y':observations.get(int(i))} for i in nearest_all],
                  'remaining_nearest':[{'index':int(i),'distance':float(distance_to_best[i])} for i in avail[np.argsort(distance_to_best[avail])[:12]]]}),flush=True)
# Fresh sparse-model diagnostic: what remains to exploit under the current model?
sparse_model = get_fitted_map_saas_model(
    train_X=torch.tensor(Z[obs],dtype=torch.double),
    train_Y=torch.tensor(yobs,dtype=torch.double).unsqueeze(-1),
    tau=0.1,outcome_transform=Standardize(m=1),
    optimizer_kwargs={'options':{'maxiter':500}})
with torch.no_grad():
    p=sparse_model.posterior(torch.tensor(Z[avail],dtype=torch.double))
    pm=p.mean.reshape(-1).numpy()
    ps=p.variance.reshape(-1).sqrt().numpy()
    ranked=np.argsort(pm-ps)[:12]
print(json.dumps({'sparse_remaining_predictions':[{'index':int(avail[i]),'mean':float(pm[i]),'sd':float(ps[i]),'best_distance':float(distance_to_best[avail[i]])} for i in ranked]}),flush=True)
