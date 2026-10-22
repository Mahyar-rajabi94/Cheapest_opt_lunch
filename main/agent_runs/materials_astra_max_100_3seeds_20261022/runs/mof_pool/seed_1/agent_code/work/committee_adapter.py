committee_predictions=globals().get('committee_predictions',[])
def committee_block(k,beta=.5,gp_weight=.25):
    begin_block(k,'gp_forest_committee')
    for _ in range(k):
        assert len(gp.y_)==len(observed_y)
        m=gp._fit_model()
        est=ExtraTreesRegressor(n_estimators=512,max_features=.5,min_samples_leaf=2,
                               bootstrap=False,random_state=1,n_jobs=1)
        est.fit(U[observed_indices],observed_y)
        av=np.asarray(h.available_indices)
        tp=np.asarray([tree.predict(U[av]) for tree in est.estimators_])
        fm=tp.mean(0); fv=tp.var(0,ddof=1)
        with torch.no_grad():
            p=m.posterior(torch.tensor(C[av],dtype=torch.double))
            gm=p.mean[:,0].numpy(); gv=p.variance[:,0].numpy()
        mean=(1-gp_weight)*fm+gp_weight*gm
        variance=(1-gp_weight)*(fv+(fm-mean)**2)+gp_weight*(gv+(gm-mean)**2)
        sd=np.sqrt(np.maximum(variance,0))
        scores=mean-beta*sd
        pos=int(np.argmin(scores)); index=int(av[pos])
        if not np.isfinite([mean[pos],sd[pos],scores[pos]]).all():
            raise RuntimeError('Nonfinite committee prediction')
        print('COMMITTEE_PREDICTION '+json.dumps({'index':index,'mean':float(mean[pos]),'sd':float(sd[pos]),
               'forest_mean':float(fm[pos]),'gp_mean':float(gm[pos]),'score':float(scores[pos])}),flush=True)
        value=purchase(index)
        if not np.isfinite(value):
            raise RuntimeError('Nonfinite objective')
        gp.tell(C[[index]],[value])
        committee_predictions.append([index,float(mean[pos]),float(sd[pos]),value])
    report()
    recent=committee_predictions[-k:]
    print('COMMITTEE_BLOCK_DIAGNOSTICS '+json.dumps({'rmse':float(np.sqrt(np.mean([(p[1]-p[3])**2 for p in recent]))),
          'mean_y':float(np.mean([p[3] for p in recent])),'block_min_y':float(min(p[3] for p in recent))}),flush=True)
