tail_predictions=globals().get('tail_predictions',[])
def tail_block(k, beta=1.0):
    begin_block(k,'extra_trees_tail_lcb')
    for _ in range(k):
        yy=np.asarray(observed_y)
        target=yy*np.abs(yy)
        est=ExtraTreesRegressor(n_estimators=512,max_features=.5,min_samples_leaf=2,
                               bootstrap=False,random_state=1,n_jobs=1)
        est.fit(U[observed_indices],target)
        av=np.asarray(h.available_indices)
        tp=np.asarray([tree.predict(U[av]) for tree in est.estimators_])
        means=tp.mean(0); sds=tp.std(0,ddof=1)
        scores=means-beta*sds
        pos=int(np.argmin(scores)); index=int(av[pos])
        inv_mean=float(np.sign(means[pos])*np.sqrt(abs(means[pos])))
        if not np.isfinite([inv_mean,scores[pos]]).all():
            raise RuntimeError('Nonfinite tail-model prediction')
        print('TAIL_PREDICTION '+json.dumps({'index':index,'inverse_transformed_mean':inv_mean,
                'transformed_sd':float(sds[pos]),'transformed_lcb':float(scores[pos])}),flush=True)
        value=purchase(index)
        if not np.isfinite(value):
            raise RuntimeError('Nonfinite objective')
        tail_predictions.append([index,inv_mean,float(sds[pos]),value])
    report()
    recent=tail_predictions[-k:]
    print('TAIL_BLOCK_DIAGNOSTICS '+json.dumps({'inverse_prediction_rmse':float(np.sqrt(np.mean([(p[1]-p[3])**2 for p in recent]))),
          'mean_y':float(np.mean([p[3] for p in recent])),'block_min_y':float(min(p[3] for p in recent))}),flush=True)
