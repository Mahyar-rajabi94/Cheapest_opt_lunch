recent_diagnostics = bo_diagnostics[-12:]
print('prospective_scores', json.dumps({
    'gp_rmse': float(np.sqrt(np.mean([(d['observed']-d['predicted_mean'])**2 for d in recent_diagnostics]))),
    'running_mean_rmse': float(np.sqrt(np.mean([(d['observed']-d['baseline_mean'])**2 for d in recent_diagnostics])))}))
probe_X = torch.tensor([r['x'] for r in observations], dtype=torch.double)
probe_Y = torch.tensor([[r['y']] for r in observations], dtype=torch.double)
probe_model = SingleTaskGP(probe_X, probe_Y, train_Yvar=torch.full_like(probe_Y, history_variance), outcome_transform=Standardize(m=1))
probe_model.load_state_dict(bo_state, strict=False)
probe_model.eval()
with torch.no_grad():
    post = probe_model.posterior(torch.tensor(np.array([probe_a, probe_b]), dtype=torch.double))
    print('anchor_model_predictions', json.dumps(dict(mean=post.mean.ravel().tolist(), sd=post.variance.sqrt().ravel().tolist())))
with torch.random.fork_rng():
    torch.manual_seed(2)
    for beta in [0.25, 4.]:
        acq = UpperConfidenceBound(probe_model, beta=beta, maximize=False)
        x_probe, _ = optimize_acqf(acq, bounds=torch.tensor([[0.]*11,[1.]*11], dtype=torch.double),q=1,num_restarts=12,raw_samples=512,options={'maxiter':150,'batch_limit':4})
        with torch.no_grad():
            post = probe_model.posterior(x_probe)
            print('acquisition_only_diagnostic', json.dumps(dict(beta=beta, x=x_probe.ravel().tolist(), model_mean=post.mean.item(), model_sd=post.variance.sqrt().item())))
