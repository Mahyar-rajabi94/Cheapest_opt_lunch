def gp_step(beta=4.):
    """Condition on all actual observations and return one evaluated GP proposal."""
    global bo_state, bo_last_fit, model
    X_tensor = torch.tensor([r['x'] for r in observations], dtype=torch.double)
    Y_tensor = torch.tensor([[r['y']] for r in observations], dtype=torch.double)
    baseline_mean = float(Y_tensor.mean())
    model = SingleTaskGP(X_tensor, Y_tensor,
                         train_Yvar=torch.full_like(Y_tensor, history_variance),
                         outcome_transform=Standardize(m=1))
    if bo_state is not None:
        model.load_state_dict(bo_state, strict=False)
    if len(observations)-bo_last_fit >= 4:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, optimizer_kwargs={'options': {'maxiter': 150}})
        bo_state = {k: v.clone() for k, v in model.state_dict().items()
                    if k.startswith(('mean_module.', 'covar_module.'))}
        bo_last_fit = len(observations)
        print('gp_fit', json.dumps(dict(n=bo_last_fit, history_variance=history_variance,
              lengthscales=model.covar_module.lengthscale.detach().numpy().tolist())), flush=True)
    model.eval()
    acquisition = UpperConfidenceBound(model, beta=beta, maximize=False)
    proposed, _ = optimize_acqf(acquisition,
                  bounds=torch.tensor([[0.]*11, [1.]*11], dtype=torch.double), q=1,
                  num_restarts=12, raw_samples=512,
                  options={'maxiter': 150, 'batch_limit': 4})
    with torch.no_grad():
        posterior = model.posterior(proposed)
        pred_mean = float(posterior.mean.item())
        pred_sd = float(posterior.variance.sqrt().item())
    point = proposed.detach().numpy()[0].clip(0., 1.)
    actual = evaluate(point)
    diagnostic = dict(n=h.evaluations_used, predicted_mean=pred_mean,
                      latent_sd=pred_sd, baseline_mean=baseline_mean, observed=actual)
    bo_diagnostics.append(diagnostic)
    print('gp_prediction_diagnostic', json.dumps(diagnostic), flush=True)
    return actual
