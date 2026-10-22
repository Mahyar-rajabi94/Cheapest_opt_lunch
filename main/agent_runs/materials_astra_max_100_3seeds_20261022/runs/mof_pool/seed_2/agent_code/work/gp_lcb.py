from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

# Same public-library GP configuration as the supplied wrapper; only acquisition changes.
def propose_gp_lcb(kappa):
    model_bounds = np.asarray(h.bounds, dtype=float).copy()
    constant = model_bounds[:, 0] == model_bounds[:, 1]
    model_bounds[constant, 1] = model_bounds[constant, 0] + 1.0
    fitted = SingleTaskGP(
        torch.tensor(X[observed_indices], dtype=torch.double),
        torch.tensor(observed_y, dtype=torch.double).unsqueeze(-1),
        input_transform=Normalize(d=X.shape[1], bounds=torch.tensor(model_bounds.T, dtype=torch.double)),
        outcome_transform=Standardize(m=1),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(fitted.likelihood, fitted))
    available = np.asarray(h.available_indices, dtype=int)
    means, sds = [], []
    with torch.no_grad():
        for start in range(0, len(available), 512):
            post = fitted.posterior(torch.tensor(X[available[start:start+512]], dtype=torch.double))
            means.extend(post.mean.reshape(-1).tolist())
            sds.extend(post.variance.reshape(-1).sqrt().tolist())
        mean, sd = np.asarray(means), np.asarray(sds)
        score = mean-kappa*sd
        best = int(np.argmin(score))
        noise_raw = float(fitted.likelihood.noise.reshape(-1)[0]) * float(fitted.outcome_transform.stdvs.reshape(-1)[0])**2
    return int(available[best]), {'model_mean': float(mean[best]), 'model_latent_sd': float(sd[best]),
                                  'model_predictive_sd': float(np.sqrt(sd[best]**2+noise_raw)),
                                  'lcb': float(score[best]), 'kappa': kappa}
