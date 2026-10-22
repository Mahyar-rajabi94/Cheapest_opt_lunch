"""Forward validation on actual observations; no objective evaluations."""
import numpy as np
A = np.asarray(X)
b = np.asarray(Y)
previous_x = np.vstack([np.full((1, h.dim), 0.5), A[:-1]])
previous_y = np.r_[0.0, b[:-1]][:, None]
features = {
    'x': A,
    'x_prev_y': np.column_stack([A, previous_y]),
    'x_prev_x': np.column_stack([A, previous_x]),
    'x_prev_x_y': np.column_stack([A, previous_x, previous_y]),
    'prev_y': previous_y,
}


def ridge_predict(F, y, test, penalty):
    mu, sd = F.mean(0), F.std(0)
    sd = np.maximum(sd, 1e-8)
    Z = (F - mu) / sd
    coeff = np.linalg.solve(Z.T @ Z + penalty * np.eye(F.shape[1]), Z.T @ (y-y.mean()))
    return y.mean() + (test-mu) / sd @ coeff


validation = []
for kind, F in features.items():
    for penalty in [1.0, 10.0, 100.0]:
        predicted = []
        for t in range(24, len(b)):
            predicted.append(float(ridge_predict(F[1:t], b[1:t], F[t], penalty)))
        truth = b[24:]
        errors = np.asarray(predicted)-truth
        validation.append({'features': kind, 'ridge': penalty, 'rmse': float(np.sqrt(np.mean(errors**2))),
                           'mae': float(np.mean(abs(errors))),
                           'correlation': float(np.corrcoef(predicted, truth)[0,1])})
reference = [float(b[1:t].mean()) for t in range(24,len(b))]
recent = [float(b[max(1,t-8):t].mean()) for t in range(24,len(b))]
print(json.dumps({'forward_test_calls': [25,len(b)], 'expanding_mean_rmse': float(np.sqrt(np.mean((np.asarray(reference)-b[24:])**2))),
                  'recent_8_mean_rmse': float(np.sqrt(np.mean((np.asarray(recent)-b[24:])**2))),
                  'ridge_validation': sorted(validation, key=lambda v:v['rmse'])}), flush=True)
