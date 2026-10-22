initial = list(map(int,rng.choice(len(X),size=6,replace=False)))
min_d2 = np.min(np.sum((Z[:,None,:]-Z[np.asarray(initial)][None,:,:])**2,axis=2),axis=1)
min_d2[initial] = -np.inf
for _ in range(6):
    idx = int(np.argmax(min_d2))
    initial.append(idx)
    min_d2 = np.minimum(min_d2,np.sum((Z-Z[idx])**2,axis=1))
    min_d2[initial] = -np.inf
initial_queue = iter(initial)
run_block('balanced_pool_design',12,lambda:next(initial_queue))
