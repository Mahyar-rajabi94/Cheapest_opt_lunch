import importlib
import work.pool_gp
importlib.reload(work.pool_gp)
from work.regularized_gp import RegularizedPoolGP
matern=RegularizedPoolGP(Z,variant='matern32_iso',seed=0,maxiter=150,chunk_size=256)
for idx,y in zip(indices,values):
    matern.tell(idx,y)
large_miss_count=0
run_block('pool_matern_logei',8,lambda:matern.choose(h.available_indices),observe=matern.tell,early_stop=gp_miss_stop)
