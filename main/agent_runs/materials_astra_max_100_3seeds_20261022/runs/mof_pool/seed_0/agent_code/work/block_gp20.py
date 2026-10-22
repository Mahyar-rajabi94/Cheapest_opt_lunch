from work.pool_gp import PoolGP
gp=PoolGP(Z,seed=0,maxiter=150,chunk_size=256)
for idx,y in zip(indices,values):
    gp.tell(idx,y)
large_miss_count=0
def gp_miss_stop(idx,y,previous,detail):
    global large_miss_count
    large_miss_count=large_miss_count+1 if abs(y-detail['mean'])>4*max(detail['sd'],1e-12) else 0
    return large_miss_count>=3
run_block('pool_gp_logei',8,lambda:gp.choose(h.available_indices),observe=gp.tell,early_stop=gp_miss_stop)
