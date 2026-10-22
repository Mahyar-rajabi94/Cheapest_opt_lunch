assert gp.ids==indices[:len(gp.ids)]
for idx,y in zip(indices[len(gp.ids):],values[len(gp.ids):]):gp.tell(idx,y)
assert len(gp.ids)==75
gp.maxiter=200
large_miss_count=0
run_block('pool_gp_logei',6,lambda:gp.choose(h.available_indices),observe=gp.tell,early_stop=gp_miss_stop)
