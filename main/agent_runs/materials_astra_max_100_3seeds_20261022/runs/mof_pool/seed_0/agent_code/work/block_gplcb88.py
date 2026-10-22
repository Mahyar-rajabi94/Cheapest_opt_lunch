from work.gp_lcb import choose_gp_lcb
large_miss_count=0
run_block('pool_gp_lcb',7,lambda:choose_gp_lcb(gp,h.available_indices,beta=1.0),observe=gp.tell,early_stop=gp_miss_stop)
