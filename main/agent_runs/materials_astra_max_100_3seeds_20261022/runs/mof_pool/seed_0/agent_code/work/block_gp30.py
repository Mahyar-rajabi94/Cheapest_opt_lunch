run_block('pool_gp_logei',10,lambda:gp.choose(h.available_indices),observe=gp.tell,early_stop=gp_miss_stop)
