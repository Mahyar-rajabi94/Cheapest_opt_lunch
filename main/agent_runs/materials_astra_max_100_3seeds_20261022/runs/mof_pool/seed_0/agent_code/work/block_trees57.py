run_block('pool_extratrees_lcb',12,lambda:trees.choose(h.available_indices),observe=trees.tell,early_stop=gp_miss_stop)
