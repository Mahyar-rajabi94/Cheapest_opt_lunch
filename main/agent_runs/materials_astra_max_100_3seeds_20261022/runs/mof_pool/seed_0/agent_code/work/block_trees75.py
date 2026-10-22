run_block('pool_extratrees_lcb',8,lambda:trees.choose(h.available_indices),observe=trees.tell,early_stop=gp_miss_stop)
