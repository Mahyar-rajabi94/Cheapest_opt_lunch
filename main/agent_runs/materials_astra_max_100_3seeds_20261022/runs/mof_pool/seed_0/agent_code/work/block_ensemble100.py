run_block('pool_gp_tree_lcb',3,lambda:choose_ensemble(gp,trees,h.available_indices,beta=.75),observe=update_both,early_stop=gp_miss_stop)
