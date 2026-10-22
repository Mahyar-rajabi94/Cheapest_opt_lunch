from work.ensemble_acquisition import choose_ensemble
assert trees.ids==indices[:len(trees.ids)]
start=len(trees.ids)
for idx,y in zip(indices[start:],values[start:]):trees.tell(idx,y)
assert trees.ids==gp.ids==indices
large_miss_count=0
def update_both(idx,y):
    gp.tell(idx,y);trees.tell(idx,y)
run_block('pool_gp_tree_lcb',6,lambda:choose_ensemble(gp,trees,h.available_indices,beta=1.5),observe=update_both,early_stop=gp_miss_stop)
