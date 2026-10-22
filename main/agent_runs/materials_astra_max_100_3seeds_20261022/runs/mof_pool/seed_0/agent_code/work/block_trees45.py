from work.pool_trees import PoolTrees
trees=PoolTrees(Z,beta=1.5,seed=0)
for idx,y in zip(indices,values):trees.tell(idx,y)
large_miss_count=0
run_block('pool_extratrees_lcb',10,lambda:trees.choose(h.available_indices),observe=trees.tell,early_stop=gp_miss_stop)
