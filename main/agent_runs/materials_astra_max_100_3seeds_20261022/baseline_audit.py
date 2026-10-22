"""Verify supplied baseline traces against the registered pools, without running optimizers."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent
REPO=next(p for p in ROOT.parents if (p/'main/src/tasks').is_dir())
sys.path.insert(0,str(REPO/'main'))
from src.tasks.opv_pool import OPVPoolTask
from src.tasks.mof_pool import MOFPoolTask

def main():
 root=ROOT/'baselines'; summary=[]
 for kind,folder,cls in [('opv_pool',root/'opv/opv_pool_pce_d11',OPVPoolTask),('mof_pool',root/'mof/results_linux/task_061',MOFPoolTask)]:
  for method in sorted(folder.iterdir()):
   if not method.is_dir():continue
   finals=[]; nrows=[]
   for f in sorted(method.glob('seed_*/trace.csv')):
    task=cls()
    with f.open(newline='') as stream:rows=list(csv.DictReader(stream))
    seen=set();best=-np.inf
    for k,row in enumerate(rows,1):
     assert int(row['eval_idx'])==k
     x=np.array([float(row[f'x{i}']) for i in range(task.dim)])
     if task.indexed_pool:
      index=int(row['candidate_index']);obs=task.offline_observation(index)
      assert row['candidate_id']==obs['candidate_id']
      assert np.array_equal(x,task.candidates[index]),(f,k,'features')
      assert float(row['y'])==obs['raw_y'],(f,k,'label')
      assert int(row['rank'])==obs['rank'],(f,k,'rank')
      y=obs['raw_y']
     else:
      index=task.nearest_untested(x)
      assert int(row['candidate_id'])==task._ids[index],(f,k,'nearest candidate')
      y=float(task._pce[index]);assert float(row['y'])==-y,(f,k,'label')
      task._tested[index]=True
     assert index not in seen,(f,k,'repeat')
     seen.add(index);best=max(best,y)
     assert float(row['y_best_so_far'])==(best if task.indexed_pool else -best),(f,k,'incumbent')
     if k==100:finals.append(-task.f_opt-best)
    assert len(rows)>=100
    nrows.append(len(rows))
   result={'task':kind,'method':method.name,'seeds':len(finals),'raw_trace_lengths':sorted(set(nrows)),
           'mean_regret_at_100':float(np.mean(finals)),
           'std_regret_at_100':float(np.std(finals,ddof=1)) if len(finals)>1 else None,
           'all_rows_verified':True}
   summary.append(result)
   print(f'{kind}/{method.name}: {len(finals)} seeds, every recorded row verified',flush=True)
 (root/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
