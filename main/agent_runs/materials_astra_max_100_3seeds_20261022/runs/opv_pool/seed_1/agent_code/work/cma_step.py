import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = str(Path('work/mplconfig').resolve())
import cma

cma_options = dict(bounds=[0.0, 1.0], seed=1, popsize=8, CMA_mu=4,
                   CMA_diagonal=True, CMA_active=False, verbose=-9, verb_log=0)
es = cma.CMAEvolutionStrategy(center.tolist(), 0.20, cma_options)
cma_generations = []

def run_cma_generations(n):
    for _ in range(n):
        if min(block_end - h.evaluations_used, h.evaluations_remaining) < es.popsize:
            raise RuntimeError('A full population would exceed the current limit')
        population = es.ask()
        values = [evaluate(x, 'cma_generation_' + str(es.countiter + 1)) for x in population]
        es.tell(population, values)
        summary = {'generation': es.countiter, 'minimum': min(values),
                   'median': float(np.median(values)), 'sigma': float(es.sigma),
                   'mean': es.mean.tolist(), 'stds': es.stds.tolist(),
                   'stop': dict(es.stop())}
        cma_generations.append(summary)
        print(json.dumps(summary), flush=True)
        if es.stop():
            break

begin_block('diag_cma', 16)
run_cma_generations(2)
checkpoint()
