from src.optimizers.sobol_search import SobolSearch
sobol = SobolSearch(bounds=h.bounds, seed=2)
method('sobol_anchor')
center = np.full(11, 0.5)
center_first = evaluate(center)
initial_design = sobol.ask(n=16)
initial_values = []
for candidate in initial_design:
    initial_values.append(evaluate(candidate))
sobol.tell(initial_design, initial_values)
center_last = evaluate(center)
print(json.dumps(dict(center_first=center_first, center_last=center_last,
                      center_change=center_last-center_first)), flush=True)
