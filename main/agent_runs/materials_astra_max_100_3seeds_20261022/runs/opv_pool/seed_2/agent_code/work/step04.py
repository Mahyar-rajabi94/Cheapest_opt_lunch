exec(compile(Path('work/gp_step.py').read_text(), 'work/gp_step.py', 'exec'), globals())
method('gp_history_lcb')
for _ in range(12):
    gp_step(beta=4.)
