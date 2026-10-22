for control in [probe_a, probe_b]:
    method('gp_history_exploit')
    for _ in range(3):
        gp_step(beta=0.25)
    method('history_control')
    evaluate(control)
