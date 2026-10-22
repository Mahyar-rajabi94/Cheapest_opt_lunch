class OptimizerExhausted(Exception):
    """Raised by an optimizer's `ask` when it has nothing further to
    propose -- e.g. a closed-loop local solver (BOBYQA, NelderMead) that
    has converged, or stopped, before the externally requested budget was
    used up. Distinct from a plain RuntimeError so callers (see
    scripts/run_campaign.py) can specifically catch this case and stop a
    campaign early, rather than treat it as an unexpected bug.
    """
