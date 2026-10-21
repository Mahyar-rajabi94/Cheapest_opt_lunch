# turbo — trust-region Bayesian optimization (TuRBO-1)
**Good for:** expensive evaluations and smallish budgets (≤ 50·dim); finding
good regions sample-efficiently.
**Weak at:** cheap evaluations with large budgets (model fitting is slow and
the final convergence is loose); it is beaten by CMA-ES + polish there.
**State:** `tr_length` shrinks on failure; a restart happens automatically at the minimum.
**Use as:** the main method when evaluations are expensive; a race candidate otherwise.
