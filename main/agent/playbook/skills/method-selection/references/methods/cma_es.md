# cma_es — Covariance Matrix Adaptation Evolution Strategy
**Good for:** continuous problems with interacting variables; ill-conditioned
valleys (it learns the scaling); moderate multimodality with restarts;
noisy or rough landscapes.
**Weak at:** smooth single-basin problems (local solvers are 2–5× faster);
very small budgets (< 20·dim).
**Working well:** best improves every round while `state.sigma` shrinks smoothly.
**Finished:** `last_event = converged` or `sigma` < 1e-8 × box width → polish `active_best`.
**Warm start:** x0 = incumbent, sigma0 = archive local scale (automatic).
**Restarts:** `restart` with `start=diverse` and `popsize_factor` 2, 4, 8 on
successive restarts (IPOP). A larger population explores more globally.
