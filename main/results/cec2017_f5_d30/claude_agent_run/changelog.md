
## Evaluation 96
**Change:** Switched from gp_bo_logei to cma_es_explore
**Reason:** y_best_so_far unchanged over the last 30 evaluations (value=852.9831980496507); triggering pre-registered stagnation deviation to inject broader exploration via CMA-ES. Note: this project's CMAES wrapper only supports starting at the box midpoint (no custom x0/sigma0 argument), so CMA-ES is started fresh from the box midpoint with its own default step size, rather than from the incumbent as originally planned in the roadmap; this is a deviation from the roadmap's wording, logged here, made necessary by the available tool's interface.
