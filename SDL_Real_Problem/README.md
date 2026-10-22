# SDL Real Problems: Cheapest Lunch on self-driving-lab science

Three optimization problems taken from real work in the SDL5 formulation lab, each written as a black-box task the "Cheapest Lunch" optimizer agent can run on. Each problem has a full definition (variables, bounds, equations, objective), its data (real where it exists, simulated where it does not), a best-known reference optimum, and baseline results from the 9 optimizers on `master`.

| # | Problem | Science | Dim | Data | Eval cost | Landscape |
|---|---|---|---|---|---|---|
| [01](01_nonnewtonian_blend/PROBLEM.md) | Blend 4 thickeners + glycerol + pH + mix time to hit a target flow curve | formulation / rheology | 7 | real AutoVisco calibration (4 thickeners) + real PEPV blend validation + 2048 simulated recipes | ~0.1 ms | nonlinear (η ∝ c^2–4.6), 3 basins, flat pH regions, active bounds |
| [02](02_viscometer_calibration/PROBLEM.md) | Joint calibration of the gap-stepping cone-plate viscometer | instrument physics / inverse problem | 26 | 100% real: 23 certified silicone standards, 448 points | ~0.015 ms | narrow curved valley (k–p coupling), 23 coupled offsets, very flat optimum |
| [03](03_robot_cell_layout/PROBLEM.md) | Lay out the UR3e arm, CNC and grasp to minimize the mixer-loading cycle time | robotics / mechanical | see folder | real cell geometry and kinematics from `sdl_dt` + simulated layouts | ~2–6 ms | discrete IK branch switches, infeasible regions, non-smooth penalties |

Together they cover three different kinds of difficulty, which is the point for an agent that is supposed to choose and switch optimizers.

## Layout

```
SDL_Real_Problem/
├── run_baselines.py          shared runner: registers a problem's task with main/'s
│                             campaign loop and sweeps all 9 master optimizers
├── 01_nonnewtonian_blend/
├── 02_viscometer_calibration/
└── 03_robot_cell_layout/
    ├── PROBLEM.md            problem definition, equations, data provenance, results
    ├── task.py               make_task() -> src.tasks.base.Task (same interface as main/ tasks)
    ├── model.py / build_*.py the physics model, and the script that builds data/ and f_opt
    ├── data/                 real and simulated data, reference_optimum.json
    ├── figures/              problem figures
    └── results/<task>/       per-optimizer traces, plots, summary.md
```

## Reproduce

From the repository root (dependencies: `main/requirements.txt` plus pandas and matplotlib):

```bash
python SDL_Real_Problem/run_baselines.py 01_nonnewtonian_blend --seeds 10 --workers 4
python SDL_Real_Problem/run_baselines.py 02_viscometer_calibration --seeds 10 --workers 4
python SDL_Real_Problem/run_baselines.py 03_robot_cell_layout --seeds 10 --workers 4
```

Sweep settings match `main/scripts/run_sweep.py`: budget 2000; GP-BO, TuRBO and BOBYQA capped at 100 evaluations; deterministic methods (BOBYQA, Nelder–Mead, L-BFGS-B) run seed 0 only. All runs here were on Windows. `main/ADDING_TASKS.md` keeps `results_linux/` for Linux runs only, so these results stay inside each problem folder.

## Results

Median regret over seeds against each problem's best-known optimum; lower is better. GP-BO and TuRBO are at 100 evaluations, the rest at 2000.

| Method | 01 blend (f_opt 0.0673) | 02 viscometer (f_opt 1.654%) | 03 robot cell (f_opt 86.19)* |
|---|---:|---:|---:|
| **v2 scripted agent** (3 seeds) | 0.0233 | 7.33 | 823 |
| CMA-ES | **0.00016** | 11.7 | **243** |
| Differential evolution | 0.0244 | 51.6 | 923 |
| Nelder–Mead | 0.0278 | 6.40 | 1076 |
| Random search | 0.0519 | 49.6 | 1009 |
| TuRBO (100) | 0.0544 | 53.1 | not run |
| Sobol | 0.0589 | 48.6 | 970 |
| L-BFGS-B | 0.0618 | **0.070** | 1238 |
| GP-BO (100) | 0.0661 | 69.5 | not run |
| BOBYQA | – (hung on Windows) | – (hung on Windows) | not run |

\*Problem 3 is a reduced sweep: 3 seeds, 6 optimizers. Its f_opt is a penalized best-known value, because no fully collision-free layout has been found yet (see its PROBLEM.md).

**The winner flips between problems.** CMA-ES wins the multimodal formulation problem and the robot layout, where L-BFGS-B is close to the worst. L-BFGS-B wins the smooth 26-D calibration, where CMA-ES is 170× worse. No single method is safe for all three, which is the case for an agent that picks and switches methods.

**The team's v2 scripted agent** (`agent-v2` branch, no LLM; run with `run_agent.py`) is never the best and never the worst. It ranks 2nd on 01 and 03 and 3rd on 02. On 01, its probes classify the landscape as "smooth single basin" and send it to L-BFGS-B, which this problem punishes. On 02, it reads the probes as "several basins", spends 600 evaluations probing and about 900 on CMA-ES, and switches to L-BFGS-B, this problem's winner, only at evaluation 1510. These are concrete cases where a better switching policy would win. The LLM policy needs an API key that isn't set on this machine, so it was not run.

Figures in each `results/<task>/` folder, in the team's format: `plot1_counterfactual_bar.png`, `plot2_regret_vs_eval.png`, and `plot3_agent_method_timeline.png` (stars mark each agent decision, from `decisions.jsonl` → `changelog.md`). Regenerate with `python SDL_Real_Problem/make_plots.py`.

## Using these with the agent

Each `task.py` can be wrapped as a blind harness in the same way as `main/src/agent_harness/task_<NNN>.py`. Copy the objective, not the `f_opt`, and build a sandbox with `main/scripts/make_sandbox.py`. The `PROBLEM.md` files name the science in full, so **do not ship them into an agent sandbox**. A blind brief should state only the dimension, bounds and budget.
