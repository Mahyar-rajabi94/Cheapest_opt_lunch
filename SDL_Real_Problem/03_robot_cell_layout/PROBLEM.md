# Problem 3: Robot cell layout for the SDL5 formulation cell

## Motivation

In the SDL5 formulation cell (Acceleration Consortium), a UR3e arm on the workstation deck tends a Thinky ARE-312 planetary mixer next to a FormulatorCNC_V2 that holds two balances. For every sample, the arm opens the mixer lid, takes the cup off the balance, puts it in the mixer's tilted holder and closes the lid. Where the arm and the CNC stand, how far the lid opens, and how the gripper holds the handle and the cup decide whether the cycle is possible at all, how close the arm passes to the housing and the open lid, and how long each sample waits. Making this cycle faster speeds up the whole lab, since every formulation passes through it.

The sdl_dt digital twin (`C:/Users/mrast/Github_Mahdi/sdl_dt`) plans the cycle exactly, but one `plan_lid_cycle()` call takes about 58 s. The repo's default layout is also infeasible: its best-effort plan has −35.5 mm clearance. This task is a reduced model of the same cycle. It uses the twin's real geometry and kinematics, runs in 2–6 ms per evaluation, and so can be searched with 2000-evaluation budgets.

It's hard in a different way from the other two problems:
- It's mixed-integer: two IK branches out of 8 each, so 64 combinations.
- The reachable set is a thin sliver. None of 2048 Sobol layouts in [data/simulated_layouts.csv](data/simulated_layouts.csv) reaches every target.
- Inside the reachable set, the objective is non-smooth: collision depths come from many min/max operations, and the ready pose switches between discrete fallbacks.
- Cycle time competes with clearance.

## Decision variables (15)

| # | symbol | meaning | units | bounds |
|---|---|---|---|---|
| 0 | b_x | UR3e base x on the deck (cell frame = mixer frame) | m | [−0.316, 0.357] (deck inset by the base radius) |
| 1 | b_y | UR3e base y | m | [−0.70, 0.265] |
| 2 | b_ψ | UR3e base yaw | rad | [−π, π] |
| 3 | c_x | CNC footprint centre x (yaw fixed: balances face the mixer side) | m | [−0.080, 0.121] |
| 4 | c_y | CNC footprint centre y | m | [−0.960, 0.018] |
| 5 | θ_o | lid opening angle | rad | [20°, 70°] (70° = `door_joint` upper limit) |
| 6, 7 | τ_x, τ_y | gripper tilt on the lid handle about tool x and y | rad | [−30°, 30°] |
| 8 | α | rim azimuth at which the cup is picked from the balance | rad | [0, 2π] |
| 9–11 | s_A, e_A, w_A | IK branch for lid handling and the pick: shoulder, elbow, wrist (≥ 0.5 → +1, else −1) | – | [0, 1] |
| 12–14 | s_B, e_B, w_B | IK branch for the moves in and out of the mixer holder | – | [0, 1] |

**Why two branches.** sdl_dt keeps one IK branch while tracking the lid and picking the cup. It plans the holder approach from fresh IK seeds, so the arm may reconfigure in the joint moves before and after it. Under the default layout, no single branch reaches both the lid arc and the holder. The branch choice is left to the optimizer instead of being searched inside the objective. That makes it a real discrete design decision, the configuration taught to the robot, and it keeps each evaluation cheap.

## Model

All geometry comes from sdl_dt via [extract_geometry.py](extract_geometry.py) and is stored in [data/cell_geometry.json](data/cell_geometry.json). Each number is tagged with its source file.

**(E1) Forward kinematics.** The UR3e URDF chain gives T_tool0(q) = T_base · Π_k O_k R_z(q_k). Here O_k are the URDF joint origins, and T_base = Trans(b_x, b_y, h_plate) · R_z(b_ψ) with h_plate = 19.05 mm. The TCP sits 0.113 m along tool z (gripper body 75 mm + fingers 50 mm − 12 mm). This FK matches sdl_dt's `tool_pose` to 3e-16 m.

**(E2) Inverse kinematics.** Closed-form UR IK on the standard UR3e DH parameters (d1 = 0.15185, a2 = −0.24355, a3 = −0.2132, d4 = 0.13105, d5 = 0.08535, d6 = 0.0921), with T_URDF = R_z(π) · T_DH. One branch (s, e, w) ∈ {±1}³ gives:
- θ1 = atan2(p05y, p05x) + s · acos(d4/r) + π/2, with p05 = p − d6 z6 and r = |p05,xy|;
- θ5 = w · acos((p_x sinθ1 − p_y cosθ1 − d4)/d6);
- θ6 from T06⁻¹;
- θ3 = e · acos((l² − a2² − a3²)/(2 a2 a3)) for the planar 2-link joints 2–4 in frame 1's x–y plane, where l = |p14,xy|;
- θ2 = atan2(p14y, p14x) − atan2(a3 sinθ3, a2 + a3 cosθ3);
- θ4 from T34.

Where an acos argument leaves [−1, 1], the step has no solution. The **reach deficit** in metres records by how much: max(d4 − r, 0) + d6 · max(|arg5| − 1, 0) + max(l − |a2| − |a3|, 0) + max(||a2| − |a3|| − l, 0). Validation: FK(IK(T)) = T to 6e-10 for every target the branch reports reachable, and every random target is recovered by some branch.

**(E3) Joint turns.** Each solution's 2π turns are chosen nearest the previous configuration, within the ±2π limits. The elbow, limited to ±π, is never turned.

**(E4) Lid and handle.** The lid frame is L(θ) = Trans(0, 0, 0.328) · Rot(−y, θ), checked against sdl_dt's `link_transforms`. The handle sits at (0.21, 0, 0.04) in the lid frame. The TCP target holding it is H(θ) = L(θ) · Trans(handle) · diag(1, −1, −1) · R_x(τ_x) · R_y(τ_y), identical to `_handle_grasp`.

**(E5) Cup.**
- **On the balance:** C_s = Trans(c_x, c_y, 0) · C_cnc. C_cnc is the cup frame on the arm-side balance in the CNC frame, from `_cup_on_scale`.
- **Grasp:** G(α) = Trans(r cosα, r sinα, h − 0.01) · R_z(α) · diag(1, −1, −1) · R_x(30°), the rim-wall pinch from `_cup_grasp`.
- **In the holder:** the cup goes to the `thinky_cup` seat (bowl at −115°), approached along the holder's axis and released 0.08 m above the seat. It is held at azimuth 130° for placing, as in `_move_cup`.

**(E6) Linear moves.** Position is interpolated linearly and orientation about a single axis (log/exp map). Each linear move is sampled at 5 poses, each solved by IK.

**(E7) Timing.** Every move uses a trapezoidal profile: t(D; v, a) = 2√(D/a) if D < v²/a, else D/v + v/a.
- **Joint moves:** max over joints of t(|Δq_j|; 1.05 rad/s, 1.4 rad/s²), the UR movej defaults.
- **Linear moves:** max(t(path; 0.25 m/s, 1.2 m/s²) [UR movel defaults], max over joints of t(Σ|Δq_j|; 1.05, 1.4)).
- **Lid arcs:** max(t(handle arc length; 0.10 m/s, 0.5 m/s²) [assumed safe handle speed], joint-path time).
- **Gripper actions:** 0.7 + 0.6 + 0.7 + 0.8 + 0.7 + 0.6 s, sdl_dt's durations.

**(E8) The cycle.** The 18 moves of `_plan_from` and `_move_cup`, in order:
1. ready → above closed handle (joint)
2. descend onto handle (line)
3. open lid 0 → θ_o (arc)
4. back off (line)
5. → above cup (joint)
6. down to rim (line)
7. lift cup (line)
8. → high waypoint between balance and bowl (joint, branch A → B)
9. → over the bowl (joint)
10. → pre-insert (line)
11. insert (line)
12. withdraw (line)
13. up (line)
14. → above open handle (joint, B → A)
15. re-engage (line)
16. close lid (arc)
17. lift off (line)
18. return to ready (joint)

Fixed planner choices come from sdl_dt: 0.05 m handle stand-off, 0.09 m cup lift, 0.08 m insert stand-off, a 0.08 m rise over the bowl, and a 0.04 m rise at the high waypoint. The ready pose is the first of sdl_dt's four fallbacks (back 0.35/0.35/0.3/0.2 of the way toward the base, lift 0.10/0.05/0/0 m) that branch A reaches. The cycle time T is the sum of (E7) over the 18 moves plus the gripper times.

**(E9) Clearance.** Each UR3e link is enclosed by 6 spheres, k-means clusters of its visual-mesh vertices with each cluster's farthest vertex on its sphere. The gripper is 4 spheres along the tool axis and the held cup 3. Joint moves are sampled at 5 configurations, linear moves at 5 and arcs at 7, about 100 configurations per cycle. For each configuration, d = min over spheres of the signed distance to the obstacles, minus the sphere radius. The obstacles are:
- the mixer body box, open over the bowl cavity as in `_mixer_body_distance`;
- the lid box at the current lid angle;
- 33 CNC frame/axis boxes and 8 balance boxes;
- the deck plane z = 0. The base housing is exempt.

In contact phases (on the handle, on the cup rim, inserting or releasing), the gripper and cup are left out of the mixer, lid and CNC checks, as in sdl_dt. Collision C = Σ_configs max(0, 5 mm − d) [m], with the 5 mm margin of `plan_lid_cycle(min_clearance=0.005)`.

**(E10) Manipulability.** The Yoshikawa measure w = √det(J Jᵀ) at the TCP, along both lid arcs. The shortfall is M = Σ max(0, 1 − w/0.01), using the 0.01 normalization of sdl_dt's plan score.

**(E11) Footprints.** O is the sum of three penetration depths: the UR3e base disk (r = 64 mm) into the mixer rectangle, the base disk into the CNC rectangle, and the CNC rectangle into the mixer rectangle, all in metres.

**(E12) Objective, minimized, in seconds.**

```
if reach_deficit > 0 or O > 0:   f = 1000 + 1000 · (reach_deficit + O)                        (≥ 1000)
else:                            P = 500·C + 20·J + 2·M     (J = Σ IK jumps above 0.25 rad within a linear move or arc)
                                 f = T + 900 · (1 − exp(−P/900))                               (< 1000)
```

The tiers guarantee that every reachable layout beats every unreachable one, and the graded deficit gives a slope back toward reachability. The saturating penalty keeps a colliding layout always worse than a clean one with the same T. A fully feasible layout (C = J = M = 0) scores exactly its cycle time.

## Data

| file | kind | content |
|---|---|---|
| [data/cell_geometry.json](data/cell_geometry.json) | real (CAD/URDF) | all geometry above, exported from sdl_dt with source tags |
| [data/validation.json](data/validation.json) | check | FK vs sdl_dt (3e-16 m); IK round trip (6e-10); handle/cup/balance targets vs sdl_dt (≤ 1e-16); default-layout terms; timing |
| [data/simulated_layouts.csv](data/simulated_layouts.csv) | simulated | 2048 scrambled-Sobol layouts with all 15 variables and every objective term |
| [data/reference_optimum.json](data/reference_optimum.json) | simulated | best-known x*, f_opt, its terms, how it was found |

**sdl_dt's default layout.** The best of its 64 branch pairs × 24 pick azimuths has cycle time **36.27 s**, against sdl_dt's own planner at 36.3 s. It is infeasible: min clearance −91 mm, upper arm over the mixer body near the holder. sdl_dt's best-effort plan is infeasible too, at −35.5 mm. The deeper collision here comes from the fixed planner choices listed under simplifications.

## Simplifications compared with `plan_lid_cycle`

- **Fixed planner choices.** One handle stand-off (sdl_dt tries 0.07/0.05/0.03 m), one rise over the bowl (sdl_dt tries 4), one holder IK branch (sdl_dt tries 33 seeds), and joint moves interpolated linearly in joint space. sdl_dt does the same, but samples at 30 Hz.
- **Sphere collision model.** Spheres instead of every mesh vertex, about 1–2 cm conservative. No self-collision.
- **No dynamics.** Timing assumes UR default speed and acceleration limits, with no payload effects.
- **CNC yaw fixed at 0** and the CNC axes parked, as in sdl_dt.

## Reference optimum

[build_data.py](build_data.py) `reference` sets f_opt:
- multi-start CMA-ES in the unit box (σ = 0.3), with the first start at sdl_dt's default layout;
- Nelder–Mead polish of the 9 continuous coordinates of the best runs, with their branch flags fixed;
- the best value found becomes f_opt.

It is a best-known value, not a proven optimum, so regret can in principle go negative. The values are in `data/reference_optimum.json`.

## Running

```bash
python SDL_Real_Problem/03_robot_cell_layout/extract_geometry.py        # needs sdl_dt; already done
python SDL_Real_Problem/03_robot_cell_layout/build_data.py validate     # needs sdl_dt
python SDL_Real_Problem/03_robot_cell_layout/build_data.py dataset
python SDL_Real_Problem/03_robot_cell_layout/build_data.py reference --workers 1
python SDL_Real_Problem/run_baselines.py 03_robot_cell_layout --seeds 10 --workers 4
```

```python
import model; model.evaluate(x)   # every term: cycle_time, reach_deficit, min_clearance, collision, ...
```

## Results

This was a reduced sweep, for time: **6 optimizers** (random search, Sobol, CMA-ES, DE, Nelder–Mead, L-BFGS-B), **3 seeds** (deterministic methods seed 0 only), **budget 2000**. GP-BO, TuRBO and BOBYQA were not run. The run used `python SDL_Real_Problem/run_baselines.py 03_robot_cell_layout --seeds 3 --workers 4 --optimizers random_search sobol cma_es de nelder_mead lbfgsb` and took 27 s wall time. Full table: [results/summary.md](results/summary.md). Plots: [results/robot_cell_layout_d15/plot1_counterfactual_bar.png](results/robot_cell_layout_d15/plot1_counterfactual_bar.png) and [plot2_regret_vs_eval.png](results/robot_cell_layout_d15/plot2_regret_vs_eval.png).

The reference is f_opt = 86.19 s, from 12 CMA-ES restarts × 15 000 evaluations plus Nelder–Mead polish. It is a best-known value and **not collision-free**: cycle time 37.65 s, min clearance −25 mm, penalty 48.5 s. No fully feasible layout has been found yet, so f_opt may still drop.

| rank | optimizer | seeds | median best f (s) | median regret | best seed | worst seed |
|---:|---|---:|---:|---:|---:|---:|
| 1 | cma_es | 3 | 329.18 | 243 | 326.37 | 391.06 |
| 2 | de | 3 | 1008.8 | 923 | 878.82 | 1034.4 |
| 3 | sobol | 3 | 1056.5 | 970 | 800.96 | 1154.3 |
| 4 | random_search | 3 | 1095.5 | 1010 | 1023.6 | 1134.6 |
| 5 | nelder_mead | 1 | 1162.6 | 1080 | 1162.6 | 1162.6 |
| 6 | lbfgsb | 1 | 1323.9 | 1240 | 1323.9 | 1323.9 |

Values above 1000 mean the layout is still unreachable (E12 tier 1). Values below 1000 are reachable layouts with some collision penalty.

**Interpretation.**
- **CMA-ES is the only method that reliably crosses into the reachable tier.** All 3 seeds reach 326–391 s: cycles near 35 s with large collision penalties. It follows the graded reach deficit across the thin reachable sliver, and its covariance adaptation handles the strongly coupled base pose, CNC position and tilts.
- **DE, Sobol and random search mostly stay unreachable** (medians above 1000). Single lucky points do get through: one Sobol seed reaches 801 and one DE seed 879. This matches the dataset, where none of 2048 Sobol layouts is reachable.
- **The local methods fail.** Nelder–Mead and L-BFGS-B start from the box centre, stall on the non-smooth reach-deficit surface, and do worse than random search. L-BFGS-B's finite-difference gradients are meaningless across the rounded branch flags and the min/max terms.
- **Nobody reaches f_opt.** Even CMA-ES ends about 240 s above the reference. The step from reachable to collision-free, which requires the right branch pair and a reach into the bowl under the open lid, is the bottleneck. That is where switching optimizers, for example a global search for the branch flags followed by local refinement of the continuous layout, should pay off.
