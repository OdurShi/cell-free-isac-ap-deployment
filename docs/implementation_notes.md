# Implementation notes

Paper: F. Xu, S. Liu, Z. Mao, S. Shi, D. Xu, D. Wang, Y. Huang, "Access Point Deployment for
Localizing Accuracy and User Rate in Cell-Free Systems", ACM MobiCom '24,
doi:10.1145/3636534.3698221. Equation, table and figure numbers below refer to it.

The training code of the original study was not preserved; only its experiment artifacts
(`experiments/`) were. Everything under `src/cfisac/` was written from the paper. The
artifacts were used as ground truth for the constants the paper does not state. This file
records how the paper maps to the code, every choice that the paper leaves open, and the
points where the paper, its figures and the archived logs disagree.

Contents

1. [Paper to code map](#1-paper-to-code-map)
2. [Scenario constants](#2-scenario-constants)
3. [Archived variant: what the original logs contain](#3-archived-variant-what-the-original-logs-contain)
4. [Unspecified details and choices](#4-unspecified-details-and-choices)
5. [Recorder format](#5-recorder-format)
6. [Known limitations](#6-known-limitations)
7. [Notes for the authors](#7-notes-for-the-authors)
8. [What was run for this release](#8-what-was-run-for-this-release)
9. [How these outputs relate to the numbers in the paper](#9-how-these-outputs-relate-to-the-numbers-in-the-paper)

## 1. Paper to code map

| Paper | Code |
|---|---|
| Positions `t_m`, `r_n`, `u_k`, `p` (Section 2) | arrays `[M, 2]`, `[N, 2]`, `[K, 2]`, `[Q, 2]` in `cfisac/metrics.py` |
| Eq. (6), bearing vectors `alpha = (cos theta, sin theta)` | `metrics.unit_directions`, `metrics.bearing_angles` |
| Eq. (5), `Phi = J0^T J0` (with `Sigma = I`) | `metrics.fim_terms` returns `A = phi_11`, `B = phi_22`, `C = phi_12` |
| Eq. (7), determinant in angular form | `metrics.fim_determinant_from_angles` |
| Eq. (8), determinant in Cartesian form | `metrics.fim_determinant(..., metric="eq8")` |
| Eq. (11)/(14), Euclidean-distance SNR | `metrics.snr` |
| Eq. (15), rate `R_k` | `metrics.user_rates` |
| Eq. (16), deployment region | `ScenarioConfig.snap`, `ScenarioConfig.action_to_positions` |
| Eq. (17), max-sum and max-min objectives | `metrics.evaluate_deployment`, `ObjectiveBreakdown.isac` |
| Table 3 comparison objectives | `ObjectiveBreakdown.value(objective="sensing" / "communication" / "weighted")` |
| Eq. (18), state | `env.APDeploymentEnv._draw_snapshot` |
| Eq. (19), action | `ScenarioConfig.action_to_positions` |
| Eq. (20), reward | `env.APDeploymentEnv.step` |
| Eq. (21), maximum-entropy objective | realised by the three SAC losses below |
| Eq. (22), actor loss | `agents/sac.py: SACAgent._actor_loss` |
| Eq. (23), critic loss, twin critics, target critics | `agents/sac.py: SACAgent._critic_loss` |
| soft update `phibar <- tau phi + (1 - tau) phibar` | `agents/networks.py: soft_update` |
| Eq. (24), temperature loss, `H_bar = -dim(A)` | `agents/sac.py: SACAgent._temperature_loss` |
| Table 1 hyper-parameters | `agents/base.py: AgentConfig` defaults, `configs/agents/sac.yaml` |
| Baselines of Fig. 3 (DDPG, TD3, A2C; PPO is in the archive) | `agents/td3.py`, `agents/ppo.py`, `configs/agents/*.yaml` |
| Fig. 3 learning curves | `analysis.aggregate_runs`, `plotting.plot_learning_curves`, `cfisac curves` |
| Figs. 4-5 deployment maps | `plotting.plot_deployment`, `cfisac deployment` |
| Tables 2-3 | `tables.compute_tables`, `cfisac tables`, `cfisac check-table3` |
| Scenario of Section 4 as fixed data files | `cfisac/data.py`, `scripts/generate_data.py`, `data/` |
| Figures 1-5 as images | `scripts/extract_paper_figures.py`, `assets/paper_figures/` |

A slow, loop-based transcription of Eqs. (8), (14), (15) and (17) lives in
`cfisac/reference.py`; the tests compare the vectorised code against it and against the
Lagrange-identity form of the determinant (`sum_{i<j} (v_i x v_j)^2` over the rows of `J0`).

All agents are compact implementations written for this repository (no Stable-Baselines3
or ElegantRL dependency) behind one interface, `agents/base.py: Agent`.

## 2. Scenario constants

Section 4 states: one target on a circular trajectory, three UEs Gaussian around it with
variance 2, 1T1R ... settings, Table 1. Everything else is a field of
`cfisac.scenario.ScenarioConfig`. The evidence for each default:

| Field | Default | Source |
|---|---|---|
| `x_min, x_max, y_min, y_max` | 0, 50, 0, 50 m | Original run names contain `50m`; every AP coordinate in the archive lies in [0, 50]; Figs. 4-5 show APs only in that square. |
| `grid_step` | 10 m | The paper says the action space is discretised. Every recorded AP coordinate (`recorder_best_location.csv`, 8739 rows over the 20 runs) is a multiple of 10 m, and the APs in Figs. 4-5 sit on a 10 m lattice. |
| `target_radius` | 10 m | Figs. 4-5 (circle from 15 to 35 m) and the archived `Deployment.png` plots. |
| `target_center` | (25, 25) m; (25.05, 25.05) m in `configs/archived/` | Figs. 4-5 show (25, 25) at plot resolution. The archived sensing values are reproduced exactly only with (25.05, 25.05), see Section 3. |
| `n_trajectory_points`, `trajectory_endpoint` | 25, `true` | The archived `Deployment.png` plots mark 24 distinct points 15 degrees apart; the archived sensing values are reproduced exactly by `linspace(0, 2 pi, 25)` with both ends included (the first point counted twice), not by 24 distinct points, see Section 3. |
| `ue_centers` | (25, 45), (10, 15), (40, 10) m | Centres of the three UE clouds in Figs. 4-5. With these, the recorded communication values are reproduced with a median deviation of 1.3 %, see Section 3. |
| `ue_variance` | 2 per coordinate | "Variance of 2" in Section 4; per-coordinate variance 2 reproduces the recorded rate level of the converged runs. |
| `log_base` | 10 | The paper writes "log". Base 10 reproduces the recorded communication values; bases e and 2 are off by factors of about 2.3 and 3.3, see Section 3. |
| `one_plus_snr` | `true` | First line of Eq. (15), see Section 4. |
| `episode_length` | 256 | `horizon_len: 256` in every archived `hyperparameter.json`; the recorded cumulative sensing value of an evaluation is 256 times the recorded per-step value. |
| `reward_scale` | 1 | `reward_scale: 1` in the archive. |
| `obs_margin` | 10 m | Axis range of Figs. 4-5 (-10 to 60 m); only bounds the observation box. |
| `weight` | 0.5 | Weight of the weighted-sum objective of Table 3; not stated. |
| `min_distance` | 1e-9 m | Numerical guard, never active on the 10 m grid. |

The original run names also contain `10.0m`. The archive does not say what it denotes; both
the candidate-site spacing and the trajectory radius are 10 m.

## 3. Archived variant: what the original logs contain

`ScenarioConfig.sensing_metric` selects the sensing value:

* `"eq8"` (library default, `configs/eq8/`): `|Phi| = A B - C^2`, Eq. (8) exactly as printed.
* `"archived"` (`configs/archived/`, `cfisac.scenario.archived_scenario`):
  `A B - C`, i.e. the cross term is **not squared**, evaluated on the trajectory centred at
  (25.05, 25.05).

The second variant is not in the paper. It was inferred from the archive as follows
(`scripts/analyze_archive.py` repeats the comparison; output in
`results/archive_analysis.md`).

**Sensing.** Each row of `recorder_my.csv` has `best_rewards_s`, the sensing factor of the
deployment stored in the same row of `recorder_best_location.csv`. Over the 20 archived runs
there are 77 distinct deployments (12 with 1T1R, 65 with 2T2R). Recomputing the trajectory
mean of the sensing value for each of them gives:

| Candidate | Max. relative deviation | Median |
|---|---|---|
| `A B - C^2` (Eq. 8 as printed), centre (25.05, 25.05), `linspace(0, 2 pi, 25)` with end point | 1.0 | 0.79 |
| `A B - C`, centre (25.05, 25.05), 256 equispaced points | 4.0e-2 | 1.5e-2 |
| `A B - C`, centre (25.05, 25.05), 24 equispaced points | 4.2e-2 | 1.5e-2 |
| `A B - C`, centre (25, 25), `linspace(0, 2 pi, 25)` with end point | 3.3e-3 | 9.4e-4 |
| `A B - C`, centre (25.05, 25.05), `linspace(0, 2 pi, 25)` with end point | **4.3e-16** | 1.2e-16 |

Eq. (8) as printed cannot produce the recorded values at all: for 1T1R it is identically
zero (Section 7), while the archive records, e.g., 4.0149 for both APs at (40, 10). The last
row reproduces all 77 recorded values to within double-precision rounding. The centre
(25.05, 25.05) was found by an exploratory least-squares fit of centre, radius and phase to
the recorded values (it returned 25.050, 25.050, 10.000, 0; the fit itself is not part of
this repository) and then fixed.
`tests/test_archive.py::test_all_recorded_deployments` asserts the agreement.

**Rate.** `rewards_c / 256` is the mean per-step communication factor of the evaluated
deployment. With the UE centres above, variance 2 and the sum over the three UEs of
`log(1 + SNR_k)`, base 10 matches the records (median deviation +1.3 %, maximum 7.3 %);
base e overshoots by +133 % and base 2 by +237 %. In an exploratory least-squares fit that
is not part of this repository, letting the three UE centres vary moved them by less than
0.3 m and reduced the deviation to 0.3 %; the defaults keep the round values read off the
figures. The spread of `rewards_c` between evaluations and the size of the recorded best
single-step values indicate that each step's rate is an average over a few tens of UE draws
rather than one draw (an estimate, not a measurement); this is why the environment draws one
UE snapshot per trajectory instant at every step (Section 4).

**Reward structure.** In all 8739 archived rows, cumulative reward = (cumulative sensing / 256)
x cumulative rate, to a relative deviation of at most 2.5e-7. The sensing factor is therefore constant within an
episode: every step's reward is the trajectory-level objective of Eq. (17), not a per-point
value, and an episode is 256 such steps.

**Table 3 and Figs. 4-5 of the paper.** Evaluating the deployments drawn in Figs. 4-5 (read
off the figures on the 10 m grid, stored in `cfisac/paper_values.py`) in the archived
scenario with **M = N = 4** reproduces the sensing entries of Table 3 to all printed digits:

| Figure | Deployment (Tx / Rx) | This code | Table 3 |
|---|---|---|---|
| 4(a) SAC, max-sum | 3 x (40, 10) + (40, 0) / the same | `sum_p abs(Phi)/Q` = 688.97 | 688.97 (ISAC) |
| 4(e) communication only | 4 x (40, 10) / 3 x (40, 10) + (40, 20) | 642.08 | 642.08 (Comm. only) |
| 4(f) sensing only | 8 x (0, 0) | 842.92, `min_p abs(Phi)` = 696.89 | 842.92, 696.89 (Sensing only, Weighted sum) |
| 5(e) communication only, max-min | (20, 40), 2 x (30, 40), (10, 10) / (30, 40), 2 x (10, 20), (40, 10) | `min_p abs(Phi)` = 188.40 | 188.40 (Comm. only) |

For the sensing-only deployment the rates do not depend on the UE draws; this code gives
`sum_k R_k / Q` = 0.014 and `min_k R_k` = 0.0013, equal to Table 3.
(`cfisac check-table3`, `tests/test_archive.py::test_table3_sensing_entries_from_figure_deployments`.)
With Eq. (8) as printed the deployments of Figs. 4(f) and 5(f) (all APs on one site) have
`|Phi| = 0`.

**Consequence for this repository.** The library default follows the printed equation. The
`archived` variant exists so that the archived logs, Fig. 3 and Tables 2-3 can be related
to code, and it is labelled as such wherever it is used. Results for both are reported
separately in the README.

## 4. Unspecified details and choices

Objective and environment

* **Eq. (15), "1 +".** The first line is `log(1 + SNR_k)`, the last line prints
  `log(sum ...)` without `1 +`. The code uses `log(1 + SNR_k)`: it follows from Eq. (14), keeps
  the rate non-negative, and is what the archived rate values correspond to (the far-field
  values are positive and match `log10(1 + SNR)`). `one_plus_snr: false` evaluates the
  literal last line.
* **Division by `Q` of the rate in Eq. (17).** `sum_k R_k / Q` only has a meaning if rates
  are accumulated over the `Q` trajectory instants. The code draws one snapshot of the `K`
  UEs per instant, `R_k` is the mean over the `Q` snapshots of `log(1 + SNR_k)`,
  `sum_k R_k / Q` in the paper's notation is `sum_k` of that mean, and `min_k R_k` is the
  minimum over UEs of that mean.
* **Eq. (17) versus Eq. (20).** Eq. (20) is Eq. (17) without the `1/Q` factors. The archived
  per-step values equal Eq. (17), so the reward is Eq. (17) times `reward_scale`; the constant
  factor does not change the optimal deployment.
* **State.** Eq. (18) lists the coordinates of the UEs and of the target. The observation is
  the snapshot of the current instant: the target point `p(eps_i)`, `i = step mod Q`, and the
  UE positions drawn for it, in metres (`2 K + 2 = 8` numbers; the archived networks also
  have an 8-dimensional input). The reward of the step is computed on all `Q` snapshots drawn
  at that step.
* **State discretisation.** The paper says states and actions are discretised. Actions are
  snapped to `grid_step`. For the state, the only discretisation applied is the sampling of
  the trajectory at `Q` points; the archive shows no rounding of target coordinates (the
  continuous fit of Section 3 is exact).
* **Action mapping.** Actions live in `[-1, 1]` and are mapped affinely onto the region, then
  snapped; values outside are clipped (Eq. 16). APs may share a site (they do in the archive
  and in Figs. 4-5).
* **Fixed training and evaluation data.** The paper generates its training data in the
  environment. Here the UE positions are materialised once with fixed seeds
  (`data/ue_positions_train.npy`, 409,600 snapshots, seed 1; `data/ue_positions_eval.npy`,
  32,768 snapshots, seed 0; see `data/README.md`). At every step the training environment
  takes a block of `Q = 25` consecutive snapshots, chosen uniformly at random with the run's
  seed; evaluation does the same on the evaluation split. The optimisers and table scripts
  score deployments on the first `n` snapshots of the evaluation split, which is identical
  to drawing `n` snapshots with `numpy.random.default_rng(0)` (tested). When the files are
  absent, when a scenario changes the UE centres or variance, or with
  `cfisac train --online-ues`, positions are drawn on the fly from the same distribution.
* **Dynamics.** The next state does not depend on the action. Episodes are cut after 256
  steps and never terminate; agents bootstrap through the time limit.
* **Weighted-sum objective.** `weight * communication + (1 - weight) * sensing` with
  `weight = 0.5`; the paper gives no weight.

Agents (Table 1 gives hidden layers, learning rate, buffer size, batch size, discount, tau)

* Taken from the archived `hyperparameter.json` files: `clip_grad_norm = 3`,
  `reward_scale = 1`, `horizon_len = 256` (steps collected between update phases), batch size
  1024 and no replay buffer for A2C / PPO. `state_value_tau = 0` in those files is taken to
  mean that no running state normalisation was used.
* **Eq. (23) as printed** lacks the temperature in front of `log pi` and brackets around the
  soft value. The code uses the standard SAC target of the cited reference [5]:
  `r + gamma (min_j Q_phibar_j(s', a') - omega log pi(a' | s'))`.
* **Network inputs.** Observations are mapped affinely from the observation box to `[-1, 1]`
  inside every network (fixed, not learned). The paper does not say how inputs were scaled.
* **Critics.** Two independent Q-networks (the archived `CriticTwin` shares a trunk).
* **Optimiser.** Adam for all networks and for `log omega`, all with the Table 1 learning rate.
* **Not stated anywhere, defaults chosen here:** one gradient step per environment step
  (`repeat_times = 1`), 1024 uniformly random steps before learning starts, initial
  temperature `exp(-1)`, log-std clamp `[-20, 2]`; DDPG / TD3 exploration noise 0.1, TD3
  target noise 0.2 clipped at 0.5 and policy delay 2 (values of the TD3 paper); A2C / PPO
  GAE lambda 0.95, entropy coefficient 0.01, PPO clip 0.25 with 8 reuse passes, minibatches
  drawn with replacement (the archived batch size 1024 exceeds the rollout length 256).
* **Evaluation.** Deterministic policy (`tanh` of the mean), after the first horizon and
  then every `eval_interval` steps. The archive evaluates every 20224 steps; the number of
  evaluation episodes is not recorded (4 by default here).

Non-RL optimisers and tables

* Deployments are scored on a fixed sample of UE snapshots (common random numbers), so that
  each deployment has one deterministic score. On the 10 m grid, 1T1R and 2T2R are solved by
  enumeration. Larger settings use coordinate ascent (one AP moved at a time) from random
  starts and from the 36 deployments with all APs on one site; when a table is computed,
  every search is restarted from the deployments found for the other objectives. The result
  is a local optimum and is not guaranteed to be global.
* `compute_tables` uses `M = N` in {1, 2, 3, 4} for Table 2 (the paper's column label is
  `M + N` in {2, 4, 6, 8}) and `M = N = 4` for Table 3; see Section 7 for why.

## 5. Recorder format

New runs write `recorder.csv` with the column names of the archived `recorder_my.csv`:

| Column | Archive | This repository |
|---|---|---|
| `steps`, `times` | environment steps, seconds | same |
| `rewards`, `std_rewards` | cumulative reward of an evaluation episode (256 steps), mean and std over evaluation episodes | same |
| `rewards_isac`, `rewards_s`, `rewards_c` (and `std_*`) | cumulative objective, sensing factor and communication factor | same; `rewards_isac` is the product objective even when a Table 3 comparison objective is trained |
| `best_rewards`, `best_rewards_isac`, `best_rewards_s`, `best_rewards_c` | values of the best **single step** seen so far | per-step **means of the best evaluation** so far |

`recorder_best_location.csv` has one row per evaluation with the best deployment so far as
`x, y` of the transmit APs followed by the receive APs, as in the archive. The best
deployment of an evaluation is the deployment the deterministic policy selects most often.
Checkpoints are `state_dict`s (`checkpoint_best.pt`, `checkpoint_last.pt`) that load with
`torch.load(..., weights_only=True)`; they are not compatible with the archived checkpoints,
which are pickled ElegantRL modules.

## 6. Known limitations

* The original environment code is not available. The archived variant reproduces the
  recorded sensing values exactly and the recorded rate values approximately; how the
  original code built its state vector, how many UE draws it used per step and how its
  `best_*` values were selected is inferred, not known.
* The training runs of Section 8 cover 163,840 of the 3e6 steps of Fig. 3 and one seed; they
  exercise the training pipeline and do not support a comparison between algorithms.
* The communication entries of Table 3 for deployments close to a UE (1.13, 1.26, 0.042,
  0.043) are not reproduced; see Section 7.
* The archived checkpoints cannot be loaded by this code and were not deserialised. Their
  pickle streams were only disassembled (no code execution) to read class names
  (`elegantrl.agents.net.ActorSAC`, `CriticTwin`, `Actor`, `Critic`, `ActorPPO`, `CriticPPO`)
  and tensor sizes (state dimension 8, action dimension 4 or 8, hidden layers 64 and 32).
* Reward landscapes on the 10 m grid are piecewise constant in the action; all agents see
  zero reward gradient inside a grid cell and rely on exploration noise to cross cells.

## 7. Notes for the authors

These are the points where the paper, its figures and the archived logs do not agree. They
are stated as observations; the fixes are the authors' call.

1. **Eq. (8) versus the numbers.** The recorded sensing values, the sensing entries of
   Table 3 and the co-located "sensing only" deployments of Figs. 4(f) / 5(f) correspond to
   `A B - C` (cross term not squared), not to the determinant `A B - C^2` of Eqs. (7)-(8).
   The two behave very differently: the determinant is zero whenever all APs are seen from
   the target under the same bearing (in particular for co-located APs, and always for
   M = N = 1), whereas `A B - C` is largest for co-located APs seen at 45 degrees to the axes
   (`16 (M N)^2 cos^2 sin^2 - 4 M N cos sin`), is not invariant to rotations of the scene and
   differs between the two diagonals. The statement in Section 4 that the best sensing
   positions "lie on a straight line and form specific angles with the target" describes the
   maximiser of `A B - C`; for the determinant of Eq. (8) collinear APs are the worst case.
2. **M = N = 1 with Eq. (8).** One transmit and one receive AP give a rank-one FIM, so
   `|Phi| = 0` for every geometry and both objectives of Eq. (17) vanish. The M + N = 2 column
   of Table 2 cannot come from Eq. (8).
3. **Which setting Tables 2-3 refer to.** The sensing entries of Table 3 are reproduced by
   eight APs (M = N = 4), as drawn in Figs. 4-5. The product of the ISAC column of Table 3
   (1.13 x 688.97 = 778.5) is close to the Table 2 entry labelled M + N = 4 (775.73), and the
   max-min entry of that column (10.48) is close to the best M = N = 4 value found here by
   local search (10.78), whereas the best 2T2R value is 0.253. The column labels of Table 2
   therefore appear to denote M = N rather than M + N. Settings with more than eight APs
   were not computed here, so the last two columns were not checked.
4. **Table 3, ISAC column, `min_p |Phi|` = 0.25.** Table 2 gives 10.48 for the max-min
   objective of the same setting and Table 3 gives `min_k R_k` = 0.042 for it, which implies
   `min_p |Phi|` of about 250; the deployment of Fig. 5(a), as read off the figure, gives
   246.66 with this code. 0.25 looks like a unit or typing slip.
5. **Communication entries of Table 3 near a UE.** For the deployments of Figs. 4(a), 4(e),
   5(a), 5(e) the mean rates computed here (base 10) are 0.63, 0.68, 0.038, 0.038, against
   1.13, 1.26, 0.042, 0.043 in Table 3, while the far-field entries (0.014, 0.0013) agree.
   A possible explanation is that the table took the `best_*` columns of the recorder, which
   hold the best single step (a favourable UE draw) rather than the mean; this could not be
   verified.
6. **Sensing-only optimum.** Within the `A B - C` objective the corner (0, 0) of Fig. 4(f) is
   not the best co-located site: (0, 50) and (50, 0) give a larger value (for 2T2R: 61.85 and
   61.53 against 47.15), because the sign of `cos sin` differs between the diagonals.
7. **Logarithm base and scenario constants.** The paper does not state the base of the
   logarithm (the logs correspond to base 10), the region, the grid spacing, the trajectory
   radius and sampling, or the UE centres.
8. **Eq. (15)** drops the `1 +` in its last line; **Eq. (16)** has `x_min` and `y_min` on both
   sides of the inequalities; **Eq. (23)** lacks the temperature in front of `log pi`.
9. **A2C and PPO in the archive.** The archive contains PPO runs in addition to the SAC,
   DDPG, TD3 and A2C runs shown in Fig. 3. The A2C and PPO recorders of the same seed are
   identical, apart from the time column, in their first 115 evaluations (1T1R, seed 1) and
   their first 8, 81 and 54 evaluations (2T2R, seeds 1, 128, 256), and differ afterwards.
10. **`best_location_origin.npy`** holds the best deployment with offsets of up to 2 m on APs
   that share a site (the positions drawn in the archived `Deployment.png`); the grid
   positions are in `recorder_best_location.csv`.


## 8. What was run for this release

Machine: Apple-silicon laptop, CPU only; Python 3.11, PyTorch 2.15, NumPy 2.4. Raw outputs
are under `results/`. Nothing below was run at the length of the paper's experiments.

**Archive analysis.** `python scripts/analyze_archive.py` -> `results/archive_analysis.md`.
The numbers are those of Section 3: 77 distinct recorded deployments; maximum relative
deviation of the recomputed sensing factor 4.3e-16 for the `archived` setting, 3.3e-3 with
the trajectory centre at (25, 25), 4.2e-2 with 24 equispaced samples, 1.0 for Eq. (8) as
printed; recorded rate factors matched by base 10 (median +1.3 %, maximum 7.3 %), base e
+133 %, base 2 +237 %.

**Fig. 3(a) re-plotted from the archived recorders.** `python scripts/plot_learning_curves.py`
-> `results/figures/fig3a_archived_2t2r_maxsum.png` and `.json`. This is a plot of logged
data, not a new experiment. Mean of the last 10 % of the evaluations of each run, 2T2R,
max-sum, seeds 1, 128, 256:

| Algorithm | Mean | Std over seeds | Min | Max |
|---|---|---|---|---|
| SAC | 6004.2 | 7.0 | 5995.8 | 6012.9 |
| DDPG | 3601.2 | 1448.5 | 2570.4 | 5649.7 |
| TD3 | 3395.7 | 1285.9 | 1769.0 | 4913.0 |
| A2C | 1887.1 | 504.9 | 1185.4 | 2352.8 |
| PPO | 2194.2 | 510.6 | 1677.5 | 2889.4 |

**Deployments drawn in Figs. 4-5.** `cfisac check-table3` -> `results/tables/figure_deployments.md`.
AP positions read off the figures by eye on the 10 m grid (`cfisac/paper_values.py`),
evaluated in the archived scenario with M = N = 4 on 8192 snapshots of the evaluation split:

| Deployment (figure) | `sum_k R_k / Q` | `sum_p abs(Phi) / Q` | `min_k R_k` | `min_p abs(Phi)` |
|---|---|---|---|---|
| SAC, max-sum (4a) | 0.631 | 688.97 | 0.0022 | 281.93 |
| Communication only, max-sum (4e) | 0.676 | 642.08 | 0.0026 | 257.16 |
| Sensing only (4f, 5f) | 0.0140 | 842.92 | 0.0013 | 696.89 |
| SAC, max-min (5a) | 0.434 | 316.12 | 0.0378 | 246.66 |
| Communication only, max-min (5e) | 0.333 | 298.38 | 0.0379 | 188.40 |

**Tables 2-3 from the non-learning optimisers.** `python scripts/make_tables.py` ->
`results/tables/tables_{archived,eq8}.{json,md}`; `python scripts/plot_deployments.py` ->
`results/figures/deployment_*.png`. 1024 snapshots of the evaluation split; M = N = 1, 2 by
enumeration of the 10 m grid, M = N = 3, 4 by local search (8 random and 36 co-located
starts, cross-seeded between objectives).

| Sensing value | Problem | M = N = 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| `archived` | max-sum | 1.435 | 23.78 | 135.8 | 475.9 |
| `archived` | max-min | 0.00158 | 0.253 | 1.970 | 10.78 |
| `eq8` | max-sum | 0 | 10.16 | 64.52 | 243.4 |
| `eq8` | max-min | 0 | 0.223 | 1.808 | 9.102 |

M = N = 4, each column optimises a different objective:

| Sensing value | Quantity | ISAC | Sensing only | Comm. only | Weighted sum |
|---|---|---|---|---|---|
| `archived` | `sum_k R_k / Q` (max-sum optimum) | 0.723 | 0.0091 | 0.723 | 0.0091 |
| `archived` | `sum_p abs(Phi) / Q` (max-sum optimum) | 658.03 | 901.22 | 658.03 | 901.22 |
| `archived` | `min_k R_k` (max-min optimum) | 0.0446 | 0.0011 | 0.0450 | 0.0013 |
| `archived` | `min_p abs(Phi)` (max-min optimum) | 242.04 | 745.81 | 98.71 | 745.81 |
| `eq8` | `sum_k R_k / Q` (max-sum optimum) | 0.598 | 0.0088 | 0.723 | 0.0174 |
| `eq8` | `sum_p abs(Phi) / Q` (max-sum optimum) | 406.95 | 472.80 | 0 | 472.80 |
| `eq8` | `min_k R_k` (max-min optimum) | 0.0373 | 0.0016 | 0.0450 | 0.0016 |
| `eq8` | `min_p abs(Phi)` (max-min optimum) | 244.10 | 422.71 | 88.00 | 422.71 |

**Reduced-length training runs.** `bash scripts/run_short_training.sh` ->
`results/runs/paper_hparams_{maxsum,maxmin}/`, curves in
`results/figures/short_runs_2t2r_*.png`. Archived 2T2R scenario, seed 1, Table 1 / archive
hyper-parameters (learning rate 1e-5), 163,840 steps (5.5 % of 3e6), evaluation every 5120
steps with two 256-step episodes. These runs were made before the files in `data/` existed,
with UE positions drawn on the fly; the script passes `--online-ues` so that it regenerates
them. Four processes in parallel took 24.5 min of wall-clock time on a heavily loaded
machine (about 13.5 min of CPU time per SAC process).

| Algorithm, problem | Return of the last evaluation | Best evaluation (per step / per episode, at step) | Grid optimum (per step / per episode) |
|---|---|---|---|
| SAC, max-sum | 3353 | 13.30 / 3404, step 153,856 | 23.78 / 6087 |
| TD3, max-sum | 89 | 6.01 / 1539, step 30,976 | 23.78 / 6087 |
| DDPG, max-sum | 41 | 0.46 / 119, step 256 (initial policy) | 23.78 / 6087 |
| SAC, max-min | 10.4 | 0.070 / 18.0, step 107,776 | 0.253 / 64.8 |

None of these runs has converged. SAC on max-sum was still rising when it stopped; TD3 and
DDPG did not improve on what they found early. At the same number of steps the archived
DDPG and TD3 recorders show returns in the thousands, so settings of the original baselines
that are not recorded (exploration noise, input scaling, update schedule) differ from the
ones chosen here.

**Quick start.** `cfisac demo` (SAC with learning rate 3e-4, 15,360 steps, UE positions from
`data/`) printed the output shown in the README: 75 % of the 2T2R max-sum grid optimum.
The README's example `cfisac -v train ... --total-steps 20480` ended with a return of 5500
per 256-step episode.

**Tests and lint.** `pytest`: 118 passed on Python 3.10, 3.11 and 3.12. `ruff check .`: clean.

**Paper figures and data.** `scripts/extract_paper_figures.py` produced the five images in
`assets/paper_figures/`; `scripts/generate_data.py` produced `data/` (regenerating it gives
byte-identical files).

## 9. How these outputs relate to the numbers in the paper

* The SAC plateau of Fig. 3(a), about 6000, is the level of the archived SAC recorders
  (6004 +/- 7) and of the exhaustive optimum of the archived 2T2R max-sum objective
  (23.78 per step, 6087 per 256-step episode, all four APs at (40, 10)); the three archived
  SAC seeds end on that deployment.
* The sensing entries of Table 3 (688.97, 842.92, 642.08, 696.89, 188.40) and its
  sensing-only rate entries (0.014, 0.0013) are obtained from the deployments drawn in
  Figs. 4-5 with the `archived` sensing value and M = N = 4. With Eq. (8) as printed a single
  Tx-Rx pair and any set of co-located APs give exactly zero, and the optima are different
  (rows `eq8` in Section 8).
* The rate entries of Table 3 for deployments next to a UE (1.13, 1.26, 0.042, 0.043) are
  larger than the averages over UE positions computed here (0.63, 0.68, 0.038, 0.038); see
  item 5 of Section 7.
* Table 2, max-sum: the paper's values are larger than the best values found here, by a
  factor of about 1.7 if the column labels are read as `M = N` (41.07 against 23.78, 775.73
  against 475.9) and of 24 to 33 if they are read as `M + N`. The difference is in the rate
  factor (previous item). Table 2, max-min: 10.48 in the column labelled 4 against 10.78
  found here for M = N = 4; 0.042 in the first column against 0.253 for 2T2R and 0.0016 for
  1T1R.
* The qualitative statements of Section 4 of the paper hold for the `archived` objective:
  the weighted-sum optimum coincides with a sensing-only optimum, the max-min optimum spreads
  the APs over the three UEs, the max-sum optimum gathers them at one UE.
* Not covered by any run here: Fig. 3 by new training over 3e6 steps, Fig. 3(b) (the archive
  holds no max-min logs), Table 2 for more than eight APs. The archived checkpoints were not
  loaded.
