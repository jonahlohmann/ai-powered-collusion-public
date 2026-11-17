# Map of the implementation choices

This file lists the choices that a rebuild of the simulation of Dou, Goldstein and Ji (2025),
"AI-Powered Trading, Algorithmic Collusion, and Price Efficiency" (**DGJ** below), has to
make, and what each of them does to the measured collusion capacity Δ^C in this repository.
For every choice it gives what DGJ describe, what this implementation does, the
configuration key, the runs that vary it, the effect and the place in the code. Technical
definitions are in `docs/parameters.md`.

All runs use one regime: ξ = 500, two speculators, ρ = 0.95, θ = 0.1, T_m = 10,000, and
σ_u = 1 unless stated. ξ, θ, ρ and the number of speculators are DGJ's values (Dou et al.,
2025, p. 23).

**How to read Δ^C.** Δ^C is 0 at the profit of the competitive Nash equilibrium and 1 at the
profit of the perfect cartel, which lies 12.5% above the Nash profit. Three yardsticks
belong to every level: the Nash equilibrium of the action grid itself (Δ^C −0.029 on the
grid of the baseline specification, −0.277 on the grid of the earlier specification),
uniform random play on the grid of the baseline specification (Δ^C +0.68), and the profit
in percent above the Nash profit. Every Δ^C below is the mean over all sessions of a run
with a bootstrap 95% confidence interval.

## Summary

| Choice | Varied in | Δ^C [95% CI], profit above Nash |
|:--|:--|:--|
| Stopping rule: window of 100,000 periods and cap of 10 million, against 1 million and 40 million | R1.1 against R1.2 | 0.07 [−0.04, 0.16], +0.8% (15 of 30 sessions converged) against 0.19 [0.16, 0.23], +2.4% (grid Nash −0.277 in both) |
| Specification: action grid, exploration counter, Q-initialisation and β together | R1.2 against R1 | 0.19 [0.16, 0.23], +2.4% (grid Nash −0.277) against 0.34 [0.32, 0.36], +4.2% (grid Nash −0.029) |
| Exploration decay β: 1e-6 against DGJ's 5e-7 | R1 against R1.6 | 0.34 against 0.35 [0.33, 0.38], +4.4% |
| Value grid n_v: 5 against DGJ's 10 (with β 5e-7, 8 sessions) | R1 against R1.7 | 0.34 against 0.31 [0.29, 0.32], +3.8% |
| Noise level σ_u: 1 against DGJ's 0.1 | R1 against R1.8 | 0.34 against 0.34 [0.31, 0.36], +4.2% |
| Price used in the reward: raw against rounded to its bin | R1 against R1.3 | 0.34 against 0.91 [0.88, 0.93], +11.3% |
| Lagged price in the state: present against removed | R1 against R1.4 and R1.5 | 0.34 against 0.22 [0.20, 0.24], +2.8%, and 0.24 [0.21, 0.27], +3.0% |
| Layout of the price grid: one grid for all values | exploratory run outside the 12 runs | no level (none of 30 sessions converged within the cap) |
| Benchmark of Δ^C | not varied | – |
| Learner family: Q–Q, Q–AC, AC–AC | R1, R2, R2.1 | 0.34, 0.14 [0.12, 0.15], +1.7%, and 0.09 [0.08, 0.11], +1.2% |
| Learning rate within Q-learning: α 0.01 and 0.1 | R1 against R2.2 | 0.34 against 0.33 [0.30, 0.35], +4.1% |
| Settings of the AC learner | pilot (3 settings × 4 sessions) | chosen by a rule fixed in advance; Δ^C was no criterion |

In short: Δ^C moves from 0.07 to 0.91 with the stopping rule, the action grid and the price
used in the reward. It does not move with n_v, β or σ_u. In every session of every run
trained with the window of 1 million periods, Δ^C lies above the Nash value of its grid; on
the raw price it stays far below the cartel level.

## Runs

Configurations are under `experiments/`, results under `results/<folder>/`
(`report.txt`, `deviation_test.txt`, one folder per session). 30 sessions per run unless
stated; the seed of a session is 42 plus its number.

| Run | What it is | Configuration | Results folder |
|:-:|:--|:--|:--|
| R1 | The baseline: two Q-learners, baseline specification | `01_replicate_baseline/config_r1_baseline.yaml` | `r1_baseline` |
| R1.1 | Earlier specification, window 100,000, cap 10 million | `01_replicate_baseline/config_r1_1_early_stop.yaml` | `r1_1_early_stop` |
| R1.2 | Earlier specification, window 1 million, cap 40 million | `01_replicate_baseline/config_r1_2_earlier_spec.yaml` | `r1_2_earlier_spec` |
| R1.3 | Baseline specification, reward on the rounded price | `01_replicate_baseline/config_r1_3_rounded_price.yaml` | `r1_3_rounded_price` |
| R1.4 | Baseline specification without the lagged price in the state | `01_replicate_baseline/config_r1_4_no_price.yaml` | `r1_4_no_price` |
| R1.5 | R1.4 with β 3e-6 | `01_replicate_baseline/config_r1_5_no_price_beta3e6.yaml` | `r1_5_no_price_beta3e6` |
| R1.6 | Baseline specification with DGJ's β 5e-7, cap 150 million | `01_replicate_baseline/config_r1_6_ref_beta.yaml` | `r1_6_ref_beta` |
| R1.7 | DGJ's value grid (n_v 10) and β 5e-7, cap 400 million, 8 sessions | `01_replicate_baseline/config_r1_7_ref_grid.yaml` | `r1_7_ref_grid` |
| R1.8 | Baseline specification at DGJ's noise level σ_u 0.1 | `01_replicate_baseline/config_r1_8_ref_noise.yaml` | `r1_8_ref_noise` |
| R2 | Q–AC: one Q-learner (agent 0), one AC learner (agent 1) | `02_heterogeneity/config_r2_q_ac.yaml` | `r2_q_ac` |
| R2.1 | AC–AC | `02_heterogeneity/config_r2_1_ac_ac.yaml` | `r2_1_ac_ac` |
| R2.2 | Q–Q with learning rates 0.01 and 0.1 | `02_heterogeneity/config_r2_2_two_alphas.yaml` | `r2_2_two_alphas` |
| Validation runs | Each learner against a rival fixed at the Nash intensity, 4 sessions | `02_heterogeneity/config_validation_{q,ac}.yaml` | `validation_q`, `validation_ac` |
| Pilot | Q–AC, three settings of the AC learner, 4 sessions each | `02_heterogeneity/config_pilot_{a,b,c}.yaml` | `pilot_{a,b,c}` |

The parameter sets are `experiments/shared/params_baseline.yaml` (the baseline
specification), `params_earlier_spec.yaml` (the earlier specification) and
`params_ref_grid.yaml` (R1.7).

## 1. Stopping rule

- **DGJ.** A session has converged when the strategies of all speculators stay unchanged for
  1,000,000 consecutive periods (Section 4.2). Their sessions take between 20 million and
  50 billion periods across all their experiments (Dou et al., 2025, p. 24). No cap is stated.
- **Here.** The same rule with a cap on the length of training. Sessions that reach the cap
  are evaluated and reported with the flag `converged = False`; a second flag, `settled`,
  marks sessions whose greedy policy changed in at most 10% of the visited states in the
  last window.
- **Keys.** `simulation.convergence_periods`, `simulation.max_periods`; diagnostics under
  `simulation.diagnostics`.
- **Runs and effect.** R1.1 (window 100,000, cap 10 million) against R1.2 (window 1 million,
  cap 40 million), both on the earlier specification. In R1.1, 15 of 30 sessions reach the
  cap and Δ^C over all sessions is 0.07 [−0.04, 0.16]; in R1.2 all 30 converge (median 13.8
  million periods) and Δ^C is 0.19 [0.16, 0.23]. A session that ends at the cap says
  something about the cap, not about collusion. Over the 15 converged sessions of R1.1
  alone, Δ^C is 0.26 (`results/r1_1_early_stop/report.txt`), which is why all sessions are
  reported first.
- **Later runs.** Cap 100 million (R1.6: 150 million, R1.7: 400 million). In the baseline, 29
  of 30 sessions converge, after 36 to 79 million periods.
- **Code.** `simulation/runner.py`: `_check_convergence_step`, `_policy_snapshot`,
  `_stability_stats`, `_resolve_max_periods`.

## 2. Action grid

- **DGJ.** 15 grid points on the interval [x^M − ι(x^N − x^M), x^N + ι(x^N − x^M)] (Section
  4.2). x^N and x^M depend on the value v; the text does not say outright whether the grid
  is built once or for each value.
- **Here.** `action_grid: per_v` (baseline specification): 15 levels of trading intensity χ
  from 120.83 to 170.83, the order at value v_k being χ·(v_k − v̄). Level 1 is closest to the
  cartel intensity (125.00), level 13 closest to the Nash intensity (166.67). This is a
  reading of DGJ's text. `action_grid: global` (earlier specification): one grid of order
  sizes for all values.
- **Why it matters.** On the global grid the one-period game has its Nash equilibrium at
  Δ^C −0.277, and Nash and cartel orders are at most two grid points apart. On the per-value
  grid the one-period game has exactly one equilibrium in pure strategies at every v ≠ v̄
  (both at level 13) with Δ^C −0.029, so the grid offers no supra-competitive equilibrium.
- **Runs and effect.** R1.2 (global) against R1 (per value): 0.19 against 0.34. The two runs
  also differ in the exploration counter, the Q-initialisation and β (Sections 3 and 4); no
  run varies these one at a time. Measured from each grid's own Nash value, R1.2 lies 0.47
  above it and R1 0.37.
- **Code.** `env/discretization.py`: `build_action_grid`, `build_chi_grid`,
  `build_action_table`; `MarketEnv.action_table`; `simulation/benchmarks.py`:
  `grid_stage_game`.

## 3. Exploration counter and decay β

- **DGJ.** The exploration probability decays with the number of visits of the current
  value, ε = exp(−β·N_v) (Equation 4.3), with β = 5e-7 at n_v = 10.
- **Here.** `exploration_counter: v` implements that rule (baseline specification);
  `exploration_counter: state` counts visits of the full state (earlier specification, with
  β = 5e-4). The baseline specification sets β = 1e-6: with n_v = 5 this gives the same 430
  random draws per state–action cell as DGJ's β at n_v = 10.
- **Runs and effect.** R1.6 uses DGJ's β = 5e-7 at n_v = 5 (twice the draws per cell, cap 150
  million): Δ^C 0.35 [0.33, 0.38] against 0.34. A slower decay of exploration does not bring
  the Q-learners closer to the Nash intensity (χ̂ 157.3 against 157.7).
- **Code.** `agents/q_learning.py`: `QLearningAgent.act`.

## 4. Q-initialisation

- **DGJ.** The initial Q-value of an action is the discounted profit against a rival who
  draws uniformly from the grid (Section 4.2).
- **Here.** `q_init: paper` implements that formula (baseline specification; the value is a
  literal identifier in the code). `q_init: optimistic` starts every cell at the discounted
  cartel profit (earlier specification).
- **Runs and effect.** Varied only together with the action grid (R1.2 against R1).
- **Code.** `agents/q_learning.py`: `QLearningAgent._initialise_q_table`.

## 5. Value grid n_v

- **DGJ.** n_v = 10.
- **Here.** n_v = 5, for computing time. The benchmarks use the variance of the grid values,
  so Δ^C is measured against what the grid can reach.
- **Runs and effect.** R1.7 has n_v = 10 and β = 5e-7 (8 sessions, cap 400 million, median
  273 million periods): Δ^C 0.31 [0.29, 0.32], +3.8%, a single deviation pays in 8 of 8.
- **Code.** `env/discretization.py`: `build_v_grid`; `simulation/benchmarks.py`:
  `_profit_variance`.

## 6. Noise level σ_u

- **DGJ.** They state a Δ^C of approximately 0.75 for σ_u = 0.1 (Dou et al., 2025, p. 30).
  For σ_u = 1 the text gives no number; Panel A of their Figure 2 (p. 28) reads roughly 0.26
  there.
- **Here.** σ_u = 1 in all runs but one. Noise moves the price by 0.002 and a price bin is
  0.0687 wide, so noise does not blur the price bin at either level.
- **Runs and effect.** R1.8 is the baseline with σ_u = 0.1 and nothing else: Δ^C 0.34 [0.31,
  0.36], +4.2%; difference to R1 0.00 [−0.03, 0.04]. The gap to DGJ therefore exists at their
  noise level (0.34 against 0.75). What does not replicate is the rise of Δ^C as the noise
  falls. Under the baseline specification, the noise level does not account for the gap.
  The high-noise regime was not run.
- **Keys.** `sweep.sigma_u`, `overrides.sigma_u`.

## 7. Price used in the reward

- **DGJ.** The profit is paid on the price that the market maker sets (Dou et al., 2025,
  p. 21).
- **Here.** `reward_price: raw` does the same. `reward_price: snapped` pays the profit on
  the price rounded to its bin, which is the grid of the lagged-price state (`snapped` is a
  literal identifier in the code).
- **Runs and effect.** R1.3: Δ^C 0.91 [0.88, 0.93], +11.3%, with a trading intensity almost
  that of the baseline (χ̂ 154.9 against 157.7). The gain comes from the rounding of the
  price in the reward. The tested deviations do not pay (0 of 30 sessions; total −2.72), and
  they lose in the deviation period itself (−1.71): one more unit of order flow tips the
  price into the next bin. This is a static payoff cliff, not a punishment. Because DGJ pay
  the profit on the market maker's price, the rounded price is no account of their result.
- **Code.** `env/market.py`: `MarketEnv.step`.

## 8. Lagged price in the state

- **DGJ.** The state holds the lagged price, the lagged value and the current value. As a
  check of the price-trigger mechanism they remove the lagged price, and their Δ^C drops to
  zero (Dou et al., 2025, pp. 32–33).
- **Here.** `state_price: lagged` (default) or `none`. With `none` the price component of
  every state is 0; prices, rewards and all random draws are unchanged.
- **Runs and effect.** R1.4: Δ^C 0.22 [0.20, 0.24], +2.8%. R1.5 adds β = 3e-6, so that
  exploration fades three times as fast and the draws per reachable cell are those of the
  baseline: 0.24 [0.21, 0.27], +3.0%. Δ^C is positive in all 30 sessions of both runs.
  About one third of the baseline's Δ^C depends on the price in the state; about two thirds
  remain where no agent can react to a deviation. The runs do not show whether the
  price-dependent part is a price trigger: the continuation term of the deviation test is
  zero by construction. Fewer sessions settle (11 and 5 of 30).
- **Code.** `env/market.py`: `MarketEnv.reset`, `MarketEnv.step`.

## 9. Layout of the price grid

- **DGJ.** 31 grid points for the price. They state that their choice of n_p ≈ 2 n_x ensures
  that a change of one grid point in one speculator's order moves the price over the grid
  (Dou et al., 2025, p. 23, footnote 25).
- **Here.** One grid for all values (range ±1.03, spacing 0.0687), used only to index the
  lagged price in the state. One action level moves the price by at most 0.13 of a bin, so a
  speculator sees a small deviation of their rival only when the price crosses the edge of
  a bin. A price grid built for each value would have the property that DGJ describe; this
  is our reading of their text (p. 22), not knowledge of their code.
- **Runs and effect.** Not varied in the 12 runs. An exploratory run outside them
  (`experiments/03_exploratory_price_grid/`: a patch for a switch `price_grid: per_v`, its
  tests, configurations and reports) repeats R1.8 on such a grid. None of its 30 sessions
  converged within 100 million periods, so they give no level. One session that ran on
  converged after 581 million periods at a Δ^C of 0.39, and a single deviation still paid
  there. One session settles nothing: for the gap to DGJ, the layout of the price grid stays
  a candidate that we could not test at this scale.
- **Code.** `env/discretization.py`: `build_price_grid`.

## 10. Benchmark of Δ^C

- **DGJ.** Δ^C compares the realised profit with Nash and cartel profits on the same path of
  values and noise (DGJ's Online Appendix, eq. IA.4.1).
- **Here.** The same per-path benchmarks, with the continuous Nash and cartel intensities
  (166.67 and 125.00) and the values of the grid. Two reference values go with it: the Nash
  equilibrium of the action grid (−0.029 or −0.277; printed in every `report.txt`) and
  uniform random play on the grid of the baseline specification (+0.68;
  `outputs/tables/numbers.tex`). Uniform random play alone thus gives a positive Δ^C, and a
  level is read against these values.
- **Code.** `simulation/benchmarks.py`: `realised_benchmark_profit`, `grid_stage_game`;
  `simulation/metrics.py`: `collusion_capacity`; `scripts/make_thesis_outputs.py`:
  `random_play_delta_c`.

## 11. Learner family and the AC learner's settings

- **DGJ.** Q-learners only. DGJ's Online Appendix (Section 4.12, p. 72) varies a parameter
  between two Q-learners.
- **Here.** A tabular softmax actor-critic learner (AC learner) as a stand-in for the
  policy-gradient family; it is not PPO and not deep. Settings: actor step 0.1, critic step
  0.05, entropy bonus 0.1 that decays on the clock of the Q-learner's exploration, reward
  scale equal to the cartel profit (`alpha_actor`, `alpha_critic`, `entropy_coef`,
  `entropy_decay`, `reward_scale`).
- **How the settings were chosen.** Two validation runs put each learner against a rival
  fixed at the Nash intensity (4 sessions, β 5e-6, cap 20 million): the learner's own Δ^C is
  −0.06 (Q-learner, χ̂ 161.5; none of its 4 sessions converged within the cap) and −0.04 (AC
  learner, χ̂ 162.8), an indication only. A pilot then compared three
  settings of actor step and entropy bonus, (0.1, 0.1), (0.03, 0.1) and (0.1, 0.02), in
  4 sessions each with a cap of 60 million. The rule was fixed in advance: the highest
  number of sessions converged or settled, a tie goes to the first setting, and the policy
  concentration must be at least 0.9. The counts were 4, 3 and 4 of 4; the tie went to
  (0.1, 0.1), whose concentration is 0.997 or higher. Δ^C was no criterion.
- **Runs and effect.** Q–Q 0.34, Q–AC 0.14 [0.12, 0.15], AC–AC 0.09 [0.08, 0.11]. In Q–AC
  the Q-learner ends near the Nash profit (Δ^C 0.03) and the AC learner earns what is left
  (0.24). The mixed pair lies above the twin AC learners (difference 0.04 [0.02, 0.07]), so
  in these populations the learner family counts and the mix does not. Two Q-learners with
  learning rates 0.01 and 0.1 (R2.2) give 0.33 [0.30, 0.35]; the difference to R1 is
  0.01 [−0.02, 0.05]. One learner stands for each family, and all three populations exist at
  σ_u = 1 only.
- **Code.** `agents/actor_critic.py`: `ActorCriticAgent`; `agents/fixed_policy.py`:
  `FixedChiAgent`.

## 12. Further departures from DGJ

| Item | DGJ | Here | Effect |
|:--|:--|:--|:--|
| Sessions per run | 1,000 | 30 (R1.7: 8) | Wider confidence intervals; not varied |
| Market maker | Rolling regressions over T_m periods | One rolling regression with known ξ and v̄ (`MarketEnv._estimate_lambda`) | The estimated price impact equals λ^N = 0.002 to seven digits in every run; this holds by construction in this regime |
| Shock in the impulse-response test | Sized to a price move of 1.2% (DGJ's Online Appendix, Section 4.5) | Sized in price bins (`irf.noise.magnitudes`), 0.125, 0.25 and 0.5 of a bin, so that some shocks move the lagged price into another bin | A response is possible only in those trials; statements on the mechanism rest on the lines "Visited post-shock states only" of `report.txt` |
| Deviation test and staleness check | – | Added: a single grid best response under frozen policies over 60 periods, and the stored Q-value of the deviation against its one-step target (`scripts/deviation_test.py`) | In the baseline a single deviation pays in 22 of 30 sessions (total 0.36 [0.22, 0.50]) |

## What the map does not show

- No run varies the action grid, the exploration counter and the Q-initialisation one at a
  time.
- The gap to the level that DGJ report at σ_u = 0.1 is not explained by β, n_v or σ_u. The
  layout of the price grid and the length of learning remain open.
- The mechanism behind the baseline's Δ^C is diagnosed, not proven.
- Nothing here is a statement about DGJ's code, which we have not seen.
