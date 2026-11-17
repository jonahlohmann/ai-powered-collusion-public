# Parameters and technical definitions

**DGJ** in this file means Dou, Goldstein and Ji (2025), "AI-Powered Trading, Algorithmic
Collusion, and Price Efficiency", the replicated study. Section, equation and figure numbers
without another source refer to DGJ; "OA" is DGJ's Online Appendix.

Three identifiers in code, configurations and saved outputs keep older words:
`q_init: paper` ("paper" is DGJ), `reward_price: snapped` (the price rounded to its bin),
and the line "pre-Phase-0 definition" in `report.txt` (Δ^C with expected-value benchmarks,
column `delta_c_expected`).

What each choice does to the results: `docs/implementation-choices.md`.

## Current specification at a glance

All results use the **baseline specification** in one regime (ξ = 500, σ_u = 1.0). The
**earlier specification** survives as one robustness row (run R1.2). The sections below define
every element; where a table has a "DGJ" column, it is the reference, not what was run.

| Parameter | DGJ | **Baseline specification** | Earlier specification (robustness) |
|:--|:--|:--|:--|
| `n_agents`, `xi`, `rho`, `theta`, `T_m` | 2, {5, 500}, 0.95, 0.1, 10 000 | 2, 500, 0.95, 0.1, 10 000 | as the baseline specification |
| `sigma_u` | {0.1, 100} | 1.0 | 1.0 |
| `n_v`, `n_x`, `n_p` | 10, 15, 31 | 5, 15, 31 | as the baseline specification |
| `alpha` | 0.01 | 0.01 | 0.01 |
| `beta` | 5e-7 | **1e-6** (430 random draws per state–action cell, as in DGJ) | 5e-4 |
| `action_grid` | not stated | `per_v`: 15 aggressiveness levels, cartel = 1, Nash = 13 | `global` |
| `exploration_counter` | per v (eq. 4.3) | `v` | `state` |
| `q_init` | §4.2 formula | `paper` | `optimistic` |
| `reward_price` | raw | `raw` (run R1.3: `snapped`) | `raw` |
| `state_price` | lagged (§5.3 test: none) | `lagged` (runs R1.4 and R1.5: `none`) | `lagged` |
| Convergence window, cap | 1 000 000, none | 1 000 000, 100M | 1 000 000, 40M |
| Sessions, seed base | 1 000 | 30, 42 | 30, 42 |
| Files | – | `shared/params_baseline.yaml`, `01_replicate_baseline/config_r1_baseline.yaml` | `shared/params_earlier_spec.yaml`, `01_replicate_baseline/config_r1_2_earlier_spec.yaml` |

Reference values at the baseline specification: χ^N 166.67, χ^M 125.00, λ^N 0.002, π^N 42.61,
π^M 47.93; grid Nash Δ^C −0.029; uniform random play Δ^C +0.68. Parameters of the AC learner:
"Heterogeneity extension" below. Runs and their results: `docs/implementation-choices.md`,
`results/<run>/report.txt` and `outputs/tables/`.

---

## Model parameters (DGJ)

| Symbol | Code name  | DGJ's value | Description                         |
|:-------|:-----------|:---------------|:------------------------------------|
| I      | `n_agents` | 2              | Number of strategic agents          |
| ξ      | `xi`       | {5, 500}       | Demand elasticity of information-insensitive investors          |
| σ_u    | `sigma_u`  | {1e-1, 1e2}    | Standard deviation of noise trader order flow                   |
| ρ      | `rho`      | 0.95           | Subjective discount factor of informed speculators (Bellman equation) |
| θ      | `theta`    | 0.1            | Market-maker inventory penalty      |

## Discretisation parameters (DGJ)

| Symbol | Code name | DGJ's value | Description                        |
|:-------|:----------|:---------------|:-----------------------------------|
| n_v    | `n_v`     | 10             | Grid points for fundamental value  |
| n_x    | `n_x`     | 15             | Grid points for the action grid X (order quantity per period) |
| n_p    | `n_p`     | 31             | Grid points for price              |
| ι      | `iota`    | 0.1            | Action-grid padding factor on (χ^N − χ^M); see "Action grid construction" below |

## Training parameters (DGJ)

| Symbol | Code name | DGJ's value | Description                        |
|:-------|:----------|:---------------|:-----------------------------------|
| T_m    | `T_m`     | 10 000         | Market-maker rolling-regression window length (§4.1) |
| α      | `alpha`   | 0.01           | Q-learning rate                    |
| β      | `beta`    | 5e-7           | Exploration decay parameter        |

## Notes

- DGJ sweep ξ ∈ {5, 500} and σ_u ∈ {1e-1, 1e2} to create four regimes (low/high noise ×
  low/high volatility). These are DGJ's reference values; the scope here is narrower (see
  "Parameter sets" below).
- The discount factor γ = ρ = 0.95.

## Action grid construction (§4.2)

### Global grid (`action_grid: global`, the earlier specification)

The action grid spans from the most-negative Nash action (with `iota` padding beyond) to the
most-positive Nash action (with `iota` padding beyond):

    dv_max  = v_grid[-1] − v̄
    dv_min  = v_grid[0]  − v̄
    pad     = ι · (χ^N − χ^M)
    x_max   = χ^N · dv_max + pad · dv_max
    x_min   = χ^N · dv_min − pad · |dv_min|
    grid    = np.linspace(x_min, x_max, n_x)

With n_v = 5, ι = 0.1, σ_u = 1.0 the grid spans ≈ [−218.9, +218.9] in 15 steps of 31.3. Nash
and cartel bins are separated by 2 positions at extreme v and by 1 at moderate v.
Implemented in `env/discretization.build_action_grid`.

**Limit of this grid.** The one-shot game on this global grid has its Nash equilibrium at
bins 0, 4, 10 and 14 for the four non-zero v-states, which gives Δ^C = −0.277 instead of 0.
Its best symmetric outcome (bins 2, 5, 9, 12) gives Δ^C = 0.993. Only three actions at
extreme v and two at moderate v are relevant. This is the reason for the per-v grid of the
baseline specification.

### Per-v action grid (`action_grid: per_v`, the baseline specification)

DGJ's interval `[x^M − ι(x^N − x^M), x^N + ι(x^N − x^M)]` is built from
x^N = χ^N·(v − v̄) and x^M = χ^M·(v − v̄), which depend on v. Read per v, the grid is a set
of n_x aggressiveness levels:

    chi_grid = np.linspace(χ^M − ι(χ^N − χ^M), χ^N + ι(χ^N − χ^M), n_x)
    action_table[k, a] = chi_grid[a] · (v_grid[k] − v̄)        # shape (n_v, n_x)

With n_x = 15, ι = 0.1: χ from **120.8333 to 170.8333 in steps of 3.5714**. The row for
v = +1.2816 runs from 154.854 to 218.932, the row for v = −1.2816 is its negative, the row
for v = v̄ is all zeros. Action index a is aggressiveness level a at every v: for v < v̄ the
quantities are negative and index 14 is still the most aggressive trade. The cartel level
is index 1, the Nash level index 13. The largest quantity (218.93) is the same as on the
global grid, so the price grid is unchanged.

The one-shot game on this table has one equilibrium at every v ≠ v̄, bin (13, 13), with
**Δ^C = −0.029**; the best symmetric outcome is bin 1 with Δ^C = 1.000. At v = v̄ all actions
trade zero ("degenerate" state, see "Switches" below).

Implemented in `env/discretization.build_chi_grid` and `build_action_table` (mode
`"global"` returns the tiled 1-D grid above). `MarketEnv.action_table` holds the table in
both modes and is what `step()` trades. In mode `per_v`, `MarketEnv.action_grid` holds the
χ grid, not quantities. DGJ do not state the implementation outright; the per-v grid is an
interpretation, and both variants are run (R1 and R1.2).

*Source of the tables above: Dou, Goldstein & Ji (2025), Table I and §3.*

---

## Parameter sets

Named parameter sets under `experiments/shared/`: `params_debug.yaml` (smoke tests),
`params_earlier_spec.yaml` (**the earlier specification**), `params_baseline.yaml` (**the
baseline specification**; table at the top of this file) and `params_ref_grid.yaml` (run
R1.7: `n_v: 10`, `beta: 5e-7`, cap 400M). Every run references one of these sets and states
its changes under `overrides:`.

### Parameter comparison table

| Parameter | DGJ | Debug | Earlier specification (`params_earlier_spec.yaml`) | Rationale |
|:----------|:------|:------|:-------|:----------|
| `n_agents` | 2 | 2 | 2 | Number of strategic informed speculators |
| `xi` | {5, 500} | 500 | 500 | Demand elasticity of information-insensitive investors |
| `sigma_u` | {0.1, 100} | [0.1] | 1.0 | Noise trader order flow std; single regime (the run configurations override the file's default 0.1) |
| `rho` | 0.95 | 0.95 | 0.95 | Subjective discount factor (Bellman equation) |
| `theta` | 0.1 | 0.1 | 0.1 | Market-maker inventory penalty |
| `n_v` | 10 | 3 | 5 | Grid points for fundamental value; reduced for computing time |
| `n_x` | 15 | 5 | 15 | Grid points for the action grid X; matches DGJ |
| `n_p` | 31 | 11 | 31 | Grid points for price; matches DGJ |
| `T_m` | 10 000 | 10 000 | 10 000 | Market-maker rolling-regression window length (§4.1) |
| `alpha` | 0.01 | 0.1 | 0.01 | Q-learning rate; higher = faster but noisier updates |
| `beta` | 5e-7 | 1e-4 | 5e-4 | Exploration decay under the earlier specification (per-state counter). **The baseline specification uses 1e-6 with the per-v counter** |
| `convergence_periods` | 1 000 000 | 1 000 | 100 000 | Consecutive unchanged-strategy periods before convergence is declared. All runs except R1.1 use 1 000 000 (set in the run configuration); `params_earlier_spec.yaml` keeps 100 000 so that `config_r1_1_early_stop.yaml` stays reproducible |
| `max_periods` | – | 100 000 | 10 000 000 | Hard cap on training. R1.2 uses 40M and the baseline specification 100M; all sessions are reported, with converged and settled flags. With the 10M cap, 15 of the 30 sessions of R1.1 ended at the cap |
| `n_sim` | 1 000 | 2 | 30 | Simulation sessions per parameter point; trades fidelity for runtime |
| `eval_periods` | 100 000 | 10 000 | 100 000 | Measurement window after training |
| `save_eval_periods` | – | (all) | 10 000 | Eval rows written to `periods.parquet` (metrics use all eval periods) |
| `eval_greedy` | (pure exploitation) | true | true | Agents frozen (greedy, no learning) during eval |
| `reward_price` | raw | raw | raw | Price used for profits; `snapped` = the price rounded to its bin (run R1.3 only) |
| `market_maker` | rolling OLS | rolling_ols | rolling_ols | `fixed_lambda` uses λ^N throughout |

`params_baseline.yaml` is the earlier specification with `action_grid: per_v`,
`exploration_counter: v`, `q_init: paper`, `beta: 1e-6`, `convergence_periods: 1 000 000` and
`max_periods: 100 000 000`. The three switch keys are absent from `params_debug.yaml` and
`params_earlier_spec.yaml`, where they take their defaults (`global`, `state`, `optimistic`).

**Reading rule for Δ^C under the baseline specification.** On the per-v grid uniformly random
play gives Δ^C = +0.68, and symmetric play spans −0.21 (level 14) to +1.00 (level 1); grid
Nash is −0.029 (level 13). A positive Δ^C is therefore expected without learning. Evidence
is read in three steps: (i) supra-competitive profits: Δ^C over all sessions with a
bootstrap CI, relative to the grid Nash line; compare the earlier and the baseline
specification by the distance from each grid's own Nash value (−0.277 and −0.029), not by
raw Δ^C; (ii) equilibrium: the deviation test (does a single grid best response pay over 60
periods?); (iii) mechanism, only if (ii) says that the deviation does not pay: the
visited-states lines of the impulse responses and the Q-table trigger test. "Price trigger"
may only be written with (iii).

### Notes

- **Regime:** ξ = 500, σ_u = 1.0, which lies in the region where DGJ report collusion
  through price-trigger strategies (Figure 1, top-left quadrant). Run R1.8 repeats the
  baseline at DGJ's σ_u = 0.1. Noise moves the price by λσ_u = 0.002, far below the width of
  a price bin (0.0687), and informed flow is about 333 times the noise: noise does not blur
  the price bin. The high-noise regime (σ_u = 100) was not run.
- **Grid-corrected benchmarks:** when `v_grid` is provided, `benchmarks.py` uses
  `Var(v_grid)` instead of σ_v² to compute π^N and π^M. Δ^C then measures performance
  against what is achievable on the discrete grid, not against the continuous optimum.
- **Q-table initialisation (the earlier specification, `q_init: optimistic`, the default):**
  optimistic at V^M = π^M/(1−ρ) (Sutton & Barto, 2018, §2.6). This departs from DGJ's §4.2
  formula, which is the perpetuity of the one-period payoff against a uniformly random
  opponent and is implemented as `q_init: paper` (the baseline specification; "Switches"
  below).
- **Exploration rate (the earlier specification, `exploration_counter: state`, the
  default):** ε_t(s) = exp(−β·N[s]) where `N[s]` is the visit count of the full state triple
  `s = (p_{t-1}_bin, v_{t-1}_bin, v_t_bin)`, an `(n_p, n_v, n_v)` `int64` array on
  `QLearningAgent`. This departs from DGJ's eq. 4.3, which counts visits of v only and is
  implemented as `exploration_counter: v` (the baseline specification).
- **RNG streams:** each session's environment and agents get independent streams from
  `np.random.SeedSequence(seed).spawn(1 + n_agents)` (env first), with
  `seed = seed_base + session_id`.

### Standards behind the parameter choices (parameter tables)

`scripts/make_thesis_outputs.py` writes the parameter table of the thesis appendix in two
halves (`make_t2_split`; rows in `Q_PARAMETER_ROWS` and `ac_parameter_rows`):
`outputs/tables/tab_parameters_baseline.tex` (the baseline with two Q-learners next to DGJ's
values, with short reasons) and `outputs/tables/tab_parameters_ac.tex` (the settings of the
AC learner, no column for DGJ). The standards of the reasons there are "DGJ's value",
"DGJ's rule", "Computing time:" and, for the two grids, the layout. A combined table
(`make_t2`, reasons in `PAPER_VALUES`, `outputs/tables/tab_parameters.tex`) is still
generated, because `tests/test_make_t2.py` pins it; the thesis does not include it. Its
reason column opens every reason with the standard behind the choice:

- **DGJ value:** the value of Dou, Goldstein and Ji (2025), for comparability.
- **DGJ rule:** DGJ's rule as written (for example the exploration counter of eq. 4.3 and
  the Q-value initialisation of §4.2).
- **DGJ rule, rescaled:** DGJ's rule with a constant rescaled to our grid (β: the same
  number of exploratory draws per state–action cell).
- **Interpretation of DGJ:** DGJ leave the point open and it is read in one defined way (the
  per-v action grid of §4.2).
- **Laptop reproducibility:** everything must run on a consumer laptop, so that anyone with
  the repository can reproduce it (n_v, maximum periods, number of sessions).
- **Regime choice:** one cell of DGJ's parameter space (ξ, σ_u).
- **Own design:** a choice without a counterpart in DGJ (the AC learner's block).

The combined table also lists the design rows `exploration_counter`, `q_init`, `action_grid`
and `max_periods` (values read from the baseline's resolved config) and the AC learner's
block (values from `resolved_config.agents[1].config` of the first
`results/r2_q_ac/s*/run_metadata.json`; keys it does not set, or a missing run, fall back to
the `ActorCriticAgent` defaults).

---

## Price grid construction (§4.2)

    p_H = v̄ + λ^N · (I · χ^N · (v_max − v̄) + 1.96 σ_u)
    p_L = v̄ + λ^N · (I · χ^N · (v_min − v̄) − 1.96 σ_u)
    grid = linspace(p_L − ι(p_H − p_L), p_H + ι(p_H − p_L), n_p)

With n_v = 5, σ_u = 1.0, ι = 0.1, n_p = 31: range **±1.029945**, spacing **0.068663**.
Implemented in `env/discretization.build_price_grid`. The grid is used **only to index the
lagged-price state**. It is one grid for all values; a variant built for each value is the
subject of `experiments/03_exploratory_price_grid/`.

## Rewards and market maker

- **Rewards use the raw clearing price** p_t = v̄ + λ̂_t·y_t (§4.1, protocol steps 3–5):
  π_i = (v_t − p_t)·x_i. `reward_price: snapped` pays profits on the price rounded to its
  bin instead. At the largest order (219), rounding by up to half a bin (0.034) changes the
  profit of a period by up to 7.5, more than the mean gap between cartel and Nash profit
  per period (5.3). Run R1.3 measures the effect.
- **Market maker:** only the (v, y) regression is estimated (known ξ, v̄), over the last
  T_m = 10 000 periods, in an O(1) ring buffer with running sums (exactly refreshed once per
  window). λ̂ = (θγ̂ + ξ)/(θ + ξ²). The weight of the learned γ̂ is θ/(θ + ξ²) ≈ 4·10⁻⁷ at
  ξ = 500, θ = 0.1, so λ̂ ≈ 1/ξ = λ^N to seven digits: "λ̂ = λ^N" is not evidence of
  market-maker learning in this regime. `market_maker: fixed_lambda` uses λ^N throughout.

## Collusion capacity Δ^C

- **`delta_c`** (primary): DGJ, OA eq. IA.4.1: Δ^C = (1/I) Σ_i (π̄_i − π̄^N)/(π̄^M − π̄^N),
  where π̄^N and π̄^M are **per-path realised benchmarks** (OA IA.4.2–IA.4.3): the profit
  each speculator would earn if all played x = χ^N(v_t − v̄) (resp. χ^M) against
  p = v̄ + λ^N(I·x + u_t) (resp. λ^M) on the **same realised (v_t, u_t)** as the agents. This
  removes the sampling noise in E[v²] over the eval window.
- **`delta_c_expected`:** the same ratio with the analytical benchmarks
  (1 − Iλχ)χ·Var(v_grid). Reports label it "pre-Phase-0 definition".
- Both use the grid v and the continuous χ^N/λ^N, χ^M/λ^M.
- **`profit_gain`** = Σ_i π̄_i / (I · π̄^N_path) (OA §4.1, "profit gain relative to
  noncollusion").
- Per-agent `delta_c_agent_i` and `chi_hat_agent_i` are also stored.

## IRF protocol

Config keys under `irf:` (values of `config_r1_1_early_stop.yaml`; the values of the baseline
specification follow in "IRF settings for the baseline specification"):

| Key | R1.1 | Meaning |
|:--|:--|:--|
| `enabled` | true | Run IRFs after eval |
| `pre_periods` | 3 | Unrecorded periods before each shock; the shock period is labelled t = 3 (DGJ's labelling) |
| `post_periods` | 6 | Recorded periods t = 3…8 on both paths |
| `n_trials` | 20 | Trials per spec |
| `noise.units` | price_bins | `price_bins`: u_shock = sgn(v_t − v̄)·m·Δp/λ^N; `sigma_u`: sgn·m·σ_u |
| `noise.magnitudes` | [0.5, 1, 2, 4] | Shock sizes (1 price bin = 34.33 noise units at n_v = 5, σ_u = 1) |
| `deviation.enabled` | true | Unilateral-deviation IRF (DGJ, Fig. 5) |
| `deviation.bins` | [1, 2] | Deviator's greedy action moved this many bins in the aggressive direction (clipped to the grid). Global grid: index shift `round(sgn(v_t − v̄)·m)`, one bin = 31.3 units of quantity. **Per-v grid: index shift `+m` at every v** (a higher index is more aggressive for both signs of v_t − v̄), one bin = one aggressiveness level (3.57 in χ); `config_r1_baseline.yaml` uses [4, 8] |
| `deviation.agents` | [0, 1] | Which agent deviates |

**Design:** after the pre-periods (and, if v_t = v̄, extra periods until it is not), env and
agents are deep-copied. The copy receives the shock ("shock" rows), the original continues
unshocked ("base" rows); both are stepped side by side with identical random draws (the env
always draws u, so `u_shock` is additive). The difference shock − base isolates the shock's
effect; the main trajectory continues from the unshocked path. Agents are frozen (eval mode)
throughout. Noise shocks are **adverse** (same sign as v_t − v̄), per §5.3.

**Deliberate departure from DGJ:** DGJ calibrate shocks to a 1.2% price deviation; here
shocks are sized in price-bin units so that they can move the lagged-price state (shocks of
1 to 2 σ_u move the price by 0.002 to 0.004 and never cross a bin of 0.0687).

**Classification** (OA §4.5, at t = pre + 1 = 4, per session, mean over trials, sign-adjusted
order-flow deviation normalised by the eval mean): price trigger if both agents' deviation
> 5·10⁻⁴; no response if both |deviation| < 5·10⁻⁵; otherwise ambiguous.

**Caveats:** (i) a forced deviation is clipped when the deviator already plays the most
aggressive bin. On the global grid Nash *is* the edge bin at the extreme v-states, so such
trials carry no deviation; on the per-v grid the Nash level is bin 13 of 0 to 14, so a
deviation of 4 or 8 bins from Nash is clipped to one bin. `scripts/baseline_report.py`
counts a trial as effective when the deviator's traded quantity differs between the shock
and the base path, and reports and excludes the others (valid for both grids). (ii) If a
shock leads to a never-visited state, the greedy response is the argmax of the initial
Q-values: bin 0 under `q_init: optimistic` (tie-break), the most aggressive bin (n_x − 1)
under `q_init: paper` at v ≠ v̄. `n_visits_agent_i` (visits of the full state, in both
exploration-counter modes) flags this and the report shows the rate. **On the per-v grid
this is not a rare case:** the lagged price is pinned down by v_{t−1} (noise moves the price
by 0.002, a price bin is 0.0687 wide), so each price bin occurs with one v_{t−1} only (bins
2 to 6 for v = −1.28, 10 to 11 for −0.52, 15 for v̄, 19 to 20 for +0.52, 24 to 28 for +1.28).
A shock of one or more price bins often leads to a state that cannot be reached in training
at all; with DGJ's Q-init both agents then play the most aggressive bin, which looks like
joint punishment. `scripts/baseline_report.py` therefore prints, under each noise shock and
each deviation, a second line restricted to trials whose post-shock state had been visited
("Visited post-shock states only", "Visited post-deviation states only"). **Mechanism claims
must rest on those lines, not on the all-trials classification.**

## Output schema

- `sessions.parquet`: `session_id, sigma_u, xi, agent_types, converged,
  periods_to_convergence, delta_c` (per-path), `delta_c_expected, delta_c_agent_i,
  pi_N_path, pi_M_path, profit_gain, chi_hat, chi_hat_agent_i, lambda_hat,
  price_informativeness, market_liquidity, mispricing, seed, avg_profit_agent_i,
  eval_periods, eval_greedy, reward_price, market_maker, wall_clock_seconds,
  periods_per_second, code_version`. Diagnostics columns: `stability_flip_share,
  stability_flip_share_weighted` (float; NaN without diagnostics or without on-path states),
  `stability_n_onpath, stability_window_periods` (int; 0 without diagnostics),
  `stability_n_onpath_degenerate` (int; the number of on-path states with v_t = v̄ on the
  per-v grid, 0 on the global grid or without diagnostics; see "Switches"),
  `stability_flip_share_nd, stability_flip_share_weighted_nd` (float; the same shares over
  the non-degenerate on-path states only; equal to the plain shares on the global grid, NaN
  without diagnostics), `settled` (bool; `stability_flip_share_nd ≤ settled_max_flip_share`;
  False without diagnostics), `diagnostics_enabled` (bool). AC learners add
  `policy_concentration_agent_i`. The results of R1.1 were written without diagnostics and
  lack these columns; `scripts/baseline_report.py` handles both.
- `trajectory.parquet` (only with `simulation.diagnostics.enabled: true`; one row per
  session and completed training block, written at every multiple of `block_periods` unless
  the convergence check stopped training in that period): `session_id, sigma_u, period,
  delta_c` (mean over agents), `delta_c_agent_i, chi_hat_agent_i` (greedy policies on a deep
  copy for `trajectory_eval_periods` periods, per-path benchmarks), `block_flip_share,
  block_flip_share_weighted, block_n_onpath` (stability statistics of that block alone,
  window = `block_periods`), `block_flip_share_nd, block_flip_share_weighted_nd` (the block's
  shares over non-degenerate on-path states; equal to the plain ones on the global grid).
- Diagnostics config keys (`simulation.diagnostics`, defaults): `enabled: false`,
  `block_periods: 1000000`, `onpath_min_share: 0.001`, `settled_max_flip_share: 0.10`,
  `trajectory_eval_periods: 20000`. Definitions under "Training diagnostics" below. Enabling
  them does not change training (same RNG streams, same Q-tables).
- `periods.parquet` (first `save_eval_periods` eval rows + all IRF rows): `session_id,
  sigma_u, period, v_t, u_t` (applied noise), `u_shock, y_t, p_t` (**raw** price), `p_bin,
  lambda_hat, phase` (eval | irf), `irf_type` ("" | noise | deviation), `irf_path` ("" |
  shock | base), `irf_trial, irf_period, shock_magnitude, deviating_agent, x_agent_i,
  profit_agent_i, n_visits_agent_i`. `x_agent_i` is the traded order quantity in both
  action-grid modes (taken from the environment's `info["x"]`).
- `qtables/session_NNN/` (with `save_qtables: true`): `agent_i.npy` (shape
  `(n_p, n_v, n_v, n_x)`), `v_grid.npy`, `price_grid.npy`, `action_grid.npy` (1-D;
  quantities on the global grid, χ levels on the per-v grid) and `action_table.npy` (shape
  `(n_v, n_x)`; the quantity of action a at value index k, for both grids). Scripts read
  quantities from `action_table.npy` and fall back to `action_grid.npy` where it is missing.
- `run_metadata.json`: config path, experiment name, code version, timestamps, CLI overrides
  (`session_id_start, n_sim, output_dir, max_periods`), resolved config, Python/numpy
  versions. `--max-periods N` overrides `simulation.max_periods` and drops
  `simulation.max_periods_by_sigma_u`.

## Departures from DGJ

| Item | DGJ | Earlier specification | Baseline specification |
|:--|:--|:--|:--|
| β | 5e-7 | 5e-4 | 1e-6 (same draws per cell as DGJ) |
| Exploration counter | per v | per full state (p, v', v) | per v |
| Q-init | perpetuity vs uniform-random opponent | constant V^M | DGJ's formula |
| Action grid | most naturally per-v aggressiveness levels | one global quantity grid (Nash–cartel 2/1/0 bins apart) | per-v aggressiveness levels |
| n_v | 10 | 5 | 5 |
| n_sim | 1 000 | 30 | 30 |
| Convergence criterion | 1 000 000 unchanged | 100 000 (run R1.1); 1 000 000 (run R1.2) | 1 000 000 |
| Market maker | two rolling OLS | one OLS + known ξ, v̄ (immaterial at θ = 0.1) | the same |
| IRF shock size | 1.2% price deviation | price-bin units (see above) | the same |
| σ_u | 0.1 | 1.0 | 1.0 (run R1.8: 0.1) |

---

## The two specifications

| | Earlier specification (robustness row) | **Baseline specification** |
|:--|:--|:--|
| `action_grid` | `global` | `per_v`: 15 levels χ ∈ [120.833, 170.833], step 3.5714; x = χ·(v − v̄) |
| `exploration_counter` | `state` | `v` (eq. 4.3) |
| `q_init` | `optimistic` | `paper` |
| `beta` | 5e-4 | **1e-6** |
| `convergence_periods` | 1 000 000 | 1 000 000 |
| `max_periods` | 40 000 000 | 100 000 000 |
| Parameter file | `params_earlier_spec.yaml` | `params_baseline.yaml` |
| Config | `config_r1_2_earlier_spec.yaml` | `config_r1_baseline.yaml` |
| Unchanged | n_v 5, n_x 15, n_p 31, α 0.01, ρ 0.95, ξ 500, σ_u 1.0, θ 0.1, T_m 10 000, raw-price rewards, 100k greedy evaluation, n_sim 30, seed_base 42 | the same |

The right-hand column is the specification of every figure and of all runs but R1.1 and
R1.2; R1.3 to R1.8 and R2.2 each change one or two settings of it. The earlier specification
(R1.2) is one row of the sensitivity table.

### Rationale

- **Convergence window 1 000 000.** DGJ's value (§4.2). With the window of 100 000 and the
  cap of 10M (R1.1), 15 of 30 sessions end at the cap.
- **Caps.** 40M for the earlier specification (R1.2: all 30 sessions converge, the last
  after 32.6M periods); 100M for the baseline specification (R1: 29 of 30 converge, after
  36M to 79M periods).
- **Per-v action grid.** DGJ (§4.2) build the interval from x^N(v) and x^M(v). The global
  grid's own Nash equilibrium sits at Δ^C = −0.277; the per-v grid's at −0.029 (bin 13 of 0
  to 14). The per-v grid has 12 to 13 levels between cartel and Nash at every v, and an
  upward deviation is always possible. DGJ do not state the implementation outright, so
  both variants are reported.
- **Per-v exploration counter.** DGJ, eq. 4.3. With the per-state counter a rarely visited
  state explores with probability near one whenever it is reached, even after tens of
  millions of periods, which makes responses in off-path states random.
- **DGJ's Q-init.** Q₀(s, x) = [(v − v̄) − λ^N(x + (I − 1)·x̄(v))]·x / (1 − ρ), with x̄(v) the
  mean quantity on the grid at v. At the baseline specification, v = 1.2816: 1852.2 (bin 0),
  1995.9 (bin 7), 2057.5 (bin 14); the most aggressive action starts highest.
- **β = 1e-6.** Random draws per state–action cell = (1/β)/(n_p·n_v·n_x). DGJ have
  (1/5e-7)/(31·10·15) = 430. With n_v = 5 the same number needs β = 1e-6. ε falls below 0.01
  after 23M periods, below 0.001 after 35M and below 5e-6 after 61M. A per-v counter with
  the earlier β = 5e-4 would leave less than one draw per cell.
- **Reference runs.** DGJ's literal β = 5e-7 at n_v = 5 (twice DGJ's exploration per cell)
  is run R1.6. n_v = 10 with β = 5e-7 is run R1.7, with 8 sessions: exploration alone lasts
  244M periods per session.
- **Computing time.** Every `report.txt` prints the throughput (periods per second) and the
  total wall-clock time of its run.

### Training diagnostics

Config block `simulation.diagnostics` (defaults): `enabled: false`, `block_periods: 1000000`,
`onpath_min_share: 0.001`, `settled_max_flip_share: 0.10`, `trajectory_eval_periods: 20000`.

- **On-path state:** visited in at least 0.1% of the window's periods.
- **Window:** the last complete block plus the current partial block (1M to 2M periods).
- **`stability_flip_share`:** share of on-path states whose greedy action changed at least
  once in the window, averaged over agents. `stability_flip_share_weighted`: weighted by
  visits.
- **`settled`:** `stability_flip_share_nd ≤ 0.10`: the flip share over the non-degenerate
  on-path states. On the global grid (the earlier specification) there are no degenerate
  states, so this is `stability_flip_share ≤ 0.10`. The threshold comes from a diagnostic
  run made before the runs of this repository (not included), in which every block with an
  unstable Δ^C had a share of at least 0.19 and every stable block at most 0.11.
- **Learning curve:** `trajectory.parquet`, one row per session and block: greedy Δ^C on a
  deep copy over 20 000 periods, per-agent Δ^C and χ̂, and the block's flip shares.
- The diagnostic is reported for every session. It never stops a run.

### Switches

Three config keys, read from the merged config (parameter file or `overrides:`). The
defaults are the earlier specification.

| Key | Values | Default | Read by | Effect |
|:--|:--|:--|:--|:--|
| `action_grid` | `global`, `per_v` | `global` | `MarketEnv` | Which action table `step()` trades (see "Per-v action grid" above). Any other value raises `ValueError` |
| `exploration_counter` | `state`, `v` | `state` | `QLearningAgent` | ε = exp(−β·N[s]) (visits of the full state) or ε = exp(−β·N_v[v_t]) (visits of the current value; eq. 4.3). Any other value raises `ValueError` |
| `q_init` | `optimistic`, `paper`, `zero` | `optimistic` | `QLearningAgent` | Initial Q-table: constant V^M, DGJ's §4.2 formula, or 0 |

- **Counters.** `N` (per full state) and `N_v` (per current value) are both incremented in
  every training `act()`, in both modes; the counter is read before it is incremented, so
  the first visit explores with probability 1. `visit_count()` and `n_visits_agent_i` always
  report the full-state count.
- **DGJ's Q-init.** For state (p, v′, v_k) and action a with x = `action_table[k, a]`,
  I = `n_agents` and m_k the mean of row k:
  Q₀ = [(v_k − v̄) − λ^N·(x + (I − 1)·m_k)]·x / (1 − ρ). This equals DGJ's average over a
  uniformly random opponent (tested against the explicit sum). It does not depend on p or
  v′. Values at the baseline specification (ρ = 0.95): v = ±1.2816: 1852.233 (a = 0),
  1995.941 (a = 7), 2053.759 (a = 13), 2057.530 (a = 14); v = ±0.5244: 310.134, 334.196,
  343.877, 344.509; v = v̄: 0. For comparison V^M = 958.68. `q_init: paper` needs the action
  table (the runner always passes it); without it the agent raises `ValueError`.
- **Degenerate states.** A value index k is degenerate if all entries of `action_table[k]`
  are equal: on the per-v grid this is v = v̄ (every action trades zero); on the global grid
  there is none. In a degenerate state a change of the greedy index is not a change of
  strategy. The runner sets the greedy-policy snapshot to 0 in those states before
  comparing snapshots (`runner._policy_snapshot`, one place for the convergence check and
  the stability diagnostic). So such a change neither resets the convergence counter nor
  counts as a flip. The degenerate states stay in the denominator of `stability_flip_share`
  (they are on-path states that cannot flip); `stability_n_onpath_degenerate` gives their
  number, so the share over the other states is
  `stability_flip_share · stability_n_onpath / (stability_n_onpath − stability_n_onpath_degenerate)`.
  The runner computes that share directly as `stability_flip_share_nd` (and the weighted
  and per-block versions), and `settled` uses it (`runner._stability_stats(…, exclude=…)`,
  mask from `_degenerate_state_mask`).
- **IRF deviation direction.** Global grid: index shift `round(sgn(v_t − v̄)·m)`. Per-v grid:
  index shift `+int(m)` at every v, clipped to n_x − 1.
- **Environment.** `MarketEnv.step()` returns the `info` entries `x` (the agents' order
  quantities in agent order) and `v_bin` (index of v_t). The runner records quantities from
  `info["x"]`.
- **Scripts.** `baseline_report.py` computes the grid stage-game line from the action table
  (Δ^C = −0.277 on the global grid, −0.029 on the per-v grid) and adds the
  visited-states-only IRF lines (see "IRF protocol", caveat ii). `deviation_test.py` reads
  quantities from the action table, moves `one_up` towards higher aggressiveness on the
  per-v grid, skips degenerate states for the deviation and does not count a changed greedy
  index at v = v̄ as an opponent reaction.
- **`state_price` (R1.4 and R1.5).** `MarketEnv` key, values `lagged` (default) and `none`;
  any other value raises `ValueError`. With `none`, the first component of every state
  returned by `reset()` and `step()` is 0, so the agents cannot condition on the lagged
  price (DGJ's check of the price-trigger mechanism, §5.3). Nothing else changes: the price
  is computed and binned as before, `info["price_bin"]` and the `p_bin` column report the
  true bin, and rewards, the market maker and every RNG draw are identical. Agents keep
  tables of shape (n_p, n_v, n_v, n_x) and only use price index 0. Configs:
  `config_r1_4_no_price.yaml` (R1.4) and `config_r1_5_no_price_beta3e6.yaml` (R1.5, β 3e-6
  to restore the exploration per reachable cell); both set `irf.enabled: false` (an agent
  without the price in its state cannot respond to a price shock).

### IRF settings for the baseline specification

Noise shocks of **0.125, 0.25 and 0.5 price bins, 100 trials** (`irf.n_trials` is shared, so
the deviation IRF also has 100 trials). Deviations of 4 and 8 aggressiveness bins (one bin
is 2.2% of the order flow), always towards higher aggressiveness, clipped at bin 14. One
price bin equals 34.33 units of order flow, or 26.8 units of total χ at |v| = 1.2816.

Why the shocks are this small: on the per-v grid the lagged price bin is pinned down by
v_{t−1} (reachable bins 2 to 6, 10 to 11, 15, 19 to 20, 24 to 28), so shocks of 1 and 2
bins mostly lead to states that training never visits, where the agents play the argmax of
DGJ's Q-init (level 14) and the report reads "price trigger". DGJ size the shock to a 1.2%
price move (OA §4.5), about 0.11 bin at |v − v̄| = 1.28 and 0.05 bin at 0.52. `report.txt`
prints, for every shock size, the share of trials in which the price bin changes ("bin-flip
rate") and the share of post-shock states never visited.

---

## Heterogeneity extension

The second learner is a tabular softmax actor-critic (the AC learner), in
`agents/actor_critic.py` (registry key `actor_critic`). The rival of the validation runs,
fixed at one aggressiveness level, is in `agents/fixed_policy.py` (key `fixed_chi`, config
`chi: <float> | "nash" | "cartel"`). Configs: `experiments/02_heterogeneity/`
(`config_r2_q_ac.yaml`, `config_r2_1_ac_ac.yaml`, `config_r2_2_two_alphas.yaml`,
`config_pilot_{a,b,c}.yaml`, `config_validation_q.yaml`, `config_validation_ac.yaml`), all on
the baseline specification (`params_baseline.yaml`).

**Populations:** Q–Q (the baseline, R1), Q–AC (R2) and AC–AC (R2.1), all at the baseline
specification, ξ = 500, σ_u = 1.0, 30 sessions each. AC–AC separates "the learners differ"
from "an AC learner is present". R2.2 has two Q-learners with α = 0.01 and 0.1, the kind of
difference that DGJ test (OA §4.12).

### Parameters of the AC learner

| Parameter | Code name | Default | Description |
|:--|:--|:--|:--|
| Actor step size | `alpha_actor` | 0.1 | Step on the action preferences H[s, a] |
| Critic step size | `alpha_critic` | 0.05 | Step on the state values V[s] |
| Entropy coefficient | `entropy_coef` | 0.1 | η₀ in η = η₀·exp(−`entropy_decay`·t(v)) |
| Entropy decay | `entropy_decay` | = `beta` | Same clock as the Q-learner's ε |
| Reward scale | `reward_scale` | = π^M | Divides the TD error in the actor update |

- Preferences start at 0 (uniform policy). The critic starts at the value of uniform random
  play: 977.53, 895.09, 878.51, 895.09, 977.53 for the five v at the baseline specification.
- Update: δ = r + ρV[s′] − V[s]; V[s] += α_critic·δ;
  H[s,b] += α_actor·[(δ/π^M)(1{b=a} − π_b) − η·π_b(ln π_b + ℋ)]; then H[s,·] is shifted to a
  maximum of 0 and floored at −30.
- Evaluation mode: greedy action (first argmax of H), no random draws, no updates.
- Training action: one uniform draw per period; the action is the smallest index whose
  cumulative probability reaches the draw.
- **Policy concentration** (session column `policy_concentration_agent_i`, AC learners
  only, NaN for other agents): the mean of max_a π(a|s) over the non-degenerate on-path
  states of the final stability window (same on-path rule as the stability diagnostic;
  states with v = v̄ excluded as for `settled`).
- Saved tables (`save_qtables: true`): `agent_<i>_prefs.npy` (n_p, n_v, n_v, n_x) and
  `agent_<i>_values.npy` (n_p, n_v, n_v) next to the Q-learner's `agent_<i>.npy`.
  `scripts/deviation_test.py` restores them; its part 3 (value staleness) is NaN for
  AC learners that deviate.

### Tuning protocol

The pick rule was fixed before the pilot; Δ^C is not a criterion.

1. Validation runs of the Q-learner (`config_validation_q.yaml`) and of the AC learner
   (`config_validation_ac.yaml`): each learner alone against a rival fixed at the Nash
   intensity, 4 sessions, β = 5e-6, cap 20M, no IRF. Pass: `delta_c_agent_0 ≥ −0.10` and
   `chi_hat_agent_0 ≥ 152` in at least 3 of 4 sessions.
2. Pilot: `config_pilot_a.yaml` (0.1, 0.1), `_b` (0.03, 0.1), `_c` (0.1, 0.02) for
   (`alpha_actor`, `entropy_coef`); 4 sessions each, cap 60M.
3. Pick: the highest share of converged or settled sessions; ties go to a; the pick must
   have policy concentration ≥ 0.9.

**Pick: setting a** (`alpha_actor` 0.1, `entropy_coef` 0.1). `config_r2_q_ac.yaml` and
`config_r2_1_ac_ac.yaml` carry these values through the agent defaults.

| Setting | (`alpha_actor`, `entropy_coef`) | converged | settled | converged or settled | min / max concentration (AC) | Δ^C mean (agent 0 / agent 1) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| a | (0.1, 0.1) | 4/4 | 2/4 | **4/4** | 0.997 / 0.999 | +0.117 (+0.018 / +0.216) |
| b | (0.03, 0.1) | 2/4 | 3/4 | 3/4 | 0.952 / 0.989 | +0.162 (−0.034 / +0.358) |
| c | (0.1, 0.02) | 3/4 | 3/4 | 4/4 | 0.996 / 1.000 | +0.150 (−0.033 / +0.334) |

a and c tie at 4/4; the rule gives the tie to a, and a's concentration is above 0.9 in every
session. Periods to convergence under a: 37.7M to 52.8M (cap 60M); the Q-learner's Δ^C stays
near zero in three of four sessions while the AC learner earns +0.15 to +0.38.

---

## Robustness and reference runs

Config-only variants of the baseline (`config_r1_baseline.yaml`); each file's header says
what differs.

| Run | Config | Change against the baseline |
|:-:|:--|:--|
| R1.3 | `01_replicate_baseline/config_r1_3_rounded_price.yaml` | `reward_price: snapped` (profits on the price rounded to its bin) |
| R1.4 | `01_replicate_baseline/config_r1_4_no_price.yaml` | `state_price: none`, IRF off |
| R1.5 | `01_replicate_baseline/config_r1_5_no_price_beta3e6.yaml` | `state_price: none`, `beta: 3e-6`, IRF off |
| R1.6 | `01_replicate_baseline/config_r1_6_ref_beta.yaml` | `beta: 5e-7`, cap 150M |
| R1.7 | `01_replicate_baseline/config_r1_7_ref_grid.yaml` with `shared/params_ref_grid.yaml` | `n_v: 10`, `beta: 5e-7`, cap 400M, 8 sessions |
| R1.8 | `01_replicate_baseline/config_r1_8_ref_noise.yaml` | `sigma_u: 0.1` |
| R2.2 | `02_heterogeneity/config_r2_2_two_alphas.yaml` | agent 1 with `alpha: 0.1` (OA §4.12) |

Descriptive scripts: `scripts/qtable_trigger_test.py --results-dir DIR --config CFG [--out FILE]`
compares the greedy quantity in each on-path state with the state one lagged-price bin
further from the centre (punish / same / forgive), visit-weighted, with untouched (never
visited) deviation cells reported separately. Under the reading rule above it is cited only
if the deviation test says that the deviation does not pay.
`scripts/irf_replay.py --results-dir DIR --config CFG --noise … --deviation … --trials N`
replays the IRF protocol on the saved tables with any shock sizes (visited-state proxy:
saved row differs from the initial row); on the baseline specification it reproduces the
runner's IRF picture.
