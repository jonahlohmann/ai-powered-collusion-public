# Exploratory run: a price grid built for each value

This folder documents one exploratory run. It is **outside the 12 runs of the thesis**: it
is not in the table of runs, its numbers are not in `outputs/tables/numbers.tex`, and no
figure or table of the thesis uses it. The thesis mentions it in the first limit of
Section 7.2. Everything needed to repeat it is here. The code under `src/` and the 245
tests under `tests/` do not contain the change; the run needs the patch of this folder.
The raw results of the run (session files and saved Q-tables) are not part of this
repository; the files under `reports/` are its record.

## Question

Dou et al. (2025, p. 23, footnote 25) state that their choice of n_p ≈ 2 n_x "ensures that,
all else equal, a one-grid point change in one informed AI speculator's order flow will
result in a change in price p_t over the grid". In the thesis the price grid is one grid
for all values, and one action level moves the price by at most 0.13 of a bin. Does the
result change when the price grid has the property that DGJ describe?

## The one change

A switch `price_grid: "global" | "per_v"` (default `global`, the behaviour of the thesis).
With `per_v` the lagged price is sorted into a grid built for the lagged value v_k, with
DGJ's formula (p. 22) read per value, as the baseline specification already reads the
action grid:

    p_H(k) = v̄ + λ^N (I · max{χ^M, χ^N} · (v_k − v̄) + 1.96 σ_u)     (v_k > v̄; mirrored below)
    p_L(k) = v̄ + λ^N (I · min{χ^M, χ^N} · (v_k − v̄) − 1.96 σ_u)
    grid_k = linspace(p_L − ι (p_H − p_L), p_H + ι (p_H − p_L), n_p)

One action level then moves the price by about one grid step (0.98 to 1.07). At the middle
value every action trades zero, so the bin is fixed at the centre. This is a reading of
DGJ's text, not knowledge of their code.

## Files

| File | Content |
|:--|:--|
| `price_grid_per_v.patch` | The change to `src/ai_trading/env/discretization.py` and `src/ai_trading/env/market.py` |
| `test_price_grid_per_v.py` | 10 tests of the switch (they need the patch) |
| `config_pg_per_v.yaml` | The 30 sessions: the configuration of R1.8 (baseline specification, σ_u = 0.1, seeds 42 + session id, cap 100 million periods) with `price_grid: per_v` and nothing else |
| `config_pg_per_v_long.yaml` | One session (session 1, seed 43) with a cap of 1 billion periods |
| `reports/pg_per_v_report.txt`, `reports/pg_per_v_deviation_test.txt` | Report and deviation test of the 30 sessions |
| `reports/pg_per_v_long_report.txt`, `reports/pg_per_v_long_deviation_test.txt` | The same for the long session |
| `reports/sessions.csv` | One row per session of both runs |
| `reports/pg_per_v_long_trajectory.csv` | The long session per million periods of training |

The reports are the output of `scripts/baseline_report.py` and `scripts/deviation_test.py`.
The path of the working folder was removed from their last two lines; nothing else was
edited. They name the configurations under `experiments/01_replicate_baseline/`, where the
two files lay when the run was made.

## How to repeat it

From the root of the repository:

```bash
git apply experiments/03_exploratory_price_grid/price_grid_per_v.patch
cp experiments/03_exploratory_price_grid/test_price_grid_per_v.py tests/
uv run pytest tests/ -q                  # 255 tests: the 245 of the repository and 10 new

# The 30 sessions, one session per call (session id 0 to 29)
uv run python -m ai_trading.simulation.runner \
    --config experiments/03_exploratory_price_grid/config_pg_per_v.yaml \
    --session-id-start 0 --n-sim 1 --output-dir results/pg_per_v/s0

# The long session
uv run python -m ai_trading.simulation.runner \
    --config experiments/03_exploratory_price_grid/config_pg_per_v_long.yaml \
    --session-id-start 1 --n-sim 1 --output-dir results/pg_per_v_long/s1

# Reports
uv run python scripts/baseline_report.py --results-dir results/pg_per_v \
    --config experiments/03_exploratory_price_grid/config_pg_per_v.yaml --out results/pg_per_v/report.txt
uv run python scripts/deviation_test.py --results-dir results/pg_per_v \
    --config experiments/03_exploratory_price_grid/config_pg_per_v.yaml --out results/pg_per_v/deviation_test.txt

# Back to the code of the thesis
git apply -R experiments/03_exploratory_price_grid/price_grid_per_v.patch
rm tests/test_price_grid_per_v.py
```

Computing time on the laptop of the thesis: 29 hours for the 30 sessions (four at a time)
and 6 hours for the long session.

## What came out

| | Price grid per value | R1.8 of the thesis |
|:--|:--|:--|
| Sessions converged within 100 million periods | 0 of 30 (1 settled) | 30 of 30 (18 settled) |
| Δ^C | 0.44 [0.41, 0.47] at the cap | 0.34 [0.31, 0.36] |
| One-period best response, total gain | 0.83 [0.60, 1.06] | 0.40 [0.31, 0.50] |
| It pays in | 20 of 30 sessions (loses in none) | 23 of 30 sessions |

The long session converged by DGJ's rule after 581 million periods at Δ^C 0.39 (R1.8 with
the same seed: 0.48 after 36 million periods). Its path fell to 0.41 by 125 million
periods, held a plateau at 0.51 from 126 to 282 million periods, fell again and stayed at
0.37 to 0.40. At the end the one-period best response still pays (total gain 0.71,
standard error 0.23), and the stored values of deviating actions lie 14 to 19 units below
their one-step target.

## How to read it, and how not to

- **The 30 sessions give no level.** None converged, and the policies were still changing
  in about 42% of the visited states when the sessions stopped. A Δ^C read at the cap says
  nothing for or against DGJ's 0.75. The same holds for the deviation test and the impulse
  responses of these sessions. The cap of 100 million periods was what the computing time
  allowed.
- **The long session is one seed.** It shows that a session on this grid can need several
  hundred million periods, and that this one ended near the level of the baseline with a
  deviation that pays. It is not a result on 30 sessions.
- **Impulse responses.** The noise shocks are still sized in bins of the global grid
  (0.125, 0.25 and 0.5 of 0.0684); on the grid per value these are about 1, 2 and 4 bins at
  the outer values.
- Whether such a price grid brings back DGJ's level or their mechanism remains open.

## Provenance

The run was made with the simulation code of this repository and the patch of this folder.
The switch, the tests and the configurations were written by Claude Code (Claude Fable 5.2)
to the author's specifications, as was all simulation code of the thesis (Section 4.8 of
the thesis).
