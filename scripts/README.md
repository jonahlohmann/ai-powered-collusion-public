# Scripts

Analysis scripts. None of them runs a simulation; they read `results/`. Run them from
the root of the repository with `uv run python scripts/<name>.py`; `--help` lists the
arguments.

| Script | Reads | Writes |
|:--|:--|:--|
| `make_thesis_outputs.py` | Every run folder under `results/` | All figures (`outputs/figures/*.pdf`), all tables and the number macros (`outputs/tables/*.tex`) of the thesis. Labels, table heads and notes are set in this script |
| `baseline_report.py` | One run folder and its configuration | `report.txt`: Δ^C over all, converged and settled sessions with confidence intervals, the grid's own Nash value, a table per session, the impulse responses |
| `deviation_test.py` | One run folder (saved tables) and its configuration | `deviation_test.txt`, `deviation_static.parquet`, `deviation_trials.parquet`: the gain of a single best response in the one-period game and under the frozen policies, and the staleness check of the stored Q-values |
| `irf_replay.py` | One run folder (saved tables) | `irf_replay.txt`, `irf_replay_periods.parquet`: impulse responses replayed from the saved policies, with the visited states marked |
| `qtable_trigger_test.py` | One run folder (saved tables) | `qtable_trigger_test.txt`, `qtable_trigger_cells.parquet`: whether the saved greedy policies respond to the lagged price |
| `make_results_explorer.py` | Every run folder under `results/`, through the functions of `make_thesis_outputs.py` | The data block inside `results_explorer.html` (root of the repository), the interactive page of the results. Layout and texts of the page are in the HTML file itself |

Four of the generated files are not used by the thesis: `outputs/tables/tab_parameters.tex`,
`tab_parameters_short.tex` and `tab_replication.tex`, and `outputs/figures/fig_policy.pdf`.

The figure functions themselves are in `src/ai_trading/viz/figures.py`.
