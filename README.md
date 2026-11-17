# AI-powered trading and algorithmic collusion under algorithmic heterogeneity

Code, run configurations and results of a bachelor's thesis at the University of
St.Gallen (2026). The thesis rebuilds the simulation of Dou, Goldstein and Ji (2025),
"AI-Powered Trading, Algorithmic Collusion, and Price Efficiency" (DGJ in this
repository), at a scale that a single laptop can run, and extends it to algorithmic
heterogeneity: a tabular Q-learner and a tabular softmax actor-critic (AC) learner trade
side by side as informed speculators in a Kyle-type market with noise traders,
information-insensitive investors and a market maker who learns by rolling regression.

Research question: does the introduction of algorithmic heterogeneity prevent or
facilitate collusive convergence among reinforcement learning trading agents in thin
market simulations?

This repository is a snapshot. It was published once, together with the thesis, and is
not developed further. Everything the thesis reports can be regenerated from it.

## What is in it

| Path | Content |
|:--|:--|
| `src/ai_trading/` | The simulation: market environment, grids, agents (Q-learner, AC learner, fixed policy), runner, benchmarks, metrics, figure functions |
| `tests/` | The test suite (245 tests) |
| `experiments/` | One YAML configuration per run and the shared parameter files ([experiments/README.md](experiments/README.md)) |
| `scripts/` | Run reports, the deviation test, and the script that draws every figure and table of the thesis ([scripts/README.md](scripts/README.md)) |
| `results/` | The outputs of every run that the thesis cites: per session the summary, the evaluation periods, the learning path, the saved tables and the log; per run the report and the deviation test ([results/README.md](results/README.md)) |
| `outputs/` | The figures (PDF), the tables and the number macros (LaTeX) of the thesis, as generated from `results/` |
| `results_explorer.html` | The results as an interactive page for the browser, one file without dependencies ([Explore the results](#explore-the-results)) |
| `docs/parameters.md` | Technical definitions: specifications, grids, switches, the impulse-response protocol, the output schema |
| `docs/implementation-choices.md` | The map of the implementation choices: what DGJ describe, what this code does, which run varies the choice and what it does to the result |
| `docs/ai-use-record.md` | The record of the use of AI tools, by working step, with the main prompts |

## Results in brief

The measure is DGJ's collusion capacity Δ^C: 0 at the profit of the competitive Nash
equilibrium, 1 at the profit of the perfect cartel. On the action grid of the baseline
specification the grid's own Nash equilibrium lies at −0.029. Values are means over all
sessions of a run with a bootstrap 95% confidence interval; the full table is
`outputs/tables/tab_runs.tex`, the reports are in `results/<run>/report.txt`.

| Population | Run | Δ^C | Profit above Nash |
|:--|:-:|:--|--:|
| Two Q-learners (the baseline) | R1 | 0.34 [0.32, 0.36] | +4.2% |
| One Q-learner and one AC learner | R2 | 0.14 [0.12, 0.15] | +1.7% |
| Two AC learners | R2.1 | 0.09 [0.08, 0.11] | +1.2% |
| Two Q-learners with different learning rates | R2.2 | 0.33 [0.30, 0.35] | +4.1% |

- In the baseline, two Q-learners earn supra-competitive profits in every session, but
  about 4% above the Nash profit. DGJ report a Δ^C of about 0.75 at their noise level (σ_u = 0.1); the
  reference run at that noise level (R1.8) gives 0.34.
- A single best response pays in 22 of the 30 sessions of the baseline. The deviation
  test finds a cost that follows a deviation, but no detectable deterrent punishment.
- The level moves from 0.07 to 0.91 with the stopping rule, the action grid and the price
  used in the reward (R1.1, R1.2, R1.3). It does not move with the value grid, the decay
  of exploration or the noise level (R1.6, R1.7, R1.8).
- A learner from another family lowers the level; a different learning rate within
  Q-learning does not.

The thesis itself is not part of this repository.

## Explore the results

[`results_explorer.html`](results_explorer.html) shows the results as an interactive page:
play the one-period game that the speculators face, watch the sessions of a run learn,
compare the runs session by session, pair the learners, and see what happens when one
speculator deviates. Every session of every run is also listed in a sortable table.

GitHub displays an HTML file as source code, not as a page. To use it, download the file
(on its GitHub page: "Download raw file"), or clone the repository or download it as a ZIP
(Code → Download ZIP), and open `results_explorer.html` with a double click. It is a single
file that contains its data; it needs no installation, no server and no network connection.

No result is typed into the page by hand. Its numbers are read from `results/` with the
functions that make the tables and figures of the thesis, and can be written anew (a
few seconds, no simulation):

```bash
uv run python scripts/make_results_explorer.py      # rewrites the data inside results_explorer.html
```

## Install and test

Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev                # environment from uv.lock
uv run pytest tests/ -q            # 245 tests
uv run python -m ai_trading.simulation.runner \
    --config experiments/01_replicate_baseline/config_debug.yaml   # small run, a few seconds
```

On macOS, `uv sync` can set the hidden flag on the editable-install file, and Python then
cannot import `ai_trading`. If that happens, run
`chflags nohidden .venv/lib/python*/site-packages/_editable_impl_*.pth` and use
`uv run --no-sync` afterwards.

## The runs

Two specifications are used. The baseline specification
(`experiments/shared/params_baseline.yaml`) has an action grid per value, DGJ's
exploration counter and DGJ's initialisation of the Q-table. The earlier specification
(`experiments/shared/params_earlier_spec.yaml`) has one grid of order sizes for all
values, among other settings. All runs use seed 42 + session id.

| Run | What it is | Configuration under `experiments/` | Folder under `results/` | Sessions |
|:-:|:--|:--|:--|--:|
| R1 | The baseline: two Q-learners, baseline specification | `01_replicate_baseline/config_r1_baseline.yaml` | `r1_baseline` | 30 |
| R1.1 | Earlier specification, training stopped early (window 100,000 periods, cap 10 million) | `01_replicate_baseline/config_r1_1_early_stop.yaml` | `r1_1_early_stop` | 30 |
| R1.2 | Earlier specification, DGJ's stopping rule (window 1 million periods, cap 40 million) | `01_replicate_baseline/config_r1_2_earlier_spec.yaml` | `r1_2_earlier_spec` | 30 |
| R1.3 | Baseline specification with the profit paid on the rounded price | `01_replicate_baseline/config_r1_3_rounded_price.yaml` | `r1_3_rounded_price` | 30 |
| R1.4 | Baseline specification without the lagged price in the state | `01_replicate_baseline/config_r1_4_no_price.yaml` | `r1_4_no_price` | 30 |
| R1.5 | R1.4 with exploration that fades three times as fast | `01_replicate_baseline/config_r1_5_no_price_beta3e6.yaml` | `r1_5_no_price_beta3e6` | 30 |
| R1.6 | Reference run with DGJ's exploration decay | `01_replicate_baseline/config_r1_6_ref_beta.yaml` | `r1_6_ref_beta` | 30 |
| R1.7 | Reference run with DGJ's value grid and exploration decay | `01_replicate_baseline/config_r1_7_ref_grid.yaml` | `r1_7_ref_grid` | 8 |
| R1.8 | Reference run at DGJ's noise level (σ_u = 0.1) | `01_replicate_baseline/config_r1_8_ref_noise.yaml` | `r1_8_ref_noise` | 30 |
| R2 | One Q-learner and one AC learner | `02_heterogeneity/config_r2_q_ac.yaml` | `r2_q_ac` | 30 |
| R2.1 | Two AC learners | `02_heterogeneity/config_r2_1_ac_ac.yaml` | `r2_1_ac_ac` | 30 |
| R2.2 | Two Q-learners with learning rates 0.01 and 0.1 | `02_heterogeneity/config_r2_2_two_alphas.yaml` | `r2_2_two_alphas` | 30 |
| Validation runs | A Q-learner and an AC learner, each against a rival fixed at the Nash intensity | `02_heterogeneity/config_validation_q.yaml`, `config_validation_ac.yaml` | `validation_q`, `validation_ac` | 4 each |
| Pilot | Three settings of the AC learner | `02_heterogeneity/config_pilot_a.yaml`, `_b`, `_c` | `pilot_a`, `pilot_b`, `pilot_c` | 4 each |

An exploratory run on the layout of the price grid stands outside these runs and is
documented in
[experiments/03_exploratory_price_grid/](experiments/03_exploratory_price_grid/README.md).

## Reproduce

Regenerate every figure, table and number of the thesis from the results in this
repository (a few seconds, no simulation):

```bash
uv run python scripts/make_thesis_outputs.py        # writes outputs/figures and outputs/tables
```

Regenerate the report and the deviation test of a run:

```bash
uv run python scripts/baseline_report.py --results-dir results/r1_baseline \
    --config experiments/01_replicate_baseline/config_r1_baseline.yaml \
    --out results/r1_baseline/report.txt
uv run python scripts/deviation_test.py --results-dir results/r1_baseline \
    --config experiments/01_replicate_baseline/config_r1_baseline.yaml \
    --out results/r1_baseline/deviation_test.txt
```

Rerun a session, or a whole run one session at a time. A session of the baseline takes
roughly a quarter of an hour on a laptop; all runs together took about 139 hours.

```bash
uv run python -m ai_trading.simulation.runner \
    --config experiments/01_replicate_baseline/config_r1_baseline.yaml \
    --session-id-start 7 --n-sim 1 --output-dir results/r1_baseline/s7
```

A session is determined by its configuration and its seed, so a rerun on the same
versions of Python and NumPy (recorded in each `run_metadata.json`) reproduces the stored
session.

## About this snapshot

- The repository is a copy of the author's private working repository at commit
  `91c16c0`. The code is the code that produced the results: against the working
  repository, only comments and docstrings were revised (internal working labels replaced
  by the run labels of the thesis, references to working documents removed), and the
  default output folders of `scripts/make_thesis_outputs.py` are `outputs/` here.
- The runs were made while the code was still being extended, under working names. The
  file `run_metadata.json` of each session records the name and the configuration file of
  that time and the commit of the working repository (`code_version`).
  [results/README.md](results/README.md) maps these names to the run labels above.
- In the logs and metadata under `results/`, the absolute path of the working folder is
  replaced by `.`. Nothing else in the result files was changed.

## Reference

Dou, W. W., Goldstein, I., & Ji, Y. (2025). AI-powered trading, algorithmic collusion,
and price efficiency. *SSRN Electronic Journal*. https://doi.org/10.2139/ssrn.4452704

## Use of AI tools

Based on the author's insights, Claude Code (Anthropic) helped write the simulation code,
the testing suite and the analysis scripts of the simulation runs.
[docs/ai-use-record.md](docs/ai-use-record.md) records the working steps and their main
prompts.

## Licence

MIT for the code, see [LICENSE](LICENSE).
