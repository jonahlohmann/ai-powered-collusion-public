# Experiments

One YAML configuration per run. A configuration names a parameter file under `shared/`
(`params:`), the agents, the simulation settings (sessions, cap, evaluation window,
impulse responses) and, where a run departs from its parameter file, `overrides:`. The
head of each file says what the run is, how it differs from the baseline and how to start
it. The keys are defined in `docs/parameters.md`; what each choice does to the result is
in `docs/implementation-choices.md`.

| Folder | Content |
|:--|:--|
| `shared/` | Parameter files: `params_baseline.yaml` (the baseline specification), `params_earlier_spec.yaml` (the earlier specification), `params_ref_grid.yaml` (R1.7: DGJ's value grid and exploration decay), `params_debug.yaml` (small grids for tests) |
| `01_replicate_baseline/` | The baseline R1 and the runs R1.1 to R1.8 (two Q-learners); `config_debug.yaml` is a run of a few seconds for a first check of an installation |
| `02_heterogeneity/` | R2 (Q-learner and AC learner), R2.1 (two AC learners), R2.2 (two learning rates), the two validation runs and the three settings of the pilot |
| `03_exploratory_price_grid/` | An exploratory run outside the runs of the thesis, with its own README, patch, tests and reports |

Start a run one session at a time (the seed is `seed_base` + session id, so the sessions
can be run in any order and in parallel):

```bash
uv run python -m ai_trading.simulation.runner \
    --config experiments/01_replicate_baseline/config_r1_baseline.yaml \
    --session-id-start 0 --n-sim 1 --output-dir results/r1_baseline/s0
```

The table of all runs with their result folders is in the [README](../README.md) of the
repository.
