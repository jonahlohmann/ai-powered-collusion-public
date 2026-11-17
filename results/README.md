# Results

The outputs of every run that the thesis cites, as the simulation and the report scripts
wrote them. `scripts/make_thesis_outputs.py` reads these folders and writes `outputs/`.

## Layout of a run folder

```
results/<run>/
├── report.txt                  # run report (scripts/baseline_report.py)
├── deviation_test.txt          # deviation test and staleness check (scripts/deviation_test.py)
├── deviation_static.parquet    # deviation test: the one-period game, per session and state
├── deviation_trials.parquet    # deviation test: the simulated deviations
├── logs/s<i>.log               # log of the runner for session i
└── s<i>/                       # one folder per session (seed = 42 + i)
    ├── sessions.parquet        # one row: convergence, Δ^C, trading intensity, profits
    ├── periods.parquet         # evaluation window and impulse-response paths, per period
    ├── trajectory.parquet      # learning path: one snapshot per million periods of training
    ├── run_metadata.json       # resolved configuration, versions, start and end time
    └── qtables/session_<i>/    # saved tables and grids (.npy): agent_<j>.npy is a Q-table,
                                #   agent_<j>_prefs.npy and agent_<j>_values.npy belong to an AC learner
```

The columns are defined in `docs/parameters.md`. Not every run has every file:

- `r1_1_early_stop` was made before the learning path and the deviation test existed, on
  a hosted runner (its metadata show the paths of that machine); it has no logs.
- `validation_q` and `validation_ac` have no report: the thesis reads them from
  `sessions.parquet` (`outputs/tables/tab_validation.tex`).
- `pilot_b` and `pilot_c` have no deviation test.
- `r1_baseline` also holds the two descriptive checks of the impulse responses:
  `irf_replay.txt` with `irf_replay_periods.parquet` (`scripts/irf_replay.py`) and
  `qtable_trigger_test.txt` with `qtable_trigger_cells.parquet`
  (`scripts/qtable_trigger_test.py`).

## Names recorded at run time

The runs were made under working names, and the configuration files were renamed
afterwards. The metadata, the logs and the header lines of the reports keep the names of
that time. The content of each configuration is the one under `experiments/` today.

| Folder | Run | Recorded experiment name | Recorded configuration file |
|:--|:-:|:--|:--|
| `r1_baseline` | R1 | `m2` | `config_m2.yaml` |
| `r1_1_early_stop` | R1.1 | `phase0_baseline` | `config_phase0.yaml` |
| `r1_2_earlier_spec` | R1.2 | `m1_long` | `config_m1_long.yaml` |
| `r1_3_rounded_price` | R1.3 | `main_snapped` | `config_m2_snapped.yaml` |
| `r1_4_no_price` | R1.4 | `m2_noprice` | `config_m2_noprice.yaml` |
| `r1_5_no_price_beta3e6` | R1.5 | `m2_noprice_beta3e6` | `config_m2_noprice_beta3e6.yaml` |
| `r1_6_ref_beta` | R1.6 | `m2_beta5e7` | `config_m2_beta5e7.yaml` |
| `r1_7_ref_grid` | R1.7 | `paper_exact` | `config_paper_exact.yaml` |
| `r1_8_ref_noise` | R1.8 | `m2_sigma01` | `config_m2_sigma01.yaml` |
| `r2_q_ac` | R2 | `qa` | `config_qa.yaml` |
| `r2_1_ac_ac` | R2.1 | `aa` | `config_aa.yaml` |
| `r2_2_two_alphas` | R2.2 | `alpha_het` | `config_alpha_het.yaml` |
| `validation_q`, `validation_ac` | validation runs | `br_q`, `br_ac` | `config_br_q.yaml`, `config_br_ac.yaml` |
| `pilot_a`, `pilot_b`, `pilot_c` | pilot | `qa_tune_a`, `_b`, `_c` | `config_qa_tune_a.yaml`, `_b`, `_c` |

In the same files, "M1" is the earlier specification, "M2" the baseline specification,
"snapped" the rounded price and "Phase 0" the run R1.1. The field `code_version` is a
commit of the author's private working repository; the sessions of a run can carry
different commits, because documentation and analysis scripts were committed while a run
was under way.

## What was changed for publication

- The absolute path of the working folder is replaced by `.` in the logs and in
  `run_metadata.json`.
- The shell scripts that started the sessions on the laptop, and their console output,
  are left out; the run command is in the head of each configuration file.
- The sessions of `r1_1_early_stop` were moved up from a nested folder, and its
  `report.txt` was written anew with `scripts/baseline_report.py` of this repository (the
  report of the time had an older format without the summary over all sessions).

Nothing else was edited.
