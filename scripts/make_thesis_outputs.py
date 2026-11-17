"""Thesis figures, tables and number macros from the result runs.

Reads every run directory of the run map (``<run>/s*/sessions.parquet``,
``periods.parquet``, ``trajectory.parquet``, ``run_metadata.json`` and the saved
tables under ``qtables/``, plus ``<run>/deviation_trials.parquet`` from
``scripts/deviation_test.py``), and writes the files below. The function that
makes a file stands in brackets; a file marked * is generated but not included
in the thesis.

Figures (``outputs/figures``):
    fig_timing.pdf                timing of one period; schematic, no run data (make_f1)
    fig_learning_curves.pdf       baseline (make_f2)
    fig_delta_c_distribution.pdf  earlier_spec, baseline, rounded_price (make_f3)
    fig_policy.pdf *              baseline; Q-tables or preferences (make_f4)
    fig_irf_deviation.pdf         baseline (make_f5)
    fig_irf_noise.pdf             baseline (make_f5b)
    fig_populations.pdf           baseline, q_ac, ac_ac (make_f6)
    fig_populations_irf.pdf       baseline, q_ac, ac_ac (make_f7)
    fig_state_space.pdf           a learner's table; schematic (make_f8)
    fig_learner_paths.pdf         q_ac: trading intensity per learner (make_f9)

Tables (``outputs/tables``):
    tab_benchmarks.tex            closed-form Nash and cartel (make_t1)
    tab_parameters.tex *          all parameters next to DGJ's values (make_t2)
    tab_parameters_short.tex *    its rows that differ from DGJ (make_t2)
    tab_parameters_baseline.tex   market and Q-learner next to DGJ (make_t2_split)
    tab_parameters_ac.tex         settings of the AC learner (make_t2_split)
    tab_validation.tex            validation runs: learner vs fixed Nash (make_t3)
    tab_replication.tex *         earlier_spec, baseline (make_t4)
    tab_sensitivity.tex           the baseline and its variations R1.1 to R1.8 (make_t5)
    tab_populations.tex           baseline, q_ac, ac_ac, two_alphas (make_t6)
    tab_deviation.tex             deviation test and value staleness (make_t7)
    tab_runs.tex                  overview of all runs, by hypothesis (make_t8)
    numbers.tex                   macros for every number the text quotes

A missing run produces a warning and its outputs are skipped. No number in
the thesis is typed by hand: cite the macros of ``numbers.tex``.

Usage:
    PYTHONPATH=src uv run --no-sync python scripts/make_thesis_outputs.py \\
        [--runs KEY=DIR ...] [--out-figures outputs/figures] \\
        [--out-tables outputs/tables]
    `--runs baseline=results/r1_2_earlier_spec` makes another run the baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.simulation import benchmarks
from ai_trading.viz import figures as viz

logger = logging.getLogger("make_thesis_outputs")

# Run keys follow the run labels of the thesis (tab_runs.tex): the baseline R1 and its
# variations R1.1 to R1.8, the mixed population R2 and its variations R2.1 and R2.2.
# A key gives the macro prefix (\DeltaC<Key>Mean, ...); a folder carries the label.
DEFAULT_RUNS = {
    "baseline": "results/r1_baseline",
    "early_stop": "results/r1_1_early_stop",
    "earlier_spec": "results/r1_2_earlier_spec",
    "rounded_price": "results/r1_3_rounded_price",
    "no_price": "results/r1_4_no_price",
    "no_price_beta3e6": "results/r1_5_no_price_beta3e6",
    "ref_beta": "results/r1_6_ref_beta",
    "ref_grid": "results/r1_7_ref_grid",
    "ref_noise": "results/r1_8_ref_noise",
    "q_ac": "results/r2_q_ac",
    "ac_ac": "results/r2_1_ac_ac",
    "two_alphas": "results/r2_2_two_alphas",
}
RUN_LABELS = {
    "baseline": "R1 Baseline",
    "early_stop": "R1.1 Earlier specification, early stop",
    "earlier_spec": "R1.2 Earlier specification",
    "rounded_price": "R1.3 Reward on the rounded price",
    "no_price": "R1.4 No lagged price",
    "no_price_beta3e6": r"R1.5 No lagged price, $\beta = 3\times10^{-6}$",
    "ref_beta": "R1.6 DGJ's exploration decay",
    "ref_grid": "R1.7 DGJ's value grid and exploration decay",
    "ref_noise": "R1.8 DGJ's noise level",
    "q_ac": "R2 Q–AC",
    "ac_ac": "R2.1 AC–AC",
    "two_alphas": "R2.2 Two learning rates",
}
# Differences of the session-mean Delta^C between two runs (A minus B), each with a
# bootstrap 95% CI over independent samples: macros \DeltaCDiff<A><B>Mean, ...Lo, ...Hi.
DIFF_PAIRS: list[tuple[str, str]] = [
    ("baseline", "q_ac"),
    ("q_ac", "ac_ac"),
    ("baseline", "ac_ac"),
    ("baseline", "two_alphas"),
    ("baseline", "no_price"),
    ("baseline", "no_price_beta3e6"),
    ("baseline", "ref_noise"),
]
# Overview of the runs (tab_runs.tex): thesis label, population, role and the change
# against the baseline specification, grouped by the hypothesis a run belongs to. Roles:
# "Result" tests a hypothesis, "Robustness" changes one implementation choice or isolates
# a cause, "Reference" moves one setting back to DGJ's value. The labels R1, R1.x, R2,
# R2.x are those of the thesis text; the logs and the metadata saved with the runs keep
# the working names under which the runs were made (listed in the README).
RUN_OVERVIEW: list[tuple[str, list[tuple[str, str, str, str, str]]]] = [
    (
        "H1: replication",
        [
            ("baseline", "R1", "Q–Q", "Result", "The baseline: baseline specification"),
            (
                "early_stop",
                "R1.1",
                "Q–Q",
                "Robustness",
                "Earlier specification; training stops after {conv} unchanged periods or at {cap}",
            ),
            (
                "earlier_spec",
                "R1.2",
                "Q–Q",
                "Robustness",
                "Earlier specification: one grid of order sizes for all values",
            ),
            (
                "rounded_price",
                "R1.3",
                "Q–Q",
                "Robustness",
                "Reward paid on the price rounded to its bin",
            ),
            (
                "no_price",
                "R1.4",
                "Q–Q",
                "Robustness",
                "Lagged price removed from the state",
            ),
            (
                "no_price_beta3e6",
                "R1.5",
                "Q–Q",
                "Robustness",
                r"Lagged price removed from the state; $\beta = {beta}$",
            ),
            (
                "ref_beta",
                "R1.6",
                "Q–Q",
                "Reference",
                r"DGJ's exploration decay: $\beta = {beta}$",
            ),
            (
                "ref_grid",
                "R1.7",
                "Q–Q",
                "Reference",
                r"DGJ's value grid, $n_v = {n_v}$, and exploration decay as in R1.6",
            ),
            (
                "ref_noise",
                "R1.8",
                "Q–Q",
                "Reference",
                r"DGJ's noise level: $\sigma_u = {sigma_u}$",
            ),
        ],
    ),
    (
        "H2: heterogeneity",
        [
            ("q_ac", "R2", "Q–AC", "Result", "One Q-learner replaced by an AC learner"),
            (
                "ac_ac",
                "R2.1",
                "AC–AC",
                "Robustness",
                "Both Q-learners replaced by AC learners",
            ),
            (
                "two_alphas",
                "R2.2",
                "Q–Q",
                "Robustness",
                r"Two Q-learners with learning rates $\alpha = 0.01$ and $0.1$",
            ),
        ],
    ),
]
PAPER_VALUES: list[tuple[str, str, str, str]] = [
    # (label, config key, DGJ value as text, reason). The reason opens with the standard
    # behind the choice.
    ("Speculators $I$", "n_agents", "2", "DGJ value; comparability."),
    (
        r"Demand elasticity $\xi$",
        "xi",
        "500",
        (
            r"Regime choice: DGJ's large-$\xi$ cell, where they find "
            "price-trigger collusion."
        ),
    ),
    (
        r"Noise-trader s.d. $\sigma_u$",
        "sigma_u",
        "0.1",
        (
            r"Regime choice: a higher noise level in DGJ's price-trigger region; "
            "at both levels noise moves the price by far less than one price bin. "
            "R1.8 repeats the baseline at 0.1."
        ),
    ),
    (r"Discount factor $\rho$", "rho", "0.95", "DGJ value; comparability."),
    (r"Inventory penalty $\theta$", "theta", "0.1", "DGJ value; comparability."),
    (
        "Value grid $n_v$",
        "n_v",
        "10",
        (
            "Laptop scale: with ten values a session trains about five times "
            "as long (run R1.7)."
        ),
    ),
    ("Action grid $n_x$", "n_x", "15", "DGJ value; comparability."),
    ("Price grid $n_p$", "n_p", "31", "DGJ value; comparability."),
    ("Market-maker window $T_m$", "T_m", "10,000", "DGJ value; comparability."),
    (r"Learning rate $\alpha$", "alpha", "0.01", "DGJ value; comparability."),
    (
        r"Exploration decay $\beta$",
        "beta",
        r"$5\times10^{-7}$",
        (
            "DGJ rule, rescaled to five values: about 430 random draws per cell "
            "of the table, as in DGJ."
        ),
    ),
    (
        "Exploration counter",
        "exploration_counter",
        "per $v$",
        "DGJ rule (Equation~4.3).",
    ),
    ("Q-value initialisation", "q_init", "Section~4.2", "DGJ rule (Section~4.2)."),
    (
        "Action grid layout",
        "action_grid",
        "not stated",
        "Interpretation of DGJ's description (p.~22): levels of trading intensity per $v$.",
    ),
    (
        "Convergence window",
        "convergence_periods",
        "1,000,000",
        "DGJ value; comparability.",
    ),
    (
        "Maximum periods",
        "max_periods",
        "none",
        "Laptop scale: one session that converges very late cannot hold up a run; about twice the median session.",
    ),
    (
        "Sessions $N_{\\mathrm{sim}}$",
        "__n_sessions__",
        "1,000",
        "Laptop scale; enough for bootstrap confidence intervals.",
    ),
]
# ---- The parameter table in two halves (appendix of the thesis) ---------------------
# The parameter table is printed in two halves: the baseline with its two Q-learners
# next to DGJ's values (tab_parameters_baseline.tex) and the settings of the AC learner
# (tab_parameters_ac.tex). PAPER_VALUES and tab_parameters.tex above are the combined
# table: still generated (tests/test_make_t2.py pins it), not included in the thesis.
# Rows: (label, config key, DGJ's value as text, reason); None starts a new group.
# DGJ's values and rules: Dou et al. (2025), pp. 22-24.
# Reasons are short and point to the section of the thesis that argues them;
# {placeholders} are filled from the runs (q_parameter_rows).
Q_PARAMETER_ROWS: list[tuple[str, str, str, str] | None] = [
    ("Speculators", "n_agents", "2", "DGJ's value."),
    (r"Investors' demand slope $\xi$", "xi", "500", "DGJ's value."),
    (r"Pricing-error weight $\theta$", "theta", "0.1", "DGJ's value."),
    (r"Discount factor $\rho$", "rho", "0.95", "DGJ's value."),
    (
        r"Noise level $\sigma_u$",
        "sigma_u",
        "0.1",
        (
            "Set early in the project. At 0.1, R1.8 gives the same $\\Delta^C$ "
            r"(Section~\ref{sec:parameters})."
        ),
    ),
    ("Market maker's window", "T_m", "10,000", "DGJ's value, in periods."),
    None,
    (
        "Value grid, points $n_v$",
        "n_v",
        "10",
        (
            "Computing time: about {minutes_base} minutes per session instead of "
            r"{minutes_big} (Section~\ref{sec:parameters})."
        ),
    ),
    (
        "Action grid, levels",
        "n_x",
        "15",
        (
            "DGJ's value, with one grid for each value: our reading of their "
            r"p.~22 (Section~\ref{sec:discretisation})."
        ),
    ),
    (
        "Price grid, bins",
        "n_p",
        "31",
        (
            "DGJ's value, with one grid for all values. Unlike on the grid that DGJ "
            "describe (p.~23), one level in an order rarely moves the price into "
            r"another bin (Section~\ref{sec:disc-limits})."
        ),
    ),
    (
        "Margin of the grids",
        "iota",
        "0.1",
        "DGJ's value: 10\\% beyond the bounds of each grid.",
    ),
    None,
    (r"Learning rate $\alpha$", "alpha", "0.01", "DGJ's value."),
    (
        r"Exploration decay $\beta$",
        "beta",
        r"$5\times10^{-7}$",
        (
            "As many random draws per cell of the table as in DGJ, about {draws} "
            r"(Section~\ref{sec:parameters})."
        ),
    ),
    (
        "Exploration counter",
        "exploration_counter",
        "per value",
        r"DGJ's rule (Section~\ref{sec:learners}).",
    ),
    (
        "Initial values",
        "q_init",
        "random rival",
        r"DGJ's rule (Section~\ref{sec:learners}).",
    ),
    None,
    (
        "Convergence window",
        "convergence_periods",
        "1,000,000",
        r"DGJ's rule and value, in periods (Section~\ref{sec:convergence}).",
    ),
    (
        "Cap on a session",
        "max_periods",
        "none",
        (
            "Computing time: a session that converges very late cannot hold up a "
            r"run (Section~\ref{sec:convergence})."
        ),
    ),
    (
        "Sessions per run",
        "__n_sessions__",
        "1,000",
        (
            "Computing time: about {hours} hours per run instead of {weeks} weeks "
            r"(Section~\ref{sec:parameters})."
        ),
    ),
]
# Placeholders of the reasons above that are filled from the runs.
Q_PARAMETER_FILL = (
    "n_x",
    "draws",
    "minutes_big",
    "minutes_base",
    "weeks",
    "hours",
)
# Thesis-column text of the rules in the split table, by config value.
Q_PARAMETER_TEXT: dict[str, dict[str, str]] = {
    "exploration_counter": {"v": "per value", "state": "per state"},
    "q_init": {"paper": "random rival", "optimistic": "optimistic", "zero": "zero"},
}
# Reason column of the two halves (text width 15 cm; both set in \footnotesize, so
# that the first half fits on the page under its paragraph).
T2Q_ALIGN = r"@{}lrr>{\raggedright\arraybackslash}p{6.75cm}@{}"
T2AC_ALIGN = r"@{}lr>{\raggedright\arraybackslash}p{8.75cm}@{}"
# DGJ's value grid and exploration decay (Dou et al., 2025, p. 23), for the number of
# random draws per cell of their table.
DGJ_N_V, DGJ_BETA = 10, 5e-7
# Thesis-column text of the design rows, keyed by (config key, config value).
DESIGN_TEXT: dict[str, dict[str, str]] = {
    "exploration_counter": {"v": "per $v$", "state": "per state"},
    "q_init": {"paper": "Section~4.2", "optimistic": "optimistic", "zero": "zero"},
    "action_grid": {"per_v": "per-$v$ levels", "global": "global"},
}
ACTOR_CRITIC_REASON = (
    "Own design: first-principles default, selected by stability in a pilot; "
    "see limitations."
)
# Config keys of the parameter rows where the thesis departs from DGJ (the rows
# of tab_parameters_short.tex).
T2_SHORT_KEYS = (
    "sigma_u",
    "n_v",
    "beta",
    "action_grid",
    "max_periods",
    "__n_sessions__",
)
# The two validation runs (agent 0 learns against a fixed Nash player); not part of
# the run map, so they get no run macros.
VALIDATION_RUNS = {
    "validation_q": ("Q-learner", "results/validation_q"),
    "validation_ac": ("AC learner", "results/validation_ac"),
}
# Runs of the deviation table (tab_deviation.tex), in row order. "no_price" (R1.4) has
# no row: tab_sensitivity.tex carries its deviation gain with the note that the
# continuation is zero by construction.
DEVIATION_KEYS = ("baseline", "rounded_price", "q_ac", "ac_ac")
# Classification thresholds of DGJ (Online Appendix 4.5), as in
# scripts/baseline_report.py: price trigger if both agents respond by more than X_UPPER.
X_UPPER = 5e-4
# DGJ's own figures for xi = 500 and sigma_u = 0.1 (Dou et al., 2025, Section 5.3,
# the paragraph on Panel A of Figure 2): an average collusion capacity of
# "approximately 0.75" and profits "about 10%" above the non-collusive benchmark.
# Quoted from the source, not computed from results/.
DGJ_DELTA_C = 0.75
DGJ_PROFIT_PCT = 10
AGENT_TYPE_TAGS = {"qlearner": "Q", "actor_critic": "AC"}
POP_LABELS = {
    "baseline": "Q–Q (R1)",
    "q_ac": "Q–AC (R2)",
    "ac_ac": "AC–AC (R2.1)",
    "two_alphas": r"Q–Q, two $\alpha$ (R2.2)",
}
# Row labels of the deviation table: the population first, as in tab_populations.tex.
DEVIATION_LABELS = {
    "baseline": "Q–Q (R1)",
    "rounded_price": "Rounded price (R1.3)",
    "no_price": "No lagged price (R1.4)",
    "q_ac": "Q–AC (R2)",
    "ac_ac": "AC–AC (R2.1)",
}


# ---- loading ---------------------------------------------------------------------


@dataclass
class Run:
    key: str
    path: Path
    sessions: pd.DataFrame
    periods: pd.DataFrame
    trajectory: pd.DataFrame
    metadata: dict[str, Any]
    qtable_dirs: dict[int, Path] = field(default_factory=dict)
    deviation_trials: pd.DataFrame | None = None

    @property
    def config(self) -> dict[str, Any]:
        return self.metadata.get("resolved_config", {})

    @property
    def n_agents(self) -> int:
        return int(self.config.get("n_agents", 2))

    @property
    def v_bar(self) -> float:
        return float(self.config.get("v_bar", 0.0))

    @property
    def pre(self) -> int:
        return int((self.config.get("irf") or {}).get("pre_periods", 3))

    @property
    def post(self) -> int:
        return int((self.config.get("irf") or {}).get("post_periods", 6))

    @property
    def label(self) -> str:
        return RUN_LABELS.get(self.key, self.key)


def _concat(files: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_run(key: str, path: Path) -> Run | None:
    """Load one run directory; None (with a warning) when it has no sessions."""
    if not path.exists():
        logger.warning("run %s: directory %s does not exist; skipped", key, path)
        return None
    sess_files = sorted(path.rglob("sessions.parquet"))
    if not sess_files:
        logger.warning("run %s: no sessions.parquet under %s; skipped", key, path)
        return None
    sessions = _concat(sess_files).sort_values("session_id").reset_index(drop=True)
    periods = _concat(sorted(path.rglob("periods.parquet")))
    trajectory = _concat(sorted(path.rglob("trajectory.parquet")))
    meta_files = sorted(path.rglob("run_metadata.json"))
    metadata = json.loads(meta_files[0].read_text()) if meta_files else {}
    qdirs: dict[int, Path] = {}
    for q in sorted(path.rglob("qtables/session_*")):
        m = re.fullmatch(r"session_(\d+)", q.name)
        if m:
            qdirs[int(m.group(1))] = q
    dev_file = path / "deviation_trials.parquet"
    dev = pd.read_parquet(dev_file) if dev_file.exists() else None
    logger.info(
        "run %s: %d sessions, %d period rows, %d trajectory rows, %d table dirs, deviation trials: %s",
        key,
        len(sessions),
        len(periods),
        len(trajectory),
        len(qdirs),
        "yes" if dev is not None else "no",
    )
    return Run(key, path, sessions, periods, trajectory, metadata, qdirs, dev)


def grids_of(run: Run) -> dict[str, np.ndarray] | None:
    """Grids saved with the first session's tables, or None."""
    if not run.qtable_dirs:
        return None
    q = run.qtable_dirs[min(run.qtable_dirs)]
    out = {}
    for name in ("v_grid", "action_grid", "price_grid"):
        f = q / f"{name}.npy"
        if not f.exists():
            return None
        out[name] = np.load(f)
    table = q / "action_table.npy"
    if table.exists():
        out["action_table"] = np.load(table)
    else:
        # Runs saved without an action table (global grid only): tile the quantity grid.
        out["action_table"] = np.tile(out["action_grid"], (len(out["v_grid"]), 1))
    return out


def benchmarks_of(run: Run) -> dict[str, float] | None:
    """Closed-form benchmarks at the run's parameters and v-grid."""
    g = grids_of(run)
    cfg = run.config
    if g is None or not cfg:
        return None
    sigma_u = float(run.sessions["sigma_u"].iloc[0])
    return benchmarks.compute_all_benchmarks(
        n_agents=run.n_agents,
        xi=float(cfg["xi"]),
        sigma_u=sigma_u,
        theta=float(cfg["theta"]),
        sigma_v=float(cfg.get("sigma_v", 1.0)),
        v_grid=g["v_grid"],
    )


def grid_nash_of(run: Run) -> float | None:
    """Δ^C of the grid Nash equilibrium of the run's action table."""
    g, b = grids_of(run), benchmarks_of(run)
    if g is None or b is None:
        return None
    game = benchmarks.grid_stage_game(
        g["action_table"], g["v_grid"], b["lambda_N"], b["pi_N"], b["pi_M"], run.v_bar
    )
    return float(game["delta_c_grid_nash"])


def greedy_policies(run: Run) -> dict[int, list[np.ndarray]]:
    """Per session, the greedy policy ``(n_p, n_v, n_v)`` of every agent from the saved tables."""
    out: dict[int, list[np.ndarray]] = {}
    for sid, q in run.qtable_dirs.items():
        pols = []
        for i in range(run.n_agents):
            if (q / f"agent_{i}.npy").exists():
                pols.append(np.argmax(np.load(q / f"agent_{i}.npy"), axis=-1))
            elif (q / f"agent_{i}_prefs.npy").exists():
                pols.append(np.argmax(np.load(q / f"agent_{i}_prefs.npy"), axis=-1))
        if len(pols) == run.n_agents:
            out[sid] = pols
    return out


# ---- statistics ------------------------------------------------------------------------


def session_stats(run: Run) -> dict[str, Any]:
    """The summary numbers of one run, as the result tables print them."""
    s = run.sessions
    d = s["delta_c"].to_numpy(dtype=float)
    lo, hi = viz.bootstrap_ci(d)
    conv = s["converged"].astype(bool)
    has_diag = (
        bool(s["diagnostics_enabled"].astype(bool).any())
        if "diagnostics_enabled" in s
        else "settled" in s
    )
    sett = (
        s["settled"].astype(bool) if "settled" in s else pd.Series(False, index=s.index)
    )
    gap = (
        np.abs(s["delta_c_agent_0"] - s["delta_c_agent_1"]).mean()
        if "delta_c_agent_1" in s
        else math.nan
    )
    out: dict[str, Any] = {
        "sessions": len(s),
        "converged": int(conv.sum()),
        "settled": int(sett.sum()) if has_diag else None,
        "settled_text": str(int(sett.sum())) if has_diag else "–",
        "converged_or_settled": int((conv | sett).sum()),
        "delta_c_mean": float(np.mean(d)),
        "delta_c_lo": lo,
        "delta_c_hi": hi,
        "delta_c_median": float(np.median(d)),
        "share_positive": float(np.mean(d > 0)),
        "chi_hat": float(s["chi_hat"].mean()),
        "profit_gain": float(s["profit_gain"].mean()),
        "profit_pct": 100.0 * (float(s["profit_gain"].mean()) - 1.0),
        "agent_gap": float(gap),
        "periods_median": float(s["periods_to_convergence"].median()),
        "hours_median": (
            float(s["wall_clock_seconds"].median()) / 3600.0
            if "wall_clock_seconds" in s
            else math.nan
        ),
        "hours_total": (
            float(s["wall_clock_seconds"].sum()) / 3600.0
            if "wall_clock_seconds" in s
            else math.nan
        ),
        "dev_gain_mean": math.nan,
        "dev_gain_lo": math.nan,
        "dev_gain_hi": math.nan,
        "dev_pays": None,
        "dev_gain_period": math.nan,
        "dev_gain_cont": math.nan,
        "stale_err_dev": math.nan,
        "stale_err_chosen": math.nan,
    }
    for i in range(run.n_agents):
        out[f"delta_c_agent_{i}"] = float(s[f"delta_c_agent_{i}"].mean())
        col = f"policy_concentration_agent_{i}"
        out[f"concentration_agent_{i}"] = float(s[col].mean()) if col in s else math.nan
        col = f"chi_hat_agent_{i}"
        out[f"chi_hat_agent_{i}"] = float(s[col].mean()) if col in s else math.nan
    if "avg_profit_agent_0" in s and "avg_profit_agent_1" in s:
        p0 = s["avg_profit_agent_0"].to_numpy(dtype=float)
        p1 = s["avg_profit_agent_1"].to_numpy(dtype=float)
        out["agent_0_ahead"] = int((p0 > p1).sum())
        out["agent_1_ahead"] = int((p1 > p0).sum())
    if run.deviation_trials is not None and not run.deviation_trials.empty:
        br = run.deviation_trials[run.deviation_trials["kind"] == "grid_br"]
        g = br.groupby("session_id")["total"]
        per, se = g.mean(), g.std() / np.sqrt(g.count())
        dlo, dhi = viz.bootstrap_ci(per.to_numpy())
        out.update(
            dev_gain_mean=float(per.mean()),
            dev_gain_lo=dlo,
            dev_gain_hi=dhi,
            dev_pays=int((per - 2 * se > 0).sum()),
            dev_gain_period=float(br.groupby("session_id")["gain0"].mean().mean()),
            dev_gain_cont=float(br.groupby("session_id")["cont"].mean().mean()),
        )
        # Value staleness: stored value minus its one-step target, pooled over the
        # trials whose deviator keeps a Q-table (as scripts/deviation_test.py, part 3).
        if "q_dev" in br and br["q_dev"].notna().any():
            out["stale_err_dev"] = float((br["q_dev"] - br["tgt_dev"]).mean())
            out["stale_err_chosen"] = float((br["q_greedy"] - br["tgt_greedy"]).mean())
    return out


def random_play_delta_c(
    action_table: np.ndarray,
    v_grid: np.ndarray,
    lambda_n: float,
    pi_n: float,
    pi_m: float,
    v_bar: float,
) -> float:
    """Δ^C when both agents draw every grid action with equal probability.

    Expected profit per agent, every value state weighted equally and the noise at
    its mean of zero (the convention of ``benchmarks.grid_stage_game``).
    """
    table = np.asarray(action_table, dtype=np.float64)
    v_arr = np.asarray(v_grid, dtype=np.float64)
    means = []
    for k in range(table.shape[0]):
        x = table[k]
        payoff = (v_arr[k] - v_bar - lambda_n * (x[:, None] + x[None, :])) * x[:, None]
        means.append(float(payoff.mean()))
    return (float(np.mean(means)) - pi_n) / (pi_m - pi_n)


def best_reply_to_nash_delta_c(
    action_table: np.ndarray,
    v_grid: np.ndarray,
    chi_n: float,
    lambda_n: float,
    pi_n: float,
    pi_m: float,
    v_bar: float,
) -> float:
    """Δ^C of one agent that plays its best grid action against a fixed Nash player.

    The fixed player trades ``chi_n * (v - v_bar)``; same weighting and noise
    convention as ``random_play_delta_c``.
    """
    table = np.asarray(action_table, dtype=np.float64)
    v_arr = np.asarray(v_grid, dtype=np.float64)
    best = []
    for k in range(table.shape[0]):
        x = table[k]
        x_opp = chi_n * (v_arr[k] - v_bar)
        best.append(float(((v_arr[k] - v_bar - lambda_n * (x + x_opp)) * x).max()))
    return (float(np.mean(best)) - pi_n) / (pi_m - pi_n)


def agent_tags(run: Run) -> list[str] | None:
    """Short names of the two agents when they differ (learner type or α), else None."""
    agents = run.config.get("agents") or []
    if len(agents) != 2:
        return None
    types = [str(a.get("type")) for a in agents]
    if types[0] != types[1]:
        return [AGENT_TYPE_TAGS.get(ty, viz.tex_escape(ty)) for ty in types]
    alphas = [
        (a.get("config") or {}).get("alpha", run.config.get("alpha")) for a in agents
    ]
    if None not in alphas and float(alphas[0]) != float(alphas[1]):
        return [rf"$\alpha = {float(a):g}$" for a in alphas]
    return None


def _no_negative_zero(text: str) -> str:
    """``-0.00`` (a rounded tiny negative number) as ``0.00``."""
    return text[1:] if re.fullmatch(r"-0(\.0+)?", text) else text


def _tex_minus(text: str) -> str:
    """Typeset the hyphen in front of a number as a minus sign (table cells)."""
    return re.sub(r"(?<![\w$^{}\\.-])-(?=\d)", "$-$", text)


def bootstrap_diff_ci(
    a: Any, b: Any, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of ``a`` minus mean of ``b`` with a percentile bootstrap 95% CI.

    The samples are independent (sessions of two different runs): each draw
    resamples ``a`` and ``b`` separately with replacement. Companion of
    ``viz.bootstrap_ci`` (kept here because ``src/`` is frozen).

    Returns:
        ``(diff, lo, hi)``; ``(diff, nan, nan)`` if either sample has fewer
        than two finite values.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    diff = float(x.mean() - y.mean()) if x.size and y.size else math.nan
    if x.size < 2 or y.size < 2:
        return diff, math.nan, math.nan
    rng = np.random.default_rng(seed)
    mx = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    my = y[rng.integers(0, y.size, size=(n_boot, y.size))].mean(axis=1)
    d = mx - my
    return diff, float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def add_diff_macros(out: Outputs, runs: dict[str, Run]) -> None:
    """Macros for the Delta^C differences in ``DIFF_PAIRS`` (both runs present)."""
    for ka, kb in DIFF_PAIRS:
        if ka not in runs or kb not in runs:
            continue
        diff, lo, hi = bootstrap_diff_ci(
            runs[ka].sessions["delta_c"].to_numpy(dtype=float),
            runs[kb].sessions["delta_c"].to_numpy(dtype=float),
        )
        name = f"DeltaCDiff{viz.macro_name(ka)}{viz.macro_name(kb)}"
        out.macro(f"{name}Mean", viz.fmt(diff))
        out.macro(f"{name}Lo", viz.fmt(lo))
        out.macro(f"{name}Hi", viz.fmt(hi))


def _sci(x: float) -> str:
    """``$m\\times10^{e}$`` with one significant digit in the mantissa."""
    if x == 0 or not math.isfinite(x):
        return viz.fmt(x)
    e = math.floor(math.log10(abs(x)))
    m = x / 10**e
    m_txt = f"{m:.0f}" if abs(m - round(m)) < 1e-9 else f"{m:.1f}"
    return f"${m_txt}\\times10^{{{e}}}$"


def _ci(mean: float, lo: float, hi: float, decimals: int = 2) -> str:
    if not math.isfinite(mean):
        return ""
    return (
        f"{viz.fmt(mean, decimals)} [{viz.fmt(lo, decimals)}, {viz.fmt(hi, decimals)}]"
    )


def relabel_deviation_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Add ``dx_dev``, ``dx_oth``, ``dprof_dev``, ``dprof_oth`` for a two-agent run."""
    if pairs.empty:
        return pairs
    p = pairs.copy()
    j = p["deviating_agent"].to_numpy().astype(int)
    for kind in ("dx", "dprof"):
        a0 = p[f"{kind}_0"].to_numpy()
        a1 = p[f"{kind}_1"].to_numpy()
        p[f"{kind}_dev"] = np.where(j == 0, a0, a1)
        p[f"{kind}_oth"] = np.where(j == 0, a1, a0)
    return p


def deviation_pairs_kept(
    run: Run, keep_sessions: set[int] | None = None
) -> pd.DataFrame:
    """Deviation IRF pairs: effective trials, visited post-deviation state, kept sessions."""
    pairs = viz.irf_pairs(run.periods, "deviation", run.v_bar, run.n_agents)
    if pairs.empty:
        return pairs
    pairs = viz.effective_deviation_trials(pairs, run.pre)
    pairs = viz.visited_trials(pairs, run.pre, run.n_agents)
    if keep_sessions is not None:
        pairs = pairs[pairs["session_id"].isin(keep_sessions)]
    return relabel_deviation_pairs(pairs)


# ---- outputs ---------------------------------------------------------------------------


class Outputs:
    def __init__(self, fig_dir: Path, tab_dir: Path) -> None:
        self.fig_dir, self.tab_dir = fig_dir, tab_dir
        fig_dir.mkdir(parents=True, exist_ok=True)
        tab_dir.mkdir(parents=True, exist_ok=True)
        self.written: list[str] = []
        self.skipped: list[tuple[str, str]] = []
        self.macros: dict[str, str] = {}

    def figure(self, name: str, fig: Any) -> None:
        path = self.fig_dir / name
        fig.savefig(path, bbox_inches="tight")
        viz.plt.close(fig)
        self.written.append(str(path))

    def table(self, name: str, text: str) -> None:
        path = self.tab_dir / name
        if name.startswith("tab_"):
            text = _tex_minus(text)
        path.write_text(text)
        self.written.append(str(path))

    def skip(self, name: str, reason: str) -> None:
        self.skipped.append((name, reason))
        logger.warning("skipped %s: %s", name, reason)

    def macro(self, name: str, value: str) -> None:
        self.macros[name] = _no_negative_zero(value)


def add_run_macros(
    out: Outputs, run: Run, st: dict[str, Any], grid_nash: float | None
) -> None:
    k = viz.macro_name(run.key)
    out.macro(f"Sessions{k}", str(st["sessions"]))
    out.macro(f"Converged{k}", str(st["converged"]))
    out.macro(f"Settled{k}", st["settled_text"])
    out.macro(f"ConvergedOrSettled{k}", str(st["converged_or_settled"]))
    out.macro(f"DeltaC{k}Mean", viz.fmt(st["delta_c_mean"]))
    out.macro(f"DeltaC{k}CIlo", viz.fmt(st["delta_c_lo"]))
    out.macro(f"DeltaC{k}CIhi", viz.fmt(st["delta_c_hi"]))
    out.macro(f"DeltaC{k}Median", viz.fmt(st["delta_c_median"]))
    out.macro(f"SharePositive{k}", viz.fmt(100 * st["share_positive"], 0))
    out.macro(f"ChiHat{k}", viz.fmt(st["chi_hat"], 1))
    out.macro(f"ProfitGain{k}", viz.fmt(st["profit_gain"], 3))
    out.macro(f"ProfitPct{k}", viz.fmt(st["profit_pct"], 1))
    for i in range(run.n_agents):
        if f"agent_{i}_ahead" in st:
            out.macro(
                f"Agent{viz.macro_name(str(i))}Ahead{k}", str(st[f"agent_{i}_ahead"])
            )
    out.macro(f"AgentGap{k}", viz.fmt(st["agent_gap"]))
    out.macro(f"PeriodsMedian{k}", viz.fmt(st["periods_median"] / 1e6, 1))
    if math.isfinite(st.get("hours_median", math.nan)):
        out.macro(f"MinutesPerSession{k}", viz.fmt(60 * st["hours_median"], 0))
        out.macro(f"HoursPerSession{k}", viz.fmt(st["hours_median"], 1))
        out.macro(f"HoursTotal{k}", viz.fmt(st["hours_total"], 0))
    for i in range(run.n_agents):
        out.macro(
            f"DeltaC{k}Agent{viz.macro_name(str(i))}", viz.fmt(st[f"delta_c_agent_{i}"])
        )
        if math.isfinite(st[f"chi_hat_agent_{i}"]):
            out.macro(
                f"ChiHat{k}Agent{viz.macro_name(str(i))}",
                viz.fmt(st[f"chi_hat_agent_{i}"], 1),
            )
        if math.isfinite(st[f"concentration_agent_{i}"]):
            out.macro(
                f"Concentration{k}Agent{viz.macro_name(str(i))}",
                viz.fmt(st[f"concentration_agent_{i}"]),
            )
    if math.isfinite(st["dev_gain_mean"]):
        out.macro(f"DevGain{k}Mean", viz.fmt(st["dev_gain_mean"]))
        out.macro(f"DevGain{k}CIlo", viz.fmt(st["dev_gain_lo"]))
        out.macro(f"DevGain{k}CIhi", viz.fmt(st["dev_gain_hi"]))
        out.macro(f"DevPays{k}", str(st["dev_pays"]))
        out.macro(f"DevGainPeriod{k}", viz.fmt(st["dev_gain_period"]))
        out.macro(f"DevGainCont{k}", viz.fmt(st["dev_gain_cont"]))
    if math.isfinite(st["stale_err_dev"]):
        out.macro(f"StaleErrDev{k}", viz.fmt(st["stale_err_dev"], 1))
        out.macro(f"StaleErrChosen{k}", viz.fmt(st["stale_err_chosen"], 1))
    if grid_nash is not None:
        out.macro(f"GridNashDeltaC{k}", viz.fmt(grid_nash, 3))
        out.macro(f"DeltaC{k}AboveGridNash", viz.fmt(st["delta_c_mean"] - grid_nash))


def _settled_legend_above(ax: Any) -> None:
    """Legend for filled and hollow session markers, above the axes (never on data)."""
    old = ax.get_legend()
    if old is not None:
        old.remove()
    common = {"marker": "o", "linestyle": "none", "markeredgecolor": viz.ACCENT}
    handles = [
        viz.plt.Line2D(
            [], [], markerfacecolor=viz.ACCENT, label="settled session", **common
        ),
        viz.plt.Line2D([], [], markerfacecolor="white", label="not settled", **common),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=2,
        borderaxespad=0.0,
        handletextpad=0.2,
        columnspacing=1.5,
    )


def make_f1(out: Outputs) -> None:
    """fig_timing.pdf: one period (schematic after DGJ's protocol, Section 4.1)."""
    steps = [
        ("1  Value", "speculators\nobserve $v_t$"),
        ("2  Orders", "speculator $i$: $x_{i,t}$\nnoise trader: $u_t$"),
        (
            "3  Price",
            "market maker sees\n$y_t = x_{1,t} + x_{2,t} + u_t$\nand sets $p_t$",
        ),
        ("4  Payoff", "investors trade $z_t$\nprofit: $(v_t - p_t)\\,x_{i,t}$"),
        ("5  Next period", "state becomes\n$(p_t,\\ v_t,\\ v_{t+1})$"),
    ]
    fig, ax = viz.new_figure(4.2)
    ax.set_axis_off()
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 30.0)
    width, gap, y0, height = 17.6, 3.0, 8.0, 13.0
    for j, (title, body) in enumerate(steps):
        x0 = j * (width + gap)
        ax.add_patch(
            viz.plt.Rectangle(
                (x0, y0), width, height, fill=False, edgecolor=viz.BLACK, linewidth=0.8
            )
        )
        ax.text(
            x0 + width / 2,
            y0 + height + 1.5,
            title,
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.text(
            x0 + width / 2,
            y0 + height / 2,
            body,
            ha="center",
            va="center",
            fontsize=8,
            linespacing=1.5,
        )
        if j < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x0 + width + gap, y0 + height / 2),
                xytext=(x0 + width, y0 + height / 2),
                arrowprops={"arrowstyle": "-|>", "color": viz.BLACK, "linewidth": 0.8},
            )
    split = 4 * (width + gap) - gap / 2
    for a, b, label in (
        (0.0, split - 1.0, "period $t$"),
        (split + 1.0, 100.0, "period $t+1$"),
    ):
        ax.plot([a, b], [4.5, 4.5], color=viz.GREY, linewidth=0.8)
        for x in (a, b):
            ax.plot([x, x], [4.5, 5.8], color=viz.GREY, linewidth=0.8)
        ax.text(
            (a + b) / 2, 3.6, label, ha="center", va="top", fontsize=9, color=viz.GREY
        )
    fig.tight_layout()
    out.figure("fig_timing.pdf", fig)


def _label_grid_nash_run(fig: Any, label: str = "grid Nash R1") -> None:
    """Name the run whose action grid the grid Nash line belongs to.

    In the figures of the replication results the line is that of the baseline
    specification (R1, R1.3); the earlier specification (R1.2) has its own value,
    given in tab_sensitivity.tex.
    """
    for ax in fig.axes:
        for txt in ax.texts:
            if txt.get_text() == "grid Nash":
                txt.set_text(label)


def _converged_or_settled(run: Run) -> set[int]:
    """Sessions that converged or settled: the sample of the forced-deviation figures."""
    s = run.sessions
    return set(
        s.loc[
            s["converged"].astype(bool)
            | s.get("settled", pd.Series(False, index=s.index)).astype(bool),
            "session_id",
        ].astype(int)
    )


def make_f2(out: Outputs, main: Run, grid_nash: float | None) -> None:
    if main.trajectory.empty:
        out.skip("fig_learning_curves.pdf", "baseline run has no trajectory.parquet")
        return
    fig = viz.fig_learning_curves(main.trajectory, main.sessions, grid_nash)
    _label_grid_nash_run(fig)
    out.figure("fig_learning_curves.pdf", fig)


def make_f3(out: Outputs, runs: dict[str, Run], grid_nash: float | None) -> None:
    groups = [
        (runs[k].label, runs[k].sessions)
        for k in ("earlier_spec", "baseline", "rounded_price")
        if k in runs
    ]
    if not groups:
        out.skip(
            "fig_delta_c_distribution.pdf",
            "none of earlier_spec, baseline, rounded_price present",
        )
        return
    fig = viz.fig_delta_c_distribution(groups, grid_nash)
    _label_grid_nash_run(fig)
    _settled_legend_above(fig.axes[0])
    out.figure("fig_delta_c_distribution.pdf", fig)


def make_f4(out: Outputs, main: Run, bench: dict[str, float] | None) -> None:
    g = grids_of(main)
    pols = greedy_policies(main)
    if g is None or not pols or bench is None or main.periods.empty:
        out.skip(
            "fig_policy.pdf", "baseline run lacks saved tables, eval rows or benchmarks"
        )
        return
    visits = viz.eval_state_visits(main.periods, g["v_grid"], main.v_bar)
    policy = viz.policy_by_price_bin(
        visits, pols, g["action_table"], g["v_grid"], main.v_bar
    )
    if policy.empty:
        out.skip("fig_policy.pdf", "no on-path state with v != v_bar")
        return
    out.figure(
        "fig_policy.pdf",
        viz.fig_policy(
            policy,
            float(bench["chi_N"]),
            float(bench["chi_M"]),
            g["price_grid"],
            main.v_bar,
        ),
    )
    for lv, blk in policy.groupby("abs_dv"):
        w = blk.groupby("session_id").apply(
            lambda b: np.average(b["chi"], weights=b["visits"]), include_groups=False
        )
        out.macro(
            f"PolicyChiBaseline{viz.macro_name(f'{lv:.3f}'.replace('.', 'p'))}",
            viz.fmt(float(w.mean()), 1),
        )


def make_f5(out: Outputs, main: Run) -> None:
    pairs = deviation_pairs_kept(main, _converged_or_settled(main))
    if pairs.empty:
        out.skip(
            "fig_irf_deviation.pdf",
            "baseline run has no effective, visited deviation trials",
        )
        return
    by_size = {float(m): blk for m, blk in pairs.groupby("shock_magnitude")}
    out.figure(
        "fig_irf_deviation.pdf", viz.fig_irf_deviation(by_size, main.pre, main.post)
    )
    for m, blk in by_size.items():
        prof = viz.irf_profile(blk, "dx_oth", [main.pre + 1]).iloc[0]
        tag = viz.macro_name(f"{m:g}")
        out.macro(f"IrfDevOtherOFBaseline{tag}", viz.fmt(100 * prof["mean"], 2))
        out.macro(f"IrfDevTrialsBaseline{tag}", str(viz.n_trials(blk)))
        # Profit in the deviation period, in percent of the eval mean: the fall of the
        # other speculator's profit and the gain of the deviator (both as positive numbers).
        for col, name, sign in (
            ("dprof_oth", "OtherProfitFall", -1.0),
            ("dprof_dev", "OwnProfitGain", 1.0),
        ):
            p0 = viz.irf_profile(blk, col, [main.pre]).iloc[0]
            out.macro(f"IrfDev{name}Baseline{tag}", viz.fmt(sign * 100 * p0["mean"], 1))
        # A response is possible only where the deviation moves the lagged price into
        # another bin: share of such trials, and the response of the other speculator
        # in them (the price bin is coarse: a small deviation mostly leaves the bin).
        keys = [k for k in viz.IRF_PAIR_KEYS if k != "irf_period"]
        at = blk[blk["irf_period"] == main.pre]
        moved = at.loc[at["p_bin_s"] != at["p_bin_b"], keys]
        out.macro(
            f"IrfDevBinMovedBaseline{tag}",
            viz.fmt(100 * len(moved) / max(len(at), 1), 0),
        )
        seen = blk.merge(moved, on=keys, how="inner")
        if not seen.empty:
            ps = viz.irf_profile(seen, "dx_oth", [main.pre + 1]).iloc[0]
            out.macro(f"IrfDevOtherOFSeenBaseline{tag}", viz.fmt(100 * ps["mean"], 1))


def make_f5b(out: Outputs, main: Run) -> None:
    pairs = viz.irf_pairs(main.periods, "noise", main.v_bar, main.n_agents)
    if pairs.empty:
        out.skip("fig_irf_noise.pdf", "baseline run has no noise-shock IRF rows")
        return
    all_n = {float(m): viz.n_trials(blk) for m, blk in pairs.groupby("shock_magnitude")}
    pairs = viz.visited_trials(pairs, main.pre, main.n_agents)
    by_size = {float(m): blk for m, blk in pairs.groupby("shock_magnitude")}
    if not by_size:
        out.skip("fig_irf_noise.pdf", "no visited-state noise trials")
        return
    out.figure(
        "fig_irf_noise.pdf",
        viz.fig_irf_noise(by_size, main.pre, main.post, main.n_agents),
    )
    triggers = []
    for m, blk in by_size.items():
        tag = viz.macro_name(f"{m:g}".replace(".", "p"))
        out.macro(f"IrfNoiseTrialsKeptBaseline{tag}", f"{viz.n_trials(blk)}/{all_n[m]}")
        # Sessions DGJ's rule classifies as "price trigger" (visited states only).
        cols = [f"dx_{i}" for i in range(main.n_agents)]
        nxt = blk[blk["irf_period"] == main.pre + 1]
        per = nxt.groupby("session_id")[cols].mean()
        triggers.append(int((per > X_UPPER).all(axis=1).sum()))
        out.macro(f"IrfNoiseTriggerBaseline{tag}", str(triggers[-1]))
        for i in range(main.n_agents):
            prof = viz.irf_profile(blk, f"dx_{i}", [main.pre + 1]).iloc[0]
            out.macro(
                f"IrfNoiseOFBaseline{tag}Agent{viz.macro_name(str(i))}",
                viz.fmt(100 * prof["mean"], 2),
            )
    out.macro("IrfNoiseTriggerBaselineMin", str(min(triggers)))
    out.macro("IrfNoiseTriggerBaselineMax", str(max(triggers)))


def make_f6(out: Outputs, runs: dict[str, Run], grid_nash: float | None) -> None:
    pops = [k for k in ("baseline", "q_ac", "ac_ac") if k in runs]
    if len(pops) < 2:
        out.skip(
            "fig_populations.pdf",
            "needs the baseline run and at least one of q_ac, ac_ac",
        )
        return
    groups: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    for k in pops:
        s = runs[k].sessions
        groups.append(
            (
                POP_LABELS[k],
                s["delta_c"].to_numpy(float),
                s["settled"].astype(bool).to_numpy() if "settled" in s else None,
            )
        )
    if "q_ac" in runs:
        s = runs["q_ac"].sessions
        mask = s["settled"].astype(bool).to_numpy() if "settled" in s else None
        groups.append(("Q–AC:\nQ-learner", s["delta_c_agent_0"].to_numpy(float), mask))
        groups.append(("Q–AC:\nAC learner", s["delta_c_agent_1"].to_numpy(float), mask))
    fig = viz.fig_populations(groups, grid_nash)
    _settled_legend_above(fig.axes[0])
    out.figure("fig_populations.pdf", fig)


def make_f7(out: Outputs, runs: dict[str, Run]) -> None:
    pops = [k for k in ("baseline", "q_ac", "ac_ac") if k in runs]
    if len(pops) < 2:
        out.skip(
            "fig_populations_irf.pdf",
            "needs the baseline run and at least one of q_ac, ac_ac",
        )
        return
    bars: list[tuple[str, float, float, float, int]] = []
    for k in pops:
        run = runs[k]
        # The same sessions as fig_irf_deviation.pdf (converged or settled), so that the
        # baseline shows the same number in both; all 30 sessions of Q–AC and AC–AC
        # qualify.
        pairs = deviation_pairs_kept(run, _converged_or_settled(run))
        if pairs.empty:
            continue
        smallest = float(pairs["shock_magnitude"].min())
        pairs = pairs[pairs["shock_magnitude"] == smallest]
        splits = [(POP_LABELS[k], "", pairs)]
        if k == "q_ac":
            splits = [
                (
                    "Q–AC (R2):\nQ-learner deviates,\nAC learner responds",
                    "QDeviates",
                    pairs[pairs["deviating_agent"] == 0],
                ),
                (
                    "Q–AC (R2):\nAC learner deviates,\nQ-learner responds",
                    "AcDeviates",
                    pairs[pairs["deviating_agent"] == 1],
                ),
            ]
        for label, tag, blk in splits:
            prof = viz.irf_profile(blk, "dx_oth", [run.pre + 1]).iloc[0]
            bars.append(
                (
                    label,
                    100 * prof["mean"],
                    100 * prof["lo"],
                    100 * prof["hi"],
                    int(prof["n_sessions"]),
                )
            )
            out.macro(
                f"IrfPopOther{viz.macro_name(k)}{tag}",
                viz.fmt(100 * prof["mean"], 2),
            )
            for end in ("lo", "hi"):
                out.macro(
                    f"IrfPopOther{viz.macro_name(k)}{tag}CI{end}",
                    viz.fmt(100 * prof[end], 2),
                )
    if not bars:
        out.skip(
            "fig_populations_irf.pdf", "no deviation trials in the population runs"
        )
        return
    fig = viz.fig_populations_irf(bars)
    # The axis says in the words of the thesis what is measured (the period and the
    # shock stand in the caption and the note, the axis has no room for them); the
    # number of sessions (the converged ones of each population) stands in the note
    # of the figure.
    ax = fig.axes[0]
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylabel("Rise in the other speculator's\norder flow (% of its mean)")
    fig.tight_layout()
    out.figure("fig_populations_irf.pdf", fig)


def _actor_critic_params(qa_dir: Path = Path("results/r2_q_ac")) -> dict[str, Any]:
    """Actor-critic step sizes and clocks of the thesis runs.

    The run R2 writes one ``run_metadata.json`` per session (``<qa_dir>/s<id>/``). The
    first one found supplies ``resolved_config["agents"][1]["config"]`` (the overrides of
    ``config_r2_q_ac.yaml``); keys it does not set, or a missing run, fall back to the defaults
    of ``ActorCriticAgent``, read off a freshly built agent (nothing typed by hand).

    Args:
        qa_dir: Directory of run R2.
    """
    from ai_trading.agents.actor_critic import ActorCriticAgent

    overrides: dict[str, Any] = {}
    meta_files = sorted(qa_dir.rglob("run_metadata.json")) if qa_dir.exists() else []
    if meta_files:
        meta = json.loads(meta_files[0].read_text())
        agents = meta.get("resolved_config", {}).get("agents", [])
        if len(agents) > 1:
            overrides = dict(agents[1].get("config", {}))
        logger.info(
            "T2 actor-critic values from %s (overrides: %s)",
            meta_files[0],
            overrides or "none, code defaults",
        )
    else:
        logger.info("T2 actor-critic values: no run under %s, code defaults", qa_dir)
    beta, pi_m = 1.0e-4, 50.0
    cfg: dict[str, Any] = {
        "rho": 0.95,
        "n_agents": 2,
        "n_v": 2,
        "n_x": 2,
        "n_p": 2,
        "lambda_N": 0.002,
        "pi_M": pi_m,
        "beta": beta,
        "v_bar": 0.0,
        "seed": 0,
        **overrides,
    }
    v_grid = np.array([-1.0, 1.0])
    chi = np.array([100.0, 200.0])
    grids = {
        "v_grid": v_grid,
        "action_table": chi[None, :] * v_grid[:, None],
        "action_grid": chi,
        "price_grid": np.array([-1.0, 1.0]),
    }
    agent = ActorCriticAgent(agent_id=0, config=cfg, grids=grids)
    return {
        "alpha_actor": agent.alpha_actor,
        "alpha_critic": agent.alpha_critic,
        "entropy_coef": agent.entropy_coef,
        "decay_is_beta": agent.entropy_decay == agent.beta,
        "scale_is_pi_m": agent.reward_scale == agent.pi_M,
    }


def _t2_thesis_text(key: str, cfg: dict[str, Any], main: Run) -> str:
    if key == "__n_sessions__":
        return str(len(main.sessions))
    if key not in cfg:
        return ""
    v = cfg[key]
    if key in DESIGN_TEXT:
        return DESIGN_TEXT[key].get(str(v), viz.tex_escape(str(v)))
    if key in ("beta", "max_periods"):
        return _sci(float(v))
    if isinstance(v, int) or float(v).is_integer() and float(v) >= 1000:
        return viz.fmt(v, 0, thousands=True)
    return str(v)


def t2_scale_reasons(stats: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Reasons of the rows ``n_v`` and sessions with the session times of the runs.

    Uses the median session time of the baseline and of the run with DGJ's value grid
    and exploration decay; empty when one of the two runs is missing.
    """
    big, base = stats.get("ref_grid"), stats.get("baseline")
    if not big or not base:
        return {}
    hb, hm = big.get("hours_median", math.nan), base.get("hours_median", math.nan)
    if not (math.isfinite(hb) and math.isfinite(hm)):
        return {}
    return {
        "n_v": (
            "Laptop scale: with DGJ's ten values and exploration decay, a session "
            f"takes about {60 * hb:.0f} minutes instead of {60 * hm:.0f} (run R1.7)."
        ),
        "__n_sessions__": (
            "Laptop scale: 1,000 such sessions would occupy one processor for about "
            f"{1000 * hb / 168:.0f} weeks per run; a run of {base['sessions']} "
            f"sessions takes about {base['hours_total']:.0f} hours."
        ),
    }


def t2_rows(main: Run, reasons: dict[str, str] | None = None) -> list[list[str]]:
    """Rows of tab_parameters.tex: ``[parameter, DGJ, thesis, reason]`` (AC block last).

    ``reasons`` replaces the reason of a row by its config key (``t2_scale_reasons``).
    """
    cfg = main.config
    reasons = reasons or {}
    rows = [
        [label, paper_text, _t2_thesis_text(key, cfg, main), reasons.get(key, reason)]
        for label, key, paper_text, reason in PAPER_VALUES
    ]
    ac = _actor_critic_params()
    rows += [
        [
            r"Actor step size $\alpha_{\mathrm{actor}}$",
            "--",
            f"{ac['alpha_actor']:g}",
            ACTOR_CRITIC_REASON,
        ],
        [
            r"Critic step size $\alpha_{\mathrm{critic}}$",
            "--",
            f"{ac['alpha_critic']:g}",
            ACTOR_CRITIC_REASON,
        ],
        [
            r"Entropy coefficient $\eta_0$",
            "--",
            f"{ac['entropy_coef']:g}",
            ACTOR_CRITIC_REASON,
        ],
        [
            "Entropy decay",
            "--",
            r"$=\beta$" if ac["decay_is_beta"] else "own",
            (
                "Own design: same clock as the Q-learner's $\\varepsilon$, "
                "for comparability."
            ),
        ],
        [
            "Reward scale",
            "--",
            r"$=\pi^M$" if ac["scale_is_pi_m"] else "own",
            "Own design: makes the actor step size unit-free.",
        ],
    ]
    return rows


# Reason column: ragged right. With 5.6 cm the table fits the 15 cm text width and one page
# when the table environment sets it in \small (compile check; 5.8 cm is 2.8 pt too wide;
# at normal size the table is taller than a page).
T2_ALIGN = r"lrr>{\raggedright\arraybackslash}p{5.6cm}"
# The short table (tab_parameters_short.tex) is set in \footnotesize with a wider reason
# column, so that every reason fits on one or two lines.
T2_SHORT_ALIGN = r"lrr>{\raggedright\arraybackslash}p{8.1cm}"


def make_t2(out: Outputs, main: Run, reasons: dict[str, str] | None = None) -> None:
    if not main.config:
        out.skip("tab_parameters.tex", "baseline run has no run_metadata.json")
        return
    rows = t2_rows(main, reasons)
    text = viz.booktabs_fragment(
        ["Parameter", "DGJ", "Thesis", "Reason"], rows, T2_ALIGN
    )
    first_ac = rows[len(PAPER_VALUES)][0]  # the actor-critic block gets its own rule
    text = text.replace(first_ac, "\\midrule\n" + first_ac, 1)
    out.table("tab_parameters.tex", text)
    by_label = {label: key for label, key, *_ in PAPER_VALUES}
    short = [r for r in rows if by_label.get(r[0]) in T2_SHORT_KEYS]
    out.table(
        "tab_parameters_short.tex",
        viz.booktabs_fragment(
            ["Parameter", "DGJ", "Thesis", "Reason"], short, T2_SHORT_ALIGN
        ),
    )


def _q_parameter_thesis_text(key: str, cfg: dict[str, Any], main: Run) -> str:
    """Thesis column of the split parameter table (words where the thesis uses words)."""
    if key in Q_PARAMETER_TEXT and key in cfg:
        return Q_PARAMETER_TEXT[key].get(str(cfg[key]), viz.tex_escape(str(cfg[key])))
    if key == "max_periods" and key in cfg and float(cfg[key]) % 1e6 == 0:
        return f"{float(cfg[key]) / 1e6:.0f} million"
    return _t2_thesis_text(key, cfg, main)


def _random_draws_per_cell(n_x: int, n_p: int, n_v: int, beta: float) -> float:
    """Random draws per state-action cell before exploration ends.

    Dou et al. (2025, p. 23): an action is drawn at random about
    ``n_v / (n_x (1 - exp(-beta)))`` times; a table has ``n_p * n_v * n_v`` states.
    """
    return n_v / (n_x * (1.0 - math.exp(-beta))) / (n_p * n_v * n_v)


def q_parameter_rows(
    main: Run, stats: dict[str, dict[str, Any]] | None = None
) -> list[list[str] | None]:
    """Rows of the baseline half: ``[parameter, DGJ, thesis, reason]``; None = gap.

    The session times in the reasons come from the baseline and from the run with
    DGJ's value grid and exploration decay (as ``t2_scale_reasons``); without those
    runs the sentences with numbers are left out.
    """
    cfg = main.config
    fill: dict[str, str] = {"n_x": str(cfg.get("n_x", ""))}
    if all(k in cfg for k in ("n_x", "n_p", "n_v", "beta")):
        ours = _random_draws_per_cell(
            int(cfg["n_x"]), int(cfg["n_p"]), int(cfg["n_v"]), float(cfg["beta"])
        )
        dgj = _random_draws_per_cell(
            int(cfg["n_x"]), int(cfg["n_p"]), DGJ_N_V, DGJ_BETA
        )
        if abs(ours - dgj) > 0.01 * dgj:
            logger.warning(
                "random draws per cell differ from DGJ: %.1f vs %.1f", ours, dgj
            )
        fill["draws"] = f"{10 * round(ours / 10):.0f}"
    big, base = (stats or {}).get("ref_grid"), (stats or {}).get("baseline")
    if big and base:
        hb, hm = big.get("hours_median", math.nan), base.get("hours_median", math.nan)
        if math.isfinite(hb) and math.isfinite(hm):
            fill.update(
                minutes_big=f"{60 * hb:.0f}",
                minutes_base=f"{60 * hm:.0f}",
                weeks=f"{1000 * hb / 168:.0f}",
                hours=f"{base['hours_total']:.0f}",
            )
    rows: list[list[str] | None] = []
    for entry in Q_PARAMETER_ROWS:
        if entry is None:
            rows.append(None)
            continue
        label, key, dgj_text, reason = entry
        # Plain replacement, not str.format: the reasons carry LaTeX braces.
        for name, value in fill.items():
            reason = reason.replace("{" + name + "}", value)
        if any("{" + name + "}" in reason for name in Q_PARAMETER_FILL):
            # a run is missing: keep the standard, drop the sentence with numbers
            reason = reason.split(":")[0] + "."
        if key == "n_x" and cfg.get("action_grid") != "per_v":
            reason = "DGJ's value. Layout: one grid of order sizes for all values."
        rows.append([label, dgj_text, _q_parameter_thesis_text(key, cfg, main), reason])
    return rows


def ac_parameter_rows() -> list[list[str]]:
    """Rows of the AC half: ``[setting, value, reason]`` (values from the run R2)."""
    ac = _actor_critic_params()
    pilot = (
        "Set before the first run with an AC learner and kept after the pilot "
        r"(Appendix~\ref{app:validation})."
    )
    return [
        ["Step size of the actor", f"{ac['alpha_actor']:g}", pilot],
        [
            "Step size of the critic",
            f"{ac['alpha_critic']:g}",
            "Set before the first run with an AC learner. The pilot did not vary it.",
        ],
        ["Entropy bonus at the start", f"{ac['entropy_coef']:g}", pilot],
        [
            "Decay of the entropy bonus",
            r"$\beta$" if ac["decay_is_beta"] else "own",
            (
                "The bonus fades at the pace at which the exploration of the "
                r"Q-learner fades (Table~\ref{tab:parameters-full}), so that the "
                "policy can narrow to one action."
            ),
        ],
        [
            "Scale of the surprise",
            "cartel profit" if ac["scale_is_pi_m"] else "own",
            (
                "The actor's step acts on the surprise divided by the cartel profit, "
                "so that the step size does not depend on the unit of profit "
                r"(Section~\ref{sec:learners})."
            ),
        ],
        [
            "Initial values",
            "random play",
            (
                "All preferences start at zero, so every action is equally likely "
                "at first. The critic starts at the value of a state when both "
                "agents play at random."
            ),
        ],
    ]


def make_t2_split(
    out: Outputs, main: Run, stats: dict[str, dict[str, Any]] | None = None
) -> None:
    """The parameter table of the appendix in two halves (baseline; AC learner)."""
    if not main.config:
        out.skip("tab_parameters_baseline.tex", "baseline run has no run_metadata.json")
        return
    rows = q_parameter_rows(main, stats)
    text = viz.booktabs_fragment(
        ["Parameter", "DGJ", "Thesis", "Reason"],
        [r for r in rows if r is not None],
        T2Q_ALIGN,
    )
    for i, r in enumerate(rows):  # a small gap where a new group starts
        if r is None:
            first = " & ".join(rows[i + 1])  # type: ignore[arg-type]
            text = text.replace(first, "\\addlinespace\n" + first, 1)
    out.table("tab_parameters_baseline.tex", text)
    out.table(
        "tab_parameters_ac.tex",
        viz.booktabs_fragment(
            ["Setting", "Value", "Reason"], ac_parameter_rows(), T2AC_ALIGN
        ),
    )


def make_t4(
    out: Outputs, runs: dict[str, Run], stats: dict[str, dict[str, Any]]
) -> None:
    keys = [k for k in ("earlier_spec", "baseline") if k in runs]
    if not keys:
        out.skip("tab_replication.tex", "neither m1 nor main present")
        return
    header = [
        "Run",
        "Sessions",
        "Conv.",
        "Settled",
        r"$\Delta^C$ mean [95\% CI]",
        "Median",
        r"Share $>0$",
        r"$\hat\chi$",
        "Profit gain",
        "Agent gap",
        "Periods (M)",
        r"Deviation gain, total [95\% CI]",
        "Pays in",
    ]
    rows = []
    for k in keys:
        st = stats[k]
        rows.append(
            [
                runs[k].label,
                str(st["sessions"]),
                str(st["converged"]),
                st["settled_text"],
                _ci(st["delta_c_mean"], st["delta_c_lo"], st["delta_c_hi"]),
                viz.fmt(st["delta_c_median"]),
                viz.fmt(100 * st["share_positive"], 0) + r"\%",
                viz.fmt(st["chi_hat"], 1),
                viz.fmt(st["profit_gain"], 3),
                viz.fmt(st["agent_gap"]),
                viz.fmt(st["periods_median"] / 1e6, 1),
                _ci(st["dev_gain_mean"], st["dev_gain_lo"], st["dev_gain_hi"]),
                "" if st["dev_pays"] is None else f"{st['dev_pays']}/{st['sessions']}",
            ]
        )
    out.table("tab_replication.tex", viz.booktabs_fragment(header, rows))


def make_t5(
    out: Outputs,
    runs: dict[str, Run],
    stats: dict[str, dict[str, Any]],
    grid_nash: dict[str, float | None],
) -> None:
    keys = [
        k
        for k in (
            "baseline",
            "early_stop",
            "earlier_spec",
            "rounded_price",
            "no_price",
            "no_price_beta3e6",
            "ref_beta",
            "ref_grid",
            "ref_noise",
        )
        if k in runs
    ]
    if not keys:
        out.skip("tab_sensitivity.tex", "no run present")
        return
    # A head on two rows, as in tab_populations.tex, so that "Deviation gain, total"
    # does not widen its column and the table prints larger.
    groups = (
        r" &  &  &  & \multicolumn{2}{c}{Collusion capacity $\Delta^C$}"
        r" &  & Deviation \\"
        "\n"
        r"\cmidrule(lr){5-6}"
    )
    header = [
        "Run",
        "Sessions",
        "Conv.",
        "Settled",
        r"Mean [95\% CI]",
        "Grid Nash",
        r"$\hat\chi$",
        "gain, total",
    ]
    rows = []
    for k in keys:
        st = stats[k]
        rows.append(
            [
                runs[k].label,
                str(st["sessions"]),
                str(st["converged"]),
                st["settled_text"],
                _ci(st["delta_c_mean"], st["delta_c_lo"], st["delta_c_hi"]),
                viz.fmt(grid_nash.get(k), 3),
                viz.fmt(st["chi_hat"], 1),
                # Without the price in the state the gain has no continuation term.
                viz.fmt(st["dev_gain_mean"])
                + (r"$^{a}$" if runs[k].config.get("state_price") == "none" else ""),
            ]
        )
    text = viz.booktabs_fragment(header, rows, align="@{}l" + "r" * 7 + "@{}")
    text = text.replace("\\toprule\n", "\\toprule\n" + groups + "\n", 1)
    out.table("tab_sensitivity.tex", text)


def make_t6(
    out: Outputs, runs: dict[str, Run], stats: dict[str, dict[str, Any]]
) -> None:
    keys = [k for k in ("baseline", "q_ac", "ac_ac", "two_alphas") if k in runs]
    if len(keys) < 2:
        out.skip(
            "tab_populations.tex",
            "needs the baseline run and at least one population run",
        )
        return
    # No column for the trading intensity or the policy concentration, and none for
    # the number of sessions (30 in every row; the note of the table says so), so that
    # the table prints at the font size of its environment instead of being scaled
    # down. The head has two rows: the quantity once, above the columns it covers, so
    # that the symbol stands free.
    groups = (
        r" &  &  & \multicolumn{3}{c}{Collusion capacity $\Delta^C$}"
        r" & \multicolumn{2}{c}{Deviation gain, total} \\"
        "\n"
        r"\cmidrule(lr){4-6} \cmidrule(l){7-8}"
    )
    header = [
        "Population",
        "Conv.",
        "Settled",
        r"Mean [95\% CI]",
        "Agent 0",
        "Agent 1",
        r"Mean [95\% CI]",
        "Pays in",
    ]
    rows = []
    for k in keys:
        st = stats[k]
        # Tags only where the learners differ in type: (Q) and (AC).
        tags = agent_tags(runs[k]) if k == "q_ac" else None
        per_agent = [
            viz.fmt(st.get(f"delta_c_agent_{i}", math.nan))
            + (f" ({tags[i]})" if tags else "")
            for i in range(2)
        ]
        rows.append(
            [
                POP_LABELS.get(k, runs[k].label),
                str(st["converged"]),
                st["settled_text"],
                _ci(st["delta_c_mean"], st["delta_c_lo"], st["delta_c_hi"]),
                per_agent[0],
                per_agent[1],
                _ci(st["dev_gain_mean"], st["dev_gain_lo"], st["dev_gain_hi"]),
                "" if st["dev_pays"] is None else f"{st['dev_pays']}/{st['sessions']}",
            ]
        )
    text = viz.booktabs_fragment(header, rows, align="@{}l" + "r" * 7 + "@{}")
    text = text.replace("\\toprule\n", "\\toprule\n" + groups + "\n", 1)
    out.table("tab_populations.tex", text)


def make_t1(out: Outputs, main: Run, grid_nash: float | None) -> None:
    """tab_benchmarks.tex: closed-form benchmarks at the thesis parameters, and macros.

    The table has no price-impact column: it is 0.0020 in both rows
    (macros ``LambdaNash`` and ``LambdaCartel``), and the table is not DGJ's. The
    lower benchmark is named as the thesis names it; DGJ say "non-collusive".
    """
    b, g = benchmarks_of(main), grids_of(main)
    if b is None or g is None:
        out.skip("tab_benchmarks.tex", "baseline run lacks saved grids or a config")
        return
    cartel_pct = 100.0 * (b["pi_M"] / b["pi_N"] - 1.0)
    header = [
        "Benchmark",
        r"Trading intensity $\chi$",
        r"Profit per agent $\pi$",
        r"Profit above Nash",
        r"$\Delta^C$",
    ]
    rows = [
        [
            "Competitive Nash equilibrium",
            viz.fmt(b["chi_N"]),
            viz.fmt(b["pi_N"]),
            r"0.0\%",
            "0",
        ],
        [
            "Perfect cartel",
            viz.fmt(b["chi_M"]),
            viz.fmt(b["pi_M"]),
            viz.fmt(cartel_pct, 1) + r"\%",
            "1",
        ],
    ]
    out.table("tab_benchmarks.tex", viz.booktabs_fragment(header, rows))
    out.macro("ChiNash", viz.fmt(b["chi_N"]))
    out.macro("ChiCartel", viz.fmt(b["chi_M"]))
    out.macro("LambdaNash", viz.fmt(b["lambda_N"], 4))
    out.macro("LambdaCartel", viz.fmt(b["lambda_M"], 4))
    out.macro("ProfitNash", viz.fmt(b["pi_N"]))
    out.macro("ProfitCartel", viz.fmt(b["pi_M"]))
    out.macro("ProfitPctCartel", viz.fmt(cartel_pct, 1))
    # DGJ's reported level, and the same level converted with the closed-form
    # benchmarks (the cartel gain in percent is the same at sigma_u = 0.1 and 1.0).
    out.macro("DeltaCDgj", viz.fmt(DGJ_DELTA_C))
    out.macro("ProfitPctDgj", str(DGJ_PROFIT_PCT))
    out.macro("ProfitPctDgjImplied", viz.fmt(DGJ_DELTA_C * cartel_pct, 1))
    out.macro(
        "RandomPlayDeltaCBaseline",
        viz.fmt(
            random_play_delta_c(
                g["action_table"],
                g["v_grid"],
                b["lambda_N"],
                b["pi_N"],
                b["pi_M"],
                main.v_bar,
            )
        ),
    )


def make_t3(out: Outputs, main: Run | None) -> None:
    """tab_validation.tex: one learner against a fixed Nash player."""
    # No column that is the same in both rows (4 sessions, the Nash intensity, the
    # best reply on the grid; the thesis text gives them from the macros), so that the
    # table prints at the font size of its environment instead of being scaled down.
    header = [
        "Learner",
        "Converged",
        r"$\Delta^C$, mean [min, max]",
        r"Trading intensity $\hat\chi$",
    ]
    rows = []
    for key, (label, path) in VALIDATION_RUNS.items():
        run = load_run(key, Path(path))
        if run is None:
            continue
        s = run.sessions
        d = s["delta_c_agent_0"].to_numpy(dtype=float)
        b, g = benchmarks_of(run), grids_of(run)
        if b is None or g is None:
            out.skip("tab_validation.tex", f"run {key} lacks saved grids or a config")
            return
        reply = best_reply_to_nash_delta_c(
            g["action_table"],
            g["v_grid"],
            b["chi_N"],
            b["lambda_N"],
            b["pi_N"],
            b["pi_M"],
            run.v_bar,
        )
        chi = float(s["chi_hat_agent_0"].mean())
        rows.append(
            [
                label,
                f"{int(s['converged'].astype(bool).sum())}/{len(s)}",
                _ci(float(d.mean()), float(d.min()), float(d.max())),
                viz.fmt(chi, 1),
            ]
        )
        k = viz.macro_name(key)
        out.macro(f"Sessions{k}", str(len(s)))
        out.macro(f"Converged{k}", str(int(s["converged"].astype(bool).sum())))
        out.macro(f"DeltaCLearner{k}", viz.fmt(float(d.mean())))
        out.macro(f"ChiHatLearner{k}", viz.fmt(chi, 1))
        out.macro("BestReplyToNashDeltaC", _no_negative_zero(viz.fmt(reply, 3)))
    if not rows:
        out.skip("tab_validation.tex", "no validation run present")
        return
    out.table(
        "tab_validation.tex",
        viz.booktabs_fragment(header, rows, align="@{}lrrr@{}"),
    )


def make_t7(
    out: Outputs, runs: dict[str, Run], stats: dict[str, dict[str, Any]]
) -> None:
    """tab_deviation.tex: the deviation test in two terms, and the value staleness."""
    keys = [
        k
        for k in DEVIATION_KEYS
        if k in runs and math.isfinite(stats[k]["dev_gain_mean"])
    ]
    if not keys:
        out.skip("tab_deviation.tex", "no run with deviation trials")
        return
    # A head on two rows, the test once above the columns it covers, and short row
    # labels, so that the table prints at the font size of its environment instead of
    # being scaled down (as tab_populations.tex).
    groups = (
        r" & \multicolumn{4}{c}{Deviation test}"
        r" & \multicolumn{2}{c}{Value error, by action} \\"
        "\n"
        r"\cmidrule(lr){2-5} \cmidrule(l){6-7}"
    )
    header = [
        "Run",
        "Gain",
        "Continuation",
        r"Total [95\% CI]",
        "Pays in",
        "Deviating",
        "Chosen",
    ]
    rows = []
    for k in keys:
        st = stats[k]
        stale = math.isfinite(st["stale_err_dev"])
        rows.append(
            [
                DEVIATION_LABELS.get(k, runs[k].label),
                viz.fmt(st["dev_gain_period"]),
                viz.fmt(st["dev_gain_cont"]),
                _ci(st["dev_gain_mean"], st["dev_gain_lo"], st["dev_gain_hi"]),
                f"{st['dev_pays']}/{st['sessions']}",
                viz.fmt(st["stale_err_dev"], 1) if stale else "–",
                _no_negative_zero(viz.fmt(st["stale_err_chosen"], 1)) if stale else "–",
            ]
        )
    text = viz.booktabs_fragment(header, rows, align="@{}l" + "r" * 6 + "@{}")
    text = text.replace("\\toprule\n", "\\toprule\n" + groups + "\n", 1)
    out.table("tab_deviation.tex", text)
    # Units by which the stored value of the deviation lies below its target, per |v|.
    # For Q-AC the mean covers the Q-learner only: trials
    # with the AC learner as deviator have no stored value (NaN) and drop out.
    for key, prefix in (("baseline", "Baseline"), ("q_ac", "QAc")):
        run = runs.get(key)
        if run is None or run.deviation_trials is None:
            continue
        br = run.deviation_trials[run.deviation_trials["kind"] == "grid_br"]
        for lv, blk in br.groupby("abs_dv"):
            tag = viz.macro_name(f"{lv:.3f}".replace(".", "p"))
            out.macro(
                f"StaleGap{prefix}{tag}",
                viz.fmt(float((blk["tgt_dev"] - blk["q_dev"]).mean()), 0),
            )


def _run_cfg_value(run: Run, key: str) -> Any:
    """A config value of a run: the ``simulation`` block first, then the top level."""
    cfg = run.config
    sim = cfg.get("simulation", {}) if isinstance(cfg.get("simulation"), dict) else {}
    return sim.get(key, cfg.get(key))


def _run_overview_text(run: Run, template: str) -> str:
    """Fill the placeholders of a ``RUN_OVERVIEW`` description from the run's config."""
    values: dict[str, str] = {}
    for key in ("beta", "n_v", "sigma_u"):
        v = _run_cfg_value(run, key)
        if v is None:
            continue
        values[key] = _sci(float(v)).strip("$") if key == "beta" else f"{v:g}"
    for key, name in (("convergence_periods", "conv"), ("max_periods", "cap")):
        v = _run_cfg_value(run, key)
        if v is None and name == "cap":  # runs without metadata: the longest session
            v = int(run.sessions["periods_to_convergence"].max())
        if v is not None:
            v = float(v)
            values[name] = (
                f"{v / 1e6:g} million" if v >= 1e6 else viz.fmt(v, 0, thousands=True)
            )
    try:
        return template.format(**values)
    except KeyError:
        return template


def make_t8(
    out: Outputs, runs: dict[str, Run], stats: dict[str, dict[str, Any]]
) -> None:
    """tab_runs.tex: overview of all runs, grouped by hypothesis."""
    if "baseline" not in runs:
        out.skip("tab_runs.tex", "baseline run missing")
        return
    lines = [
        r"\begin{tabular}{llp{8.6cm}lr}",
        r"\toprule",
        r"Run & Population & Change against the baseline specification & Role & Sessions \\",
    ]
    n_runs = n_sessions = 0
    hours = 0.0
    for group, entries in RUN_OVERVIEW:
        present = [e for e in entries if e[0] in runs]
        if not present:
            continue
        lines += [r"\midrule", rf"\multicolumn{{5}}{{l}}{{{group}}} \\"]
        for key, label, population, role, template in present:
            st = stats[key]
            n_runs += 1
            n_sessions += st["sessions"]
            if math.isfinite(st.get("hours_total", math.nan)):
                hours += st["hours_total"]
            lines.append(
                " & ".join(
                    [
                        label,
                        population,
                        _run_overview_text(runs[key], template),
                        role,
                        str(st["sessions"]),
                    ]
                )
                + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.table("tab_runs.tex", "\n".join(lines) + "\n")
    out.macro("RunsTotal", str(n_runs))
    out.macro("SessionsTotal", str(n_sessions))
    out.macro("HoursTotalAllRuns", viz.fmt(hours, 0))


def add_scale_macros(out: Outputs, stats: dict[str, dict[str, Any]]) -> None:
    r"""What DGJ's scale would cost on the laptop: macros for the method chapter.

    ``\DgjScaleRunWeeks``: DGJ's 1,000 sessions at the median session time of the run
    with DGJ's value grid and exploration decay, in weeks of one processor.
    ``\SessionTimeFactorRefGrid``: that median session time over the baseline's.
    """
    big, base = stats.get("ref_grid"), stats.get("baseline")
    if not big or not base:
        return
    if not (math.isfinite(big["hours_median"]) and math.isfinite(base["hours_median"])):
        return
    out.macro("DgjScaleRunWeeks", viz.fmt(1000 * big["hours_median"] / (24 * 7), 0))
    out.macro(
        "SessionTimeFactorRefGrid",
        viz.fmt(big["hours_median"] / base["hours_median"], 1),
    )


def add_long_session_macros(out: Outputs, main: Run) -> None:
    r"""How long the baseline trains, and what DGJ's longest sessions would cost.

    ``\ConvPeriodsMinBaseline``, ``…Median…``, ``…Max…``: periods to convergence of the
    converged sessions, in millions. ``\LongSessionDays``: one session of 50 billion
    periods (the upper end of DGJ's range, Dou et al. 2025, p. 24) at the baseline's
    median throughput, in days of one processor. ``\LongRunMonths``: 30 such sessions,
    in months.
    """
    s = main.sessions
    if "periods_per_second" not in s or "periods_to_convergence" not in s:
        return
    conv = s.loc[s["converged"].astype(bool), "periods_to_convergence"].astype(float)
    if conv.empty:
        return
    for name, value in (
        ("Min", conv.min()),
        ("Median", conv.median()),
        ("Max", conv.max()),
    ):
        out.macro(f"ConvPeriods{name}Baseline", viz.fmt(value / 1e6, 0))
    pps = float(s["periods_per_second"].median())
    if not math.isfinite(pps) or pps <= 0:
        return
    days = 50e9 / pps / 86400.0
    out.macro("LongSessionDays", viz.fmt(days, 0))
    out.macro("LongRunMonths", viz.fmt(30 * days / (365.25 / 12), 0))


def make_f9(out: Outputs, runs: dict[str, Run]) -> None:
    r"""fig_learner_paths.pdf: trading intensity of the two learners in Q–AC.

    Median and interquartile range over the sessions of the mixed population, read from
    the snapshots of ``trajectory.parquet`` (one per million periods). A session that
    has converged keeps its last value, so every session counts at every period. The
    upper axis gives the probability with which the Q-learner still explores,
    ``exp(-beta * t / n_v)``: its counter runs per value and each of the ``n_v`` values
    occurs in one n_v-th of the periods.

    Macros (medians of the figure, and the exploration probability, in percent):
    ``\PathChiQAcQFifteen``, ``\PathChiQAcQThirty``, ``\PathChiQAcAcFifteen``,
    ``\PathChiQAcAcThirty``, ``\ExploreProbQFifteen``, ``\ExploreProbQThirty``.
    """
    name = "fig_learner_paths.pdf"
    run = runs.get("q_ac")
    if run is None or run.trajectory.empty:
        out.skip(name, "needs the q_ac run with trajectory.parquet")
        return
    tr = run.trajectory
    cols = ("chi_hat_agent_0", "chi_hat_agent_1")
    if any(c not in tr for c in cols):
        out.skip(name, "trajectory.parquet has no trading intensity per agent")
        return
    bench = benchmarks_of(run)
    cfg = run.config
    beta, n_v = float(cfg["beta"]), int(cfg["n_v"])
    x_max = 60.0
    fig, ax = viz.new_figure(6.2)
    styles = (
        ("Q-learner", viz.BLACK, viz.LIGHT_GREY),
        ("AC learner", viz.ACCENT, "#f1c9cf"),
    )
    medians: dict[str, pd.Series] = {}
    for col, (label, colour, band) in zip(cols, styles, strict=True):
        wide = (
            tr.pivot(index="period", columns="session_id", values=col)
            .sort_index()
            .ffill()
        )
        x = wide.index.to_numpy(float) / 1e6
        lo, med, hi = (wide.quantile(q, axis=1) for q in (0.25, 0.5, 0.75))
        ax.fill_between(x, lo, hi, color=band, alpha=0.6, linewidth=0, zorder=2)
        ax.plot(x, med, color=colour, linewidth=1.4, label=label, zorder=3)
        medians[label] = med
    if bench is not None:
        chi_n = float(bench["chi_N"])
        ax.axhline(chi_n, color=viz.GREY, linewidth=0.8, linestyle="--", zorder=1)
        ax.text(1.0, chi_n + 0.3, "Nash intensity", ha="left", va="bottom", fontsize=9)
        ax.set_ylim(top=chi_n + 2.6)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Training periods (millions)")
    ax.set_ylabel(r"Trading intensity $\hat\chi$")
    ax.legend(loc="lower right")
    # Upper axis: the Q-learner's exploration probability at that point of training.
    top = ax.secondary_xaxis("top")
    eps_ticks = (0.5, 0.1, 0.01, 0.001)
    top.set_xticks([n_v * math.log(1.0 / e) / beta / 1e6 for e in eps_ticks])
    top.set_xticklabels(["50%", "10%", "1%", "0.1%"])
    top.set_xlabel("Probability with which the Q-learner explores")
    top.spines["top"].set_visible(True)
    fig.tight_layout()
    out.figure(name, fig)

    for label, tag in (("Q-learner", "Q"), ("AC learner", "Ac")):
        for periods, word in ((15e6, "Fifteen"), (30e6, "Thirty")):
            med = medians[label]
            if periods in med.index:
                out.macro(f"PathChiQAc{tag}{word}", viz.fmt(med.loc[periods], 1))
    for periods, word, decimals in ((15e6, "Fifteen", 0), (30e6, "Thirty", 2)):
        eps = math.exp(-beta * periods / n_v)
        out.macro(f"ExploreProbQ{word}", viz.fmt(100 * eps, decimals))


def make_f8(out: Outputs, main: Run) -> None:
    """fig_state_space.pdf: a learner's table as a block (sizes from the config)."""
    cfg = main.config
    n_v, n_x, n_p = int(cfg["n_v"]), int(cfg["n_x"]), int(cfg["n_p"])
    bench, grids = benchmarks_of(main), grids_of(main)
    levels: dict[str, int] = {}
    if bench and grids and cfg.get("action_grid") == "per_v":
        chi = np.asarray(grids["action_grid"], dtype=float)
        levels = {
            "cartel": int(np.argmin(np.abs(chi - bench["chi_M"]))),
            "Nash": int(np.argmin(np.abs(chi - bench["chi_N"]))),
        }
    n_states = n_p * n_v * n_v
    out.macro("StatesBaseline", viz.fmt(n_states, 0, thousands=True))
    out.macro("CellsBaseline", viz.fmt(n_states * n_x, 0, thousands=True))
    out.macro("CellsPerStateBlockBaseline", viz.fmt(n_p * n_v * n_x, 0, thousands=True))

    fig, ax = viz.new_figure(5.6)
    ax.set_axis_off()
    ax.set_aspect("equal")
    cw, ch = 0.24, 0.22  # width and height of one cell on the front face
    dx, dy = 0.17, 0.12  # offset of one step into the depth (current value)
    x0, y0 = 1.95, 0.7
    width, height = n_p * cw, n_x * ch
    thin = {"color": viz.GREY, "linewidth": 0.3}
    edge = {"color": viz.BLACK, "linewidth": 0.8}
    k_state, v_state = 20, 0  # the highlighted state: price bin 21, front row
    # one state: its column of n_x cells on the front face and its cell on top
    ax.add_patch(
        viz.plt.Rectangle(
            (x0 + k_state * cw, y0),
            cw,
            height,
            facecolor=viz.LIGHT_GREY,
            edgecolor="none",
        )
    )
    top = [
        (x0 + k_state * cw + v_state * dx, y0 + height + v_state * dy),
        (x0 + (k_state + 1) * cw + v_state * dx, y0 + height + v_state * dy),
        (
            x0 + (k_state + 1) * cw + (v_state + 1) * dx,
            y0 + height + (v_state + 1) * dy,
        ),
        (x0 + k_state * cw + (v_state + 1) * dx, y0 + height + (v_state + 1) * dy),
    ]
    ax.add_patch(viz.plt.Polygon(top, facecolor=viz.LIGHT_GREY, edgecolor="none"))
    # one cell of that column, at the Nash level
    lv = levels.get("Nash", n_x - 2)
    ax.add_patch(
        viz.plt.Rectangle(
            (x0 + k_state * cw, y0 + lv * ch),
            cw,
            ch,
            facecolor=viz.BLACK,
            edgecolor="none",
        )
    )
    # front face (price bins by action levels)
    for i in range(n_p + 1):
        ax.plot(
            [x0 + i * cw] * 2, [y0, y0 + height], **(edge if i in (0, n_p) else thin)
        )
    for j in range(n_x + 1):
        ax.plot(
            [x0, x0 + width], [y0 + j * ch] * 2, **(edge if j in (0, n_x) else thin)
        )
    # top face (price bins by current values)
    for i in range(n_p + 1):
        ax.plot(
            [x0 + i * cw, x0 + i * cw + n_v * dx],
            [y0 + height, y0 + height + n_v * dy],
            **(edge if i in (0, n_p) else thin),
        )
    for m in range(1, n_v + 1):
        ax.plot(
            [x0 + m * dx, x0 + width + m * dx],
            [y0 + height + m * dy] * 2,
            **(edge if m == n_v else thin),
        )
    # right face (current values by action levels)
    for j in range(n_x + 1):
        ax.plot(
            [x0 + width, x0 + width + n_v * dx],
            [y0 + j * ch, y0 + j * ch + n_v * dy],
            **(edge if j in (0, n_x) else thin),
        )
    for m in range(1, n_v + 1):
        ax.plot(
            [x0 + width + m * dx] * 2,
            [y0 + m * dy, y0 + height + m * dy],
            **(edge if m == n_v else thin),
        )
    # labels of the three sides
    ax.text(
        x0 + width / 2,
        y0 - 0.2,
        f"lagged price $p_{{t-1}}$: {n_p} price bins",
        ha="center",
        va="top",
        fontsize=8,
    )
    ax.text(
        x0 - 1.45,
        y0 + height / 2,
        f"action: {n_x} levels of\ntrading intensity",
        ha="center",
        va="center",
        fontsize=8,
        rotation=90,
        linespacing=1.4,
    )
    for name, level in levels.items():
        ax.text(
            x0 - 0.1,
            y0 + (level + 0.5) * ch,
            f"level {level}: {name}",
            ha="right",
            va="center",
            fontsize=7,
        )
    ax.text(
        x0 + width + n_v * dx,
        y0 + height + n_v * dy + 0.12,
        f"current value $v_t$: {n_v} points",
        ha="right",
        va="bottom",
        fontsize=8,
    )
    # what a column and a cell are
    xr = x0 + width + n_v * dx + 0.45
    notes = [
        (
            y0 + height + 0.6,
            (
                "Grey column: one state.\nIt holds one cell for each\n"
                f"of the {n_x} actions."
            ),
        ),
        (
            y0 + height - 0.95,
            (
                "Black cell: one number.\nQ-learner: the value $\\hat{Q}(s, a)$.\n"
                "AC learner: the preference $H(s, a)$."
            ),
        ),
        (
            y0 + height - 2.5,
            (
                f"One block for each of the {n_v}\nlagged values $v_{{t-1}}$:\n"
                f"{viz.fmt(n_states, 0, thousands=True)} states, "
                f"{viz.fmt(n_states * n_x, 0, thousands=True)} cells."
            ),
        ),
    ]
    for y, text in notes:
        ax.text(xr, y, text, ha="left", va="top", fontsize=8, linespacing=1.4)
    ax.set_xlim(0.0, 15.0)
    ax.set_ylim(0.0, 5.6)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    out.figure("fig_state_space.pdf", fig)


def write_numbers(out: Outputs) -> None:
    lines = ["% Generated by scripts/make_thesis_outputs.py — do not edit by hand."]
    for name in sorted(out.macros):
        if not name.isalpha():
            raise ValueError(f"macro name must be letters only: {name}")
        lines.append(f"\\newcommand{{\\{name}}}{{{out.macros[name]}}}")
    out.table("numbers.tex", "\n".join(lines) + "\n")


# ---- main -------------------------------------------------------------------------------


def parse_runs(items: list[str] | None) -> dict[str, Path]:
    runs = {k: Path(v) for k, v in DEFAULT_RUNS.items()}
    for item in items or []:
        key, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"--runs expects KEY=DIR, got {item!r}")
        runs[key] = Path(value)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runs", nargs="*", help="KEY=DIR overrides of the default run map"
    )
    parser.add_argument("--out-figures", default="outputs/figures")
    parser.add_argument("--out-tables", default="outputs/tables")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    viz.set_thesis_style()

    run_paths = parse_runs(args.runs)
    runs: dict[str, Run] = {}
    for key, path in run_paths.items():
        run = load_run(key, path)
        if run is not None:
            runs[key] = run
    out = Outputs(Path(args.out_figures), Path(args.out_tables))

    stats = {k: session_stats(r) for k, r in runs.items()}
    grid_nash = {k: grid_nash_of(r) for k, r in runs.items()}
    for k, r in runs.items():
        add_run_macros(out, r, stats[k], grid_nash[k])
    add_diff_macros(out, runs)

    main_run = runs.get("baseline")
    if main_run is None:
        for name in (
            "fig_learning_curves.pdf",
            "fig_policy.pdf",
            "fig_irf_deviation.pdf",
            "fig_irf_noise.pdf",
            "tab_parameters.tex",
            "tab_parameters_baseline.tex",
            "tab_parameters_ac.tex",
            "tab_benchmarks.tex",
            "fig_state_space.pdf",
        ):
            out.skip(name, "baseline run missing")
        gn_main = None
    else:
        gn_main = grid_nash["baseline"]
        make_t1(out, main_run, gn_main)
        make_f2(out, main_run, gn_main)
        make_f4(out, main_run, benchmarks_of(main_run))
        make_f5(out, main_run)
        make_f5b(out, main_run)
        make_t2(out, main_run, t2_scale_reasons(stats))
        make_t2_split(out, main_run, stats)
        make_f8(out, main_run)
        add_long_session_macros(out, main_run)
    make_f1(out)
    make_f3(out, runs, gn_main)
    make_f6(out, runs, gn_main)
    make_f7(out, runs)
    make_f9(out, runs)
    make_t4(out, runs, stats)
    make_t5(out, runs, stats, grid_nash)
    make_t6(out, runs, stats)
    make_t7(out, runs, stats)
    make_t3(out, main_run)
    make_t8(out, runs, stats)
    add_scale_macros(out, stats)
    write_numbers(out)

    print("\nWritten:")
    for w in out.written:
        print(f"  {w}")
    print("Skipped:")
    for name, reason in out.skipped or [("(none)", "")]:
        print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
