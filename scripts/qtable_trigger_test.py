"""Q-table trigger test: price-conditional punishment in the saved greedy policies.

Direct mechanism evidence without shocks, complementing the best-response deviation
test (``scripts/deviation_test.py``). For every session and agent with a saved policy
table (``agent_i.npy`` or ``agent_i_prefs.npy``) and the session's eval rows:

1. **On-path cells**: the states (p_bin, v_prev_bin, v_bin) that occur in the eval rows
   (the state of a row is the previous row's p_bin and value bin, and the row's value bin),
   with their eval visit counts. States whose current v is degenerate (every action trades
   zero, i.e. v = v̄ on the per-v grid) are skipped.
2. **Deviation cell** of an on-path cell: the same (v_prev_bin, v_bin) with the lagged-price
   bin moved one bin further from the centre in the direction of v_prev (p + 1 if
   v_prev > v̄, p − 1 if v_prev < v̄); skipped when out of range or when v_prev = v̄. That is
   the price a more aggressive trade last period would have produced.
3. **Comparison** of the greedy quantity magnitude |x| in the deviation cell with the
   on-path cell: "punish" if larger, "same" if equal, "forgive" if smaller.
4. **Report** per run: visit-weighted shares of punish / same / forgive over on-path cells,
   per agent type and overall; the mean difference in aggressiveness χ (deviation minus
   on-path); the share of sessions in which both agents have a punish share above 0.5;
   how many deviation cells have fewer than 100 training visits according to the eval rows'
   ``n_visits_agent_i`` columns (little learned information).

Untouched deviation cells: a deviation cell that training never visited keeps
its initial table row (under ``q_init: paper`` the argmax is the most aggressive level, so
it would read as "punish" without having been learned). A cell is *untouched* when its
saved row equals the agent's initial row (the initial Q-table is rebuilt with the agent's
own initialisation; an actor-critic preference row is untouched when it is all zeros).
Those cells are counted and reported separately; the headline shares use visited (touched)
deviation cells only.

Usage:
    PYTHONPATH=src uv run --no-sync python scripts/qtable_trigger_test.py \\
        --results-dir results/r1_baseline --config experiments/01_replicate_baseline/config_r1_baseline.yaml \\
        [--out results/r1_baseline/qtable_trigger_test.txt]
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.simulation import runner

logger = logging.getLogger("qtable_trigger_test")

LOW_VISITS = 100


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        self.lines.append(text)
        print(text)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n")


# ---- pure helpers ----------------------------------------------------------------


def deviation_cell(
    p_bin: int, v_prev_bin: int, v_bin: int, v_grid: np.ndarray, v_bar: float, n_p: int
) -> tuple[int, int, int] | None:
    """The cell one lagged-price bin further from the centre in the direction of v_prev.

    Returns ``None`` when v_prev = v̄ (no direction) or the shifted bin is out of range.
    """
    dv_prev = float(v_grid[v_prev_bin] - v_bar)
    if abs(dv_prev) < 1e-12:
        return None
    p_dev = p_bin + 1 if dv_prev > 0 else p_bin - 1
    if p_dev < 0 or p_dev >= n_p:
        return None
    return p_dev, v_prev_bin, v_bin


def eval_states(
    periods: pd.DataFrame, v_grid: np.ndarray, n_agents: int
) -> pd.DataFrame:
    """On-path cells of one session's eval rows with visit counts and training visits.

    Returns one row per (p_bin, v_prev_bin, v_bin) with ``visits`` (eval occurrences) and
    ``train_visits_agent_i`` (the maximum of the row's ``n_visits_agent_i`` over its
    occurrences; −1 when the runner did not record a count).
    """
    ev = periods[
        (periods["phase"] == "eval") & (periods["irf_type"] == "")
    ].sort_values("period")
    if len(ev) < 2:
        return pd.DataFrame()
    v_arr = np.asarray(v_grid, dtype=np.float64)
    v_bin = np.abs(ev["v_t"].to_numpy()[:, None] - v_arr[None, :]).argmin(axis=1)
    p_bin = ev["p_bin"].to_numpy()
    data: dict[str, Any] = {
        "p_bin": p_bin[:-1],
        "v_prev_bin": v_bin[:-1],
        "v_bin": v_bin[1:],
    }
    for i in range(n_agents):
        col = f"n_visits_agent_{i}"
        data[f"train_visits_agent_{i}"] = (
            ev[col].to_numpy()[1:] if col in ev else np.full(len(ev) - 1, -1)
        )
    st = pd.DataFrame(data)
    agg: dict[str, Any] = {"visits": ("p_bin", "size")}
    for i in range(n_agents):
        agg[f"train_visits_agent_{i}"] = (f"train_visits_agent_{i}", "max")
    return st.groupby(["p_bin", "v_prev_bin", "v_bin"]).agg(**agg).reset_index()


def classify_cells(
    states: pd.DataFrame,
    greedy: np.ndarray,
    table_rows: np.ndarray,
    init_rows: np.ndarray,
    action_table: np.ndarray,
    v_grid: np.ndarray,
    v_bar: float,
    train_visits: np.ndarray | None,
) -> pd.DataFrame:
    """Classify every on-path cell of one agent: punish / same / forgive.

    Args:
        states: Output of :func:`eval_states` (one session).
        greedy: Greedy action index per state, shape ``(n_p, n_v, n_v)``.
        table_rows: The agent's saved table, shape ``(n_p, n_v, n_v, n_x)``.
        init_rows: The agent's initial table, same shape (to detect untouched cells).
        action_table: Quantities, shape ``(n_v, n_x)``.
        v_grid, v_bar: Value grid and its mean.
        train_visits: Training visits of the on-path cells for this agent (from the eval
            rows), aligned with ``states``; None if not recorded.

    Returns:
        One row per on-path cell with a deviation cell: ``visits`` (eval weight),
        ``label``, ``chi_diff`` (χ_dev − χ_on), ``untouched`` (deviation row equals the
        initial row), ``dev_train_visits`` (training visits of the deviation cell if it is
        itself an on-path cell, else −1).
    """
    v_arr = np.asarray(v_grid, dtype=np.float64)
    table = np.asarray(action_table, dtype=np.float64)
    n_p = greedy.shape[0]
    degenerate = np.ptp(table, axis=1) == 0.0
    key_to_visits: dict[tuple[int, int, int], int] = {}
    if train_visits is not None:
        for (p, vp, v), tv in zip(
            states[["p_bin", "v_prev_bin", "v_bin"]].to_numpy(), train_visits
        ):
            key_to_visits[(int(p), int(vp), int(v))] = int(tv)
    rows = []
    for r in states.itertuples(index=False):
        p, vp, v = int(r.p_bin), int(r.v_prev_bin), int(r.v_bin)
        if degenerate[v]:
            continue
        dev = deviation_cell(p, vp, v, v_arr, v_bar, n_p)
        if dev is None:
            continue
        pd_, _, _ = dev
        dv = float(v_arr[v] - v_bar)
        x_on = float(table[v, int(greedy[p, vp, v])])
        x_dev = float(table[v, int(greedy[pd_, vp, v])])
        if abs(x_dev) > abs(x_on) + 1e-12:
            label = "punish"
        elif abs(x_dev) < abs(x_on) - 1e-12:
            label = "forgive"
        else:
            label = "same"
        rows.append(
            {
                "p_bin": p,
                "v_prev_bin": vp,
                "v_bin": v,
                "visits": int(r.visits),
                "label": label,
                "chi_diff": (x_dev - x_on) / dv,
                "untouched": bool(
                    np.array_equal(table_rows[pd_, vp, v], init_rows[pd_, vp, v])
                ),
                "dev_train_visits": key_to_visits.get((pd_, vp, v), -1),
            }
        )
    return pd.DataFrame(rows)


def weighted_shares(cells: pd.DataFrame) -> dict[str, float]:
    """Visit-weighted shares of the three labels (NaN when there are no cells)."""
    if cells.empty or cells["visits"].sum() == 0:
        return {k: float("nan") for k in ("punish", "same", "forgive")}
    w = cells.groupby("label")["visits"].sum()
    total = float(w.sum())
    return {k: float(w.get(k, 0.0)) / total for k in ("punish", "same", "forgive")}


# ---- loading -------------------------------------------------------------------------


def find_sessions(results_dir: Path) -> list[tuple[int, Path, Path]]:
    out = []
    for qdir in sorted(results_dir.glob("**/qtables/session_*")):
        m = re.fullmatch(r"session_(\d+)", qdir.name)
        if m:
            out.append((int(m.group(1)), qdir, qdir.parent.parent / "periods.parquet"))
    return sorted(out)


def load_tables(
    qdir: Path, agents: list[Any]
) -> list[tuple[str, np.ndarray, np.ndarray] | None]:
    """Per agent: (type, saved table, initial table) or None when nothing is saved."""
    out: list[tuple[str, np.ndarray, np.ndarray] | None] = []
    for i, agent in enumerate(agents):
        q_file, p_file = qdir / f"agent_{i}.npy", qdir / f"agent_{i}_prefs.npy"
        if q_file.exists() and hasattr(agent, "Q"):
            out.append(("qlearner", np.load(q_file), np.asarray(agent.Q)))
        elif p_file.exists() and hasattr(agent, "get_preferences"):
            out.append(("actor_critic", np.load(p_file), agent.get_preferences()))
        else:
            out.append(None)
    return out


# ---- main ------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    results_dir = Path(args.results_dir).resolve()
    config = runner.load_config(args.config)
    sigma_u = float(config["sigma_u"])
    sessions = find_sessions(results_dir)
    if not sessions:
        raise SystemExit(f"No saved tables under {results_dir}.")

    all_cells: list[pd.DataFrame] = []
    for sid, qdir, periods_file in sessions:
        if not periods_file.exists():
            logger.warning("session %d: no periods.parquet; skipped", sid)
            continue
        env, agents, _bench = runner.build_session(config, sigma_u, seed=sid)
        periods = pd.read_parquet(periods_file)
        periods = periods[periods["session_id"] == sid]
        states = eval_states(periods, env.v_grid, len(agents))
        if states.empty:
            continue
        tables = load_tables(qdir, agents)
        for i, entry in enumerate(tables):
            if entry is None:
                continue
            kind, saved, init = entry
            greedy = np.argmax(saved, axis=-1)
            tv_col = f"train_visits_agent_{i}"
            tv = states[tv_col].to_numpy() if tv_col in states else None
            cells = classify_cells(
                states, greedy, saved, init, env.action_table, env.v_grid, env.v_bar, tv
            )
            cells["session_id"] = sid
            cells["agent"] = i
            cells["agent_type"] = kind
            all_cells.append(cells)

    r = Report()
    r(f"Q-table trigger test | {len(sessions)} session(s) | results {results_dir}")
    r(
        "Deviation cell = lagged-price bin one step further from the centre in the direction of v_prev."
    )
    r(
        "punish: |x| larger in the deviation cell; same: equal; forgive: smaller. Shares weighted by eval visits."
    )
    if not all_cells:
        r("No on-path cells with a deviation cell.")
        if args.out:
            r.save(Path(args.out))
        return
    cells = pd.concat(all_cells, ignore_index=True)
    touched = cells[~cells["untouched"]]
    n_cells, n_untouched = len(cells), int(cells["untouched"].sum())
    r(
        f"\nOn-path cells with a deviation cell: {n_cells} | deviation cell untouched "
        f"(initial table row, never visited in training): {n_untouched} "
        f"({n_untouched / n_cells:.1%}); headline shares use the {len(touched)} touched cells."
    )
    known = cells[cells["dev_train_visits"] >= 0]
    low = int((known["dev_train_visits"] < LOW_VISITS).sum())
    r(
        f"Deviation cells that are themselves on-path (training visits known from the eval rows): "
        f"{len(known)}; of these fewer than {LOW_VISITS} training visits: {low}."
    )

    r("\n=== HEADLINE (touched deviation cells only) ===")
    r(
        f"{'group':<16} {'cells':>6} {'punish':>7} {'same':>7} {'forgive':>8} {'mean Δχ':>8}"
    )
    groups = [("all agents", touched)] + [
        (f"type {t}", g) for t, g in touched.groupby("agent_type")
    ]
    for name, g in groups:
        sh = weighted_shares(g)
        chi = (
            float(np.average(g["chi_diff"], weights=g["visits"]))
            if len(g)
            else float("nan")
        )
        r(
            f"{name:<16} {len(g):>6} {sh['punish']:>7.3f} {sh['same']:>7.3f} "
            f"{sh['forgive']:>8.3f} {chi:>+8.3f}"
        )

    r("\n=== UNTOUCHED DEVIATION CELLS (initial row; not learned) ===")
    unt = cells[cells["untouched"]]
    sh = weighted_shares(unt)
    r(
        f"{'all agents':<16} {len(unt):>6} {sh['punish']:>7.3f} {sh['same']:>7.3f} "
        f"{sh['forgive']:>8.3f}"
    )

    r("\n=== PER SESSION (touched cells; punish share per agent) ===")
    per = (
        touched.groupby(["session_id", "agent"])
        .apply(lambda g: weighted_shares(g)["punish"], include_groups=False)
        .unstack("agent")
    )
    both = (per > 0.5).all(axis=1)
    r(
        f"{'sess':>4} "
        + " ".join(f"{'agent ' + str(a):>8}" for a in per.columns)
        + "  both>0.5"
    )
    for sid, row in per.iterrows():
        r(
            f"{sid:>4} "
            + " ".join(f"{row[a]:>8.3f}" for a in per.columns)
            + f"  {'yes' if both.loc[sid] else 'no'}"
        )
    r(
        f"Sessions in which both agents punish in more than half of the (visit-weighted) cells: "
        f"{int(both.sum())}/{len(both)}"
    )
    r(f"\nConfig: {Path(args.config).resolve()}")
    if args.out:
        out = Path(args.out)
        r.save(out)
        cells.to_parquet(out.parent / "qtable_trigger_cells.parquet", index=False)


if __name__ == "__main__":
    main()
