"""Embed the numbers of ``results/`` into ``results_explorer.html``.

``results_explorer.html`` (root of the repository) is a single page without any
dependency: it opens from disk in a browser and needs no server and no network.
Its numbers sit in one JSON block inside the file,

    <script id="explorer-data" type="application/json"> ... </script>

and this script rewrites that block from the result folders. Nothing else in
the page is touched: layout, texts and charts are written by hand in the HTML
file, and no number of the results is typed there.

The numbers are computed with the functions of ``scripts/make_thesis_outputs.py``
and ``src/ai_trading/viz/figures.py`` (session statistics, bootstrap intervals,
the pairs of the forced-deviation impulse responses), so the page shows what the
tables and figures of the thesis show. Per run the block holds: the settings
read from ``run_metadata.json``, the summary, one row per session, the result
of the deviation test per session (``deviation_trials.parquet``), the learning
path (``trajectory.parquet``) and the impulse responses to a forced deviation
(``periods.parquet``; baseline specification only). The two validation runs
add their session rows; the block also holds the benchmarks of the one-period
game and the differences between runs that the thesis reports. The output is deterministic.

Usage (from the root of the repository, a few seconds, no simulation):
    uv run python scripts/make_results_explorer.py \\
        [--results results] [--html results_explorer.html]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # ai_trading, also without an installed package
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_thesis_outputs as mto

from ai_trading.viz import figures as viz

logger = logging.getLogger("make_results_explorer")

DATA_BLOCK = re.compile(
    r'(<script id="explorer-data" type="application/json">)(.*?)(</script>)',
    re.DOTALL,
)
# The forced-deviation impulse responses are shown for the runs of the baseline
# specification that record them (deviations of 4 and 8 levels on the per-value
# grid); docs/parameters.md, "IRF protocol", says why the earlier specification
# cannot be read the same way.
IRF_RUNS = (
    "baseline",
    "rounded_price",
    "ref_beta",
    "ref_grid",
    "ref_noise",
    "q_ac",
    "ac_ac",
    "two_alphas",
)
IRF_COLUMNS = ("dx_dev", "dx_oth", "dprof_dev", "dprof_oth")


def num(x: Any, decimals: int = 6) -> float | None:
    """A float for the JSON block; None for NaN, infinity and missing values.

    The page rounds for display, so the block keeps more digits than the page shows
    (a number rounded twice can differ from the number of the thesis in the last digit).
    """
    if x is None:
        return None
    x = float(x)
    return round(x, decimals) if math.isfinite(x) else None


def settings_of(run: mto.Run) -> dict[str, Any]:
    """The settings of a run that the page names, from the resolved configuration."""
    cfg = run.config
    agents = []
    for a in cfg.get("agents") or []:
        own = a.get("config") or {}
        agents.append(
            {
                "type": str(a.get("type")),
                "alpha": num(own.get("alpha", cfg.get("alpha")), 6),
                "chi": own.get("chi"),
            }
        )
    return {
        "agents": agents,
        "n_v": int(cfg["n_v"]),
        "beta": float(cfg["beta"]),
        "sigma_u": float(run.sessions["sigma_u"].iloc[0]),
        "action_grid": cfg.get("action_grid", "global"),
        "exploration_counter": cfg.get("exploration_counter", "state"),
        "q_init": cfg.get("q_init", "optimistic"),
        "reward_price": cfg.get("reward_price", "raw"),
        "state_price": cfg.get("state_price", "lagged"),
        "convergence_periods": int(mto._run_cfg_value(run, "convergence_periods")),
        "max_periods": int(mto._run_cfg_value(run, "max_periods")),
    }


def deviation_by_session(run: mto.Run) -> pd.DataFrame | None:
    """Deviation test per session: gain, continuation, total and its standard error.

    The single grid best response of ``scripts/deviation_test.py`` (kind ``grid_br``),
    mean over agents and trials, as ``session_stats`` reads it.
    """
    dev = run.deviation_trials
    if dev is None or dev.empty:
        return None
    g = dev[dev["kind"] == "grid_br"].groupby("session_id")
    out = g[["gain0", "cont", "total"]].mean()
    out["se"] = g["total"].std() / np.sqrt(g["total"].count())
    return out


def session_rows(run: mto.Run, dev: pd.DataFrame | None) -> list[dict[str, Any]]:
    rows = []
    for _, s in run.sessions.iterrows():
        sid = int(s["session_id"])
        has_diag = bool(s.get("diagnostics_enabled", False))
        row = {
            "id": sid,
            "seed": int(s["seed"]),
            "conv": bool(s["converged"]),
            "settled": bool(s["settled"]) if has_diag else None,
            "flip": num(s.get("stability_flip_share_nd")) if has_diag else None,
            "periods": int(s["periods_to_convergence"]),
            "dc": num(s["delta_c"]),
            "dc_agent": [num(s[f"delta_c_agent_{i}"]) for i in range(run.n_agents)],
            "chi_agent": [num(s[f"chi_hat_agent_{i}"], 4) for i in range(run.n_agents)],
            "profit_agent": [
                num(s[f"avg_profit_agent_{i}"], 4) for i in range(run.n_agents)
            ],
            "profit_gain": num(s["profit_gain"]),
            "minutes": num(s.get("wall_clock_seconds", math.nan) / 60.0, 1),
        }
        if dev is not None and sid in dev.index:
            d = dev.loc[sid]
            row["dev"] = {k: num(d[k], 5) for k in ("gain0", "cont", "total", "se")}
        rows.append(row)
    return rows


def summary_of(
    run: mto.Run, stats: dict[str, Any], dev: pd.DataFrame | None
) -> dict[str, Any]:
    out = {
        "sessions": stats["sessions"],
        "converged": stats["converged"],
        "settled": stats["settled"],
        "dc": [num(stats[k]) for k in ("delta_c_mean", "delta_c_lo", "delta_c_hi")],
        "dc_median": num(stats["delta_c_median"]),
        "positive": int((run.sessions["delta_c"] > 0).sum()),
        "dc_agent": [num(stats[f"delta_c_agent_{i}"]) for i in range(run.n_agents)],
        "chi": num(stats["chi_hat"]),
        "chi_agent": [num(stats[f"chi_hat_agent_{i}"]) for i in range(run.n_agents)],
        "profit_pct": num(stats["profit_pct"]),
        "ahead": [stats.get("agent_0_ahead"), stats.get("agent_1_ahead")],
        "periods_median": num(stats["periods_median"], 0),
        "hours_total": num(stats["hours_total"]),
        "grid_nash": num(mto.grid_nash_of(run)),
        "dev": None,
    }
    if dev is not None:
        out["dev"] = {
            "total": [
                num(stats[k]) for k in ("dev_gain_mean", "dev_gain_lo", "dev_gain_hi")
            ],
            "gain0": num(stats["dev_gain_period"]),
            "cont": num(stats["dev_gain_cont"]),
            "pays": int(stats["dev_pays"]),
            "not_pays": int((dev["total"] + 2 * dev["se"] < 0).sum()),
            "sessions": len(dev),
        }
    return out


def trajectory_of(run: mto.Run) -> dict[str, Any] | None:
    """Learning path per session: one snapshot per million periods of training.

    Δ^C is stored in thousandths and the trading intensity in tenths, as integers;
    entry k of a list belongs to period (k + 1) million.
    """
    tr = run.trajectory
    if tr.empty:
        return None
    out: dict[str, Any] = {}
    for sid, blk in tr.sort_values("period").groupby("session_id"):
        t = blk["period"].to_numpy()
        if not np.array_equal(t, 1_000_000 * np.arange(1, len(t) + 1)):
            raise ValueError(f"{run.key} s{sid}: snapshots are not one per million")
        entry = {"dc": np.rint(1000 * blk["delta_c"]).astype(int).tolist()}
        for i in range(run.n_agents):
            entry[f"dc{i}"] = (
                np.rint(1000 * blk[f"delta_c_agent_{i}"]).astype(int).tolist()
            )
            entry[f"chi{i}"] = (
                np.rint(10 * blk[f"chi_hat_agent_{i}"]).astype(int).tolist()
            )
        out[str(int(sid))] = entry
    return out


def irf_of(run: mto.Run) -> dict[str, Any] | None:
    """Responses to a forced deviation, in percent of the session's evaluation mean.

    The sample of ``fig_irf_deviation.pdf``: sessions that converged or settled,
    trials in which the deviation was effective and led to a visited state. Per size
    of the deviation and per deviating agent ("all", "0", "1"): the mean over sessions
    with a bootstrap 95% band for the deviator's and the other speculator's order flow
    and profit, from the period of the deviation on.
    """
    pairs = mto.deviation_pairs_kept(run, mto._converged_or_settled(run))
    if pairs.empty:
        return None
    periods = list(range(run.pre, run.pre + run.post))
    out: dict[str, Any] = {"t": periods, "sizes": {}}
    for m, by_size in pairs.groupby("shock_magnitude"):
        splits = {"all": by_size}
        for j, blk in by_size.groupby("deviating_agent"):
            splits[str(int(j))] = blk
        block = {}
        for name, blk in splits.items():
            entry: dict[str, Any] = {
                "trials": viz.n_trials(blk),
                "sessions": int(blk["session_id"].nunique()),
            }
            for col in IRF_COLUMNS:
                prof = viz.irf_profile(blk, col, periods)
                entry[col] = {
                    k: [num(100 * v, 5) for v in prof[k]] for k in ("mean", "lo", "hi")
                }
            block[name] = entry
        out["sizes"][f"{float(m):g}"] = block
    return out


def game_of(run: mto.Run) -> dict[str, Any]:
    """Benchmarks and grid of the one-period game at the baseline specification.

    The page computes the profit of level i against level j as
    ``var_v * chi_i * (1 - lambda_N * (chi_i + chi_j))``: the mean over the values of
    the grid of ``(v - v_bar - lambda_N (x_i + x_j)) x_i`` with ``x = chi (v - v_bar)``,
    the convention of ``benchmarks.grid_stage_game``. Checked here against that
    function and against ``random_play_delta_c``.
    """
    g, b = mto.grids_of(run), mto.benchmarks_of(run)
    if g is None or b is None or run.config.get("action_grid") != "per_v":
        raise ValueError("the baseline run needs saved tables on the per-value grid")
    chi = np.asarray(g["action_grid"], dtype=float)
    v = np.asarray(g["v_grid"], dtype=float)
    var_v, lam = float(np.var(v)), float(b["lambda_N"])
    pi_n, pi_m = float(b["pi_N"]), float(b["pi_M"])
    payoff = var_v * chi[:, None] * (1.0 - lam * (chi[:, None] + chi[None, :]))
    game = mto.benchmarks.grid_stage_game(
        g["action_table"], v, lam, pi_n, pi_m, run.v_bar
    )
    # At v = v_bar every action trades zero and every pair is an equilibrium; the
    # game of the page is the one at the other values.
    nash = sorted(
        {
            pair
            for k, per_v in enumerate(game["nash_bins"])
            if abs(v[k] - run.v_bar) > 1e-12
            for pair in per_v
        }
    )
    if len(nash) != 1 or nash[0][0] != nash[0][1]:
        raise ValueError(f"expected one symmetric grid equilibrium, found {nash}")
    level = nash[0][0]
    dc_nash = (payoff[level, level] - pi_n) / (pi_m - pi_n)
    dc_random = (payoff.mean() - pi_n) / (pi_m - pi_n)
    random_play = mto.random_play_delta_c(
        g["action_table"], v, lam, pi_n, pi_m, run.v_bar
    )
    if abs(dc_nash - game["delta_c_grid_nash"]) > 1e-9:
        raise ValueError("grid Nash of the page formula differs from grid_stage_game")
    if abs(dc_random - random_play) > 1e-9:
        raise ValueError("random play of the page formula differs from the thesis")
    best_reply = mto.best_reply_to_nash_delta_c(
        g["action_table"], v, float(b["chi_N"]), lam, pi_n, pi_m, run.v_bar
    )
    return {
        "chi_grid": [num(c, 8) for c in chi],
        "var_v": num(var_v, 8),
        "lambda_N": num(lam, 8),
        "chi_N": num(b["chi_N"]),
        "chi_M": num(b["chi_M"]),
        "pi_N": num(pi_n, 8),
        "pi_M": num(pi_m, 8),
        "nash_level": int(level),
        "cartel_level": int(game["best_symmetric_bins"][-1]),
        "dc_grid_nash": num(dc_nash),
        "dc_random_play": num(dc_random),
        "dc_best_reply_to_nash": num(best_reply),
        "dgj_dc": mto.DGJ_DELTA_C,
        "dgj_profit_pct": mto.DGJ_PROFIT_PCT,
    }


def build(results: Path) -> dict[str, Any]:
    runs: dict[str, mto.Run] = {}
    for key, path in mto.DEFAULT_RUNS.items():
        run = mto.load_run(key, results / Path(path).name)
        if run is not None:
            runs[key] = run
    if "baseline" not in runs:
        raise SystemExit("the baseline run (results/r1_baseline) is missing")
    data: dict[str, Any] = {"game": game_of(runs["baseline"]), "runs": {}}
    for key, run in runs.items():
        stats = mto.session_stats(run)
        dev = deviation_by_session(run)
        data["runs"][key] = {
            "folder": run.path.name,
            "settings": settings_of(run),
            "summary": summary_of(run, stats, dev),
            "sessions": session_rows(run, dev),
            "trajectory": trajectory_of(run),
            "irf": irf_of(run) if key in IRF_RUNS else None,
        }
    for key, (_, path) in mto.VALIDATION_RUNS.items():
        run = mto.load_run(key, results / Path(path).name)
        if run is None:
            continue
        stats = mto.session_stats(run)
        data["runs"][key] = {
            "folder": run.path.name,
            "settings": settings_of(run),
            "summary": summary_of(run, stats, None),
            "sessions": session_rows(run, None),
            "trajectory": trajectory_of(run),
            "irf": None,
        }
    # Differences of the session-mean Delta^C that the thesis reports (A minus B, with
    # a bootstrap 95% interval), under the key "A|B".
    data["diffs"] = {}
    for ka, kb in mto.DIFF_PAIRS:
        if ka in runs and kb in runs:
            diff = mto.bootstrap_diff_ci(
                runs[ka].sessions["delta_c"].to_numpy(dtype=float),
                runs[kb].sessions["delta_c"].to_numpy(dtype=float),
            )
            data["diffs"][f"{ka}|{kb}"] = [num(x) for x in diff]
    main = [data["runs"][k]["summary"] for k in runs]
    data["totals"] = {
        "runs": len(runs),
        "sessions": int(sum(s["sessions"] for s in main)),
        "hours": num(sum(s["hours_total"] or 0.0 for s in main)),
    }
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", default="results")
    parser.add_argument("--html", default="results_explorer.html")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    html_path = Path(args.html)
    html = html_path.read_text(encoding="utf-8")
    if not DATA_BLOCK.search(html):
        raise SystemExit(f'{html_path}: no <script id="explorer-data"> block')
    data = build(Path(args.results))
    # "</" must not appear inside a script element; JSON allows the escaped form.
    block = json.dumps(
        data, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).replace("</", "<\\/")
    html = DATA_BLOCK.sub(
        lambda m: f"{m.group(1)}\n{block}\n{m.group(3)}", html, count=1
    )
    html_path.write_text(html, encoding="utf-8")
    print(
        f"{html_path}: {len(data['runs'])} runs, "
        f"{data['totals']['sessions']} sessions of the 12 runs of the thesis, "
        f"data block {len(block) / 1024:.0f} KiB"
    )


if __name__ == "__main__":
    main()
