"""Run report: collusion capacity, convergence and impulse responses of a run.

Reads every ``sessions.parquet`` / ``periods.parquet`` under ``--results-dir``
(one directory with 30 sessions or 30 directories with one session each) and
prints: Δ^C summary statistics with a bootstrap CI for all, converged and (if
training diagnostics were enabled) settled sessions, the grid stage-game
reference, a per-session table, a Δ^C histogram, the noise-shock IRF
(shock − counterfactual), and the unilateral-deviation IRF. Model parameters
are read from ``--config``. Results without the diagnostics columns (a run made
with the training diagnostics off, as R1.1) are supported.

Usage:
    uv run --no-sync python scripts/baseline_report.py \
        --results-dir results/r1_1_early_stop \
        --config experiments/01_replicate_baseline/config_r1_1_early_stop.yaml \
        [--out results/r1_1_early_stop/report.txt]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.env.discretization import build_v_grid
from ai_trading.simulation import benchmarks, runner

# Thresholds of DGJ's Online Appendix §4.5 on the normalised order-flow deviation.
X_LOWER = 5e-5  # |dev| below this for both agents => "no response"
X_UPPER = 10 * X_LOWER  # dev above this for both agents => "price trigger"
N_BOOT = 10_000

PAIR_KEYS = [
    "source",
    "session_id",
    "sigma_u",
    "irf_type",
    "shock_magnitude",
    "deviating_agent",
    "irf_trial",
    "irf_period",
]


class Report:
    """Collects output lines so they can go to stdout and, optionally, a file."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        self.lines.append(text)
        print(text)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n")


# ---- loading ---------------------------------------------------------------


def _load(results_dir: Path, name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for f in sorted(results_dir.glob(f"**/{name}")):
        df = pd.read_parquet(f)
        rel = f.parent.relative_to(results_dir)
        df["source"] = str(rel) if str(rel) != "." else "."
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate all sessions/periods parquet files under ``results_dir``."""
    sessions = _load(results_dir, "sessions.parquet")
    periods = _load(results_dir, "periods.parquet")
    if sessions.empty:
        raise SystemExit(f"No sessions.parquet found under {results_dir}.")
    return sessions, periods


# ---- summary ----------------------------------------------------------------


def _bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _describe(values: np.ndarray) -> str:
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return (
        f"mean={values.mean():+.4f}  median={np.median(values):+.4f}  "
        f"std={values.std(ddof=0):.4f}  min={values.min():+.4f}  "
        f"max={values.max():+.4f}  IQR=[{q1:+.4f}, {q3:+.4f}]"
    )


def _print_group(r: Report, label: str, group: pd.DataFrame) -> None:
    """Δ^C statistics of one session group (all, converged or settled)."""
    n = len(group)
    r(f"--- {label}: n={n} ---")
    if n == 0:
        r("    (no sessions)")
        return
    dc = group["delta_c"].to_numpy(dtype=float)
    lo, hi = _bootstrap_ci(dc)
    gap = (group["delta_c_agent_0"] - group["delta_c_agent_1"]).abs().mean()
    r(
        f"    Δ^C: mean={dc.mean():+.4f}  median={np.median(dc):+.4f}  "
        f"sd={dc.std(ddof=0):.4f}  min={dc.min():+.4f}  max={dc.max():+.4f}"
    )
    r(f"    bootstrap 95% CI of the mean: [{lo:+.4f}, {hi:+.4f}]")
    r(
        f"    share with Δ^C > 0: {float((dc > 0).mean()):.1%}  |  "
        f"mean |Δ^C_0 − Δ^C_1|: {gap:.4f}"
    )


def _has_diagnostics(sessions: pd.DataFrame) -> bool:
    return (
        "settled" in sessions.columns
        and "diagnostics_enabled" in sessions.columns
        and bool(sessions["diagnostics_enabled"].fillna(False).astype(bool).any())
    )


def print_summary(
    r: Report,
    sessions: pd.DataFrame,
    bench: dict[str, float],
    grid_game: dict[str, Any],
    settled_max: float = 0.10,
) -> None:
    r("=== SUMMARY ===")
    n = len(sessions)
    conv = sessions[sessions["converged"].astype(bool)]
    r(f"Sessions: {n} | converged: {len(conv)} | non-converged: {n - len(conv)}")
    dup = sessions.groupby("session_id")["source"].nunique()
    if (dup > 1).any():
        r(f"WARNING: session_id in multiple sources: {dup[dup > 1].index.tolist()}")
    r("Δ^C per-path benchmarks (paper OA eq. IA.4.1):")
    _print_group(r, "ALL sessions", sessions)
    _print_group(r, "converged sessions", conv)
    if _has_diagnostics(sessions):
        settled = sessions[sessions["settled"].fillna(False).astype(bool)]
        _print_group(
            r,
            "settled (flip share over non-degenerate on-path states "
            f"<= {settled_max:.2f})",
            settled,
        )
    r(
        f"Grid Nash equilibrium: Δ^C = {grid_game['delta_c_grid_nash']:+.3f}; "
        f"best symmetric grid outcome: Δ^C = "
        f"{grid_game['delta_c_grid_best_symmetric']:+.3f}"
    )
    dce = sessions["delta_c_expected"].to_numpy(dtype=float)
    r(
        "Δ^C (expected-value benchmarks, pre-Phase-0 definition, all sessions): "
        f"{_describe(dce)}"
    )
    r(
        f"χ̂ (all sessions): mean={sessions['chi_hat'].mean():.2f} "
        f"std={sessions['chi_hat'].std(ddof=0):.2f}"
        f"  (χ^N={bench['chi_N']:.2f}, χ^M={bench['chi_M']:.2f})"
    )
    r(
        f"λ̂ (all sessions): mean={sessions['lambda_hat'].mean():.6f}  "
        f"(λ^N={bench['lambda_N']:.6f})"
    )
    prof = sessions[["avg_profit_agent_0", "avg_profit_agent_1"]].mean(axis=1)
    r(
        f"Profit per agent (all sessions): mean={prof.mean():.3f}  "
        f"(π^N={bench['pi_N']:.3f}, π^M={bench['pi_M']:.3f}, "
        f"mean π^N on path={sessions['pi_N_path'].mean():.3f})"
    )
    r(
        "Profit gain vs Nash (Σπ / Σπ^N_path, all sessions): "
        f"mean={sessions['profit_gain'].mean():.4f}"
    )
    if not conv.empty:
        pc = conv["periods_to_convergence"].to_numpy(dtype=float)
        r(
            f"Convergence period (converged sessions): mean={pc.mean() / 1e6:.2f}M "
            f"median={np.median(pc) / 1e6:.2f}M min={pc.min() / 1e6:.2f}M "
            f"max={pc.max() / 1e6:.2f}M"
        )
    r(
        f"Throughput: mean={sessions['periods_per_second'].mean():,.0f} periods/s | "
        f"total wall-clock={sessions['wall_clock_seconds'].sum() / 3600:.2f} h"
    )
    versions = sorted(sessions["code_version"].astype(str).unique())
    r(f"Code version(s): {versions}")
    if len(versions) > 1 or any(v.endswith("+dirty") for v in versions):
        r("WARNING: mixed or dirty code versions across sessions.")


def print_session_table(r: Report, sessions: pd.DataFrame) -> None:
    r("\n=== PER-SESSION TABLE ===")
    diag = _has_diagnostics(sessions)
    header = (
        f"{'sess':>4} {'source':<10} {'conv':<4} {'settled':<7} {'flip':>6} "
        f"{'period':>11} {'Δ^C':>8} {'Δ^C_exp':>8} {'χ̂':>7} {'prof_0':>7} "
        f"{'prof_1':>7}"
    )
    r(header)
    r("-" * len(header))
    for _, row in sessions.sort_values("session_id").iterrows():
        row_diag = diag and bool(row.get("diagnostics_enabled", False))
        settled = ("yes" if row["settled"] else "no") if row_diag else "-"
        # Share over non-degenerate on-path states (the one the settled flag
        # uses); results without it only have the plain share,
        # which is the same on the global grid.
        flip_col = (
            "stability_flip_share_nd"
            if "stability_flip_share_nd" in row.index
            and np.isfinite(row.get("stability_flip_share_nd", np.nan))
            else "stability_flip_share"
        )
        flip = row.get(flip_col, np.nan) if row_diag else np.nan
        flip_s = f"{flip:>6.3f}" if np.isfinite(flip) else f"{'-':>6}"
        r(
            f"{int(row['session_id']):>4} {str(row['source'])[:10]:<10} "
            f"{'yes' if row['converged'] else 'no':<4} {settled:<7} {flip_s} "
            f"{int(row['periods_to_convergence']):>11,} {row['delta_c']:>+8.4f} "
            f"{row['delta_c_expected']:>+8.4f} {row['chi_hat']:>7.2f} "
            f"{row['avg_profit_agent_0']:>7.2f} {row['avg_profit_agent_1']:>7.2f}"
        )


def print_histogram(r: Report, sessions: pd.DataFrame) -> None:
    r("\n=== Δ^C DISTRIBUTION (per-path, converged sessions) ===")
    dc = sessions.loc[sessions["converged"], "delta_c"].to_numpy(dtype=float)
    if len(dc) == 0:
        r("(no converged sessions — histogram over all sessions)")
        dc = sessions["delta_c"].to_numpy(dtype=float)
    lo, hi = float(dc.min()), float(dc.max())
    if hi - lo < 1e-9:
        r(f"All Δ^C ≈ {lo:+.4f}")
        return
    counts, edges = np.histogram(dc, bins=np.linspace(lo, hi, 11))
    top = max(int(counts.max()), 1)
    for i, c in enumerate(counts):
        bar = "█" * int(40 * c / top)
        r(f"  [{edges[i]:+.3f}, {edges[i + 1]:+.3f}]: {int(c):>3} {bar}")


# ---- IRF --------------------------------------------------------------------


def _eval_baselines(periods: pd.DataFrame, v_bar: float, n_agents: int) -> pd.DataFrame:
    ev = periods[periods["phase"] == "eval"].copy()
    sgn = np.sign(ev["v_t"].to_numpy() - v_bar)
    agg: dict[str, Any] = {}
    for i in range(n_agents):
        ev[f"xt_{i}"] = ev[f"x_agent_{i}"].to_numpy() * sgn
        agg[f"xt_{i}"] = (f"xt_{i}", "mean")
        agg[f"prof_{i}"] = (f"profit_agent_{i}", "mean")
    ev["pt"] = (ev["p_t"].to_numpy() - v_bar) * sgn
    agg["pt"] = ("pt", "mean")
    return ev.groupby(["source", "session_id", "sigma_u"]).agg(**agg).reset_index()


def _irf_pairs(
    periods: pd.DataFrame, irf_type: str, v_bar: float, n_agents: int, r: Report
) -> pd.DataFrame:
    irf = periods[(periods["phase"] == "irf") & (periods["irf_type"] == irf_type)]
    if irf.empty:
        return irf
    s = irf[irf["irf_path"] == "shock"]
    b = irf[irf["irf_path"] == "base"]
    pairs = s.merge(b, on=PAIR_KEYS, suffixes=("_s", "_b"))
    if not np.array_equal(pairs["v_t_s"].to_numpy(), pairs["v_t_b"].to_numpy()):
        r("WARNING: shock/base pairs differ in v_t — common random numbers broken.")
    base = _eval_baselines(periods, v_bar, n_agents)
    pairs = pairs.merge(base, on=["source", "session_id", "sigma_u"], how="left")
    sgn = np.sign(pairs["v_t_s"].to_numpy() - v_bar)
    for i in range(n_agents):
        pairs[f"dx_{i}"] = (
            (pairs[f"x_agent_{i}_s"] - pairs[f"x_agent_{i}_b"]) * sgn / pairs[f"xt_{i}"]
        )
        pairs[f"dprof_{i}"] = (
            pairs[f"profit_agent_{i}_s"] - pairs[f"profit_agent_{i}_b"]
        ) / pairs[f"prof_{i}"]
    pairs["dp"] = (pairs["p_t_s"] - pairs["p_t_b"]) * sgn / pairs["pt"]
    return pairs


def _classify(d0: float, d1: float) -> str:
    if d0 > X_UPPER and d1 > X_UPPER:
        return "price_trigger"
    if abs(d0) < X_LOWER and abs(d1) < X_LOWER:
        return "no_response"
    return "ambiguous"


def print_noise_irf(
    r: Report, periods: pd.DataFrame, pre: int, v_bar: float, n_agents: int
) -> None:
    r("\n=== NOISE-SHOCK IRF (shock − counterfactual, sign-adjusted, normalised) ===")
    pairs = _irf_pairs(periods, "noise", v_bar, n_agents, r)
    if pairs.empty:
        r("No noise-shock IRF rows.")
        return
    t_max = int(pairs["irf_period"].max())
    for m in sorted(pairs["shock_magnitude"].unique()):
        blk = pairs[pairs["shock_magnitude"] == m]
        at_shock = blk[blk["irf_period"] == pre]
        flip = float((at_shock["p_bin_s"] != at_shock["p_bin_b"]).mean())
        nxt = blk[blk["irf_period"] == pre + 1]
        unvisited = np.zeros(len(nxt), dtype=bool)
        for i in range(n_agents):
            unvisited |= nxt[f"n_visits_agent_{i}_s"].to_numpy() == 0
        r(
            f"\nShock {m:g} price bins: bin-flip rate at t={pre}: {flip:.1%} | "
            f"post-shock state never visited (t={pre + 1}): "
            f"{float(unvisited.mean()) if len(nxt) else float('nan'):.1%}"
        )
        r(
            f"  {'t':>3} {'price %':>10} "
            + " ".join(f"{'OF_' + str(i) + ' %':>10}" for i in range(n_agents))
        )
        for t in range(pre, min(pre + 6, t_max + 1)):
            row = blk[blk["irf_period"] == t]
            cells = " ".join(
                f"{100 * row[f'dx_{i}'].mean():>+10.4f}" for i in range(n_agents)
            )
            r(f"  {t:>3} {100 * row['dp'].mean():>+10.4f} {cells}")
        per = nxt.groupby(["source", "session_id"])[["dx_0", "dx_1"]].mean()
        labels = [_classify(a, b) for a, b in per.to_numpy()]
        counts = {
            k: labels.count(k) for k in ("price_trigger", "no_response", "ambiguous")
        }
        r(f"  Classification at t={pre + 1} (per session, mean over trials): {counts}")
        # In a state that was never visited in training the greedy action is
        # the argmax of the initial Q-values (the most aggressive bin under
        # q_init "paper"), not a learned response. On the per-v grid many
        # shocked states cannot be reached at all, because the lagged price
        # is pinned down by v_{t-1}. Repeat the statistics without them.
        vis = nxt[~unvisited]
        if len(vis):
            per_v = vis.groupby(["source", "session_id"])[["dx_0", "dx_1"]].mean()
            labels_v = [_classify(a, b) for a, b in per_v.to_numpy()]
            counts_v = {
                k: labels_v.count(k)
                for k in ("price_trigger", "no_response", "ambiguous")
            }
            cells_v = " ".join(
                f"OF_{i} {100 * vis[f'dx_{i}'].mean():+.4f}%" for i in range(n_agents)
            )
            r(
                f"  Visited post-shock states only ({len(vis)}/{len(nxt)} trials, "
                f"{len(per_v)} session(s)): t={pre + 1} {cells_v} | "
                f"classification: {counts_v}"
            )
        else:
            r(f"  Visited post-shock states only: 0/{len(nxt)} trials.")


def print_deviation_irf(
    r: Report, periods: pd.DataFrame, pre: int, v_bar: float, n_agents: int
) -> None:
    r("\n=== UNILATERAL-DEVIATION IRF (shock − counterfactual) ===")
    pairs = _irf_pairs(periods, "deviation", v_bar, n_agents, r)
    if pairs.empty:
        r("No deviation IRF rows.")
        return
    trial_keys = [c for c in PAIR_KEYS if c != "irf_period"]
    for (k, j), blk_all in pairs.groupby(["shock_magnitude", "deviating_agent"]):
        other = 1 - int(j) if n_agents == 2 else None
        # A forced deviation is clipped to the action grid. If the deviator's
        # greedy action is already the most aggressive bin (e.g. Nash at the
        # extreme v-states on the global grid, bin n_x − 1 on the per-v grid),
        # no deviation happens; keep only effective trials. The test compares
        # traded quantities, so it holds for both action-grid modes.
        at_shock = blk_all[blk_all["irf_period"] == pre]
        moved = at_shock.loc[
            at_shock[f"x_agent_{int(j)}_s"] != at_shock[f"x_agent_{int(j)}_b"],
            trial_keys,
        ]
        blk = blk_all.merge(moved, on=trial_keys, how="inner")
        r(
            f"\nAgent {int(j)} deviates by {k:g} bin(s): effective in "
            f"{len(moved)}/{len(at_shock)} trials (others clipped at the grid edge; "
            "excluded below)"
        )
        if blk.empty:
            continue
        r(
            f"  {'t':>3} {'dev OF %':>10} {'dev prof %':>11} {'other OF %':>11} {'other prof %':>13}"
        )
        for t in (pre, pre + 1, pre + 2):
            row = blk[blk["irf_period"] == t]
            if row.empty or other is None:
                continue
            r(
                f"  {t:>3} {100 * row[f'dx_{int(j)}'].mean():>+10.4f} "
                f"{100 * row[f'dprof_{int(j)}'].mean():>+11.4f} "
                f"{100 * row[f'dx_{other}'].mean():>+11.4f} "
                f"{100 * row[f'dprof_{other}'].mean():>+13.4f}"
            )
        if other is not None:
            nxt = blk[blk["irf_period"] == pre + 1]
            per = nxt.groupby(["source", "session_id"])[f"dx_{other}"].mean()
            r(
                f"  Sessions where the non-deviator punishes (mean OF dev > "
                f"{X_UPPER:.0e} at t={pre + 1}): {int((per > X_UPPER).sum())}/{len(per)}"
            )
            # Same caveat as for the noise shocks: where the non-deviator has
            # never visited the post-deviation state, its action is the argmax
            # of the initial Q-values, not a learned reaction.
            seen = nxt[nxt[f"n_visits_agent_{other}_s"].to_numpy() > 0]
            if len(seen):
                per_s = seen.groupby(["source", "session_id"])[f"dx_{other}"].mean()
                r(
                    f"  Visited post-deviation states only ({len(seen)}/{len(nxt)} "
                    f"trials): other OF {100 * seen[f'dx_{other}'].mean():+.4f}% at "
                    f"t={pre + 1} | sessions where the non-deviator punishes: "
                    f"{int((per_s > X_UPPER).sum())}/{len(per_s)}"
                )
            else:
                r(f"  Visited post-deviation states only: 0/{len(nxt)} trials.")


# ---- main -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/r1_1_early_stop")
    parser.add_argument(
        "--config",
        default="experiments/01_replicate_baseline/config_r1_1_early_stop.yaml",
    )
    parser.add_argument("--out", default=None, help="Also write the report here.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    config = runner.load_config(args.config)
    n_agents = int(config["n_agents"])
    v_bar = float(config.get("v_bar", 0.0))
    sigma_v = float(config.get("sigma_v", 1.0))
    pre = int((config.get("irf") or {}).get("pre_periods", 3))
    v_grid = build_v_grid(int(config["n_v"]), v_bar, sigma_v)

    r = Report()
    sessions, periods = load_results(results_dir)
    sigmas = sorted(sessions["sigma_u"].unique())
    if len(sigmas) > 1:
        r(f"NOTE: {len(sigmas)} σ_u values present; reporting each separately.")
    for sigma_u in sigmas:
        r(f"\n################ σ_u = {sigma_u:g} ################")
        bench = benchmarks.compute_all_benchmarks(
            n_agents=n_agents,
            xi=float(config["xi"]),
            sigma_u=float(sigma_u),
            theta=float(config["theta"]),
            sigma_v=sigma_v,
            v_grid=v_grid,
        )
        env, _agents, env_bench = runner.build_session(config, float(sigma_u), seed=0)
        # env.action_table holds the quantity of every action at every v, for
        # the global grid (tiled 1-D grid) and for the per-v grid alike.
        grid_game = benchmarks.grid_stage_game(
            env.action_table,
            env.v_grid,
            env_bench["lambda_N"],
            env_bench["pi_N"],
            env_bench["pi_M"],
            v_bar=env.v_bar,
        )
        s_blk = sessions[sessions["sigma_u"] == sigma_u]
        p_blk = periods[periods["sigma_u"] == sigma_u] if not periods.empty else periods
        settled_max = float(
            ((config.get("simulation") or {}).get("diagnostics") or {}).get(
                "settled_max_flip_share", 0.10
            )
        )
        print_summary(r, s_blk, bench, grid_game, settled_max)
        print_session_table(r, s_blk)
        print_histogram(r, s_blk)
        if not p_blk.empty:
            print_noise_irf(r, p_blk, pre, v_bar, n_agents)
            print_deviation_irf(r, p_blk, pre, v_bar, n_agents)

    r("\n=== NOTES ===")
    r(
        f"Classification thresholds (paper OA §4.5): no response if |dev| < {X_LOWER:.0e}"
        f" for both agents; price trigger if dev > {X_UPPER:.0e} for both agents."
    )
    r("Deviations are shock-path minus counterfactual-path at the same period,")
    r("normalised by the session's eval-window mean (sign-adjusted).")
    r(f"Results dir: {results_dir}")
    r(f"Config: {Path(args.config).resolve()}")
    if args.out:
        r.save(Path(args.out))


if __name__ == "__main__":
    main()
