"""Post-hoc impulse responses from saved policies.

Replays the runner's IRF protocol on the saved tables of a finished run, so that the
shock design (sizes, trials) can change without re-running training. Approach (as in
``scripts/deviation_test.py``): rebuild each session's environment with
``runner.build_session`` (seed ``seed_base + session + 10,000``, so the paths never
coincide with training), load the saved tables into the agents and freeze them (greedy
policies), burn in, then for every trial advance ``pre`` periods, make sure v_t ≠ v̄,
deep-copy the environment, apply the shock on the copy at the first post period (noise:
``u_shock = sgn(v_t − v̄)·m·bin_width/λ^N``; deviation: ``+m`` levels on the per-v grid,
``round(sgn(v_t − v̄)·m)`` bins on the global grid) and step the shock and the base path
side by side with common random numbers for ``post`` periods.

Visited states: the training visit counts are not saved with the tables, so a state
counts as visited when the agent's saved row differs from its initial row (the initial
table is rebuilt with the agent's own initialisation; an actor-critic preference row is
"visited" when it is not all zeros; a fixed-policy agent always counts as visited). The
``n_visits_agent_i`` columns of the replay rows hold this proxy (1 or 0), and the report
says so. Trials whose post-shock state was not visited by both agents are excluded from the
tables, as in the visited-state lines of ``scripts/baseline_report.py``.

The rows have the schema of ``periods.parquet`` (``phase = "irf"``), and the session's own
eval rows are taken from ``periods.parquet`` so that responses are normalised by the same
eval means. Output: the tables of ``baseline_report.py`` (order flow and price at t = pre to
pre + 5, classification at t = pre + 1; deviation: deviator and non-deviator responses,
punish sessions) for the visited-state trials, and ``irf_replay_periods.parquet`` next to
``--out``.

Usage:
    PYTHONPATH=src uv run --no-sync python scripts/irf_replay.py \\
        --results-dir results/r1_baseline --config experiments/01_replicate_baseline/config_r1_baseline.yaml \\
        --noise 0.125 0.25 0.5 --deviation 4 8 --trials 100 --out results/r1_baseline/irf_replay.txt
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.simulation import runner
from ai_trading.viz import figures as viz

logger = logging.getLogger("irf_replay")

SEED_OFFSET = 10_000
BURN_IN = 2_000
X_LOWER = 5e-5
X_UPPER = 10 * X_LOWER


def _load_deviation_test() -> Any:
    spec = importlib.util.spec_from_file_location(
        "deviation_test", Path(__file__).resolve().parent / "deviation_test.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        self.lines.append(text)
        print(text)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n")


# ---- helpers -----------------------------------------------------------------------


def find_sessions(results_dir: Path) -> list[tuple[int, Path, Path]]:
    out = []
    for qdir in sorted(results_dir.glob("**/qtables/session_*")):
        m = re.fullmatch(r"session_(\d+)", qdir.name)
        if m:
            out.append((int(m.group(1)), qdir, qdir.parent.parent / "periods.parquet"))
    return sorted(out)


def visited_masks(
    agents: list[Any], q_init: list[np.ndarray | None]
) -> list[np.ndarray]:
    """Per agent, a bool array ``(n_p, n_v, n_v)``: saved row differs from the initial row.

    A fixed-policy agent (no table) counts as visited everywhere.
    """
    masks: list[np.ndarray | None] = []
    for agent, init in zip(agents, q_init):
        q = getattr(agent, "Q", None)
        if q is not None and init is not None:
            masks.append(np.any(np.asarray(q) != init, axis=-1))
        elif hasattr(agent, "get_preferences"):
            masks.append(np.any(agent.get_preferences() != 0.0, axis=-1))
        else:
            masks.append(None)
    shape = next((m.shape for m in masks if m is not None), None)
    if shape is None:
        raise ValueError("no agent with a saved table; nothing to replay")
    return [np.ones(shape, dtype=bool) if m is None else m for m in masks]


def replay_session(
    env: Any,
    agents: list[Any],
    visited: list[np.ndarray],
    session_id: int,
    sigma_u: float,
    *,
    noise: list[float],
    deviation: list[int],
    trials: int,
    pre: int,
    post: int,
) -> list[dict[str, Any]]:
    """Replay the IRF protocol for one session; returns periods-schema rows (phase irf)."""
    n_agents = len(agents)
    v_bar = float(env.v_bar)
    bin_width = float(env.price_grid[1] - env.price_grid[0])
    per_v = getattr(env, "action_mode", "global") == "per_v"
    n_x = env.action_table.shape[1]
    specs: list[tuple[str, float, int]] = [("noise", float(m), -1) for m in noise]
    for m in deviation:
        for j in range(n_agents):
            specs.append(("deviation", float(m), j))

    def act(st: dict[int, np.ndarray]) -> dict[int, int]:
        return {i: a.act(st[i]) for i, a in enumerate(agents)}

    def vis(st: dict[int, np.ndarray]) -> list[int]:
        return [int(visited[i][tuple(int(z) for z in st[i])]) for i in range(n_agents)]

    def row(
        info: dict[str, Any],
        rewards: dict[int, float],
        visits: list[int],
        **common: Any,
    ) -> dict[str, Any]:
        r: dict[str, Any] = {
            "session_id": session_id,
            "sigma_u": sigma_u,
            "v_t": float(info["v_t"]),
            "u_t": float(info["noise"]),
            "u_shock": float(info["u_shock"]),
            "y_t": float(info["order_flow"]),
            "p_t": float(info["price"]),
            "p_bin": int(info["price_bin"]),
            "lambda_hat": float(info["lambda_hat"]),
            "phase": "irf",
            **common,
        }
        for i in range(n_agents):
            r[f"x_agent_{i}"] = float(info["x"][i])
            r[f"profit_agent_{i}"] = float(rewards[i])
            r[f"n_visits_agent_{i}"] = int(visits[i])
        return r

    state = env.reset()
    for _ in range(BURN_IN):
        state, *_ = env.step(act(state))
    period = BURN_IN
    rows: list[dict[str, Any]] = []
    for irf_type, magnitude, dev_agent in specs:
        for trial in range(trials):
            for _ in range(pre):
                state, *_ = env.step(act(state))
                period += 1
            tries = 0
            while env.v_current == v_bar and tries < 50:
                state, *_ = env.step(act(state))
                period += 1
                tries += 1
            if env.v_current == v_bar:
                continue
            sign = 1.0 if env.v_current > v_bar else -1.0
            env_s = copy.deepcopy(env)
            st_s = {i: s.copy() for i, s in state.items()}
            st_b = state
            for k in range(post):
                period += 1
                a_s, a_b = act(st_s), act(st_b)
                u_shock = 0.0
                if k == 0:
                    if irf_type == "noise":
                        u_shock = sign * magnitude * bin_width / env.lambda_N
                    else:
                        shift = int(magnitude) if per_v else round(sign * magnitude)
                        a_s[dev_agent] = int(
                            np.clip(a_s[dev_agent] + shift, 0, n_x - 1)
                        )
                v_s, v_b = vis(st_s), vis(st_b)
                ns_s, r_s, _d, i_s = env_s.step(a_s, u_shock=u_shock)
                ns_b, r_b, _d, i_b = env.step(a_b)
                common = {
                    "period": period,
                    "irf_type": irf_type,
                    "irf_trial": trial,
                    "irf_period": pre + k,
                    "shock_magnitude": magnitude,
                    "deviating_agent": dev_agent,
                }
                rows.append(row(i_s, r_s, v_s, irf_path="shock", **common))
                rows.append(row(i_b, r_b, v_b, irf_path="base", **common))
                st_s, st_b = ns_s, ns_b
            state = st_b
    return rows


def _classify(d0: float, d1: float) -> str:
    if d0 > X_UPPER and d1 > X_UPPER:
        return "price_trigger"
    if abs(d0) < X_LOWER and abs(d1) < X_LOWER:
        return "no_response"
    return "ambiguous"


def print_noise(
    r: Report, periods: pd.DataFrame, pre: int, v_bar: float, n_agents: int
) -> None:
    pairs = viz.irf_pairs(periods, "noise", v_bar, n_agents)
    r("\n=== NOISE-SHOCK IRF, REPLAY (shock − base, sign-adjusted, % of eval mean) ===")
    if pairs.empty:
        r("No noise shocks replayed.")
        return
    for m, blk in pairs.groupby("shock_magnitude"):
        at = blk[blk["irf_period"] == pre]
        flip = float((at["p_bin_s"] != at["p_bin_b"]).mean())
        n_all = viz.n_trials(blk)
        vis = viz.visited_trials(blk, pre, n_agents)
        r(
            f"\nShock {m:g} price bins: bin-flip rate at t={pre}: {flip:.1%} | visited post-shock "
            f"state (proxy: saved row differs from the initial row): {viz.n_trials(vis)}/{n_all} trials"
        )
        if vis.empty:
            continue
        r(
            f"  {'t':>3} {'price %':>10} "
            + " ".join(f"{'OF_' + str(i) + ' %':>10}" for i in range(n_agents))
        )
        for t in range(pre, pre + 6):
            rt = vis[vis["irf_period"] == t]
            if rt.empty:
                continue
            cells = " ".join(
                f"{100 * rt[f'dx_{i}'].mean():>+10.4f}" for i in range(n_agents)
            )
            r(f"  {t:>3} {100 * rt['dp'].mean():>+10.4f} {cells}")
        nxt = vis[vis["irf_period"] == pre + 1]
        per = nxt.groupby("session_id")[[f"dx_{i}" for i in range(n_agents)]].mean()
        labels = [_classify(a, b) for a, b in per.to_numpy()[:, :2]]
        counts = {
            k: labels.count(k) for k in ("price_trigger", "no_response", "ambiguous")
        }
        r(f"  Classification at t={pre + 1} (per session, visited trials): {counts}")


def print_deviation(
    r: Report, periods: pd.DataFrame, pre: int, v_bar: float, n_agents: int
) -> None:
    pairs = viz.irf_pairs(periods, "deviation", v_bar, n_agents)
    r("\n=== UNILATERAL-DEVIATION IRF, REPLAY (shock − base) ===")
    if pairs.empty:
        r("No deviations replayed.")
        return
    eff = viz.effective_deviation_trials(pairs, pre)
    for (m, j), blk in eff.groupby(["shock_magnitude", "deviating_agent"]):
        j = int(j)
        other = 1 - j if n_agents == 2 else None
        n_all = viz.n_trials(
            pairs[(pairs["shock_magnitude"] == m) & (pairs["deviating_agent"] == j)]
        )
        vis = viz.visited_trials(blk, pre, n_agents)
        r(
            f"\nAgent {j} deviates by {m:g} bin(s): effective {viz.n_trials(blk)}/{n_all} trials; "
            f"visited post-deviation state: {viz.n_trials(vis)}"
        )
        if vis.empty or other is None:
            continue
        r(
            f"  {'t':>3} {'dev OF %':>10} {'dev prof %':>11} {'other OF %':>11} {'other prof %':>13}"
        )
        for t in (pre, pre + 1, pre + 2):
            rt = vis[vis["irf_period"] == t]
            r(
                f"  {t:>3} {100 * rt[f'dx_{j}'].mean():>+10.4f} {100 * rt[f'dprof_{j}'].mean():>+11.4f} "
                f"{100 * rt[f'dx_{other}'].mean():>+11.4f} {100 * rt[f'dprof_{other}'].mean():>+13.4f}"
            )
        nxt = vis[vis["irf_period"] == pre + 1]
        per = nxt.groupby("session_id")[f"dx_{other}"].mean()
        r(
            f"  Sessions where the non-deviator punishes (mean OF dev > {X_UPPER:.0e} at t={pre + 1}): {int((per > X_UPPER).sum())}/{len(per)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--noise", type=float, nargs="*", default=[], help="Noise shocks in price bins"
    )
    parser.add_argument(
        "--deviation",
        type=int,
        nargs="*",
        default=[],
        help="Deviation sizes in bins/levels",
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument(
        "--pre",
        type=int,
        default=None,
        help="Default: the config's irf.pre_periods (3)",
    )
    parser.add_argument(
        "--post",
        type=int,
        default=None,
        help="Default: the config's irf.post_periods (6)",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if not args.noise and not args.deviation:
        raise SystemExit("Give at least one --noise size or one --deviation size.")

    dt = _load_deviation_test()
    results_dir = Path(args.results_dir).resolve()
    config = runner.load_config(args.config)
    sigma_u = float(config["sigma_u"])
    irf_cfg = config.get("irf") or {}
    pre = int(irf_cfg.get("pre_periods", 3)) if args.pre is None else args.pre
    post = int(irf_cfg.get("post_periods", 6)) if args.post is None else args.post
    seed_base = int(
        config.get("simulation", {}).get("seed_base", config.get("seed", 0))
    )
    sessions = find_sessions(results_dir)
    if not sessions:
        raise SystemExit(f"No saved tables under {results_dir}.")

    frames: list[pd.DataFrame] = []
    n_agents = int(config["n_agents"])
    v_bar = float(config.get("v_bar", 0.0))
    for sid, qdir, periods_file in sessions:
        env, agents, _bench = runner.build_session(
            config, sigma_u, seed=seed_base + sid + SEED_OFFSET
        )
        q_init = dt.restore_agents(agents, qdir)
        visited = visited_masks(agents, q_init)
        rows = replay_session(
            env,
            agents,
            visited,
            sid,
            sigma_u,
            noise=args.noise,
            deviation=args.deviation,
            trials=args.trials,
            pre=pre,
            post=post,
        )
        replay = pd.DataFrame(rows)
        if periods_file.exists():
            ev = pd.read_parquet(periods_file)
            ev = ev[(ev["session_id"] == sid) & (ev["phase"] == "eval")]
            replay = pd.concat([ev, replay], ignore_index=True)
        else:
            logger.warning(
                "session %d: no periods.parquet (eval rows); responses cannot be normalised",
                sid,
            )
        frames.append(replay)
        logger.info("session %d replayed", sid)
    periods = pd.concat(frames, ignore_index=True)

    r = Report()
    r(
        f"IRF replay | {len(sessions)} session(s) | noise {args.noise} price bins | deviation "
        f"{args.deviation} | trials {args.trials} | pre {pre} post {post} | seeds +{SEED_OFFSET}"
    )
    r(
        "Visited-state proxy: a state counts as visited when the agent's saved table row differs from its initial row."
    )
    print_noise(r, periods, pre, v_bar, n_agents)
    print_deviation(r, periods, pre, v_bar, n_agents)
    r(f"\nResults dir: {results_dir}\nConfig: {Path(args.config).resolve()}")
    if args.out:
        out = Path(args.out)
        r.save(out)
        periods[periods["phase"] == "irf"].to_parquet(
            out.parent / "irf_replay_periods.parquet", index=False
        )


if __name__ == "__main__":
    main()
