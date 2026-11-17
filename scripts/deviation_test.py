"""Best-response deviation test.

Asks whether the learned strategies of a run are an equilibrium of the learned
game, and if not, why the agents do not deviate. Three parts:

1. **Static incentives** (eval play in ``periods.parquet``): in every eval
   period with v ≠ v̄, the one-period expected payoff of each grid action
   against the opponent's actual quantity, ``(v − v̄ − λ^N (x + x_opp)) · x``.
   Reports how often the agent plays the grid best response, the available
   one-shot gain on the grid and on the continuum, and their ratio (how much of
   the deviation gain the grid keeps).
2. **Dynamic test** (saved Q-tables, frozen greedy policies, common random
   numbers): at random on-path states one agent deviates once, either to the
   one-period grid best response (``grid_br``) or one bin in the aggressive
   direction (``one_up``, the IRF deviation), then both follow their greedy
   policies. The discounted profit difference over ``--horizon`` periods
   against the same path without the deviation is split into the deviation
   period and the continuation. The opponent's reaction in the next period is
   recorded, signed so that positive means more aggressive.
3. **Value staleness** (Q-learners only): in the deviation state, the stored
   Q-value of the greedy and of the deviation action against their one-step
   Bellman targets computed from the agent's own current Q-table. A large
   negative error for the deviation action alone means that the agent's
   estimate of the deviation is outdated (an exploration effect), not that the
   deviation is punished.

Works with the global action grid (the earlier specification) and with the per-v
grid (the baseline specification, ``action_grid: per_v``): quantities come from
``env.action_table`` and the
``one_up`` direction from ``env.action_mode``. Value states in which every
action is the same trade (per-v grid at v = v̄) are skipped.

Agents: the tabular Q-learner is restored from ``agent_<i>.npy``, the
actor-critic from ``agent_<i>_prefs.npy`` and ``agent_<i>_values.npy``, the
fixed-policy agent from its config; all are frozen (eval mode). Part 3 applies
to Q-learners only: for any other deviator ``q_greedy``, ``q_dev``,
``tgt_greedy`` and ``tgt_dev`` are NaN, and ``opp_untouched_actions`` is NaN
when the opponent is not a Q-learner.

Seeds: the simulation of session ``s`` uses ``seed_base + s + 10_000`` so it
never coincides with the training stream. Output is deterministic.

Usage:
    PYTHONPATH=src uv run --no-sync python scripts/deviation_test.py \\
        --results-dir results/r1_2_earlier_spec \\
        --config experiments/01_replicate_baseline/config_r1_2_earlier_spec.yaml \\
        --out results/r1_2_earlier_spec/deviation_test.txt
    (about 1.5 min for 30 sessions; writes deviation_trials.parquet and deviation_static.parquet next to --out)
"""

from __future__ import annotations

import argparse
import copy
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.agents.actor_critic import ActorCriticAgent
from ai_trading.agents.fixed_policy import FixedChiAgent
from ai_trading.agents.q_learning import QLearningAgent
from ai_trading.simulation import runner

logger = logging.getLogger("deviation_test")

SEED_OFFSET = 10_000
BURN_IN = 2_000
KINDS = ("grid_br", "one_up")


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


# ---- pure helpers ------------------------------------------------------------


def action_table(env: Any) -> np.ndarray:
    """Order quantity of action a at value index k, shape ``(n_v, n_x)``.

    Uses ``env.action_table`` (set by ``MarketEnv`` for both action-grid
    modes); for an environment object without it, tiles the global 1-D
    action grid.
    """
    table = getattr(env, "action_table", None)
    if table is not None:
        return np.asarray(table, dtype=np.float64)
    return np.tile(np.asarray(env.action_grid, dtype=np.float64), (len(env.v_grid), 1))


def grid_payoffs(
    quantities: np.ndarray, dv: float, x_opp: float, lam: float
) -> np.ndarray:
    """Expected one-period payoff of every grid quantity against ``x_opp``.

    Args:
        quantities: Grid quantities available at the current value state.
        dv: v_t − v̄.
        x_opp: Sum of the other agents' quantities.
        lam: Price-impact coefficient.

    Returns:
        Array of ``(dv − lam·(x + x_opp))·x`` over the grid.
    """
    q = np.asarray(quantities, dtype=np.float64)
    return (dv - lam * (q + x_opp)) * q


def continuous_best_payoff(dv: float | np.ndarray, x_opp: Any, lam: float) -> Any:
    """Payoff of the unrestricted best response, ``(dv − lam·x_opp)² / (4·lam)``."""
    return (dv - lam * x_opp) ** 2 / (4.0 * lam)


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean (sessions are the resampling unit)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=np.float64)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def one_up_index(a: int, dv: float, n_x: int, per_v: bool) -> int:
    """Action index one bin more aggressive than ``a`` (clipped to the grid)."""
    step = 1 if per_v else int(np.sign(dv))
    return int(np.clip(a + step, 0, n_x - 1))


# ---- loading -------------------------------------------------------------------


def find_sessions(results_dir: Path) -> list[tuple[int, Path, Path]]:
    """Return ``(session_id, qtable_dir, periods_file)`` for every saved session."""
    out: list[tuple[int, Path, Path]] = []
    for qdir in sorted(results_dir.glob("**/qtables/session_*")):
        m = re.fullmatch(r"session_(\d+)", qdir.name)
        if not m:
            continue
        periods = qdir.parent.parent / "periods.parquet"
        out.append((int(m.group(1)), qdir, periods))
    return sorted(out)


def restore_agents(agents: list[Any], qdir: Path) -> list[np.ndarray | None]:
    """Load the saved policies into the agents and freeze them.

    Returns the initial Q-table of every Q-learner (to flag untouched actions)
    and ``None`` for every other agent.
    """
    q_init: list[np.ndarray | None] = []
    for i, agent in enumerate(agents):
        if isinstance(agent, QLearningAgent):
            q_init.append(agent.Q.copy())
            agent.Q[...] = np.load(qdir / f"agent_{i}.npy")
        elif isinstance(agent, ActorCriticAgent):
            agent.set_preferences(np.load(qdir / f"agent_{i}_prefs.npy"))
            agent.set_values(np.load(qdir / f"agent_{i}_values.npy"))
            q_init.append(None)
        elif isinstance(agent, FixedChiAgent):
            q_init.append(None)
        else:
            raise NotImplementedError(
                f"agent {i} is {type(agent).__name__}; deviation_test.py restores "
                "qlearner, actor_critic and fixed_chi agents only."
            )
        agent.set_eval_mode(True)
    return q_init


# ---- part 1: static incentives --------------------------------------------------


def static_incentives(
    periods: pd.DataFrame,
    table: np.ndarray,
    v_grid: np.ndarray,
    v_bar: float,
    lam: float,
) -> pd.DataFrame:
    """Per agent and |v − v̄| level: BR share and one-shot deviation gains."""
    ev = periods[(periods["phase"] == "eval") & (periods["irf_type"] == "")]
    n_agents = sum(1 for c in ev.columns if re.fullmatch(r"x_agent_\d+", c))
    v = ev["v_t"].to_numpy()
    dv = v - v_bar
    k = np.abs(v[:, None] - v_grid[None, :]).argmin(axis=1)
    keep = np.abs(dv) > 1e-9
    rows: list[dict[str, Any]] = []
    for i in range(n_agents):
        x_i = ev[f"x_agent_{i}"].to_numpy()
        x_opp = sum(ev[f"x_agent_{j}"].to_numpy() for j in range(n_agents) if j != i)
        q = table[k]  # (T, n_x)
        pay = (dv[:, None] - lam * (q + x_opp[:, None])) * q
        cur = (dv - lam * (x_i + x_opp)) * x_i
        best = pay.max(axis=1)
        cont = continuous_best_payoff(dv, x_opp, lam)
        level = np.round(np.abs(dv), 3)
        for lv in np.unique(level[keep]):
            m = keep & (level == lv)
            rows.append(
                {
                    "session_id": int(ev["session_id"].iloc[0]),
                    "agent": i,
                    "abs_dv": float(lv),
                    "n": int(m.sum()),
                    "br_share": float(np.mean(np.isclose(cur[m], best[m]))),
                    "gain_grid": float(np.mean(best[m] - cur[m])),
                    "gain_cont": float(np.mean(cont[m] - cur[m])),
                    "profit": float(np.mean(cur[m])),
                }
            )
    return pd.DataFrame(rows)


# ---- part 2 and 3: dynamic test and value staleness -----------------------------


def dynamic_test(
    env: Any,
    agents: list[Any],
    q_init: list[np.ndarray | None],
    session_id: int,
    *,
    trials: int,
    horizon: int,
    rho: float,
) -> pd.DataFrame:
    """One-shot deviations under frozen greedy policies with common random numbers.

    Args:
        env: Fresh environment of the session (its RNG drives the paths).
        agents: Frozen agents (eval mode) with the learned policies.
        q_init: The agents' initial Q-tables (to flag untouched actions);
            ``None`` for agents without a Q-table (part 3 is NaN for them).
        session_id: Session id, stored in the output.
        trials: Effective deviations per (deviating agent, kind).
        horizon: Periods simulated after the deviation (deviation period included).
        rho: Discount factor for the discounted sums.

    Returns:
        One row per trial.
    """
    n_agents = len(agents)
    table = action_table(env)
    n_x = table.shape[1]
    lam, v_bar = float(env.lambda_N), float(env.v_bar)
    per_v = getattr(env, "action_mode", "global") == "per_v"
    disc = rho ** np.arange(horizon)

    def act(st: dict[int, np.ndarray]) -> dict[int, int]:
        return {i: a.act(st[i]) for i, a in enumerate(agents)}

    nan = float("nan")

    def qrow(i: int, st: dict[int, np.ndarray]) -> np.ndarray | None:
        q = getattr(agents[i], "Q", None)
        return None if q is None else q[tuple(int(z) for z in st[i])]

    state = env.reset()
    for _ in range(BURN_IN):
        state, *_ = env.step(act(state))

    rows: list[dict[str, Any]] = []
    for dev in range(n_agents):
        others = [j for j in range(n_agents) if j != dev]
        for kind in KINDS:
            done, guard = 0, 0
            while done < trials and guard < 50 * trials:
                guard += 1
                for _ in range(int(env.rng.integers(3, 12))):
                    state, *_ = env.step(act(state))
                acts = act(state)
                k = int(state[dev][2])
                dv = float(env.v_grid[k] - v_bar)
                if abs(dv) < 1e-9 or np.ptp(table[k]) == 0.0:
                    continue
                x_opp = float(sum(table[k, acts[j]] for j in others))
                if kind == "grid_br":
                    target = int(np.argmax(grid_payoffs(table[k], dv, x_opp, lam)))
                else:
                    target = one_up_index(acts[dev], dv, n_x, per_v)
                if target == acts[dev]:
                    continue
                q_here = qrow(dev, state)
                rec: dict[str, Any] = {
                    "session_id": session_id,
                    "dev": dev,
                    "kind": kind,
                    "abs_dv": round(abs(dv), 3),
                    "q_greedy": nan if q_here is None else float(q_here[acts[dev]]),
                    "q_dev": nan if q_here is None else float(q_here[target]),
                }
                env_s = copy.deepcopy(env)
                st_s = {i: s.copy() for i, s in state.items()}
                st_b = state
                diff = np.zeros(horizon)
                for t in range(horizon):
                    a_s, a_b = act(st_s), act(st_b)
                    if t == 0:
                        a_s[dev] = target
                    if t == 1:
                        j = others[0]
                        k1 = int(st_s[j][2])
                        sgn = float(np.sign(env.v_grid[k1] - v_bar))
                        dx = float(table[k1, a_s[j]] - table[int(st_b[j][2]), a_b[j]])
                        # A reaction is a change of the traded quantity. Where
                        # every action is the same trade (per-v grid at
                        # v = v̄) a different greedy index is not a reaction.
                        rec["opp_reacts"] = dx != 0.0 or (
                            a_s[j] != a_b[j] and np.ptp(table[k1]) > 0.0
                        )
                        rec["opp_aggr"] = sgn * dx
                        rec["opp_state_changed"] = not np.array_equal(st_s[j], st_b[j])
                        q_opp, q_opp0 = qrow(j, st_s), q_init[j]
                        rec["opp_untouched_actions"] = (
                            nan
                            if q_opp is None or q_opp0 is None
                            else float(
                                np.sum(q_opp == q_opp0[tuple(int(z) for z in st_s[j])])
                            )
                        )
                    nst_s, r_s, *_ = env_s.step(a_s)
                    nst_b, r_b, *_ = env.step(a_b)
                    diff[t] = r_s[dev] - r_b[dev]
                    if t == 0:
                        q_ns, q_nb = qrow(dev, nst_s), qrow(dev, nst_b)
                        rec["tgt_dev"] = (
                            nan if q_ns is None else float(r_s[dev] + rho * q_ns.max())
                        )
                        rec["tgt_greedy"] = (
                            nan if q_nb is None else float(r_b[dev] + rho * q_nb.max())
                        )
                    st_s, st_b = nst_s, nst_b
                state = st_b
                rec["gain0"] = float(diff[0])
                rec["cont"] = float((disc[1:] * diff[1:]).sum())
                rec["total"] = float((disc * diff).sum())
                rows.append(rec)
                done += 1
            if done < trials:
                logger.warning(
                    "session %d agent %d %s: only %d of %d effective trials",
                    session_id,
                    dev,
                    kind,
                    done,
                    trials,
                )
    return pd.DataFrame(rows)


# ---- report --------------------------------------------------------------------


def print_static(r: Report, st: pd.DataFrame, gap: float, n_v: int) -> None:
    r("=== 1. STATIC INCENTIVES (eval play, v ≠ v̄) ===")
    r(
        f"{'|v−v̄|':>7} {'BR share':>9} {'gain grid':>10} {'gain cont':>10} "
        f"{'grid/cont':>10} {'profit':>8}"
    )
    for lv, g in st.groupby("abs_dv"):
        r(
            f"{lv:>7.3f} {g.br_share.mean():>9.3f} {g.gain_grid.mean():>10.3f} "
            f"{g.gain_cont.mean():>10.3f} {g.gain_grid.mean() / g.gain_cont.mean():>10.3f} "
            f"{g.profit.mean():>8.2f}"
        )
    w = st.assign(wg=st.gain_grid * st.n, wc=st.gain_cont * st.n, wb=st.br_share * st.n)
    agg = w.groupby(["session_id", "agent"])[["wg", "wc", "wb", "n"]].sum()
    gg, gc, br = (agg.wg / agg.n), (agg.wc / agg.n), (agg.wb / agg.n)
    share_nonzero = (n_v - 1) / n_v if n_v % 2 == 1 else 1.0
    r(
        f"Pooled: BR share {br.mean():.3f} | grid gain {gg.mean():.3f} | continuous gain "
        f"{gc.mean():.3f} | grid keeps {gg.mean() / gc.mean():.1%} of the deviation gain"
    )
    r(
        f"Grid one-shot gain per period (all v) / (π^M − π^N): "
        f"{gg.mean() * share_nonzero / gap:.3f}"
    )


def print_dynamic(r: Report, dt: pd.DataFrame, horizon: int, rho: float) -> None:
    r(
        f"\n=== 2. DYNAMIC TEST (frozen policies, horizon {horizon}, ρ = {rho:g}, CRN) ==="
    )
    r("gain0 = deviation period; cont = discounted continuation; total = gain0 + cont.")
    r(
        "opp react = opponent's action changes at t+1; aggr = share of those that are more "
        "aggressive."
    )
    hdr = (
        f"{'kind':<8} {'|v−v̄|':>6} {'n':>5} {'gain0':>7} {'cont':>7} {'total':>7} "
        f"{'P(tot>0)':>8} {'opp react':>9} {'aggr':>5} {'untouched':>9}"
    )
    r(hdr)
    for (kind, lv), g in dt.groupby(["kind", "abs_dv"]):
        react = g[g.opp_reacts.astype(bool)]
        aggr = float((react.opp_aggr > 0).mean()) if len(react) else float("nan")
        r(
            f"{kind:<8} {lv:>6.3f} {len(g):>5} {g.gain0.mean():>+7.2f} {g.cont.mean():>+7.2f} "
            f"{g.total.mean():>+7.2f} {(g.total > 0).mean():>8.2f} "
            f"{g.opp_reacts.astype(bool).mean():>9.2f} {aggr:>5.2f} "
            f"{(g.opp_untouched_actions > 0).mean():>9.2f}"
        )
    br = dt[dt.kind == "grid_br"]
    g = br.groupby("session_id")["total"]
    per, se = g.mean(), g.std() / np.sqrt(g.count())
    pos, neg = int((per - 2 * se > 0).sum()), int((per + 2 * se < 0).sum())
    lo, hi = bootstrap_ci(per.to_numpy())
    r(
        f"Grid-BR deviation, total gain: mean over sessions {per.mean():+.3f} "
        f"[bootstrap 95% CI {lo:+.3f}, {hi:+.3f}]"
    )
    r(
        f"Sessions: pays (mean − 2 SE > 0) {pos}/{len(per)} | does not pay (mean + 2 SE < 0) "
        f"{neg}/{len(per)} | unclear {len(per) - pos - neg}/{len(per)} | point estimate > 0 "
        f"{int((per > 0).sum())}/{len(per)}"
    )


def print_staleness(r: Report, dt: pd.DataFrame) -> None:
    r("\n=== 3. VALUE STALENESS (stored Q − one-step Bellman target, own Q-table) ===")
    n_nonq = int(dt["q_greedy"].isna().sum())
    if n_nonq:
        r(
            f"Q-learners only: {n_nonq} of {len(dt)} trials have a deviator without a "
            "Q-table (actor-critic or fixed policy); their values are NaN and are left "
            "out of the means below."
        )
    r(
        f"{'kind':<8} {'|v−v̄|':>6} {'err greedy':>10} {'err dev':>8} {'Q gap':>7} "
        f"{'P(refreshed dev > greedy)':>26}"
    )
    for (kind, lv), g in dt.groupby(["kind", "abs_dv"]):
        r(
            f"{kind:<8} {lv:>6.3f} {(g.q_greedy - g.tgt_greedy).mean():>+10.2f} "
            f"{(g.q_dev - g.tgt_dev).mean():>+8.2f} {(g.q_dev - g.q_greedy).mean():>+7.2f} "
            f"{(g.tgt_dev > g.tgt_greedy).mean():>26.2f}"
        )
    r(
        "Reading: err dev ≪ err greedy ≈ 0 means the agent's value of the deviation is outdated"
    )
    r(
        "(it is not refreshed because the action is no longer explored), not that it is punished."
    )


def print_sessions(r: Report, dt: pd.DataFrame) -> None:
    r("\n=== PER SESSION (grid-BR deviation, mean over agents and trials) ===")
    g = dt[dt.kind == "grid_br"].groupby("session_id")
    per = g[["gain0", "cont", "total"]].mean()
    per["se"] = g["total"].std() / np.sqrt(g["total"].count())
    r(f"{'sess':>4} {'gain0':>7} {'cont':>7} {'total':>7} {'SE':>6}")
    for sid, row in per.iterrows():
        r(
            f"{sid:>4} {row.gain0:>+7.2f} {row.cont:>+7.2f} {row.total:>+7.2f} {row.se:>6.2f}"
        )


# ---- main ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--trials",
        type=int,
        default=500,
        help="Effective deviations per agent, kind and session.",
    )
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--out", default=None, help="Also write the report here.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    results_dir = Path(args.results_dir).resolve()
    config = runner.load_config(args.config)
    sigma_u = float(config["sigma_u"])
    rho = float(config["rho"])
    seed_base = int(
        config.get("simulation", {}).get("seed_base", config.get("seed", 0))
    )
    sessions = find_sessions(results_dir)
    if not sessions:
        raise SystemExit(f"No saved Q-tables under {results_dir}.")

    static_frames, dyn_frames = [], []
    gap, n_v = float("nan"), 0
    for sid, qdir, periods_file in sessions:
        env, agents, bench = runner.build_session(
            config, sigma_u, seed=seed_base + sid + SEED_OFFSET
        )
        gap = float(bench["pi_M"] - bench["pi_N"])
        n_v = len(env.v_grid)
        q_init = restore_agents(agents, qdir)
        if periods_file.exists():
            p = pd.read_parquet(periods_file)
            p = p[p["session_id"] == sid]
            static_frames.append(
                static_incentives(
                    p, action_table(env), env.v_grid, env.v_bar, env.lambda_N
                )
            )
        dyn_frames.append(
            dynamic_test(
                env,
                agents,
                q_init,
                sid,
                trials=args.trials,
                horizon=args.horizon,
                rho=rho,
            )
        )
        logger.info("session %d done", sid)

    dt = pd.concat(dyn_frames, ignore_index=True)
    r = Report()
    r(
        f"Best-response deviation test | {len(sessions)} session(s) | trials {args.trials} | "
        f"horizon {args.horizon}"
    )
    if static_frames:
        st = pd.concat(static_frames, ignore_index=True)
        print_static(r, st, gap, n_v)
    else:
        st = pd.DataFrame()
        r("No periods.parquet found: static part skipped.")
    print_dynamic(r, dt, args.horizon, rho)
    print_staleness(r, dt)
    print_sessions(r, dt)
    r(f"\nResults dir: {results_dir}")
    r(f"Config: {Path(args.config).resolve()}")
    if args.out:
        out = Path(args.out)
        r.save(out)
        dt.to_parquet(out.parent / "deviation_trials.parquet", index=False)
        if not st.empty:
            st.to_parquet(out.parent / "deviation_static.parquet", index=False)


if __name__ == "__main__":
    main()
