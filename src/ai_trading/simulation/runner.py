"""Simulation runner — orchestrates training, evaluation, and IRF shocks.

Top-level entry point :func:`run_experiment` loads a YAML config, sweeps over
``sigma_u``, runs ``n_sim`` independent training sessions per σ_u, and writes
two-tier Parquet output plus run metadata:

- ``sessions.parquet`` — one row per session: convergence status, Δ^C (per-path
  and expected-value definitions), λ̂, χ̂, per-agent profits, market-quality
  metrics, throughput, and code version.
- ``periods.parquet`` — one row per saved period (the first
  ``save_eval_periods`` eval periods and every IRF period; training periods
  are discarded).
- ``trajectory.parquet`` — only with ``simulation.diagnostics.enabled``: one
  row per completed training block (Δ^C learning curve and block stability).
- ``run_metadata.json`` — resolved config, CLI overrides, code version,
  timestamps, library versions.

Each session has four phases:

1. **Training** — ``reset → act → step → update`` until convergence or the
   ``max_periods`` cap. Convergence: every ``check_every`` periods, snapshot
   each Q-learner's greedy policy and compare to the previous snapshot;
   declare converged when ``unchanged_checks * check_every`` meets or
   exceeds ``convergence_periods``. States in which every action is the
   same trade (``action_grid: per_v`` at v = v̄) are ignored in the
   comparison (:func:`_policy_snapshot`).
2. **Evaluation** — ``eval_periods`` periods. With ``eval_greedy: true``
   (default) agents are frozen in evaluation mode (greedy, no learning).
3. **IRF** — impulse responses with a no-shock counterfactual path under
   common random numbers (see :func:`_run_irf`): adverse noise shocks sized
   in price-bin units, and unilateral deviations by one agent.
4. **Metrics** — Δ^C against per-path realised benchmarks (Online Appendix
   eq. IA.4.1; ``delta_c``) and against the expected-value grid-corrected
   benchmarks (``delta_c_expected``), χ̂^C, and average L^C / E^C / I^C.

Session RNG streams come from ``np.random.SeedSequence(seed).spawn(1 + I)``
(environment first, then one per agent), with ``seed = seed_base +
session_id``.

CLI usage::

    uv run python -m ai_trading.simulation.runner --config <path.yaml> \
        [--session-id-start K] [--n-sim N] [--output-dir DIR] [--max-periods N]

References:
    Paper §3.2 (Definition 3.1, eq. 3.6), §5.3 (IRFs); Online Appendix §4.1
    (eqs. IA.4.1–IA.4.7), §4.5 (IRF classification).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ai_trading.agents.actor_critic import ActorCriticAgent
from ai_trading.agents.base import Agent, Transition
from ai_trading.agents.fixed_policy import FixedChiAgent
from ai_trading.agents.q_learning import QLearningAgent
from ai_trading.env.market import MarketEnv
from ai_trading.simulation import benchmarks, metrics

logger = logging.getLogger(__name__)


AGENT_REGISTRY: dict[str, type[Agent]] = {
    "qlearner": QLearningAgent,
    "actor_critic": ActorCriticAgent,
    "fixed_chi": FixedChiAgent,
}


_EXPERIMENT_SECTIONS: tuple[str, ...] = (
    "experiment",
    "agents",
    "sweep",
    "simulation",
    "irf",
    "runner",
    "output",
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and merge an experiment YAML with its referenced shared params.

    The shared file forms the base dict; the experiment's top-level
    sections (``experiment``, ``agents``, ``sweep``, ``simulation``,
    ``irf``, ``runner``, ``output``) are added on top; finally an
    optional ``overrides:`` block in the experiment YAML replaces matching
    keys in the base dict.

    Args:
        config_path: Path to the experiment YAML.

    Returns:
        The merged config dict.
    """
    exp_path = Path(config_path).resolve()
    with exp_path.open() as f:
        experiment = yaml.safe_load(f) or {}

    params_rel = experiment.get("params")
    if not params_rel:
        raise ValueError(
            f"Experiment config {exp_path} must include a 'params:' key "
            "pointing to a shared params YAML."
        )
    params_path = (exp_path.parent / params_rel).resolve()
    with params_path.open() as f:
        shared = yaml.safe_load(f) or {}

    merged: dict[str, Any] = dict(shared)
    for section in _EXPERIMENT_SECTIONS:
        if section in experiment:
            merged[section] = experiment[section]

    for key, value in experiment.get("overrides", {}).items():
        merged[key] = value

    return merged


def _make_agent(
    entry: dict[str, Any],
    agent_id: int,
    base_config: dict[str, Any],
    grids: dict[str, np.ndarray],
) -> Agent:
    """Instantiate one agent from a YAML entry via :data:`AGENT_REGISTRY`."""
    agent_type = entry["type"]
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent type {agent_type!r}; "
            f"registered types: {sorted(AGENT_REGISTRY)}."
        )
    cls = AGENT_REGISTRY[agent_type]
    agent_cfg = dict(base_config)
    agent_cfg.update(entry.get("config", {}))
    return cls(agent_id=agent_id, config=agent_cfg, grids=grids)


def _degenerate_v_mask(action_table: np.ndarray) -> np.ndarray | None:
    """Value indices at which every action is the same trade, or None.

    A value index k is degenerate if all entries of ``action_table[k]`` are
    equal. This happens in ``action_grid: per_v`` mode at v = v̄, where every
    action trades zero. Returns a bool array of shape ``(n_v,)``, or ``None``
    when no row is degenerate (always the case for the global grid).
    """
    table = np.asarray(action_table)
    mask = np.ptp(table, axis=1) == 0.0
    return mask if bool(mask.any()) else None


def _policy_snapshot(
    agent: Agent, degenerate_v: np.ndarray | None = None
) -> np.ndarray | None:
    """Greedy-policy snapshot for convergence checking, or None if N/A.

    This is the one place where policy snapshots are taken (convergence
    check and stability diagnostic). In states whose current value index is
    degenerate (see :func:`_degenerate_v_mask`) every action is the same
    trade, so a change of the greedy index there is not a change of
    strategy; those entries are set to 0 before snapshots are compared.

    Args:
        agent: The agent to snapshot.
        degenerate_v: Bool mask over value indices, shape ``(n_v,)``, or
            ``None`` if there is nothing to mask.

    Returns:
        Greedy action index per state, shape ``(n_p, n_v, n_v)`` with the
        current value on the last axis, or ``None`` for non-tabular agents.
    """
    snap = agent.greedy_policy()
    if snap is None or degenerate_v is None:
        return snap
    snap = np.array(snap, copy=True)
    snap[..., degenerate_v] = 0
    return snap


def _check_convergence_step(
    agents: list[Agent],
    snapshots: list[np.ndarray] | None,
    unchanged_checks: int,
    check_every: int,
    convergence_periods: int,
    degenerate_v: np.ndarray | None = None,
) -> tuple[bool, list[np.ndarray] | None, int]:
    """One convergence check, called every ``check_every`` periods.

    Returns ``(converged, new_snapshots, new_unchanged_checks)``. If any
    agent lacks a Q-table the check is skipped (returns the prior state
    untouched and ``converged=False``).

    Args:
        agents: Live agent list.
        snapshots: Previous greedy-policy snapshots (or ``None`` on the
            first check).
        unchanged_checks: Count of consecutive unchanged checks observed
            so far (excluding the initial snapshot).
        check_every: Period cadence at which this function is invoked.
        convergence_periods: Convergence threshold (in periods of
            unchanged policy).
        degenerate_v: Value indices to ignore in the comparison (see
            :func:`_policy_snapshot`); ``None`` compares all states.

    Returns:
        Tuple ``(converged, snapshots, unchanged_checks)``.
    """
    new_snapshots: list[np.ndarray] = []
    for agent in agents:
        snap = _policy_snapshot(agent, degenerate_v)
        if snap is None:
            return False, snapshots, unchanged_checks
        new_snapshots.append(snap)

    if snapshots is None:
        return False, new_snapshots, 0

    all_match = all(
        np.array_equal(old, new) for old, new in zip(snapshots, new_snapshots)
    )
    if all_match:
        new_unchanged = unchanged_checks + 1
    else:
        new_unchanged = 0

    converged = new_unchanged * check_every >= convergence_periods
    return converged, new_snapshots, new_unchanged


def _ols_slope(y: np.ndarray, x: np.ndarray) -> float:
    """Slope of OLS y ~ x with intercept. Returns 0.0 when ``Var(x) = 0``."""
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x_centered, y - y.mean()) / denom)


def _onpath_mask(visits: np.ndarray, window_len: int, min_share: float) -> np.ndarray:
    """States visited in at least ``min_share`` of the window's periods."""
    return np.asarray(visits) >= min_share * window_len


def _degenerate_state_mask(
    degenerate_v: np.ndarray | None, n_p: int, n_v: int
) -> np.ndarray | None:
    """Flat-state mask of degenerate states, or None.

    A flat state ``f = p_bin·n_v·n_v + v_prev_bin·n_v + v_bin`` (the index
    used by the training diagnostics) is degenerate when its CURRENT value
    index ``v_bin`` is degenerate (see :func:`_degenerate_v_mask`).

    Args:
        degenerate_v: Bool mask over value indices, shape ``(n_v,)``, or None.
        n_p: Number of price bins.
        n_v: Number of value bins.

    Returns:
        Bool array of shape ``(n_p·n_v·n_v,)``, or None if ``degenerate_v`` is
        None (global grid).
    """
    if degenerate_v is None:
        return None
    mask = np.asarray(degenerate_v, dtype=bool)
    return np.broadcast_to(mask, (n_p, n_v, n_v)).ravel()


def _stability_stats(
    flipped: list[np.ndarray],
    visits: np.ndarray,
    window_len: int,
    min_share: float,
    exclude: np.ndarray | None = None,
) -> tuple[float, float, int]:
    """Share of on-path states whose greedy action changed in a window.

    A state is on-path if it was visited in at least ``min_share`` of the
    window's periods. Shares are averaged over the agents in ``flipped``
    (one bool mask per agent with a tabular policy).

    Args:
        flipped: Per-agent bool arrays of shape ``(n_states,)``; True where
            the greedy action changed at least once in the window.
        visits: Visits per flat state in the window, shape ``(n_states,)``.
        window_len: Number of training periods in the window.
        min_share: On-path threshold as a share of ``window_len``.
        exclude: Optional bool array of shape ``(n_states,)``. States marked
            True are removed from the on-path set before both shares are
            computed (used for the degenerate states of the per-v grid,
            which can never flip). ``None`` keeps every state.

    Returns:
        Tuple ``(flip_share, flip_share_weighted, n_onpath)``;
        ``(nan, nan, 0)`` if no state is on-path or no mask is given.
        ``n_onpath`` counts the on-path states after the exclusion.
    """
    visits = np.asarray(visits)
    onpath = _onpath_mask(visits, window_len, min_share)
    if exclude is not None:
        onpath = onpath & ~np.asarray(exclude, dtype=bool)
    n_onpath = int(np.count_nonzero(onpath))
    n_tab = len(flipped)
    if n_onpath == 0 or n_tab == 0:
        return math.nan, math.nan, 0
    onpath_visits = float(visits[onpath].sum())
    n_flipped = sum(int(np.count_nonzero(f & onpath)) for f in flipped)
    w_flipped = sum(float(visits[f & onpath].sum()) for f in flipped)
    flip_share = n_flipped / (n_tab * n_onpath)
    flip_share_weighted = (
        w_flipped / (n_tab * onpath_visits) if onpath_visits > 0 else math.nan
    )
    return float(flip_share), float(flip_share_weighted), n_onpath


def _trajectory_eval(
    env: MarketEnv,
    agents: list[Agent],
    state: dict[int, np.ndarray],
    n_periods: int,
    bench: dict[str, float],
    v_bar: float,
) -> tuple[list[float], list[float]]:
    """Greedy-policy Δ^C and χ̂ per agent on deep copies (learning curve).

    The live env, agents and their RNG streams are not touched: everything
    runs on ``copy.deepcopy`` copies in evaluation mode. Δ^C uses the
    per-path benchmarks of the eval block (Online Appendix eq. IA.4.1).

    Returns:
        Tuple ``(delta_c_agents, chi_hat_agents)``.
    """
    n_agents = len(agents)
    env_c = copy.deepcopy(env)
    agents_c = copy.deepcopy(agents)
    state_c = {i: s.copy() for i, s in state.items()}
    for a in agents_c:
        a.set_eval_mode(True)
    v_rec = np.empty(n_periods, dtype=np.float64)
    u_rec = np.empty(n_periods, dtype=np.float64)
    x_rec = np.empty((n_agents, n_periods), dtype=np.float64)
    prof_rec = np.empty((n_agents, n_periods), dtype=np.float64)
    for k in range(n_periods):
        actions = {i: a.act(state_c[i]) for i, a in enumerate(agents_c)}
        state_c, rewards, _done, info = env_c.step(actions)
        v_rec[k] = float(info["v_t"])
        u_rec[k] = float(info["noise"])
        for i in range(n_agents):
            x_rec[i, k] = float(info["x"][i])
            prof_rec[i, k] = float(rewards[i])
    pi_n_path = float(
        np.mean(
            benchmarks.realised_benchmark_profit(
                v_rec, u_rec, bench["chi_N"], bench["lambda_N"], n_agents, v_bar
            )
        )
    )
    pi_m_path = float(
        np.mean(
            benchmarks.realised_benchmark_profit(
                v_rec, u_rec, bench["chi_M"], bench["lambda_M"], n_agents, v_bar
            )
        )
    )
    delta_agents = [
        float(
            metrics.collusion_capacity(
                pi_C=float(np.mean(prof_rec[i])), pi_N=pi_n_path, pi_M=pi_m_path
            )
        )
        for i in range(n_agents)
    ]
    chi_agents = [_ols_slope(x_rec[i], v_rec - v_bar) for i in range(n_agents)]
    return delta_agents, chi_agents


def _resolve_max_periods(sim: dict[str, Any], sigma_u: float) -> int:
    """Pick ``max_periods`` for the current σ_u session.

    If the config provides ``simulation.max_periods_by_sigma_u`` (a mapping
    from σ_u to integer cap), use the entry for the current σ_u; otherwise
    fall back to the scalar ``simulation.max_periods``. Allows the cap to
    vary across the sweep without splitting the experiment YAML.
    """
    overrides = sim.get("max_periods_by_sigma_u")
    if overrides:
        for key, value in overrides.items():
            if float(key) == float(sigma_u):
                return int(value)
        raise KeyError(
            f"max_periods_by_sigma_u missing entry for σ_u={sigma_u}; "
            f"available keys: {list(overrides)}."
        )
    return int(sim["max_periods"])


def _save_qtables(
    config: dict[str, Any],
    session_id: int,
    agents: list[Agent],
    env: MarketEnv,
) -> None:
    """Persist per-agent Q-tables and the env's grids for one session.

    Called when ``config["save_qtables"]`` is truthy, after the training
    loop converges (or hits ``max_periods``) and before the eval phase
    so the saved Q-tables reflect the converged policy without
    eval-phase drift.

    Layout under the configured output directory::

        <output_dir>/qtables/session_<NNN>/
            agent_0.npy        # shape (n_p, n_v, n_v, n_x)
            agent_1.npy
            v_grid.npy
            action_grid.npy    # 1-D: quantities (global) or χ levels (per_v)
            action_table.npy   # shape (n_v, n_x): quantity of action a at v_k
            price_grid.npy

    Reshape to flat ``(n_states, n_x)`` with ``.reshape(-1, n_x)`` where
    ``n_states = n_p * n_v * n_v`` and the flat index is
    ``p_idx * n_v * n_v + v_prev_idx * n_v + v_curr_idx`` (C-order).

    An actor-critic agent (``get_preferences()`` / ``get_values()``) is
    saved as ``agent_<i>_prefs.npy`` (shape ``(n_p, n_v, n_v, n_x)``) and
    ``agent_<i>_values.npy`` (shape ``(n_p, n_v, n_v)``). Agents with
    neither a ``Q`` attribute nor preferences (the fixed-policy agent) are
    skipped with a debug log; this keeps the option safe to leave on in
    heterogeneous experiments.
    """
    output_dir = Path(config["output"]["directory"]).resolve()
    qtable_dir = output_dir / "qtables" / f"session_{session_id:03d}"
    qtable_dir.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(agents):
        q = getattr(a, "Q", None)
        if q is not None:
            np.save(qtable_dir / f"agent_{i}.npy", np.asarray(q))
            continue
        get_prefs = getattr(a, "get_preferences", None)
        if callable(get_prefs):
            np.save(qtable_dir / f"agent_{i}_prefs.npy", np.asarray(get_prefs()))
            np.save(qtable_dir / f"agent_{i}_values.npy", np.asarray(a.get_values()))
            continue
        logger.debug(
            "agent %d (%s) has no Q table or preferences — skipping save",
            i,
            type(a).__name__,
        )
    np.save(qtable_dir / "v_grid.npy", np.asarray(env.v_grid))
    np.save(qtable_dir / "action_grid.npy", np.asarray(env.action_grid))
    np.save(qtable_dir / "action_table.npy", np.asarray(env.action_table))
    np.save(qtable_dir / "price_grid.npy", np.asarray(env.price_grid))
    logger.info("saved Q-tables session=%d to %s", session_id, qtable_dir)


def _spawn_seed_sequences(
    seed: int, n_agents: int
) -> tuple[np.random.SeedSequence, list[np.random.SeedSequence]]:
    """Independent child streams for the env and each agent of one session.

    Replaces the old ``seed`` / ``seed + agent_id`` scheme, under which
    session k's agent 0 shared its stream with session k+1's environment.
    """
    children = np.random.SeedSequence(int(seed)).spawn(1 + n_agents)
    return children[0], list(children[1:])


def build_session(
    config: dict[str, Any], sigma_u: float, seed: int
) -> tuple[MarketEnv, list[Agent], dict[str, float]]:
    """Construct the environment, agents, and grid-corrected benchmarks.

    Args:
        config: Merged config dict from :func:`load_config`.
        sigma_u: σ_u for this session.
        seed: Integer session seed (``seed_base + session_id``).

    Returns:
        Tuple ``(env, agents, bench)``.
    """
    n_agents = int(config["n_agents"])
    if len(config["agents"]) != n_agents:
        raise ValueError(
            f"config lists {len(config['agents'])} agents but n_agents={n_agents}."
        )
    env_ss, agent_ss = _spawn_seed_sequences(seed, n_agents)

    env_cfg = dict(config)
    env_cfg["sigma_u"] = sigma_u
    env_cfg["seed"] = seed
    env_cfg["rng_seed_sequence"] = env_ss
    env = MarketEnv(env_cfg)

    grids = {
        "v_grid": env.v_grid,
        "action_grid": env.action_grid,
        "action_table": env.action_table,
        "price_grid": env.price_grid,
    }
    bench = benchmarks.compute_all_benchmarks(
        n_agents=n_agents,
        xi=float(config["xi"]),
        sigma_u=sigma_u,
        theta=float(config["theta"]),
        sigma_v=float(config.get("sigma_v", 1.0)),
        v_grid=env.v_grid,
    )
    agent_cfg = dict(config)
    agent_cfg["sigma_u"] = sigma_u
    agent_cfg.update(bench)
    agent_cfg["seed"] = seed
    agents: list[Agent] = []
    for i, entry in enumerate(config["agents"]):
        cfg_i = dict(agent_cfg)
        cfg_i["rng_seed_sequence"] = agent_ss[i]
        agents.append(_make_agent(entry, i, cfg_i, grids))
    return env, agents, bench


def _visits(agents: list[Agent], state: dict[int, np.ndarray]) -> list[int]:
    """Per-agent visit count of the state each agent is about to act in."""
    out: list[int] = []
    for i, a in enumerate(agents):
        n = a.visit_count(state[i])
        out.append(-1 if n is None else int(n))
    return out


def _period_row(
    *,
    session_id: int,
    sigma_u: float,
    period: int,
    phase: str,
    info: dict[str, Any],
    rewards: dict[int, float],
    visits: list[int],
    irf_type: str = "",
    irf_path: str = "",
    irf_trial: int = -1,
    irf_period: int = -1,
    shock_magnitude: float = math.nan,
    deviating_agent: int = -1,
) -> dict[str, Any]:
    """One periods.parquet row (eval or IRF). Schema in docs/parameters.md.

    ``x_agent_i`` is the order quantity the environment traded
    (``info["x"][i]``), which is valid for both action-grid modes.
    """
    row: dict[str, Any] = {
        "session_id": int(session_id),
        "sigma_u": float(sigma_u),
        "period": int(period),
        "v_t": float(info["v_t"]),
        "u_t": float(info["noise"]),
        "u_shock": float(info["u_shock"]),
        "y_t": float(info["order_flow"]),
        "p_t": float(info["price"]),
        "p_bin": int(info["price_bin"]),
        "lambda_hat": float(info["lambda_hat"]),
        "phase": phase,
        "irf_type": irf_type,
        "irf_path": irf_path,
        "irf_trial": int(irf_trial),
        "irf_period": int(irf_period),
        "shock_magnitude": float(shock_magnitude),
        "deviating_agent": int(deviating_agent),
    }
    for i in range(len(rewards)):
        row[f"x_agent_{i}"] = float(info["x"][i])
        row[f"profit_agent_{i}"] = float(rewards[i])
        row[f"n_visits_agent_{i}"] = int(visits[i])
    return row


_LEGACY_IRF_KEYS: tuple[str, ...] = ("shock_periods", "shock_magnitudes")


def _irf_specs(irf_cfg: dict[str, Any], n_agents: int) -> list[tuple[str, float, int]]:
    """Validate the IRF config and expand it into (type, magnitude, agent) specs.

    ``type`` is ``"noise"`` (magnitude in ``irf.noise.units``) or
    ``"deviation"`` (magnitude = action bins; agent = deviating speculator).
    One bin is one step of the action grid in use: 31.3 units of quantity on
    the global grid, one aggressiveness level (3.57 in χ) on the per-v grid
    (thesis parameters).
    """
    for key in _LEGACY_IRF_KEYS:
        if key in irf_cfg:
            raise ValueError(
                f"IRF config uses the pre-Phase-0 key {key!r}. Use irf.pre_periods, "
                "irf.post_periods, irf.n_trials, irf.noise and irf.deviation "
                "(see docs/parameters.md, 'IRF protocol')."
            )
    specs: list[tuple[str, float, int]] = []
    noise_cfg = irf_cfg.get("noise") or {}
    units = str(noise_cfg.get("units", "price_bins"))
    if units not in ("price_bins", "sigma_u"):
        raise ValueError(
            f"irf.noise.units must be 'price_bins' or 'sigma_u', got {units!r}."
        )
    for m in noise_cfg.get("magnitudes", []) or []:
        specs.append(("noise", float(m), -1))
    dev_cfg = irf_cfg.get("deviation") or {}
    if dev_cfg.get("enabled", False):
        for k in dev_cfg.get("bins", []) or []:
            if float(k) != int(k) or int(k) < 1:
                raise ValueError(
                    f"irf.deviation.bins must be positive integers, got {k!r}."
                )
            for j in dev_cfg.get("agents", list(range(n_agents))):
                if not 0 <= int(j) < n_agents:
                    raise ValueError(f"irf.deviation.agents entry {j!r} out of range.")
                specs.append(("deviation", float(int(k)), int(j)))
    if not specs:
        raise ValueError(
            "irf.enabled is true but no noise magnitudes or deviation bins are set."
        )
    return specs


def _irf_step(
    env: MarketEnv,
    agents: list[Agent],
    state: dict[int, np.ndarray],
    *,
    shift: dict[int, int] | None = None,
    u_shock: float = 0.0,
) -> tuple[
    dict[int, np.ndarray], dict[int, int], dict[int, float], dict[str, Any], list[int]
]:
    """One frozen-policy period; optional forced action shift and noise shock."""
    visits = _visits(agents, state)
    actions = {i: a.act(state[i]) for i, a in enumerate(agents)}
    if shift:
        n_x = len(env.action_grid)
        for j, delta in shift.items():
            actions[j] = int(np.clip(actions[j] + delta, 0, n_x - 1))
    next_state, rewards, _done, info = env.step(actions, u_shock=u_shock)
    return next_state, actions, rewards, info, visits


def _run_irf(
    config: dict[str, Any],
    env: MarketEnv,
    agents: list[Agent],
    state: dict[int, np.ndarray],
    sigma_u: float,
    session_id: int,
    period_offset: int,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray], int]:
    """Impulse responses with a no-shock counterfactual (common random numbers).

    For every spec × trial:
      1. Advance the main trajectory ``pre_periods`` periods (not recorded),
         then until v_t ≠ v̄ (at most 50 extra periods), so the shock has an
         adverse direction.
      2. Deep-copy env + agents. On the copy apply the shock at the first
         post-period: noise → ``u_shock = sgn(v_t − v̄)·m·scale`` with
         scale = price-bin width / λ^N (units "price_bins") or σ_u (units
         "sigma_u"); deviation → the deviating agent's greedy action is moved
         ``m`` bins in the aggressive direction, clipped to the grid. On the
         global grid that is the index shift ``round(sgn(v_t − v̄)·m)``
         (larger |x|). On the per-v grid a higher index is more aggressive
         at every v, so the shift is ``+m`` for both signs of v_t − v̄.
      3. Step the shocked copy ("shock" rows) and the main trajectory
         ("base" rows) side by side for ``post_periods`` periods. The RNG
         streams are identical, so shock − base isolates the shock's effect.
      4. Continue from the main (unshocked) trajectory.

    Agents are frozen (eval mode) throughout. ``irf_period`` follows the
    DGJ's labelling: the shock period is ``t = pre_periods``.
    """
    irf_cfg = config["irf"]
    n_agents = len(agents)
    specs = _irf_specs(irf_cfg, n_agents)
    pre = int(irf_cfg.get("pre_periods", 3))
    post = int(irf_cfg.get("post_periods", 6))
    n_trials = int(irf_cfg.get("n_trials", 20))
    if pre < 0 or post < 2 or n_trials < 1:
        raise ValueError(
            "irf requires pre_periods ≥ 0, post_periods ≥ 2, n_trials ≥ 1."
        )
    units = str((irf_cfg.get("noise") or {}).get("units", "price_bins"))
    bin_width = float(env.price_grid[1] - env.price_grid[0])
    v_bar = env.v_bar

    for a in agents:
        a.set_eval_mode(True)

    rows: list[dict[str, Any]] = []
    for irf_type, magnitude, dev_agent in specs:
        for trial in range(n_trials):
            for _ in range(pre):
                state, *_ = _irf_step(env, agents, state)
                period_offset += 1
            tries = 0
            while env.v_current == v_bar and tries < 50:
                state, *_ = _irf_step(env, agents, state)
                period_offset += 1
                tries += 1
            if env.v_current == v_bar:
                logger.warning(
                    "irf: v_t stayed at v̄ for 50 periods; skipping trial "
                    "(session=%d type=%s m=%s trial=%d)",
                    session_id,
                    irf_type,
                    magnitude,
                    trial,
                )
                continue
            sign = 1.0 if env.v_current > v_bar else -1.0

            env_s = copy.deepcopy(env)
            agents_s = copy.deepcopy(agents)
            state_s = {i: s.copy() for i, s in state.items()}

            for k in range(post):
                period_offset += 1
                u_shock = 0.0
                shift: dict[int, int] | None = None
                if k == 0:
                    if irf_type == "noise":
                        scale = (
                            bin_width / env.lambda_N
                            if units == "price_bins"
                            else sigma_u
                        )
                        u_shock = sign * magnitude * scale
                    elif env.action_mode == "per_v":
                        shift = {dev_agent: int(magnitude)}
                    else:
                        shift = {dev_agent: round(sign * magnitude)}
                ns_s, _a_s, r_s, i_s, vis_s = _irf_step(
                    env_s, agents_s, state_s, shift=shift, u_shock=u_shock
                )
                ns_b, _a_b, r_b, i_b, vis_b = _irf_step(env, agents, state)
                common: dict[str, Any] = {
                    "session_id": session_id,
                    "sigma_u": sigma_u,
                    "period": period_offset,
                    "phase": "irf",
                    "irf_type": irf_type,
                    "irf_trial": trial,
                    "irf_period": pre + k,
                    "shock_magnitude": magnitude,
                    "deviating_agent": dev_agent,
                }
                rows.append(
                    _period_row(
                        **common,
                        irf_path="shock",
                        info=i_s,
                        rewards=r_s,
                        visits=vis_s,
                    )
                )
                rows.append(
                    _period_row(
                        **common,
                        irf_path="base",
                        info=i_b,
                        rewards=r_b,
                        visits=vis_b,
                    )
                )
                state_s = ns_s
                state = ns_b
    return rows, state, period_offset


def run_session(
    config: dict[str, Any],
    sigma_u: float,
    session_id: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one (σ_u, session_id) training → eval → IRF → metrics loop.

    With ``simulation.diagnostics.enabled`` the training loop also tracks,
    per ``block_periods`` block, the visits per state and which states'
    greedy actions changed (stability diagnostic), and evaluates the greedy
    policies on a deep copy at every block boundary (Δ^C learning curve).

    Args:
        config: Merged config dict from :func:`load_config`.
        sigma_u: σ_u for this session (one point in the sweep).
        session_id: 0-indexed session counter.
        seed: Integer session seed; child RNG streams for the env and each
            agent are spawned from it (:func:`_spawn_seed_sequences`).

    Returns:
        Tuple ``(session_row, period_rows, trajectory_rows)``;
        ``trajectory_rows`` is empty unless diagnostics are enabled.
    """
    n_agents = int(config["n_agents"])
    sim = config["simulation"]
    convergence_periods = int(sim["convergence_periods"])
    check_every = int(sim["check_every"])
    max_periods = _resolve_max_periods(sim, sigma_u)
    eval_periods = int(sim["eval_periods"])

    env, agents, bench = build_session(config, sigma_u, seed)
    agent_types = "_".join(entry["type"] for entry in config["agents"])
    v_bar = float(config.get("v_bar", 0.0))
    save_eval_periods = int(sim.get("save_eval_periods", eval_periods))
    eval_greedy = bool(sim.get("eval_greedy", True))
    t_session_start = time.perf_counter()

    state = env.reset()

    log_every = int(config.get("runner", {}).get("log_every", 50000))
    snapshots: list[np.ndarray] | None = None
    unchanged_checks = 0
    converged = False
    periods_to_convergence = max_periods

    diag_cfg = sim.get("diagnostics") or {}
    diag_enabled = bool(diag_cfg.get("enabled", False))
    block_periods = int(diag_cfg.get("block_periods", 1_000_000))
    onpath_min_share = float(diag_cfg.get("onpath_min_share", 0.001))
    settled_max_flip_share = float(diag_cfg.get("settled_max_flip_share", 0.10))
    traj_eval_periods = int(diag_cfg.get("trajectory_eval_periods", 20000))
    n_v = len(env.v_grid)
    n_states = len(env.price_grid) * n_v * n_v
    # Value indices where all actions are the same trade (per_v grid at
    # v = v̄); None on the global grid. Greedy-index changes there are
    # ignored by the convergence check and the stability diagnostic.
    degenerate_v = _degenerate_v_mask(env.action_table)
    # The same as a flat-state mask for the stability diagnostic: the "_nd"
    # shares and the settled flag leave these states out.
    # None on the global grid, where _nd equals the plain share.
    degenerate_states = _degenerate_state_mask(degenerate_v, len(env.price_grid), n_v)
    tabular = [a.greedy_policy() is not None for a in agents]
    tab_ids = [i for i, t in enumerate(tabular) if t]
    visits_cur = np.zeros(n_states, dtype=np.int64)
    visits_prev = np.zeros(n_states, dtype=np.int64)
    flipped_cur = {i: np.zeros(n_states, dtype=bool) for i in tab_ids}
    flipped_prev = {i: np.zeros(n_states, dtype=bool) for i in tab_ids}
    len_cur = 0
    len_prev = 0
    diag_snaps: list[np.ndarray | None] | None = None  # mixed populations only
    trajectory_rows: list[dict[str, Any]] = []

    for period in range(1, max_periods + 1):
        if diag_enabled:
            s0 = state[0]
            visits_cur[int(s0[0]) * n_v * n_v + int(s0[1]) * n_v + int(s0[2])] += 1
            len_cur += 1
        actions = {i: a.act(state[i]) for i, a in enumerate(agents)}
        next_state, rewards, done, _info = env.step(actions)
        for i, a in enumerate(agents):
            a.update(
                Transition(
                    state=state[i],
                    action=actions[i],
                    reward=rewards[i],
                    next_state=next_state[i],
                    done=done,
                )
            )
        state = next_state

        if period % check_every == 0:
            old_snapshots = snapshots
            converged, snapshots, unchanged_checks = _check_convergence_step(
                agents,
                snapshots,
                unchanged_checks,
                check_every,
                convergence_periods,
                degenerate_v,
            )
            if diag_enabled:
                if snapshots is not old_snapshots:
                    if old_snapshots is not None:
                        for i in tab_ids:
                            flipped_cur[i] |= (snapshots[i] != old_snapshots[i]).ravel()
                else:
                    # The convergence check is skipped when an agent has no
                    # tabular policy; track the tabular agents separately.
                    new_snaps = [
                        _policy_snapshot(a, degenerate_v) if tabular[i] else None
                        for i, a in enumerate(agents)
                    ]
                    if diag_snaps is not None:
                        for i in tab_ids:
                            flipped_cur[i] |= (new_snaps[i] != diag_snaps[i]).ravel()
                    diag_snaps = new_snaps
            if converged:
                periods_to_convergence = period
                break

        if diag_enabled and period % block_periods == 0:
            b_flips = [flipped_cur[i] for i in tab_ids]
            b_flip, b_flip_w, b_onpath = _stability_stats(
                b_flips, visits_cur, block_periods, onpath_min_share
            )
            if degenerate_states is None:
                b_flip_nd, b_flip_w_nd = b_flip, b_flip_w
            else:
                b_flip_nd, b_flip_w_nd, _ = _stability_stats(
                    b_flips,
                    visits_cur,
                    block_periods,
                    onpath_min_share,
                    exclude=degenerate_states,
                )
            traj_delta, traj_chi = _trajectory_eval(
                env, agents, state, traj_eval_periods, bench, v_bar
            )
            traj_row: dict[str, Any] = {
                "session_id": int(session_id),
                "sigma_u": float(sigma_u),
                "period": int(period),
                "delta_c": float(np.mean(traj_delta)),
            }
            for i in range(n_agents):
                traj_row[f"delta_c_agent_{i}"] = float(traj_delta[i])
                traj_row[f"chi_hat_agent_{i}"] = float(traj_chi[i])
            traj_row["block_flip_share"] = float(b_flip)
            traj_row["block_flip_share_weighted"] = float(b_flip_w)
            traj_row["block_flip_share_nd"] = float(b_flip_nd)
            traj_row["block_flip_share_weighted_nd"] = float(b_flip_w_nd)
            traj_row["block_n_onpath"] = int(b_onpath)
            trajectory_rows.append(traj_row)
            logger.info(
                "block session=%d period=%d delta_c=%.4f flip_share=%.3f n_onpath=%d",
                session_id,
                period,
                traj_row["delta_c"],
                b_flip,
                b_onpath,
            )
            flipped_prev = flipped_cur
            visits_prev = visits_cur
            len_prev = block_periods
            flipped_cur = {i: np.zeros(n_states, dtype=bool) for i in tab_ids}
            visits_cur = np.zeros(n_states, dtype=np.int64)
            len_cur = 0

        if period % log_every == 0:
            logger.info(
                "training session=%d sigma_u=%.3g period=%d unchanged=%d",
                session_id,
                sigma_u,
                period,
                unchanged_checks,
            )

    t_train = time.perf_counter() - t_session_start
    stab_onpath_degenerate = 0
    # Policy concentration (actor-critic only; NaN otherwise): mean of
    # max_a π(a|s) over the final window's non-degenerate on-path states.
    concentration: list[float] = [math.nan] * n_agents
    if diag_enabled:
        stab_flips = [flipped_prev[i] | flipped_cur[i] for i in tab_ids]
        stab_visits = visits_prev + visits_cur
        stab_window = len_prev + len_cur
        stab_flip, stab_flip_w, stab_onpath = _stability_stats(
            stab_flips, stab_visits, stab_window, onpath_min_share
        )
        conc_mask = _onpath_mask(stab_visits, stab_window, onpath_min_share)
        if degenerate_states is not None:
            conc_mask = conc_mask & ~degenerate_states
        for i, a in enumerate(agents):
            c = a.policy_concentration(conc_mask)
            concentration[i] = math.nan if c is None else float(c)
        if degenerate_states is None:
            stab_flip_nd, stab_flip_w_nd = stab_flip, stab_flip_w
        else:
            # Shares over the non-degenerate on-path states only.
            stab_flip_nd, stab_flip_w_nd, _ = _stability_stats(
                stab_flips,
                stab_visits,
                stab_window,
                onpath_min_share,
                exclude=degenerate_states,
            )
            # On-path states that are degenerate: in the denominator of the
            # plain shares, but they can never count as flipped.
            stab_onpath_degenerate = int(
                np.count_nonzero(
                    _onpath_mask(stab_visits, stab_window, onpath_min_share)
                    & degenerate_states
                )
            )
    else:
        stab_flip, stab_flip_w, stab_onpath, stab_window = math.nan, math.nan, 0, 0
        stab_flip_nd, stab_flip_w_nd = math.nan, math.nan
    # Settled: flip share over the NON-degenerate on-path states.
    # Identical to the plain share on the global grid (earlier specification).
    settled = bool(diag_enabled and stab_flip_nd <= settled_max_flip_share)
    if not converged:
        logger.info(
            "max_periods reached without convergence (session=%d sigma_u=%.3g)",
            session_id,
            sigma_u,
        )
    period_offset = periods_to_convergence

    if config.get("save_qtables", False):
        _save_qtables(config, session_id, agents, env)

    if eval_greedy:
        for a in agents:
            a.set_eval_mode(True)

    period_rows: list[dict[str, Any]] = []
    v_record: list[float] = []
    u_record: list[float] = []
    x_records: list[list[float]] = [[] for _ in range(n_agents)]
    profit_records: list[list[float]] = [[] for _ in range(n_agents)]
    lambda_records: list[float] = []

    logger.info(
        "eval start session=%d sigma_u=%.3g periods=%d greedy=%s",
        session_id,
        sigma_u,
        eval_periods,
        eval_greedy,
    )
    for k in range(eval_periods):
        visits = _visits(agents, state)
        actions = {i: a.act(state[i]) for i, a in enumerate(agents)}
        next_state, rewards, done, info = env.step(actions)
        if not eval_greedy:
            for i, a in enumerate(agents):
                a.update(
                    Transition(
                        state=state[i],
                        action=actions[i],
                        reward=rewards[i],
                        next_state=next_state[i],
                        done=done,
                    )
                )
        v_record.append(float(info["v_t"]))
        u_record.append(float(info["noise"]))
        lambda_records.append(float(info["lambda_hat"]))
        for i in range(n_agents):
            x_records[i].append(float(info["x"][i]))
            profit_records[i].append(float(rewards[i]))
        if k < save_eval_periods:
            period_rows.append(
                _period_row(
                    session_id=session_id,
                    sigma_u=sigma_u,
                    period=period_offset + k + 1,
                    phase="eval",
                    info=info,
                    rewards=rewards,
                    visits=visits,
                )
            )
        state = next_state
    period_offset += eval_periods

    irf_cfg = config.get("irf", {}) or {}
    if irf_cfg.get("enabled", False):
        logger.info("irf start session=%d sigma_u=%.3g", session_id, sigma_u)
        irf_rows, state, period_offset = _run_irf(
            config, env, agents, state, sigma_u, session_id, period_offset
        )
        period_rows.extend(irf_rows)

    sigma_v_hat = float(env.v_grid.std())
    lambda_arr = np.asarray(lambda_records, dtype=np.float64)
    lambda_end = float(lambda_arr[-1]) if lambda_arr.size else float(env.lambda_N)

    if lambda_arr.size:
        v_arr = np.asarray(v_record, dtype=np.float64)
        u_arr = np.asarray(u_record, dtype=np.float64)
        x_arrs = [np.asarray(x_records[i], dtype=np.float64) for i in range(n_agents)]
        x_total = np.sum(x_arrs, axis=0)
        chi_hat = _ols_slope(x_total, v_arr - v_bar) / n_agents
        chi_hat_agents = [_ols_slope(x, v_arr - v_bar) for x in x_arrs]
        avg_profits = [float(np.mean(profit_records[i])) for i in range(n_agents)]
        pi_n_path = float(
            np.mean(
                benchmarks.realised_benchmark_profit(
                    v_arr, u_arr, bench["chi_N"], bench["lambda_N"], n_agents, v_bar
                )
            )
        )
        pi_m_path = float(
            np.mean(
                benchmarks.realised_benchmark_profit(
                    v_arr, u_arr, bench["chi_M"], bench["lambda_M"], n_agents, v_bar
                )
            )
        )
        delta_agents = [
            metrics.collusion_capacity(pi_C=p, pi_N=pi_n_path, pi_M=pi_m_path)
            for p in avg_profits
        ]
        delta_c = float(np.mean(delta_agents))  # DGJ OA eq. IA.4.1
        # Earlier definition, against the expected-value benchmarks.
        delta_c_expected = metrics.collusion_capacity(
            pi_C=float(np.mean(avg_profits)), pi_N=bench["pi_N"], pi_M=bench["pi_M"]
        )
        profit_gain = (
            float(np.sum(avg_profits) / (n_agents * pi_n_path))
            if pi_n_path != 0.0
            else math.nan
        )
        pi_info = metrics.price_informativeness(
            chi_C=chi_hat,
            sigma_v_hat=sigma_v_hat,
            sigma_u=sigma_u,
            n_agents=n_agents,
        )
        liquidity = float(
            np.mean(
                [
                    metrics.market_liquidity(lambda_hat=lh, xi=float(config["xi"]))
                    for lh in lambda_arr
                ]
            )
        )
        mispricing_avg = float(
            np.mean(
                [
                    metrics.mispricing(
                        lambda_hat=lh,
                        chi_C=chi_hat,
                        n_agents=n_agents,
                        v_t=vt,
                        v_bar=v_bar,
                    )
                    for lh, vt in zip(lambda_arr, v_arr)
                ]
            )
        )
    else:
        chi_hat = math.nan
        chi_hat_agents = [math.nan] * n_agents
        avg_profits = [math.nan] * n_agents
        delta_agents = [math.nan] * n_agents
        delta_c = math.nan
        delta_c_expected = math.nan
        pi_n_path = math.nan
        pi_m_path = math.nan
        profit_gain = math.nan
        pi_info = math.nan
        liquidity = math.nan
        mispricing_avg = math.nan

    session_row: dict[str, Any] = {
        "session_id": session_id,
        "sigma_u": sigma_u,
        "xi": float(config["xi"]),
        "agent_types": agent_types,
        "converged": bool(converged),
        "periods_to_convergence": int(periods_to_convergence),
        "delta_c": float(delta_c),
        "delta_c_expected": float(delta_c_expected),
        "pi_N_path": float(pi_n_path),
        "pi_M_path": float(pi_m_path),
        "profit_gain": float(profit_gain),
        "chi_hat": float(chi_hat),
        "lambda_hat": float(lambda_end),
        "price_informativeness": float(pi_info),
        "market_liquidity": float(liquidity),
        "mispricing": float(mispricing_avg),
        "seed": int(seed),
        "eval_periods": int(eval_periods),
        "eval_greedy": bool(eval_greedy),
        "reward_price": str(env.reward_price),
        "market_maker": str(env.market_maker),
        "wall_clock_seconds": float(time.perf_counter() - t_session_start),
        "periods_per_second": float(periods_to_convergence / max(t_train, 1e-9)),
        "stability_flip_share": float(stab_flip),
        "stability_flip_share_weighted": float(stab_flip_w),
        "stability_flip_share_nd": float(stab_flip_nd),
        "stability_flip_share_weighted_nd": float(stab_flip_w_nd),
        "stability_n_onpath": int(stab_onpath),
        "stability_n_onpath_degenerate": int(stab_onpath_degenerate),
        "stability_window_periods": int(stab_window),
        "settled": settled,
        "diagnostics_enabled": bool(diag_enabled),
    }
    for i in range(n_agents):
        session_row[f"avg_profit_agent_{i}"] = avg_profits[i]
        session_row[f"delta_c_agent_{i}"] = float(delta_agents[i])
        session_row[f"chi_hat_agent_{i}"] = float(chi_hat_agents[i])
        session_row[f"policy_concentration_agent_{i}"] = float(concentration[i])

    return session_row, period_rows, trajectory_rows


def _git_commit() -> str:
    """Current git commit hash ('+dirty' if uncommitted changes), or 'unknown'."""
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=here,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=here,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return (head or "unknown") + ("+dirty" if dirty else "")


def run_experiment(
    config_path: str | Path,
    *,
    session_id_start: int | None = None,
    n_sim: int | None = None,
    output_dir: str | Path | None = None,
    max_periods: int | None = None,
) -> dict[str, Path]:
    """Run a full experiment: sweep × n_sim sessions, write Parquet output.

    Args:
        config_path: Path to the experiment YAML.
        session_id_start: Overrides ``simulation.session_id_start``.
        n_sim: Overrides ``simulation.n_sim``.
        output_dir: Overrides ``output.directory``.
        max_periods: Overrides ``simulation.max_periods``; when given,
            ``simulation.max_periods_by_sigma_u`` is ignored.

    Returns:
        Mapping of artefact name (``"sessions"``, ``"periods"``,
        ``"trajectory"``, ``"metadata"``) to the absolute output path
        written.
    """
    config = load_config(config_path)
    if session_id_start is not None:
        config["simulation"]["session_id_start"] = int(session_id_start)
    if n_sim is not None:
        config["simulation"]["n_sim"] = int(n_sim)
    if output_dir is not None:
        config["output"]["directory"] = str(output_dir)
    if max_periods is not None:
        config["simulation"]["max_periods"] = int(max_periods)
        config["simulation"].pop("max_periods_by_sigma_u", None)
    code_version = _git_commit()
    started_utc = datetime.now(UTC).isoformat()

    output_dir_path = Path(config["output"]["directory"]).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    sweep_sigmas = list(config["sweep"]["sigma_u"])
    n_sim_eff = int(config["simulation"]["n_sim"])
    seed_base = int(config["simulation"]["seed_base"])
    sid_start = int(config["simulation"].get("session_id_start", 0))

    logger.info(
        "experiment start config=%s sweep_size=%d n_sim=%d session_id_start=%d "
        "code_version=%s",
        Path(config_path).resolve(),
        len(sweep_sigmas),
        n_sim_eff,
        sid_start,
        code_version,
    )

    all_session_rows: list[dict[str, Any]] = []
    all_period_rows: list[dict[str, Any]] = []
    all_trajectory_rows: list[dict[str, Any]] = []

    for sigma_u in sweep_sigmas:
        for session_id in range(sid_start, sid_start + n_sim_eff):
            seed = seed_base + session_id
            session_row, period_rows, trajectory_rows = run_session(
                config=config,
                sigma_u=float(sigma_u),
                session_id=session_id,
                seed=seed,
            )
            session_row["code_version"] = code_version
            all_session_rows.append(session_row)
            all_period_rows.extend(period_rows)
            all_trajectory_rows.extend(trajectory_rows)
            logger.info(
                "session done sigma_u=%.3g session=%d converged=%s periods=%d "
                "delta_c=%.4f delta_c_expected=%.4f periods/s=%.0f",
                sigma_u,
                session_id,
                session_row["converged"],
                session_row["periods_to_convergence"],
                session_row["delta_c"],
                session_row["delta_c_expected"],
                session_row["periods_per_second"],
            )

    sessions_path = output_dir_path / "sessions.parquet"
    pd.DataFrame(all_session_rows).to_parquet(sessions_path, index=False)
    logger.info("wrote %s (%d rows)", sessions_path, len(all_session_rows))
    artefacts: dict[str, Path] = {"sessions": sessions_path}

    if all_period_rows:
        periods_path = output_dir_path / "periods.parquet"
        pd.DataFrame(all_period_rows).to_parquet(periods_path, index=False)
        logger.info("wrote %s (%d rows)", periods_path, len(all_period_rows))
        artefacts["periods"] = periods_path

    if all_trajectory_rows:
        trajectory_path = output_dir_path / "trajectory.parquet"
        pd.DataFrame(all_trajectory_rows).to_parquet(trajectory_path, index=False)
        logger.info("wrote %s (%d rows)", trajectory_path, len(all_trajectory_rows))
        artefacts["trajectory"] = trajectory_path

    metadata = {
        "config_path": str(Path(config_path).resolve()),
        "experiment_name": (config.get("experiment") or {}).get("name"),
        "code_version": code_version,
        "started_utc": started_utc,
        "finished_utc": datetime.now(UTC).isoformat(),
        "session_id_start": sid_start,
        "n_sim": n_sim_eff,
        "output_dir": str(output_dir_path),
        "sweep_sigma_u": sweep_sigmas,
        "python": sys.version,
        "numpy": np.__version__,
        "cli_overrides": {
            "session_id_start": session_id_start,
            "n_sim": n_sim,
            "output_dir": None if output_dir is None else str(output_dir),
            "max_periods": max_periods,
        },
        "resolved_config": config,
    }
    metadata_path = output_dir_path / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    artefacts["metadata"] = metadata_path

    return artefacts


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run an ai_trading experiment from a YAML config."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the experiment YAML file."
    )
    parser.add_argument(
        "--session-id-start",
        type=int,
        default=None,
        help="Override simulation.session_id_start (seed = seed_base + id).",
    )
    parser.add_argument(
        "--n-sim",
        type=int,
        default=None,
        help="Override simulation.n_sim (sessions per sigma_u).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output.directory.",
    )
    parser.add_argument(
        "--max-periods",
        type=int,
        default=None,
        help="Override simulation.max_periods (ignores max_periods_by_sigma_u).",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    run_experiment(
        args.config,
        session_id_start=args.session_id_start,
        n_sim=args.n_sim,
        output_dir=args.output_dir,
        max_periods=args.max_periods,
    )


if __name__ == "__main__":
    _cli()
