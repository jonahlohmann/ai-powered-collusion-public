"""Tests for scripts/deviation_test.py (best-response deviation test)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from ai_trading.simulation import runner

ROOT = Path(__file__).resolve().parents[1]
DEBUG_CFG = ROOT / "experiments/01_replicate_baseline/config_debug.yaml"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "deviation_test", ROOT / "scripts/deviation_test.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dt = _load_script()


def test_grid_payoffs_never_beat_continuous_best_response() -> None:
    grid = np.linspace(-218.93, 218.93, 15)
    lam = 0.002
    for dv in (-1.2816, -0.5244, 0.5244, 1.2816):
        for x_opp in grid:
            pay = dt.grid_payoffs(grid, dv, x_opp, lam)
            assert pay.max() <= dt.continuous_best_payoff(dv, x_opp, lam) + 1e-9


def test_continuous_best_payoff_matches_best_response_quantity() -> None:
    lam, dv, x_opp = 0.002, 1.2816, 156.38
    x_star = (dv - lam * x_opp) / (2 * lam)
    expected = dt.grid_payoffs(np.array([x_star]), dv, x_opp, lam)[0]
    assert dt.continuous_best_payoff(dv, x_opp, lam) == pytest.approx(expected)


def test_grid_best_response_against_cartel_quantity_is_top_bin() -> None:
    # Thesis grid at v = +1.2816: the continuous BR to 156.38 is ≈ 242, so the
    # grid BR is the largest quantity.
    grid = np.linspace(-218.93, 218.93, 15)
    pay = dt.grid_payoffs(grid, 1.2816, 156.38, 0.002)
    assert int(np.argmax(pay)) == 14


def test_one_up_index_direction_and_clipping() -> None:
    assert dt.one_up_index(7, 0.5, 15, per_v=False) == 8
    assert dt.one_up_index(7, -0.5, 15, per_v=False) == 6
    assert dt.one_up_index(0, -0.5, 15, per_v=False) == 0
    assert dt.one_up_index(14, 0.5, 15, per_v=False) == 14
    assert dt.one_up_index(7, -0.5, 15, per_v=True) == 8


def test_bootstrap_ci_brackets_the_mean() -> None:
    v = np.random.default_rng(1).normal(1.0, 0.5, size=30)
    lo, hi = dt.bootstrap_ci(v)
    assert lo < v.mean() < hi


def _debug_session(tmp_path: Path) -> tuple[object, list, Path]:
    config = runner.load_config(str(DEBUG_CFG))
    env, agents, _ = runner.build_session(config, 0.1, seed=3)
    rng = np.random.default_rng(0)
    qdir = tmp_path / "qtables" / "session_000"
    qdir.mkdir(parents=True)
    for i, agent in enumerate(agents):
        np.save(qdir / f"agent_{i}.npy", agent.Q + rng.normal(0, 1, agent.Q.shape))
    return env, agents, qdir


def test_action_table_tiles_global_grid(tmp_path: Path) -> None:
    env, _agents, _ = _debug_session(tmp_path)
    table = dt.action_table(env)
    assert table.shape == (len(env.v_grid), len(env.action_grid))
    np.testing.assert_allclose(table[0], env.action_grid)


def test_find_sessions_and_restore(tmp_path: Path) -> None:
    _env, agents, qdir = _debug_session(tmp_path)
    found = dt.find_sessions(tmp_path)
    assert [(sid, q) for sid, q, _ in found] == [(0, qdir)]
    q_init = dt.restore_agents(agents, qdir)
    assert all(a.eval_mode for a in agents)
    np.testing.assert_allclose(agents[0].Q, np.load(qdir / "agent_0.npy"))
    assert not np.allclose(q_init[0], agents[0].Q)


def test_dynamic_test_smoke_and_determinism(tmp_path: Path) -> None:
    frames = []
    for _ in range(2):
        env, agents, qdir = _debug_session(tmp_path / str(len(frames)))
        q_init = dt.restore_agents(agents, qdir)
        frames.append(
            dt.dynamic_test(env, agents, q_init, 0, trials=3, horizon=5, rho=0.95)
        )
    a, b = frames
    assert len(a) > 0
    assert set(a["kind"]) <= {"grid_br", "one_up"}
    np.testing.assert_allclose(a["total"], a["gain0"] + a["cont"])
    np.testing.assert_allclose(a["total"].to_numpy(), b["total"].to_numpy())


# ---- per-v action grid --------------------------------------------------------

_PER_V_KEYS = {"action_grid": "per_v", "exploration_counter": "v", "q_init": "paper"}


def _per_v_session(tmp_path: Path, greedy: int = 1) -> tuple[object, list, Path]:
    """Debug session on the per-v grid; saved Q-tables make ``greedy`` the policy."""
    config = runner.load_config(str(DEBUG_CFG))
    config.update(_PER_V_KEYS)
    env, agents, _ = runner.build_session(config, 0.1, seed=3)
    qdir = tmp_path / "qtables" / "session_000"
    qdir.mkdir(parents=True)
    for i, agent in enumerate(agents):
        q = np.zeros_like(agent.Q)
        q[..., greedy] = 1.0
        np.save(qdir / f"agent_{i}.npy", q)
    return env, agents, qdir


def test_action_table_is_the_env_table_on_the_per_v_grid(tmp_path: Path) -> None:
    env, _agents, _ = _per_v_session(tmp_path)
    assert env.action_mode == "per_v"
    table = dt.action_table(env)
    np.testing.assert_array_equal(table, env.action_table)
    assert np.ptp(table[1]) == 0.0  # v = v̄ (n_v = 3): every action trades zero
    assert not np.array_equal(table[0], table[2])


def test_dynamic_test_on_the_per_v_grid(tmp_path: Path) -> None:
    """Degenerate v = v̄ states are skipped; ``one_up`` raises the aggressiveness."""
    env, agents, qdir = _per_v_session(tmp_path, greedy=1)
    q_init = dt.restore_agents(agents, qdir)
    out = dt.dynamic_test(env, agents, q_init, 0, trials=6, horizon=5, rho=0.95)
    assert set(out["kind"]) == {"grid_br", "one_up"}
    assert (out.groupby(["dev", "kind"]).size() == 6).all()
    assert (out["abs_dv"] > 0).all()  # no trial at v = v̄
    np.testing.assert_allclose(out["total"], out["gain0"] + out["cont"])
    # Both play level 1 (χ = 133.3), below the static best response. Moving one
    # level up, or to the grid best response, raises the deviation-period profit
    # for both signs of v − v̄ (noise is common to both paths).
    assert (out["gain0"] > 0).all()
    assert not out["opp_reacts"].any()  # constant policies: nothing to react with


def test_index_change_at_v_bar_is_not_an_opponent_reaction(tmp_path: Path) -> None:
    """Per-v grid: at v = v̄ all actions trade zero, so nothing counts as a reaction."""
    config = runner.load_config(str(DEBUG_CFG))
    config.update(_PER_V_KEYS)
    env, agents, _ = runner.build_session(config, 0.1, seed=3)
    qdir = tmp_path / "qtables" / "session_000"
    qdir.mkdir(parents=True)
    for i, agent in enumerate(agents):
        # The greedy action depends on the lagged price bin, so a deviation
        # that moves the price changes the opponent's greedy index next period.
        q = np.zeros_like(agent.Q)
        for p in range(q.shape[0]):
            q[p, :, :, p % q.shape[-1]] = 1.0
        np.save(qdir / f"agent_{i}.npy", q)
    q_init = dt.restore_agents(agents, qdir)
    out = dt.dynamic_test(env, agents, q_init, 0, trials=40, horizon=4, rho=0.95)
    reacts = out["opp_reacts"].astype(bool)
    changed = out["opp_state_changed"].astype(bool)
    assert reacts.any()
    # Every counted reaction is a change of the traded quantity ...
    assert (out.loc[reacts, "opp_aggr"] != 0.0).all()
    # ... and there are trials in which the opponent's state changed but its
    # trade did not (next value = v̄): these must not be counted.
    assert (changed & ~reacts).any()


def test_static_incentives_on_the_per_v_grid(tmp_path: Path) -> None:
    import pandas as pd

    env, _agents, _ = _per_v_session(tmp_path)
    table, lam = dt.action_table(env), float(env.lambda_N)
    rows = []
    for k in (0, 1, 2, 0, 2):
        v = float(env.v_grid[k])
        rows.append(
            {
                "session_id": 0,
                "phase": "eval",
                "irf_type": "",
                "v_t": v,
                "x_agent_0": table[k, 1],
                "x_agent_1": table[k, 3],
            }
        )
    st = dt.static_incentives(pd.DataFrame(rows), table, env.v_grid, env.v_bar, lam)
    assert len(st) == 2  # one |v − v̄| level, two agents; the v = v̄ row is dropped
    assert (st["n"] == 4).all()
    assert (st["gain_grid"] >= -1e-9).all()
    assert (st["gain_cont"] >= st["gain_grid"] - 1e-9).all()
    # Agent 0 plays level 1 against level 3: not a grid best response.
    assert st.loc[st["agent"] == 0, "br_share"].iloc[0] == 0.0


def _q_ac_session(tmp_path: Path) -> tuple[object, list, Path]:
    """Debug Q–AC session on the per-v grid with saved Q-table, preferences, values."""
    config = runner.load_config(str(DEBUG_CFG))
    config.update(_PER_V_KEYS)
    config["agents"] = [{"type": "qlearner"}, {"type": "actor_critic"}]
    env, agents, _ = runner.build_session(config, 0.1, seed=3)
    rng = np.random.default_rng(1)
    qdir = tmp_path / "qtables" / "session_000"
    qdir.mkdir(parents=True)
    q = np.zeros_like(agents[0].Q)
    q[..., 1] = 1.0  # a middle action, so that a deviation exists
    np.save(qdir / "agent_0.npy", q)
    prefs = agents[1].get_preferences() + rng.normal(0, 1, (11, 3, 3, 5))
    np.save(qdir / "agent_1_prefs.npy", prefs)
    np.save(qdir / "agent_1_values.npy", agents[1].get_values() + 1.0)
    return env, agents, qdir


def test_restore_and_dynamic_test_on_a_q_ac_session(tmp_path: Path) -> None:
    env, agents, qdir = _q_ac_session(tmp_path)
    q_init = dt.restore_agents(agents, qdir)
    assert q_init[0] is not None and q_init[1] is None
    assert agents[1].eval_mode
    np.testing.assert_array_equal(
        agents[1].get_preferences(), np.load(qdir / "agent_1_prefs.npy")
    )
    frame = dt.dynamic_test(env, agents, q_init, 0, trials=3, horizon=5, rho=0.95)
    assert set(frame["dev"]) == {0, 1}
    q_dev = frame[frame.dev == 0]
    ac_dev = frame[frame.dev == 1]
    assert (
        np.isfinite(q_dev[["q_greedy", "q_dev", "tgt_greedy", "tgt_dev"]]).all().all()
    )
    assert ac_dev[["q_greedy", "q_dev", "tgt_greedy", "tgt_dev"]].isna().all().all()
    # The Q-learner's opponent is the actor-critic: no untouched-action count.
    assert q_dev["opp_untouched_actions"].isna().all()
    assert np.isfinite(ac_dev["opp_untouched_actions"]).all()
    assert np.isfinite(frame[["gain0", "cont", "total"]]).all().all()


def test_restore_rejects_unknown_agent_types(tmp_path: Path) -> None:
    class Odd(runner.Agent):
        def act(self, state):
            return 0

        def update(self, transition):
            return None

        def reset(self):
            return None

    with pytest.raises(NotImplementedError):
        dt.restore_agents([Odd(agent_id=0, n_actions=5)], tmp_path)
