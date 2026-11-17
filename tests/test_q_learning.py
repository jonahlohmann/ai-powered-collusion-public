"""Tests for QLearningAgent (agents/q_learning.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ai_trading.agents.base import Agent, Transition
from ai_trading.agents.q_learning import QLearningAgent
from ai_trading.env import discretization
from ai_trading.simulation import benchmarks


def _baseline_config(**overrides: object) -> dict[str, object]:
    """DGJ's baseline (n_p=31, n_v=10, n_x=15)."""
    n_agents = 2
    xi = 500.0
    sigma_u = 0.1
    theta = 0.1
    sigma_v = 1.0
    cfg: dict[str, object] = {
        "alpha": 0.01,
        "beta": 5.0e-7,
        "rho": 0.95,
        "n_agents": n_agents,
        "n_v": 10,
        "n_x": 15,
        "n_p": 31,
        "v_bar": 0.0,
        "sigma_v": sigma_v,
        "chi_N": benchmarks.x_N(n_agents, xi, sigma_u, theta, sigma_v),
        "lambda_N": benchmarks.lambda_N(n_agents, xi, sigma_u, theta, sigma_v),
        "pi_N": benchmarks.pi_N(n_agents, xi, sigma_u, theta, sigma_v),
        "pi_M": benchmarks.pi_M(n_agents, xi, sigma_u, theta, sigma_v),
        "seed": 42,
    }
    cfg.update(overrides)
    return cfg


def _baseline_grids(cfg: dict[str, object]) -> dict[str, np.ndarray]:
    n_agents = int(cfg["n_agents"])
    xi = 500.0
    sigma_u = 0.1
    theta = 0.1
    sigma_v = float(cfg["sigma_v"])
    v_bar = float(cfg["v_bar"])
    return {
        "v_grid": discretization.build_v_grid(
            n_v=int(cfg["n_v"]), v_bar=v_bar, sigma_v=sigma_v
        ),
        "action_grid": discretization.build_action_grid(
            n_x=int(cfg["n_x"]),
            n_agents=n_agents,
            xi=xi,
            sigma_u=sigma_u,
            theta=theta,
            sigma_v=sigma_v,
        ),
        "price_grid": discretization.build_price_grid(
            n_p=int(cfg["n_p"]),
            n_agents=n_agents,
            xi=xi,
            sigma_u=sigma_u,
            theta=theta,
            sigma_v=sigma_v,
            v_bar=v_bar,
        ),
    }


@pytest.fixture
def cfg() -> dict[str, object]:
    return _baseline_config()


@pytest.fixture
def grids(cfg: dict[str, object]) -> dict[str, np.ndarray]:
    return _baseline_grids(cfg)


@pytest.fixture
def agent(cfg: dict[str, object], grids: dict[str, np.ndarray]) -> QLearningAgent:
    return QLearningAgent(agent_id=0, config=cfg, grids=grids)


def test_q_table_shape(agent: QLearningAgent) -> None:
    assert agent.Q.shape == (31, 10, 10, 15)
    assert agent.Q.dtype == np.float64


def test_visit_count_shape(agent: QLearningAgent) -> None:
    assert agent.N.shape == (31, 10, 10)
    assert agent.N.dtype == np.int64
    assert agent.N.sum() == 0


def test_subclasses_agent_abc(agent: QLearningAgent) -> None:
    assert isinstance(agent, Agent)


def test_n_actions_matches_action_grid(
    agent: QLearningAgent, cfg: dict[str, object]
) -> None:
    assert agent.n_actions == int(cfg["n_x"])


def test_initial_q_invariant_in_p_and_v_prev(agent: QLearningAgent) -> None:
    """Q_init does not depend on p_{t-1} or v_{t-1} (i.i.d. v)."""
    slice_ref = agent.Q[0, 0]
    for p_bin in range(agent.Q.shape[0]):
        for vp_bin in range(agent.Q.shape[1]):
            assert np.array_equal(agent.Q[p_bin, vp_bin], slice_ref)


def test_initial_q_is_optimistic_constant_at_v_m(
    agent: QLearningAgent, cfg: dict[str, object]
) -> None:
    """Optimistic init: every cell equals V^M = π^M / (1 − ρ)."""
    pi_m = float(cfg["pi_M"])
    rho = float(cfg["rho"])
    expected_v_m = pi_m / (1.0 - rho)

    assert agent.Q.min() == pytest.approx(expected_v_m, rel=1e-12, abs=1e-12)
    assert agent.Q.max() == pytest.approx(expected_v_m, rel=1e-12, abs=1e-12)
    assert np.all(agent.Q == expected_v_m)


def test_initial_q_is_strictly_above_nash_continuation(
    agent: QLearningAgent, cfg: dict[str, object]
) -> None:
    """Sanity: the optimistic anchor exceeds the Nash continuation V^N."""
    pi_n = float(cfg["pi_N"])
    pi_m = float(cfg["pi_M"])
    rho = float(cfg["rho"])
    assert pi_m > pi_n  # cartel > Nash (precondition for V^M > V^N)
    v_n = pi_n / (1.0 - rho)
    assert float(agent.Q.min()) > v_n


def test_one_update_moves_q_by_alpha(
    agent: QLearningAgent, cfg: dict[str, object]
) -> None:
    state = np.array([5, 3, 4], dtype=np.int64)
    next_state = np.array([6, 4, 7], dtype=np.int64)
    action = 8
    reward = 1.25

    q_old = float(agent.Q[5, 3, 4, 8])
    q_next_max = float(np.max(agent.Q[6, 4, 7, :]))
    target = reward + float(cfg["rho"]) * q_next_max
    expected = (1.0 - float(cfg["alpha"])) * q_old + float(cfg["alpha"]) * target

    agent.update(
        Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=False,
        )
    )
    assert agent.Q[5, 3, 4, 8] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_eps_one_on_first_visit_to_state(agent: QLearningAgent) -> None:
    """N[s]=0 ⇒ ε = exp(0) = 1 ⇒ first action is uniform random."""
    state = np.array([0, 0, 0], dtype=np.int64)
    assert agent.N[0, 0, 0] == 0
    rng_pre = np.random.default_rng(42)
    expected_random = int(rng_pre.random() < 1.0)
    assert expected_random == 1
    agent.act(state)
    assert agent.N[0, 0, 0] == 1


def test_eps_decays_with_visits(cfg: dict[str, object]) -> None:
    """After 1000 visits with β=5e-7, ε ≈ exp(−5e-4) ≈ 0.9995."""
    grids = _baseline_grids(cfg)
    agent = QLearningAgent(agent_id=0, config=cfg, grids=grids)
    agent.N[2, 3, 4] = 1000
    eps = math.exp(-float(cfg["beta"]) * 1000)
    assert eps == pytest.approx(0.9995, abs=1e-3)
    assert eps < 1.0


def test_act_exploitation_returns_argmax(
    cfg: dict[str, object], grids: dict[str, np.ndarray]
) -> None:
    """With N pushed high enough that ε ≈ 0, act() returns argmax Q[s, :]."""
    big_beta_cfg = {**cfg, "beta": 10.0}
    agent = QLearningAgent(agent_id=0, config=big_beta_cfg, grids=grids)
    p, vp, v = 5, 3, 4
    agent.N[p, vp, v] = 1000

    agent.Q[p, vp, v, :] = -100.0
    agent.Q[p, vp, v, 11] = 999.0

    state = np.array([p, vp, v], dtype=np.int64)
    for _ in range(20):
        action = agent.act(state)
        assert action == 11


def test_act_returns_valid_action_index(agent: QLearningAgent) -> None:
    state = np.array([5, 3, 4], dtype=np.int64)
    for _ in range(50):
        a = agent.act(state)
        assert 0 <= a < agent.n_actions


def test_reset_does_not_clear_q_or_visits(agent: QLearningAgent) -> None:
    state = np.array([5, 3, 4], dtype=np.int64)
    for _ in range(10):
        agent.act(state)
    agent.update(
        Transition(
            state=state,
            action=2,
            reward=3.0,
            next_state=np.array([6, 4, 7], dtype=np.int64),
            done=False,
        )
    )
    q_before = agent.Q.copy()
    n_before = agent.N.copy()

    agent.reset()
    assert np.array_equal(agent.Q, q_before)
    assert np.array_equal(agent.N, n_before)
    assert int(agent.N[5, 3, 4]) == 10


def test_smoke_100_random_steps(
    cfg: dict[str, object], grids: dict[str, np.ndarray]
) -> None:
    """100 random (act, update) iterations without exceptions, deterministic."""
    rng = np.random.default_rng(0)

    def run() -> tuple[float, float]:
        agent = QLearningAgent(agent_id=0, config=cfg, grids=grids)
        local_rng = np.random.default_rng(0)
        for _ in range(100):
            state = np.array(
                [
                    int(local_rng.integers(0, agent.Q.shape[0])),
                    int(local_rng.integers(0, agent.Q.shape[1])),
                    int(local_rng.integers(0, agent.Q.shape[2])),
                ],
                dtype=np.int64,
            )
            a = agent.act(state)
            next_state = np.array(
                [
                    int(local_rng.integers(0, agent.Q.shape[0])),
                    int(local_rng.integers(0, agent.Q.shape[1])),
                    int(local_rng.integers(0, agent.Q.shape[2])),
                ],
                dtype=np.int64,
            )
            agent.update(
                Transition(
                    state=state,
                    action=a,
                    reward=float(local_rng.normal()),
                    next_state=next_state,
                    done=False,
                )
            )
        return float(agent.Q.sum()), float(agent.N.sum())

    a, b = run(), run()
    assert a == b
    del rng


def test_get_q_table_returns_live_view(agent: QLearningAgent) -> None:
    q = agent.get_q_table()
    assert q is agent.Q


def test_grid_shape_validation(cfg: dict[str, object]) -> None:
    grids = _baseline_grids(cfg)
    bad_grids = {**grids, "v_grid": np.zeros(7, dtype=np.float64)}
    with pytest.raises(ValueError):
        QLearningAgent(agent_id=0, config=cfg, grids=bad_grids)


def test_zero_q_init_produces_zero_table(cfg: dict[str, object]) -> None:
    """``q_init='zero'`` fills the Q-table with zeros at the correct shape."""
    cfg_zero = {**cfg, "q_init": "zero"}
    grids = _baseline_grids(cfg_zero)
    agent = QLearningAgent(agent_id=0, config=cfg_zero, grids=grids)

    n_p = int(cfg_zero["n_p"])
    n_v = int(cfg_zero["n_v"])
    n_x = int(cfg_zero["n_x"])
    assert agent.Q.shape == (n_p, n_v, n_v, n_x)
    assert agent.Q.dtype == np.float64
    assert np.all(agent.Q == 0.0)


def test_unknown_q_init_raises(cfg: dict[str, object]) -> None:
    cfg_bad = {**cfg, "q_init": "pessimistic"}
    grids = _baseline_grids(cfg_bad)
    with pytest.raises(ValueError, match="Unknown q_init mode"):
        QLearningAgent(agent_id=0, config=cfg_bad, grids=grids)


def test_eval_mode_act_is_greedy_without_side_effects(
    cfg: dict[str, object], grids: dict[str, np.ndarray]
) -> None:
    agent = QLearningAgent(agent_id=0, config=cfg, grids=grids)  # β tiny ⇒ ε≈1
    p, vp, v = 5, 3, 4
    agent.Q[p, vp, v, :] = 0.0
    agent.Q[p, vp, v, 6] = 1.0
    agent.set_eval_mode(True)
    rng_state = agent.rng.bit_generator.state
    state = np.array([p, vp, v], dtype=np.int64)
    for _ in range(20):
        assert agent.act(state) == 6
    assert agent.N[p, vp, v] == 0
    assert agent.rng.bit_generator.state == rng_state


def test_eval_mode_update_is_noop(agent: QLearningAgent) -> None:
    agent.set_eval_mode(True)
    q_before = agent.Q.copy()
    agent.update(
        Transition(
            state=np.array([5, 3, 4], dtype=np.int64),
            action=2,
            reward=100.0,
            next_state=np.array([6, 4, 7], dtype=np.int64),
            done=False,
        )
    )
    assert np.array_equal(agent.Q, q_before)
    agent.set_eval_mode(False)
    assert agent.eval_mode is False


def test_visit_count_reports_n(agent: QLearningAgent) -> None:
    state = np.array([5, 3, 4], dtype=np.int64)
    for _ in range(3):
        agent.act(state)
    assert agent.visit_count(state) == 3


def test_rng_seed_sequence_is_used(
    cfg: dict[str, object], grids: dict[str, np.ndarray]
) -> None:
    seq_a = np.random.SeedSequence(9)
    seq_b = np.random.SeedSequence(9)
    a = QLearningAgent(0, {**cfg, "rng_seed_sequence": seq_a}, grids)
    b = QLearningAgent(1, {**cfg, "rng_seed_sequence": seq_b}, grids)
    assert a.rng.random() == b.rng.random()


def test_greedy_policy_equals_argmax_of_q(agent: QLearningAgent) -> None:
    rng = np.random.default_rng(0)
    agent.Q[...] = rng.normal(size=agent.Q.shape)
    policy = agent.greedy_policy()
    assert policy.shape == agent.Q.shape[:-1]
    assert np.array_equal(policy, np.argmax(agent.Q, axis=-1))


# ---- DGJ's Q-initialisation and per-v exploration counter ---------------------

_THESIS = {"n_agents": 2, "xi": 500.0, "sigma_u": 1.0, "theta": 0.1, "sigma_v": 1.0}


def _thesis_setup(
    action_mode: str = "per_v", **overrides: object
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Thesis parameters (n_v=5, n_x=15, n_p=31, σ_u=1) with an action table."""
    v_grid = discretization.build_v_grid(n_v=5)
    bench = benchmarks.compute_all_benchmarks(**_THESIS, v_grid=v_grid)
    cfg: dict[str, object] = {
        "alpha": 0.01,
        "beta": 1.0e-6,
        "rho": 0.95,
        "n_agents": 2,
        "n_v": 5,
        "n_x": 15,
        "n_p": 31,
        "v_bar": 0.0,
        "sigma_v": 1.0,
        "seed": 42,
        **bench,
    }
    cfg.update(overrides)
    if action_mode == "per_v":
        action_grid = discretization.build_chi_grid(n_x=15, **_THESIS)
    else:
        action_grid = discretization.build_action_grid(n_x=15, n_v=5, **_THESIS)
    grids = {
        "v_grid": v_grid,
        "action_grid": action_grid,
        "action_table": discretization.build_action_table(
            action_mode, n_x=15, v_grid=v_grid, **_THESIS
        ),
        "price_grid": discretization.build_price_grid(n_p=31, n_v=5, **_THESIS),
    }
    return cfg, grids


def test_default_switches_are_the_legacy_behaviour(agent: QLearningAgent) -> None:
    assert agent.q_init == "optimistic"
    assert agent.exploration_counter == "state"
    assert agent.action_table is None  # the fixture passes no action table
    assert agent.N_v.shape == (10,)
    assert agent.N_v.dtype == np.int64
    assert agent.N_v.sum() == 0


def test_paper_q_init_values_thesis_parameters() -> None:
    """DGJ §4.2 initialisation on the per-v table at the thesis parameters."""
    cfg, grids = _thesis_setup(q_init="paper")
    a = QLearningAgent(agent_id=0, config=cfg, grids=grids)
    assert a.Q.shape == (31, 5, 5, 15)
    assert a.Q.dtype == np.float64
    assert a.Q.flags.writeable
    expected = {
        4: {0: 1852.233, 7: 1995.941, 13: 2053.759, 14: 2057.530},  # v = +1.2816
        3: {0: 310.134, 7: 334.196, 13: 343.877, 14: 344.509},  # v = +0.5244
    }
    for k, values in expected.items():
        for action, value in values.items():
            assert a.Q[0, 0, k, action] == pytest.approx(value, abs=0.01)
            # Symmetric for negative v (index 4 − k).
            assert a.Q[0, 0, 4 - k, action] == pytest.approx(value, abs=0.01)
    assert np.all(a.Q[:, :, 2, :] == 0.0)  # v = v̄
    # Independent of p_{t-1} and v_{t-1}.
    assert np.all(a.Q == a.Q[0:1, 0:1, :, :])


def test_paper_q_init_equals_the_papers_average_over_opponent_actions() -> None:
    """Q_0 = 1/((1−ρ)n_x) · Σ_{x'} [v − (v̄ + λ^N (x + (I−1)x'))] · x.

    The average over the opponent's actions of DGJ §4.2.
    """
    for mode in ("per_v", "global"):
        cfg, grids = _thesis_setup(action_mode=mode, q_init="paper")
        a = QLearningAgent(agent_id=0, config=cfg, grids=grids)
        table, v_grid = grids["action_table"], grids["v_grid"]
        lam, rho, n_x = float(cfg["lambda_N"]), 0.95, 15
        for k in range(5):
            for action in range(n_x):
                x = table[k, action]
                total = sum((v_grid[k] - lam * (x + xo)) * x for xo in table[k])
                assert a.Q[3, 1, k, action] == pytest.approx(
                    total / ((1 - rho) * n_x), rel=1e-10, abs=1e-9
                )


def test_paper_q_init_requires_an_action_table() -> None:
    cfg, grids = _thesis_setup(q_init="paper")
    del grids["action_table"]
    with pytest.raises(ValueError, match="action_table"):
        QLearningAgent(agent_id=0, config=cfg, grids=grids)


def test_action_table_of_the_wrong_shape_raises() -> None:
    cfg, grids = _thesis_setup()
    grids["action_table"] = grids["action_table"][:, :-1]
    with pytest.raises(ValueError, match="action_table"):
        QLearningAgent(agent_id=0, config=cfg, grids=grids)


def test_unknown_exploration_counter_raises() -> None:
    cfg, grids = _thesis_setup(exploration_counter="price")
    with pytest.raises(ValueError, match="exploration_counter"):
        QLearningAgent(agent_id=0, config=cfg, grids=grids)


class _ScriptedRng:
    """Stands in for the agent's RNG: fixed uniform draw, marker random action."""

    def __init__(self, marker: int) -> None:
        self.u = 0.0
        self.marker = marker

    def random(self) -> float:
        return self.u

    def integers(self, _low: int, _high: int) -> int:
        return self.marker


def _explores(agent: QLearningAgent, state: np.ndarray, u: float) -> bool:
    """True if ``act`` takes the exploration branch when the uniform draw is u."""
    agent.rng.u = u  # type: ignore[attr-defined]
    return agent.act(state) == agent.rng.marker  # type: ignore[attr-defined]


def _scripted_pair(counter: str) -> tuple[QLearningAgent, QLearningAgent]:
    """Two identical agents: one probes just below ε, the other just above."""
    agents = []
    for _ in range(2):
        cfg, grids = _thesis_setup(beta=0.1, exploration_counter=counter)
        a = QLearningAgent(agent_id=0, config=cfg, grids=grids)
        a.rng = _ScriptedRng(marker=9)  # type: ignore[assignment]
        agents.append(a)  # optimistic init: the greedy action is index 0, not 9
    return agents[0], agents[1]


def test_exploration_counter_v_uses_visits_of_v_regardless_of_price() -> None:
    """Mode "v": ε at the n-th visit of a v is exp(−β(n−1)), whatever the price bin."""
    below, above = _scripted_pair("v")
    visits = {k: 0 for k in range(5)}
    rng = np.random.default_rng(0)
    for _ in range(200):
        state = np.array(
            [rng.integers(0, 31), rng.integers(0, 5), rng.integers(0, 5)],
            dtype=np.int64,
        )
        k = int(state[2])
        eps = math.exp(-0.1 * visits[k])  # n-th visit: n − 1 earlier visits
        assert _explores(below, state, eps - 1e-9)
        assert not _explores(above, state, eps + 1e-9)
        visits[k] += 1
    assert below.N_v.tolist() == [visits[k] for k in range(5)]
    assert int(below.N.sum()) == 200  # the per-state counter still counts


def test_exploration_counter_state_uses_visits_of_the_full_state() -> None:
    """Mode "state" (default): ε = exp(−β · visits of the full state triple)."""
    below, above = _scripted_pair("state")
    visits: dict[tuple[int, int, int], int] = {}
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = (int(rng.integers(0, 3)), int(rng.integers(0, 2)), int(rng.integers(0, 5)))
        state = np.array(s, dtype=np.int64)
        eps = math.exp(-0.1 * visits.get(s, 0))
        assert _explores(below, state, eps - 1e-9)
        assert not _explores(above, state, eps + 1e-9)
        visits[s] = visits.get(s, 0) + 1
        assert below.visit_count(state) == visits[s]
    assert int(below.N_v.sum()) == 200  # N_v is counted in both modes


def test_eval_mode_leaves_both_counters_untouched() -> None:
    cfg, grids = _thesis_setup(exploration_counter="v", q_init="paper")
    a = QLearningAgent(agent_id=0, config=cfg, grids=grids)
    a.set_eval_mode(True)
    state = np.array([4, 1, 3], dtype=np.int64)
    rng_state = a.rng.bit_generator.state
    for _ in range(5):
        assert a.act(state) == int(np.argmax(a.Q[4, 1, 3]))
    assert a.N.sum() == 0 and a.N_v.sum() == 0
    assert a.rng.bit_generator.state == rng_state


def test_visit_count_returns_full_state_visits_in_mode_v() -> None:
    cfg, grids = _thesis_setup(exploration_counter="v")
    a = QLearningAgent(agent_id=0, config=cfg, grids=grids)
    s1 = np.array([4, 1, 3], dtype=np.int64)
    s2 = np.array([5, 1, 3], dtype=np.int64)  # same v_t, other price bin
    for _ in range(3):
        a.act(s1)
    a.act(s2)
    assert a.visit_count(s1) == 3
    assert a.visit_count(s2) == 1
    assert a.N_v[3] == 4


def test_agent_smoke_100_steps_paper_switches_deterministic() -> None:
    """Smoke test with the three DGJ-style switches on."""

    def run() -> tuple[list[int], np.ndarray]:
        cfg, grids = _thesis_setup(
            beta=0.01, exploration_counter="v", q_init="paper", seed=7
        )
        a = QLearningAgent(agent_id=0, config=cfg, grids=grids)
        rng = np.random.default_rng(1)
        state = np.array([15, 2, 2], dtype=np.int64)
        actions = []
        for _ in range(100):
            action = a.act(state)
            nxt = np.array(
                [rng.integers(0, 31), state[2], rng.integers(0, 5)], dtype=np.int64
            )
            a.update(Transition(state, action, float(rng.normal()), nxt, False))
            actions.append(action)
            state = nxt
        return actions, a.Q.copy()

    actions_a, q_a = run()
    actions_b, q_b = run()
    assert actions_a == actions_b
    assert np.array_equal(q_a, q_b)
    assert all(0 <= x < 15 for x in actions_a)
