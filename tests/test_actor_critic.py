"""Tests for ActorCriticAgent (agents/actor_critic.py)."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from ai_trading.agents.actor_critic import ActorCriticAgent
from ai_trading.agents.base import Agent, Transition
from ai_trading.simulation import runner

# Config of the baseline R1 (the baseline specification).
_M2_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "01_replicate_baseline"
    / "config_r1_baseline.yaml"
)


def _small_config(**overrides: object) -> dict[str, object]:
    cfg: dict[str, object] = {
        "rho": 0.95,
        "n_agents": 2,
        "n_v": 3,
        "n_x": 3,
        "n_p": 2,
        "lambda_N": 0.002,
        "pi_M": 50.0,
        "beta": 1.0e-4,
        "v_bar": 0.0,
        "seed": 7,
    }
    cfg.update(overrides)
    return cfg


def _small_grids(cfg: dict[str, object]) -> dict[str, np.ndarray]:
    n_v, n_x = int(cfg["n_v"]), int(cfg["n_x"])
    v_grid = np.linspace(-1.0, 1.0, n_v)
    chi = np.linspace(100.0, 200.0, n_x)
    return {
        "v_grid": v_grid,
        "action_table": chi[None, :] * v_grid[:, None],
        "action_grid": chi,
        "price_grid": np.linspace(-1.0, 1.0, int(cfg["n_p"])),
    }


def _agent(**overrides: object) -> ActorCriticAgent:
    cfg = _small_config(**overrides)
    return ActorCriticAgent(agent_id=0, config=cfg, grids=_small_grids(cfg))


def _state(p: int = 0, vp: int = 0, v: int = 0) -> np.ndarray:
    return np.array([p, vp, v], dtype=np.int64)


# ---- construction ------------------------------------------------------------


def test_actor_critic_is_an_agent_with_uniform_initial_policy() -> None:
    agent = _agent()
    assert isinstance(agent, Agent)
    for s in range(agent.n_states):
        assert agent.H[s] == [0.0, 0.0, 0.0]
    p = agent.policy(_state(1, 2, 0))
    np.testing.assert_allclose(p, np.full(3, 1.0 / 3.0))
    assert math.isclose(float(p.sum()), 1.0)


def test_critic_initial_values_at_the_m2_parameters() -> None:
    config = runner.load_config(_M2_CONFIG)
    config["agents"] = [{"type": "actor_critic"}, {"type": "actor_critic"}]
    _env, agents, _bench = runner.build_session(config, float(config["sigma_u"]), 1)
    agent = agents[0]
    assert isinstance(agent, ActorCriticAgent)
    v0 = agent.critic_init_by_value()
    np.testing.assert_allclose(v0, [977.53, 895.09, 878.51, 895.09, 977.53], atol=0.01)
    values = agent.get_values()
    assert values.shape == (31, 5, 5)
    np.testing.assert_allclose(values[3, 1, :], v0)
    assert agent.reward_scale == pytest.approx(float(_bench["pi_M"]))
    assert agent.entropy_decay == pytest.approx(1.0e-6)


def test_actor_critic_requires_the_action_table() -> None:
    cfg = _small_config()
    grids = _small_grids(cfg)
    grids.pop("action_table")
    with pytest.raises(ValueError, match="action_table"):
        ActorCriticAgent(agent_id=0, config=cfg, grids=grids)


# ---- update --------------------------------------------------------------------


def test_hand_computed_update_without_entropy() -> None:
    agent = _agent(
        alpha_critic=0.05, alpha_actor=0.1, reward_scale=10.0, entropy_coef=0.0
    )
    s, s2 = _state(0, 0, 0), _state(1, 0, 1)
    fs, fs2 = agent._flat(s, 3), agent._flat(s2, 3)
    agent.V[fs], agent.V[fs2] = 10.0, 12.0
    agent.H[fs] = [0.0, 0.0, 0.0]
    agent.update(Transition(state=s, action=0, reward=5.0, next_state=s2, done=False))
    assert agent.V[fs] == pytest.approx(10.32, abs=1e-12)
    np.testing.assert_allclose(agent.H[fs], [0.0, -0.064, -0.064], atol=1e-12)


def test_entropy_bonus_raises_the_policy_entropy() -> None:
    agent = _agent(entropy_coef=0.1, entropy_decay=0.0)
    s = _state(0, 0, 2)
    fs = agent._flat(s, 3)
    agent.H[fs] = [0.0, -1.0, -2.0]
    # delta = r + rho V[s] - V[s] = 0  <=>  r = (1 - rho) V[s]
    r = (1.0 - agent.rho) * agent.V[fs]

    def entropy(p: np.ndarray) -> float:
        return float(-(p * np.log(p)).sum())

    before = entropy(agent.policy(s))
    agent.update(Transition(state=s, action=0, reward=r, next_state=s, done=False))
    after = entropy(agent.policy(s))
    assert after > before


def test_preferences_are_shifted_to_zero_and_floored() -> None:
    agent = _agent(entropy_coef=0.0, reward_scale=1.0, alpha_actor=100.0)
    s = _state(0, 0, 0)
    fs = agent._flat(s, 3)
    agent.V[fs] = 0.0
    agent.update(Transition(state=s, action=2, reward=1.0, next_state=s, done=False))
    assert max(agent.H[fs]) == 0.0
    assert min(agent.H[fs]) >= -30.0


# ---- acting --------------------------------------------------------------------


def test_training_act_consumes_one_draw_and_counts_visits() -> None:
    agent = _agent()
    s = _state(1, 2, 1)
    rng_copy = np.random.default_rng(7)  # seed + agent_id = 7 + 0
    a = agent.act(s)
    u = rng_copy.random()
    # Uniform policy: the action is the smallest a with (a + 1) / 3 >= u.
    assert a == min(math.ceil(u * 3) - 1, 2)
    assert agent.N[agent._flat(s, 3)] == 1
    assert agent.N_v[1] == 1
    assert agent.rng.bit_generator.state == rng_copy.bit_generator.state


def test_eval_mode_is_deterministic_and_side_effect_free() -> None:
    agent = _agent()
    s = _state(1, 1, 2)
    fs = agent._flat(s, 3)
    agent.H[fs] = [-1.0, 0.0, -0.5]
    agent.set_eval_mode(True)
    rng_state = copy.deepcopy(agent.rng.bit_generator.state)
    h_before = copy.deepcopy(agent.H)
    v_before = list(agent.V)
    for _ in range(5):
        assert agent.act(s) == 1
    agent.update(Transition(state=s, action=0, reward=100.0, next_state=s, done=False))
    assert agent.rng.bit_generator.state == rng_state
    assert agent.N[fs] == 0 and sum(agent.N_v) == 0
    assert agent.H == h_before and agent.V == v_before


def test_greedy_policy_shape_and_first_argmax() -> None:
    agent = _agent()
    fs = agent._flat(_state(1, 0, 2), 3)
    agent.H[fs] = [0.0, 0.0, -1.0]  # tie between 0 and 1 → first index
    gp = agent.greedy_policy()
    assert gp.shape == (2, 3, 3) and gp.dtype.kind == "i"
    assert gp[1, 0, 2] == 0
    assert agent.visit_count(_state(1, 0, 2)) == 0


def test_one_state_bandit_learns_the_best_arm() -> None:
    cfg = _small_config(
        n_p=1, n_v=1, rho=0.0, reward_scale=1.0, entropy_decay=0.01, seed=3
    )
    grids = {
        "v_grid": np.array([0.0]),
        "action_table": np.array([[0.0, 0.0, 0.0]]),
        "action_grid": np.zeros(3),
        "price_grid": np.zeros(1),
    }
    agent = ActorCriticAgent(agent_id=0, config=cfg, grids=grids)
    noise = np.random.default_rng(11)
    means = np.array([1.0, 2.0, 3.0])
    s = _state(0, 0, 0)
    for _ in range(5000):
        a = agent.act(s)
        r = float(means[a] + noise.normal(0.0, 0.5))
        agent.update(Transition(state=s, action=a, reward=r, next_state=s, done=False))
    assert int(agent.greedy_policy()[0, 0, 0]) == 2
    assert agent.policy(s)[2] > 0.9
    assert agent.policy_concentration(np.array([True])) > 0.9


def test_policy_concentration_mask_semantics() -> None:
    agent = _agent()
    agent.H[0] = [0.0, -30.0, -30.0]  # concentrated
    mask = np.zeros(agent.n_states, dtype=bool)
    assert math.isnan(agent.policy_concentration(mask))
    mask[0] = True
    assert agent.policy_concentration(mask) == pytest.approx(1.0, abs=1e-12)
    mask[1] = True  # uniform state
    assert agent.policy_concentration(mask) == pytest.approx((1.0 + 1.0 / 3.0) / 2)
    with pytest.raises(ValueError):
        agent.policy_concentration(np.ones(3, dtype=bool))


def test_determinism_for_a_fixed_seed_sequence() -> None:
    def run() -> tuple[list[list[float]], list[float]]:
        cfg = _small_config(rng_seed_sequence=np.random.SeedSequence(99))
        agent = ActorCriticAgent(agent_id=0, config=cfg, grids=_small_grids(cfg))
        env_rng = np.random.default_rng(5)
        s = _state(0, 0, 0)
        for _ in range(300):
            a = agent.act(s)
            s2 = _state(int(env_rng.integers(2)), int(s[2]), int(env_rng.integers(3)))
            r = float(env_rng.normal(0.0, 10.0))
            agent.update(Transition(s, a, r, s2, False))
            s = s2
        return agent.H, agent.V

    h1, v1 = run()
    h2, v2 = run()
    assert h1 == h2 and v1 == v2


def test_preferences_and_values_round_trip() -> None:
    agent = _agent()
    prefs = agent.get_preferences()
    vals = agent.get_values()
    assert prefs.shape == (2, 3, 3, 3) and vals.shape == (2, 3, 3)
    prefs[1, 2, 0, :] = [0.0, -1.0, -2.0]
    vals[1, 2, 0] = 123.0
    agent.set_preferences(prefs)
    agent.set_values(vals)
    fs = agent._flat(_state(1, 2, 0), 3)
    assert agent.H[fs] == [0.0, -1.0, -2.0] and agent.V[fs] == 123.0
    np.testing.assert_array_equal(agent.get_preferences(), prefs)
