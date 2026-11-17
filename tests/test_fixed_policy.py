"""Tests for FixedChiAgent (agents/fixed_policy.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ai_trading.agents.base import Transition
from ai_trading.agents.fixed_policy import FixedChiAgent
from ai_trading.simulation import runner

# Config of the baseline R1 (the baseline specification).
_M2_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "01_replicate_baseline"
    / "config_r1_baseline.yaml"
)


def _m2_fixed(chi: object) -> tuple[object, FixedChiAgent]:
    config = runner.load_config(_M2_CONFIG)
    config["agents"] = [
        {"type": "qlearner"},
        {"type": "fixed_chi", "config": {"chi": chi}},
    ]
    env, agents, bench = runner.build_session(config, float(config["sigma_u"]), 1)
    agent = agents[1]
    assert isinstance(agent, FixedChiAgent)
    env.bench = bench  # for the tests below
    return env, agent


def test_nash_on_the_per_v_grid_is_bin_13_at_every_v_but_v_bar() -> None:
    env, agent = _m2_fixed("nash")
    assert agent.chi == pytest.approx(float(env.bench["chi_N"]))
    assert list(agent.action_by_v[[0, 1, 3, 4]]) == [13, 13, 13, 13]
    assert agent.act(np.array([5, 0, 4])) == 13
    assert agent.act(np.array([5, 0, 0])) == 13
    # At v = v̄ every action trades zero; any index is the same trade.
    assert env.action_table[2, agent.act(np.array([15, 2, 2]))] == 0.0


def test_cartel_picks_the_nearest_level_and_floats_are_accepted() -> None:
    _env, cartel = _m2_fixed("cartel")
    assert list(cartel.action_by_v[[0, 4]]) == [1, 1]
    _env, by_value = _m2_fixed(170.83)
    assert list(by_value.action_by_v[[0, 4]]) == [14, 14]
    with pytest.raises(ValueError, match="nash"):
        _m2_fixed("random")


def test_fixed_agent_never_learns_and_reports_a_constant_policy() -> None:
    _env, agent = _m2_fixed("nash")
    before = agent.action_by_v.copy()
    s = np.array([1, 2, 3])
    agent.update(Transition(s, 0, 1e6, s, False))
    agent.reset()
    np.testing.assert_array_equal(agent.action_by_v, before)
    gp = agent.greedy_policy()
    assert gp.shape == (31, 5, 5)
    assert np.all(gp[:, :, 4] == 13)
    assert agent.visit_count(s) is None
    assert agent.policy_concentration(np.ones(31 * 25, dtype=bool)) is None
