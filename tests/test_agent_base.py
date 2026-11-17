"""Tests for the Agent ABC and Transition NamedTuple."""

from __future__ import annotations

import numpy as np
import pytest

from ai_trading.agents.base import Agent, Transition


class DummyAgent(Agent):
    """Minimal concrete subclass used only by these tests."""

    def act(self, state: np.ndarray) -> int:
        return 0

    def update(self, transition: Transition) -> None:
        return None

    def reset(self) -> None:
        return None


def test_transition_is_a_named_tuple() -> None:
    s = np.zeros(2, dtype=np.float64)
    s_prime = np.ones(2, dtype=np.float64)
    t = Transition(state=s, action=3, reward=1.5, next_state=s_prime, done=False)

    assert isinstance(t, tuple)
    assert isinstance(t, Transition)
    assert t._fields == ("state", "action", "reward", "next_state", "done")


def test_transition_fields_accessible_by_name() -> None:
    s = np.array([1.0, 2.0])
    s_prime = np.array([3.0, 4.0])
    t = Transition(state=s, action=7, reward=0.25, next_state=s_prime, done=True)

    assert np.array_equal(t.state, s)
    assert t.action == 7
    assert t.reward == 0.25
    assert np.array_equal(t.next_state, s_prime)
    assert t.done is True


def test_agent_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Agent(agent_id=0, n_actions=10)  # type: ignore[abstract]


def test_concrete_subclass_instantiates() -> None:
    agent = DummyAgent(agent_id=0, n_actions=15)
    assert agent.agent_id == 0
    assert agent.n_actions == 15


def test_concrete_subclass_accepts_extra_kwargs() -> None:
    agent = DummyAgent(agent_id=1, n_actions=15, seed=42, foo="bar")
    assert agent.agent_id == 1
    assert agent.n_actions == 15


def test_act_returns_int() -> None:
    agent = DummyAgent(agent_id=0, n_actions=15)
    state = np.zeros(4, dtype=np.float64)
    action = agent.act(state)
    assert isinstance(action, int)


def test_name_property_combines_class_and_agent_id() -> None:
    agent = DummyAgent(agent_id=0, n_actions=15)
    assert agent.name == "DummyAgent_0"

    other = DummyAgent(agent_id=7, n_actions=15)
    assert other.name == "DummyAgent_7"


def test_update_and_reset_callable_without_error() -> None:
    agent = DummyAgent(agent_id=0, n_actions=15)
    transition = Transition(
        state=np.zeros(2),
        action=1,
        reward=0.5,
        next_state=np.ones(2),
        done=False,
    )
    assert agent.update(transition) is None
    assert agent.reset() is None


def test_subclass_missing_method_remains_abstract() -> None:
    class HalfBaked(Agent):
        def act(self, state: np.ndarray) -> int:
            return 0

    with pytest.raises(TypeError):
        HalfBaked(agent_id=0, n_actions=5)  # type: ignore[abstract]


def test_eval_mode_defaults_and_visit_count_none() -> None:
    agent = DummyAgent(agent_id=0, n_actions=3)
    assert agent.eval_mode is False
    agent.set_eval_mode(True)
    assert agent.eval_mode is True
    assert agent.visit_count(np.zeros(3, dtype=np.int64)) is None


def test_greedy_policy_default_is_none() -> None:
    agent = DummyAgent(agent_id=0, n_actions=3)
    assert agent.greedy_policy() is None
