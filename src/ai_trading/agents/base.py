"""Agent ABC — the contract every speculator must satisfy.

All agents in this project inherit from Agent and implement two methods:
    act(state)          — choose a discrete action given the current state.
    update(transition)  — learn from a (s, a, r, s') transition.

The Transition named-tuple is also defined here for shared use.

All speculator implementations (the tabular Q-learner, the tabular softmax
actor-critic and the fixed-policy speculator) implement the same
(act, update) interface. The runner is unaware of which family it is calling.

Evaluation mode (all agent families):
    After ``set_eval_mode(True)`` an agent must (i) act deterministically on
    its learned policy (no exploration), (ii) not consume its RNG, and (iii)
    not change learned parameters in ``update()``. The runner relies on this
    for the eval window and for the IRF's shock-vs-counterfactual paths.

Usage:
    from ai_trading.agents.base import Agent, Transition
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NamedTuple

import numpy as np


class Transition(NamedTuple):
    """One agent timestep: ``(s, a, r, s', done)``.

    Attributes:
        state: State observed by the agent before acting.
        action: Discrete action bin chosen by the agent.
        reward: Profit received this period.
        next_state: State observed after the market cleared.
        done: Whether the episode terminated this period.
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class Agent(ABC):
    """Abstract base class for all speculator agents.

    Subclasses must implement :meth:`act`, :meth:`update`, and :meth:`reset`.
    The base class only stores ``agent_id`` and ``n_actions``; everything
    else (RNG, learned parameters, replay buffers, config) is the
    subclass's responsibility.

    Args:
        agent_id: Integer identifier for this speculator within the run
            (0-indexed). Used by the runner to route per-agent state and
            by :attr:`name` for logging.
        n_actions: Size of the discrete action grid.
        **kwargs: Subclass-specific keyword arguments; ignored at this
            level so subclasses can extend ``__init__`` without
            forwarding boilerplate.
    """

    def __init__(self, agent_id: int, n_actions: int, **kwargs: Any) -> None:
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.eval_mode: bool = False

    @property
    def name(self) -> str:
        """Identifier of the form ``"<ClassName>_<agent_id>"``."""
        return f"{type(self).__name__}_{self.agent_id}"

    def set_eval_mode(self, enabled: bool) -> None:
        """Toggle evaluation mode.

        In evaluation mode an agent must (i) act deterministically on its
        learned policy (no exploration), (ii) not consume its RNG, and (iii)
        not change learned parameters in :meth:`update`. The runner relies on
        this for the eval window and for the IRF's counterfactual paths.
        """
        self.eval_mode = bool(enabled)

    def visit_count(self, state: np.ndarray) -> int | None:
        """Number of times ``state`` was visited (None if not tracked)."""
        return None

    def greedy_policy(self) -> np.ndarray | None:
        """Greedy action index per state, or None without a tabular policy.

        Returns:
            Integer array of shape ``(n_p, n_v, n_v)`` holding the greedy
            action index for every state, or ``None`` if the agent has no
            tabular policy. The runner uses it for the convergence check and
            the stability diagnostic.
        """
        return None

    def policy_concentration(self, onpath_mask: np.ndarray) -> float | None:
        """Mean of ``max_a π(a | s)`` over the states where ``onpath_mask`` is True.

        Only stochastic-policy agents (the actor-critic) report this; the
        runner stores it per agent as ``policy_concentration_agent_i`` and
        it shows whether the policy has become near-deterministic (target
        ≥ 0.9). ``None`` means not applicable.

        Args:
            onpath_mask: Bool array over flat states
                (``p_bin · n_v · n_v + v_prev_bin · n_v + v_bin``).
        """
        return None

    @abstractmethod
    def act(self, state: np.ndarray) -> int:
        """Choose a discrete action bin for the current state.

        Args:
            state: Observation array for this agent.

        Returns:
            Integer index in ``[0, n_actions)``.
        """

    @abstractmethod
    def update(self, transition: Transition) -> None:
        """Learn from a completed transition.

        Args:
            transition: ``(s, a, r, s', done)`` for the just-finished step.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset episode-level state.

        Implementations should clear any per-episode buffers, counters,
        or last-observed-state caches. They must NOT reset learned
        parameters (Q-tables, network weights, fitted estimators) — those
        persist across episodes.
        """
