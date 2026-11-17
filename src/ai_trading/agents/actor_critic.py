"""Tabular softmax actor-critic speculator (heterogeneity extension).

A policy-based learner on the same state table and action grid as the
tabular Q-learner (:class:`ai_trading.agents.q_learning.QLearningAgent`), so
that the only difference between the two is value-based against
policy-based learning.

Algorithm: one-step actor-critic with a softmax policy over action
preferences and an entropy bonus whose coefficient decays on the same clock
as the Q-learner's exploration rate.

State (as for the Q-learner):
    ``s_t = (p_{t-1}, v_{t-1}, v_t)`` as three bin indices. Flat index
    ``s = p_bin · n_v · n_v + v_prev_bin · n_v + v_bin`` (the index used by the
    runner's training diagnostics).

Tables (Python lists for speed; numpy copies via :meth:`get_preferences`
and :meth:`get_values`):
    ``H[s][a]`` action preferences, all 0.0 at the start (uniform policy).
    ``V[s]`` critic, initialised at the value of uniform random play by both
    agents::

        V_0(k) = r̄(k) + ρ · mean_k(r̄) / (1 − ρ)

    where ``k`` is the state's current value index and ``r̄(k)`` is the mean
    over own action ``a`` and opponent action ``a'`` of
    ``((v_k − v̄) − λ^N · (x_a + (I − 1) · x_{a'})) · x_a`` with ``x`` from
    ``action_table[k]``. Under the baseline specification
    ``V_0 = [977.53, 895.09, 878.51, 895.09, 977.53]``.
    ``N[s]`` visits per state, ``N_v[k]`` visits per current value index.

Policy:
    ``π(a | s) = exp(H[s][a] − max_b H[s][b]) / Σ_b exp(H[s][b] − max_b H[s][b])``.

Acting in training: one uniform draw ``u`` per call; the action is the
smallest ``a`` with cumulative probability ``≥ u`` (the last index if
rounding leaves none). Then ``N[s] += 1`` and ``N_v[k] += 1``.

Update after ``(s, a, r, s')`` (``k`` = current value index of ``s``)::

    δ      = r + ρ · V[s'] − V[s]
    V[s]  += α_critic · δ
    d      = δ / reward_scale
    η      = entropy_coef · exp(−entropy_decay · (N_v[k] − 1))   # visits of k before this period
    p      = π(· | s) from H[s] before the change
    ℋ      = −Σ_b p_b · ln p_b
    H[s][b] += α_actor · ( d · (1{b = a} − p_b) + η · (−p_b · (ln p_b + ℋ)) )   for every b
    H[s][·] ← max(H[s][·] − max_b H[s][b], −30)

The first term is the REINFORCE-style score-function step with the TD error
as the advantage estimate; the second is the gradient of the policy entropy
with respect to the preferences. Shifting the preferences to a maximum of 0
leaves the policy unchanged and keeps the exponentials bounded; the floor at
−30 keeps every action at a probability of at least about ``e^-30``.

Why the entropy bonus decays: with a constant bonus the policy never becomes
deterministic, and the frozen greedy evaluation would measure a different
opponent from the one the Q-learner trained against. The runner reports the
mean of ``max_a π(a | s)`` over on-path states (:meth:`policy_concentration`)
so that this can be checked (target ≥ 0.9).

Evaluation mode (see :class:`ai_trading.agents.base.Agent`): :meth:`act`
returns the first index of ``max(H[s])`` without a random draw and without
touching a counter; :meth:`update` is a no-op.

References (section level):
    Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd
    ed., ch. 13 (policy-gradient methods): softmax in action preferences
    (§13.1), one-step actor-critic (§13.5). Entropy regularisation of the
    policy: Williams & Peng (1991), *Connection Science* 3(3); Mnih et al.
    (2016), ICML (asynchronous advantage actor-critic). The decaying entropy
    coefficient on the exploration clock is this thesis's design.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from ai_trading.agents.base import Agent, Transition

logger = logging.getLogger(__name__)

_PREF_FLOOR = -30.0


class ActorCriticAgent(Agent):
    """One-step tabular actor-critic with a softmax policy and an entropy bonus.

    Attributes:
        rho: Discount factor.
        n_agents: Number of strategic speculators (DGJ's symbol I).
        lambda_N: Closed-form Nash price impact (critic initialisation).
        pi_M: Closed-form cartel profit (default ``reward_scale``).
        beta: The Q-learner's exploration-decay rate (default
            ``entropy_decay``).
        v_bar: Mean of the fundamental value.
        alpha_actor: Step size on the preferences.
        alpha_critic: Step size on the critic.
        entropy_coef: η₀, the initial entropy coefficient.
        entropy_decay: Decay rate of η per earlier visit of the current v.
        reward_scale: Divides the TD error in the actor update.
        n_p, n_v, n_x: Grid sizes.
        n_states: ``n_p · n_v · n_v``.
        H: Preferences, ``n_states`` lists of ``n_x`` floats.
        V: Critic, ``n_states`` floats.
        N: Visits per state (ints). ``N_v``: visits per value index (ints).
        rng: Local ``numpy.random.Generator``.
    """

    def __init__(
        self,
        agent_id: int,
        config: dict[str, Any],
        grids: dict[str, np.ndarray],
    ) -> None:
        """Build the preference and critic tables.

        Args:
            agent_id: Integer identifier (0-indexed).
            config: Mapping with keys ``rho``, ``n_v``, ``n_x``, ``n_p``,
                ``n_agents``, ``lambda_N``, ``pi_M``, ``beta``; optional
                ``v_bar`` (default 0.0), ``rng_seed_sequence`` or ``seed``
                (default 0; offset by ``agent_id``), and the agent's own keys
                ``alpha_actor`` (0.1), ``alpha_critic`` (0.05),
                ``entropy_coef`` (0.1), ``entropy_decay`` (default: ``beta``),
                ``reward_scale`` (default: ``pi_M``).
            grids: Mapping with ``v_grid`` (shape ``(n_v,)``) and
                ``action_table`` (shape ``(n_v, n_x)``); ``action_grid`` and
                ``price_grid`` are accepted and ignored.

        Raises:
            ValueError: On a missing or mis-shaped grid, or a non-positive
                ``reward_scale``.
        """
        n_v = int(config["n_v"])
        n_x = int(config["n_x"])
        n_p = int(config["n_p"])
        super().__init__(agent_id=agent_id, n_actions=n_x)
        self.n_p, self.n_v, self.n_x = n_p, n_v, n_x
        self.n_states: int = n_p * n_v * n_v

        self.rho: float = float(config["rho"])
        self.n_agents: int = int(config["n_agents"])
        self.lambda_N: float = float(config["lambda_N"])
        self.pi_M: float = float(config["pi_M"])
        self.beta: float = float(config["beta"])
        self.v_bar: float = float(config.get("v_bar", 0.0))

        self.alpha_actor: float = float(config.get("alpha_actor", 0.1))
        self.alpha_critic: float = float(config.get("alpha_critic", 0.05))
        self.entropy_coef: float = float(config.get("entropy_coef", 0.1))
        decay = config.get("entropy_decay")
        self.entropy_decay: float = self.beta if decay is None else float(decay)
        scale = config.get("reward_scale")
        self.reward_scale: float = self.pi_M if scale is None else float(scale)
        if not self.reward_scale > 0.0:
            raise ValueError(f"reward_scale must be positive, got {self.reward_scale}.")

        if "action_table" not in grids or grids["action_table"] is None:
            raise ValueError("ActorCriticAgent needs grids['action_table'].")
        self.v_grid: np.ndarray = np.asarray(grids["v_grid"], dtype=np.float64)
        self.action_table: np.ndarray = np.asarray(
            grids["action_table"], dtype=np.float64
        )
        if self.v_grid.shape != (n_v,):
            raise ValueError(
                f"v_grid has shape {self.v_grid.shape}, expected ({n_v},)."
            )
        if self.action_table.shape != (n_v, n_x):
            raise ValueError(
                f"action_table has shape {self.action_table.shape}, "
                f"expected ({n_v}, {n_x})."
            )

        seed_source = config.get("rng_seed_sequence")
        if seed_source is None:
            seed_source = int(config.get("seed", 0)) + agent_id
        self.rng: np.random.Generator = np.random.default_rng(seed_source)

        v0 = self.critic_init_by_value()
        self.V: list[float] = [float(v0[s % n_v]) for s in range(self.n_states)]
        self.H: list[list[float]] = [[0.0] * n_x for _ in range(self.n_states)]
        self.N: list[int] = [0] * self.n_states
        self.N_v: list[int] = [0] * n_v

        logger.debug(
            "ActorCriticAgent_%d initialised: n_states=%d, alpha_actor=%.3g, "
            "alpha_critic=%.3g, entropy_coef=%.3g, entropy_decay=%.3g, "
            "reward_scale=%.6g",
            agent_id,
            self.n_states,
            self.alpha_actor,
            self.alpha_critic,
            self.entropy_coef,
            self.entropy_decay,
            self.reward_scale,
        )

    # ---- initialisation -------------------------------------------------------

    def critic_init_by_value(self) -> np.ndarray:
        """``V_0(k)`` for every value index ``k`` (see the module docstring)."""
        x = self.action_table  # (n_v, n_x)
        dv = (self.v_grid - self.v_bar)[:, None, None]  # (n_v, 1, 1)
        own = x[:, :, None]  # (n_v, n_x, 1)
        opp = x[:, None, :]  # (n_v, 1, n_x)
        payoff = (dv - self.lambda_N * (own + (self.n_agents - 1) * opp)) * own
        rbar = payoff.mean(axis=(1, 2))  # (n_v,)
        return rbar + self.rho * float(rbar.mean()) / (1.0 - self.rho)

    # ---- helpers ---------------------------------------------------------------

    @staticmethod
    def _flat(state: np.ndarray, n_v: int) -> int:
        return int(state[0]) * n_v * n_v + int(state[1]) * n_v + int(state[2])

    def _probs(self, h: list[float]) -> list[float]:
        m = max(h)
        e = [math.exp(x - m) for x in h]
        z = sum(e)
        return [x / z for x in e]

    def policy(self, state: np.ndarray) -> np.ndarray:
        """``π(· | s)`` as a numpy array of length ``n_x``."""
        return np.asarray(self._probs(self.H[self._flat(state, self.n_v)]))

    # ---- Agent contract ----------------------------------------------------------

    def act(self, state: np.ndarray) -> int:
        """Sample from the softmax policy (training) or take its mode (eval).

        Training consumes exactly one uniform draw per call and increments
        ``N[s]`` and ``N_v[v_t]``. Evaluation mode consumes no randomness and
        changes no counter.
        """
        s = self._flat(state, self.n_v)
        h = self.H[s]
        if self.eval_mode:
            return h.index(max(h))
        p = self._probs(h)
        u = float(self.rng.random())
        acc = 0.0
        action = self.n_x - 1
        for a, pa in enumerate(p):
            acc += pa
            if acc >= u:
                action = a
                break
        self.N[s] += 1
        self.N_v[int(state[2])] += 1
        return action

    def update(self, transition: Transition) -> None:
        """One-step actor-critic update (see the module docstring)."""
        if self.eval_mode:
            return
        n_v = self.n_v
        s = self._flat(transition.state, n_v)
        s2 = self._flat(transition.next_state, n_v)
        a = int(transition.action)
        k = int(transition.state[2])

        delta = float(transition.reward) + self.rho * self.V[s2] - self.V[s]
        self.V[s] += self.alpha_critic * delta
        d = delta / self.reward_scale
        eta = self.entropy_coef * math.exp(
            -self.entropy_decay * max(self.N_v[k] - 1, 0)
        )

        h = self.H[s]
        p = self._probs(h)
        ent = -sum(pb * math.log(pb) for pb in p if pb > 0.0)
        aa = self.alpha_actor
        for b in range(self.n_x):
            pb = p[b]
            grad_logp = (1.0 - pb) if b == a else -pb
            grad_ent = -pb * (math.log(pb) + ent) if pb > 0.0 else 0.0
            h[b] += aa * (d * grad_logp + eta * grad_ent)
        m = max(h)
        for b in range(self.n_x):
            h[b] = max(h[b] - m, _PREF_FLOOR)

    def reset(self) -> None:
        """No-op: preferences, critic and counters persist across episodes."""

    def visit_count(self, state: np.ndarray) -> int:
        """Visits of the full state triple."""
        return int(self.N[self._flat(state, self.n_v)])

    def greedy_policy(self) -> np.ndarray:
        """First argmax of ``H`` per state, shape ``(n_p, n_v, n_v)``."""
        out = np.fromiter(
            (h.index(max(h)) for h in self.H), dtype=np.int64, count=self.n_states
        )
        return out.reshape(self.n_p, self.n_v, self.n_v)

    def policy_concentration(self, onpath_mask: np.ndarray) -> float:
        """Mean over the masked states of ``max_a π(a | s)``; NaN if none."""
        mask = np.asarray(onpath_mask, dtype=bool).ravel()
        if mask.shape != (self.n_states,):
            raise ValueError(
                f"onpath_mask has shape {mask.shape}, expected ({self.n_states},)."
            )
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return math.nan
        return float(np.mean([max(self._probs(self.H[int(s)])) for s in idx]))

    # ---- diagnostics -----------------------------------------------------------

    def get_preferences(self) -> np.ndarray:
        """Copy of ``H`` with shape ``(n_p, n_v, n_v, n_x)``."""
        return np.asarray(self.H, dtype=np.float64).reshape(
            self.n_p, self.n_v, self.n_v, self.n_x
        )

    def get_values(self) -> np.ndarray:
        """Copy of ``V`` with shape ``(n_p, n_v, n_v)``."""
        return np.asarray(self.V, dtype=np.float64).reshape(
            self.n_p, self.n_v, self.n_v
        )

    def set_preferences(self, prefs: np.ndarray) -> None:
        """Load preferences from an array of shape ``(n_p, n_v, n_v, n_x)``."""
        arr = np.asarray(prefs, dtype=np.float64).reshape(self.n_states, self.n_x)
        self.H = arr.tolist()

    def set_values(self, values: np.ndarray) -> None:
        """Load the critic from an array of shape ``(n_p, n_v, n_v)``."""
        self.V = np.asarray(values, dtype=np.float64).ravel().tolist()
