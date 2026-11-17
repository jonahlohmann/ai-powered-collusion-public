"""Tabular Q-learning agent — DGJ's baseline informed speculator.

Implements the ε-greedy Q-learning algorithm from Dou, Goldstein & Ji (2025)
§3. Two choices are config switches whose defaults are those of the earlier
specification; DGJ's variants are ``q_init: "paper"`` and
``exploration_counter: "v"`` (the baseline specification).

State (DGJ §4):
    s_t = (p_{t-1}, v_{t-1}, v_t) — three discrete bin indices produced by
    :mod:`ai_trading.env.discretization`.

Action:
    Discrete index a ∈ {0, …, n_x − 1}. The order quantity it stands for is
    ``action_table[k, a]`` at value index k (see
    :class:`ai_trading.env.market.MarketEnv`, "Actions").

Q-table:
    ``np.ndarray`` of shape ``(n_p, n_v, n_v, n_x)``, dtype ``float64``.
    Axes correspond to ``(p_{t-1}-bin, v_{t-1}-bin, v_t-bin, action-bin)``.

Q-table initialisation — three modes (config key ``q_init``)
------------------------------------------------------------
``"paper"`` — **DGJ's initialisation** (DGJ §4.2). The discounted
payoff the speculator would earn forever if the other speculators chose
uniformly at random over the action grid and noise were zero::

    Q_0(s, x) = 1 / ((1 − ρ) n_x) · Σ_{x' ∈ X} [v − (v̄ + λ^N (x + (I − 1) x'))] · x

The average over a uniformly random opponent equals the opponent's mean
quantity, so with ``x = action_table[k, a]`` and ``m_k`` the mean of row k::

    Q_0 = ((v_k − v̄) − λ^N · (x + (I − 1) · m_k)) · x / (1 − ρ)

The value depends on the current value v_k and the action only, not on
``p_{t-1}`` or ``v_{t-1}``. It needs ``grids["action_table"]``. At the
thesis parameters with the per-v grid the values run from 1,852 (least
aggressive action) to 2,058 (most aggressive) at |v − v̄| = 1.28, from 310
to 345 at |v − v̄| = 0.52, and are 0 at v = v̄. Within a state the init
therefore tilts mildly towards aggressive actions. For comparison, the
constant of the optimistic mode is V^M = 959.

``"optimistic"`` (default; **not DGJ's**) — every cell is set to the
cartel discounted continuation value::

    Q_0(s, x) = π^M / (1 − ρ) = V^M  ∀ s, x

where ``π^M`` is the closed-form per-period cartel profit from
:mod:`ai_trading.simulation.benchmarks` on the simulation's v-grid. This is
optimistic initialisation (Sutton & Barto 2018, §2.6). It is the
initialisation of the earlier specification and departs from DGJ. The mode
stays the default so that the configs of that specification reproduce their
results.

``"zero"`` — every cell is 0.0. Not used by any run of the thesis.

Visit counters and exploration (config key ``exploration_counter``):
    ``N`` of shape ``(n_p, n_v, n_v)`` counts the visits of each full state
    triple. ``N_v`` of shape ``(n_v,)`` counts the visits of each current
    value v_t. Both are ``int64`` and both are incremented in every training
    ``act()`` call, whatever the mode. The exploration rate is::

        "state" (default):  ε = exp(−β · N[s])       s = full state triple
        "v" (DGJ eq. 4.3): ε = exp(−β · N_v[v_t])

    The counter is read before it is incremented, so the first visit has
    ``ε = exp(0) = 1`` (pure exploration). With ``"state"`` rarely visited
    states keep exploring for a very long time; with ``"v"`` all states with
    the same v_t share one exploration rate, as in DGJ. β must be
    chosen for the counter in use (see ``docs/parameters.md``).

Evaluation mode:
    After ``set_eval_mode(True)`` (see :class:`ai_trading.agents.base.Agent`)
    ``act()`` returns the greedy action without incrementing ``N`` or using
    the RNG, and ``update()`` is a no-op. Used for the eval window and IRFs.

Bellman update (DGJ eqs. 2.6–2.7):
    Q[s, a] ← (1 − α) · Q[s, a] + α · (r + ρ · max_a′ Q[s′, a′])

    α and β are held constant; no schedules (DGJ §4.2). reset() is a
    no-op — the Q-table and visit counts persist across episodes.

References:
    Paper §3.3 (Q-Learning Algorithm), §4.2 (Numerical Specifications:
    initial Q-matrix, exploration rate eq. 4.3), Table I (baseline
    parameters); Sutton & Barto (2018) §2.6 (optimistic initial values).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ai_trading.agents.base import Agent, Transition

logger = logging.getLogger(__name__)


class QLearningAgent(Agent):
    """ε-greedy tabular Q-learning speculator.

    Attributes:
        alpha: Learning / forgetting rate for the Bellman update.
        beta: Exploration-decay parameter (DGJ eq. 4.3).
        rho: Subjective discount factor in the Bellman equation.
        n_agents: Number of strategic informed speculators (DGJ's symbol I).
        v_bar: Mean of the fundamental value distribution.
        sigma_v: Standard deviation of the fundamental value.
        chi_N: Closed-form Nash aggressiveness coefficient (kept for
            diagnostics; not used in any init).
        lambda_N: Closed-form Nash price-impact coefficient. Used by the
            ``"paper"`` Q-init.
        pi_N: Closed-form per-period Nash profit (kept for diagnostics).
        pi_M: Closed-form per-period cartel profit. Drives the optimistic
            Q-init scalar ``V^M = π^M / (1 − ρ)``.
        q_init: ``"optimistic"`` (default), ``"paper"`` or ``"zero"``.
        exploration_counter: ``"state"`` (default) or ``"v"``.
        v_grid: 1D ``np.ndarray`` of fundamental-value grid points.
        action_grid: 1D ``np.ndarray`` of length ``n_x``: the quantity grid
            (global action grid) or the aggressiveness grid (per-v grid).
        action_table: ``np.ndarray`` of shape ``(n_v, n_x)`` with the order
            quantity of action a at value index k, or ``None`` if the
            caller did not pass one (allowed unless ``q_init == "paper"``).
        price_grid: 1D ``np.ndarray`` of price grid points.
        Q: Q-table of shape ``(n_p, n_v, n_v, n_x)``, dtype ``float64``.
        N: Visit counts per full state, shape ``(n_p, n_v, n_v)``, ``int64``.
        N_v: Visit counts per current value, shape ``(n_v,)``, ``int64``.
        rng: Local ``numpy.random.Generator`` for ε-greedy exploration.
    """

    def __init__(
        self,
        agent_id: int,
        config: dict[str, Any],
        grids: dict[str, np.ndarray],
    ) -> None:
        """Build the Q-table (init mode ``q_init``) and the visit counters.

        Args:
            agent_id: Integer identifier (0-indexed).
            config: Mapping with required keys ``alpha``, ``beta``, ``rho``,
                ``n_v``, ``n_x``, ``n_p``, ``n_agents``, ``chi_N``,
                ``lambda_N``, ``pi_N``, ``pi_M``. Optional keys: ``v_bar``
                (default 0.0), ``sigma_v`` (default 1.0), ``seed``
                (default 0; the local RNG is offset by ``agent_id`` so
                distinct agents explore independently with the same base
                seed), ``q_init`` (``"optimistic"`` | ``"paper"`` |
                ``"zero"``, default ``"optimistic"``),
                ``exploration_counter`` (``"state"`` | ``"v"``, default
                ``"state"``). ``pi_M`` drives the optimistic init scalar,
                ``lambda_N`` the ``"paper"`` init; ``chi_N`` and ``pi_N`` are
                retained for diagnostics. Optional ``rng_seed_sequence``
                (``np.random.SeedSequence``) sets the RNG stream; the runner
                always passes one, the ``seed + agent_id`` fallback is kept
                for unit tests.
            grids: Mapping with keys ``v_grid``, ``action_grid``,
                ``price_grid`` — the arrays built by
                :mod:`ai_trading.env.discretization` — and, optionally,
                ``action_table`` of shape ``(n_v, n_x)`` (required for
                ``q_init: "paper"``; the runner always passes it).

        Raises:
            ValueError: On a grid of the wrong shape, an unknown ``q_init``
                or ``exploration_counter``, or ``q_init: "paper"`` without
                ``grids["action_table"]``.
        """
        n_v = int(config["n_v"])
        n_x = int(config["n_x"])
        n_p = int(config["n_p"])
        super().__init__(agent_id=agent_id, n_actions=n_x)

        self.alpha: float = float(config["alpha"])
        self.beta: float = float(config["beta"])
        self.rho: float = float(config["rho"])
        self.n_agents: int = int(config["n_agents"])
        self.v_bar: float = float(config.get("v_bar", 0.0))
        self.sigma_v: float = float(config.get("sigma_v", 1.0))

        self.chi_N: float = float(config["chi_N"])
        self.lambda_N: float = float(config["lambda_N"])
        self.pi_N: float = float(config["pi_N"])
        self.pi_M: float = float(config["pi_M"])
        self.q_init: str = str(config.get("q_init", "optimistic"))
        self.exploration_counter: str = str(config.get("exploration_counter", "state"))
        if self.exploration_counter not in ("state", "v"):
            raise ValueError(
                "exploration_counter must be 'state' or 'v', "
                f"got {self.exploration_counter!r}."
            )
        self._explore_per_v: bool = self.exploration_counter == "v"

        self.v_grid: np.ndarray = np.asarray(grids["v_grid"], dtype=np.float64)
        self.action_grid: np.ndarray = np.asarray(
            grids["action_grid"], dtype=np.float64
        )
        self.price_grid: np.ndarray = np.asarray(grids["price_grid"], dtype=np.float64)

        if self.v_grid.shape != (n_v,):
            raise ValueError(
                f"v_grid has shape {self.v_grid.shape}, expected ({n_v},)."
            )
        if self.action_grid.shape != (n_x,):
            raise ValueError(
                f"action_grid has shape {self.action_grid.shape}, expected ({n_x},)."
            )
        if self.price_grid.shape != (n_p,):
            raise ValueError(
                f"price_grid has shape {self.price_grid.shape}, expected ({n_p},)."
            )
        table = grids.get("action_table")
        self.action_table: np.ndarray | None = (
            None if table is None else np.asarray(table, dtype=np.float64)
        )
        if self.action_table is not None and self.action_table.shape != (n_v, n_x):
            raise ValueError(
                f"action_table has shape {self.action_table.shape}, "
                f"expected ({n_v}, {n_x})."
            )

        seed_source = config.get("rng_seed_sequence")
        if seed_source is None:
            seed_source = int(config.get("seed", 0)) + agent_id
        self.rng: np.random.Generator = np.random.default_rng(seed_source)

        self.Q: np.ndarray = self._initialise_q_table(n_p, n_v, n_x)
        self.N: np.ndarray = np.zeros((n_p, n_v, n_v), dtype=np.int64)
        self.N_v: np.ndarray = np.zeros(n_v, dtype=np.int64)

        logger.debug(
            "QLearningAgent_%d initialised: Q.shape=%s, alpha=%.3g, "
            "beta=%.3g, rho=%.3g, lambda_N=%.6g",
            agent_id,
            self.Q.shape,
            self.alpha,
            self.beta,
            self.rho,
            self.lambda_N,
        )

    def _initialise_q_table(self, n_p: int, n_v: int, n_x: int) -> np.ndarray:
        """Q-init dispatcher: ``"optimistic"`` (default), ``"paper"`` or ``"zero"``.

        See the module docstring for the three modes. Optimistic mode fills
        every cell with ``V^M = π^M / (1 − ρ)``. Paper mode fills cell
        ``(p, v_prev, k, a)`` with the perpetuity of the one-period payoff of
        ``x = action_table[k, a]`` against opponents who each trade the mean
        quantity ``m_k`` of row k (DGJ §4.2). Zero mode fills with 0.0.
        """
        shape = (n_p, n_v, n_v, n_x)
        if self.q_init == "optimistic":
            v_m = self.pi_M / (1.0 - self.rho)
            return np.full(shape, v_m, dtype=np.float64)
        if self.q_init == "paper":
            if self.action_table is None:
                raise ValueError(
                    "q_init 'paper' needs grids['action_table'] (shape (n_v, n_x))."
                )
            x = self.action_table  # (n_v, n_x)
            dv = (self.v_grid - self.v_bar)[:, None]  # (n_v, 1)
            m = x.mean(axis=1, keepdims=True)  # opponent's mean quantity per v
            q0 = (dv - self.lambda_N * (x + (self.n_agents - 1) * m)) * x
            q0 = (q0 / (1.0 - self.rho)).astype(np.float64, copy=False)
            return np.broadcast_to(q0[None, None, :, :], shape).copy()
        if self.q_init == "zero":
            return np.zeros(shape, dtype=np.float64)
        raise ValueError(
            f"Unknown q_init mode {self.q_init!r}; "
            "expected 'optimistic', 'paper' or 'zero'."
        )

    def act(self, state: np.ndarray) -> int:
        """ε-greedy action selection.

        Reads ``ε = exp(−β · N[s])`` for the current state (mode
        ``"state"``) or ``ε = exp(−β · N_v[v_t])`` for the current value
        (mode ``"v"``, DGJ eq. 4.3), then with probability ``ε`` returns a
        uniform random action and otherwise returns ``argmax_a Q[s, a]``.
        Both counters are incremented after ``ε`` is read but before
        returning, so the first visit has ``ε = 1`` (pure exploration).

        Args:
            state: ``np.ndarray`` of length 3, ``[p_bin, v_prev_bin, v_bin]``.

        Returns:
            Integer action bin in ``[0, n_actions)``.
        """
        p_bin = int(state[0])
        vp_bin = int(state[1])
        v_bin = int(state[2])

        if self.eval_mode:
            return int(np.argmax(self.Q[p_bin, vp_bin, v_bin, :]))

        if self._explore_per_v:
            eps = float(np.exp(-self.beta * self.N_v[v_bin]))
        else:
            eps = float(np.exp(-self.beta * self.N[p_bin, vp_bin, v_bin]))
        if self.rng.random() < eps:
            action = int(self.rng.integers(0, self.n_actions))
        else:
            action = int(np.argmax(self.Q[p_bin, vp_bin, v_bin, :]))

        self.N[p_bin, vp_bin, v_bin] += 1
        self.N_v[v_bin] += 1
        return action

    def update(self, transition: Transition) -> None:
        """Apply a single-sample Bellman update to the Q-table.

        ``Q[s, a] ← (1 − α) · Q[s, a] + α · (r + ρ · max_a′ Q[s′, a′])``.

        Args:
            transition: ``(s, a, r, s', done)`` for the just-finished step.
                ``done`` is ignored — this model has no episode termination.
        """
        if self.eval_mode:
            return
        s = transition.state
        sp = transition.next_state
        p, vp, v = int(s[0]), int(s[1]), int(s[2])
        pn, vpn, vn = int(sp[0]), int(sp[1]), int(sp[2])
        a = int(transition.action)

        q_next_max = float(np.max(self.Q[pn, vpn, vn, :]))
        target = float(transition.reward) + self.rho * q_next_max
        self.Q[p, vp, v, a] = (1.0 - self.alpha) * self.Q[p, vp, v, a] + (
            self.alpha * target
        )

    def reset(self) -> None:
        """No-op.

        Tabular Q-learning accumulates experience across episodes; the
        Q-table and visit counts must persist between :meth:`reset` calls.
        The runner still calls :meth:`reset` for interface uniformity.
        """

    def visit_count(self, state: np.ndarray) -> int:
        """Visits of the full state triple, in both exploration-counter modes.

        In mode ``"state"`` this count drives ε = exp(−β·N[s]); in mode
        ``"v"`` it is a diagnostic only (the IRF report uses it to flag
        states that were never visited in training).
        """
        return int(self.N[int(state[0]), int(state[1]), int(state[2])])

    def greedy_policy(self) -> np.ndarray:
        """Greedy action index per state, shape ``(n_p, n_v, n_v)``."""
        return np.argmax(self.Q, axis=-1)

    def get_q_table(self) -> np.ndarray:
        """Return the Q-table (live view, not a copy) for diagnostics."""
        return self.Q
