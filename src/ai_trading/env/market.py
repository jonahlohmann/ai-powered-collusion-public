"""MarketEnv — the multi-agent trading environment.

Implements the market microstructure of Dou, Goldstein & Ji (2025) with a
rolling-regression adaptive market maker (DGJ §4.1, eq. 4.2).

Economic environment (DGJ §2, §3.1)
--------------------------------------
- Fundamental value v_t drawn i.i.d. from a discretised N(v̄, σ²_v) grid each
  period. Uniform sampling over the Φ⁻¹-based grid (DGJ §4.2) reproduces
  the discretised Gaussian by construction.
- ``n_agents`` strategic informed speculators submit discrete demand orders.
- Noise trader order flow u_t ~ N(0, σ²_u).
- Information-insensitive investors: z_t = −ξ·(p_t − v̄) (DGJ eq. 3.2).
- Total order flow y_t = Σ_i x_{i,t} + u_t.

State (DGJ §4):
    s_t = (p_{t-1}, v_{t-1}, v_t)
returned as a length-3 ``np.int64`` array of bin indices into the price and
value grids built in :mod:`ai_trading.env.discretization`.

Config key ``state_price`` selects the first component:

- ``"lagged"`` (default): the bin of the lagged price p_{t-1}, as above.
- ``"none"``: always 0, so the agents cannot condition on the lagged price
  (DGJ's check of the price-trigger mechanism, Dou et al. 2025, §5.3;
  runs R1.4 and R1.5). Nothing else changes: the price is still computed and binned,
  ``info["price_bin"]`` reports the true bin, and rewards, the market maker
  and every RNG draw (and their order) are identical in both modes.

Rewards and the price grid (DGJ §4.1, protocol steps 3–5)
-----------------------------------------------------------
The market clears at the continuous price p_t = v̄ + λ̂_t · y_t and each
speculator earns (v_t − p_t) · x_{i,t} on that **raw** price. The price grid
is used only to index the lagged-price component of the next state.
``reward_price: "snapped"`` pays the profits on the grid-rounded price
instead; only the run R1.3 uses it.

Actions (DGJ §4.2)
--------------------
An action is an index a ∈ {0, …, n_x − 1}. The order quantity it stands for
is ``action_table[k, a]``, where k is the index of the current value v_t.
Config key ``action_grid`` selects the table:

- ``"global"`` (default): one quantity grid for all v (every row of the table
  is ``action_grid``, the 1-D quantity grid).
- ``"per_v"``: row k is ``chi_grid · (v_k − v̄)``: n_x aggressiveness levels
  between the cartel and the Nash benchmark at every v (DGJ's interval
  ``[x^M − ι(x^N − x^M), x^N + ι(x^N − x^M)]`` read as a function of v). The
  highest index is the most aggressive trade at every v, also for v < v̄
  where quantities are negative. At v = v̄ all actions trade zero.
  ``action_grid`` then holds the aggressiveness (χ) grid, not quantities.

Market-maker pricing (DGJ §4.1, eq. 4.2; Online Appendix §2.1, eq. IA.2.9)
----------------------------------------------------------------------------
DGJ's adaptive market maker runs two rolling OLS regressions over the
last ``T_m`` periods:

    z_{t-τ} = ξ̂_0 − ξ̂_1 · p_{t-τ} + ε_z      (DGJ eq. 4.1, left)
    v_{t-τ} = γ̂_0 +   γ̂_1 · y_{t-τ} + ε_v     (DGJ eq. 4.1, right)

and prices via the plug-in rule

    λ̂_t = (θ · γ̂_1 + ξ̂_1) / (θ + ξ̂_1²)
    p_t = γ̂_0 + λ̂_t · y_t.

Simplification used here.
    Because z = −ξ(p − v̄) is deterministic, ξ̂_1 → ξ almost immediately, and
    because v is i.i.d. with mean v̄, γ̂_0 → v̄. We substitute the known ξ for
    ξ̂_1 and v̄ for γ̂_0 and maintain only the (v, y) regression. That
    regression is kept in an O(1) ring buffer with running sums (exactly
    refreshed once per window). During the first ``T_m`` periods (warm-up)
    the analytical λ^N is used. ``market_maker: "fixed_lambda"`` uses λ^N
    throughout. Note that the weight of the learned slope γ̂ in λ̂ is
    θ / (θ + ξ²) ≈ 4·10⁻⁷ at the thesis calibration (ξ = 500, θ = 0.1), so
    λ̂ ≈ 1/ξ in that regime.

Public API
----------
    MarketEnv(config: dict)
    MarketEnv.reset() -> dict[int, np.ndarray]
    MarketEnv.step(actions, u_override=None, u_shock=0.0) ->
        (states, rewards, done, info)
    MarketEnv.v_current -> float

References:
    Paper §2 (Model Setup, eqs. 2.2–2.5), §3.1 (Market Clearing),
    §4.1 (Adaptive Market Maker, eq. 4.2), §4.2 (Numerical Specifications).
    Online Appendix §2.1 (eq. IA.2.9 for λ).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ai_trading.env.discretization import (
    build_action_grid,
    build_action_table,
    build_chi_grid,
    build_price_grid,
    build_v_grid,
    continuous_to_bin,
)
from ai_trading.simulation import benchmarks

logger = logging.getLogger(__name__)


class MarketEnv:
    """Multi-agent trading environment with a rolling-regression market maker.

    Each :meth:`step` advances the market by one period: agents submit
    actions (bin indices into the action grid), the noise trader and the
    market maker do their work, and per-agent rewards are returned.

    Attributes:
        n_agents: Number of strategic informed speculators (DGJ's symbol I).
        xi: Demand elasticity of information-insensitive investors.
        sigma_u: Standard deviation of noise trader order flow.
        theta: Market-maker inventory penalty.
        sigma_v: Standard deviation of the fundamental value.
        v_bar: Mean of the fundamental value distribution.
        rho: Subjective discount factor (stored for downstream use).
        T_m: Rolling-regression window length (DGJ §4.1).
        reward_price: ``"raw"`` (DGJ) or ``"snapped"`` (rounded price; run R1.3).
        state_price: ``"lagged"`` (DGJ) or ``"none"`` (price component of
            the state fixed at 0; see the module docstring).
        market_maker: ``"rolling_ols"`` (DGJ) or ``"fixed_lambda"``.
        v_grid: 1D ``np.ndarray`` of fundamental-value grid points.
        action_mode: ``"global"`` (one quantity grid for all v) or
            ``"per_v"`` (one quantity grid per value state).
        action_table: ``np.ndarray`` of shape ``(n_v, n_x)``; the order
            quantity of action index a at value index k. This is what
            :meth:`step` trades.
        action_grid: 1D ``np.ndarray`` of length ``n_x``. In mode
            ``"global"`` the order-quantity grid (every row of
            ``action_table``). In mode ``"per_v"`` the aggressiveness (χ)
            grid, so that ``action_table[k] = action_grid · (v_k − v̄)``;
            it does NOT hold quantities in that mode.
        price_grid: 1D ``np.ndarray`` of price grid points (state index only).
        rng: Local ``numpy.random.Generator`` instance.
        lambda_N: Analytical Nash price-impact coefficient, used as the
            warm-up stand-in for ``λ̂_t``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Build grids, initialise the RNG, and clear the regression buffers.

        Args:
            config: Mapping of model parameters. Required keys: ``n_agents``,
                ``xi``, ``sigma_u``, ``theta``, ``rho``, ``n_v``, ``n_x``,
                ``n_p``, ``T_m``, ``seed``. Optional keys: ``sigma_v``
                (default 1.0), ``v_bar`` (default 0.0), ``iota`` (default
                0.1), ``reward_price`` (``"raw"`` | ``"snapped"``, default
                ``"raw"``), ``state_price`` (``"lagged"`` | ``"none"``,
                default ``"lagged"``), ``market_maker`` (``"rolling_ols"`` |
                ``"fixed_lambda"``, default ``"rolling_ols"``),
                ``action_grid`` (``"global"`` | ``"per_v"``, default
                ``"global"``; see the module docstring),
                ``rng_seed_sequence`` (a ``np.random.SeedSequence``; overrides
                ``seed`` for the RNG stream).
        """
        self.n_agents: int = int(config["n_agents"])
        self.xi: float = float(config["xi"])
        self.sigma_u: float = float(config["sigma_u"])
        self.theta: float = float(config["theta"])
        self.sigma_v: float = float(config.get("sigma_v", 1.0))
        self.v_bar: float = float(config.get("v_bar", 0.0))
        self.rho: float = float(config["rho"])
        self.T_m: int = int(config["T_m"])
        self._seed: int = int(config["seed"])
        self.reward_price: str = str(config.get("reward_price", "raw"))
        if self.reward_price not in ("raw", "snapped"):
            raise ValueError(
                f"reward_price must be 'raw' or 'snapped', got {self.reward_price!r}."
            )
        self.state_price: str = str(config.get("state_price", "lagged"))
        if self.state_price not in ("lagged", "none"):
            raise ValueError(
                f"state_price must be 'lagged' or 'none', got {self.state_price!r}."
            )
        self.market_maker: str = str(config.get("market_maker", "rolling_ols"))
        if self.market_maker not in ("rolling_ols", "fixed_lambda"):
            raise ValueError(
                "market_maker must be 'rolling_ols' or 'fixed_lambda', "
                f"got {self.market_maker!r}."
            )
        if self.T_m < 2:
            raise ValueError(f"T_m must be ≥ 2, got {self.T_m}.")
        self.action_mode: str = str(config.get("action_grid", "global"))
        if self.action_mode not in ("global", "per_v"):
            raise ValueError(
                f"action_grid must be 'global' or 'per_v', got {self.action_mode!r}."
            )

        n_v = int(config["n_v"])
        n_x = int(config["n_x"])
        n_p = int(config["n_p"])
        iota = float(config.get("iota", 0.1))

        self.v_grid: np.ndarray = build_v_grid(
            n_v=n_v, v_bar=self.v_bar, sigma_v=self.sigma_v
        )
        if self.action_mode == "global":
            self.action_grid: np.ndarray = build_action_grid(
                n_x=n_x,
                n_agents=self.n_agents,
                xi=self.xi,
                sigma_u=self.sigma_u,
                theta=self.theta,
                sigma_v=self.sigma_v,
                n_v=n_v,
                v_bar=self.v_bar,
                iota=iota,
            )
        else:
            # per_v: the 1-D grid holds aggressiveness levels χ, not quantities.
            self.action_grid = build_chi_grid(
                n_x=n_x,
                n_agents=self.n_agents,
                xi=self.xi,
                sigma_u=self.sigma_u,
                theta=self.theta,
                sigma_v=self.sigma_v,
                iota=iota,
            )
        self.action_table: np.ndarray = build_action_table(
            self.action_mode,
            n_x=n_x,
            v_grid=self.v_grid,
            n_agents=self.n_agents,
            xi=self.xi,
            sigma_u=self.sigma_u,
            theta=self.theta,
            sigma_v=self.sigma_v,
            v_bar=self.v_bar,
            iota=iota,
        )
        # Plain-float copy of the table for the per-period lookup in step().
        self._n_x: int = n_x
        self._x_rows: list[list[float]] = self.action_table.tolist()
        self.price_grid: np.ndarray = build_price_grid(
            n_p=n_p,
            n_agents=self.n_agents,
            xi=self.xi,
            sigma_u=self.sigma_u,
            theta=self.theta,
            sigma_v=self.sigma_v,
            v_bar=self.v_bar,
            n_v=n_v,
            iota=iota,
        )

        seed_source = config.get("rng_seed_sequence")
        self.rng: np.random.Generator = np.random.default_rng(
            seed_source if seed_source is not None else self._seed
        )

        self.lambda_N: float = benchmarks.lambda_N(
            n_agents=self.n_agents,
            xi=self.xi,
            sigma_u=self.sigma_u,
            theta=self.theta,
            sigma_v=self.sigma_v,
        )

        # Rolling-regression ring buffer (O(1) update; see _push_observation).
        self._y_buf: np.ndarray = np.zeros(self.T_m, dtype=np.float64)
        self._v_buf: np.ndarray = np.zeros(self.T_m, dtype=np.float64)
        self._buf_n: int = 0
        self._buf_pos: int = 0
        self._sum_y: float = 0.0
        self._sum_v: float = 0.0
        self._sum_yy: float = 0.0
        self._sum_yv: float = 0.0

        self._v_prev: float = self.v_bar
        self._v_curr: float = self.v_bar
        self._p_prev: float = self.v_bar
        self._v_curr_bin: int = continuous_to_bin(self._v_curr, self.v_grid)

        logger.debug(
            "MarketEnv initialised: n_agents=%d, xi=%.3g, sigma_u=%.3g, "
            "theta=%.3g, T_m=%d, lambda_N=%.6g, reward_price=%s, state_price=%s, "
            "market_maker=%s",
            self.n_agents,
            self.xi,
            self.sigma_u,
            self.theta,
            self.T_m,
            self.lambda_N,
            self.reward_price,
            self.state_price,
            self.market_maker,
        )

    @property
    def v_current(self) -> float:
        """Fundamental value v_t that the next :meth:`step` call trades on."""
        return self._v_curr

    def reset(self) -> dict[int, np.ndarray]:
        """Reset the environment to a fresh episode.

        Draws v_{-1} and v_0 uniformly from the value grid (uniform draws
        on the Φ⁻¹-based grid reproduce the discretised Gaussian by
        construction), sets p_{-1} = v̄, and clears the rolling-regression
        buffers.

        Returns:
            Mapping ``{agent_id: state}`` where ``state`` is a length-3
            ``np.int64`` array ``[p_{t-1}_bin, v_{t-1}_bin, v_t_bin]``
            (first component 0 if ``state_price == "none"``).
        """
        self._clear_buffers()

        v_prev_bin = int(self.rng.integers(0, len(self.v_grid)))
        v_curr_bin = int(self.rng.integers(0, len(self.v_grid)))
        self._v_prev = float(self.v_grid[v_prev_bin])
        self._v_curr = float(self.v_grid[v_curr_bin])
        self._v_curr_bin = v_curr_bin

        self._p_prev = float(self.v_bar)
        p_prev_bin = continuous_to_bin(self._p_prev, self.price_grid)

        p_state = p_prev_bin if self.state_price == "lagged" else 0
        state = np.array([p_state, v_prev_bin, v_curr_bin], dtype=np.int64)
        return {i: state.copy() for i in range(self.n_agents)}

    def step(
        self,
        actions: dict[int, int],
        u_override: float | None = None,
        u_shock: float = 0.0,
    ) -> tuple[dict[int, np.ndarray], dict[int, float], bool, dict[str, Any]]:
        """Advance the market by one period.

        The noise draw u ~ N(0, σ_u²) is ALWAYS taken from the RNG, so the RNG
        stream stays aligned between a shocked path and its no-shock
        counterfactual (common random numbers for the IRF).

        Agent i trades ``x_i = action_table[k, actions[i]]``, where k is the
        index of the current value v_t (see the module docstring, "Actions").

        Args:
            actions: Mapping ``{agent_id: action_bin}`` covering all agents.
            u_override: If not ``None``, replaces the noise order flow
                (``u_t = u_override``). Used by tests only.
            u_shock: Added to the drawn noise (``u_t = u_draw + u_shock``).
                Used by the runner's IRF. Ignored if ``u_override`` is set.

        Returns:
            Tuple ``(states, rewards, done, info)``:
                states: Per-agent next state ``[p_t_bin, v_t_bin, v_{t+1}_bin]``
                    (first component 0 if ``state_price == "none"``).
                rewards: Per-agent profit ``(v_t − p_t) · x_{i,t}`` where
                    ``p_t`` is the raw clearing price (or the grid-snapped
                    price if ``reward_price == "snapped"``).
                done: Always ``False``.
                info: ``price`` (price used for rewards), ``price_raw``,
                    ``price_snapped``, ``price_bin``, ``order_flow``,
                    ``noise`` (u_t actually applied), ``u_shock``, ``v_t``,
                    ``lambda_hat``, ``z_t``, ``x`` (list of the agents' order
                    quantities in agent order), ``v_bin`` (index of v_t).

        Raises:
            ValueError: If ``actions`` does not cover exactly the agent ids.
            IndexError: If an action index is outside ``[0, n_x)``.
        """
        if set(actions.keys()) != set(range(self.n_agents)):
            raise ValueError(
                f"actions must cover agent ids 0..{self.n_agents - 1}; "
                f"got keys {sorted(actions.keys())}."
            )

        v_bin = self._v_curr_bin
        x_row = self._x_rows[v_bin]
        x_vals: dict[int, float] = {}
        for i in range(self.n_agents):
            a = int(actions[i])
            if a < 0 or a >= self._n_x:
                raise IndexError(f"action index {a} out of range [0, {self._n_x}).")
            x_vals[i] = x_row[a]
        u_draw = float(self.rng.normal(loc=0.0, scale=self.sigma_u))
        if u_override is not None:
            u_t = float(u_override)
            applied_shock = 0.0
        else:
            applied_shock = float(u_shock)
            u_t = u_draw + applied_shock
        y_t = float(sum(x_vals.values()) + u_t)

        lambda_hat = self._estimate_lambda()
        p_raw = self.v_bar + lambda_hat * y_t
        p_bin = continuous_to_bin(p_raw, self.price_grid)
        p_snapped = float(self.price_grid[p_bin])
        p_t = p_raw if self.reward_price == "raw" else p_snapped

        v_t = self._v_curr
        rewards = {i: float((v_t - p_t) * x_vals[i]) for i in range(self.n_agents)}
        z_t = float(-self.xi * (p_t - self.v_bar))

        v_next_bin = int(self.rng.integers(0, len(self.v_grid)))
        v_next = float(self.v_grid[v_next_bin])

        self._push_observation(y_t, v_t)

        p_state = p_bin if self.state_price == "lagged" else 0
        state = np.array([p_state, v_bin, v_next_bin], dtype=np.int64)
        states = {i: state.copy() for i in range(self.n_agents)}

        self._p_prev = p_t
        self._v_prev = v_t
        self._v_curr = v_next
        self._v_curr_bin = v_next_bin

        info: dict[str, Any] = {
            "price": p_t,
            "price_raw": p_raw,
            "price_snapped": p_snapped,
            "price_bin": int(p_bin),
            "order_flow": y_t,
            "noise": u_t,
            "u_shock": applied_shock,
            "v_t": v_t,
            "lambda_hat": lambda_hat,
            "z_t": z_t,
            "x": [x_vals[i] for i in range(self.n_agents)],
            "v_bin": v_bin,
        }
        return states, rewards, False, info

    def _clear_buffers(self) -> None:
        """Empty the rolling-regression ring buffer and its running sums."""
        self._y_buf.fill(0.0)
        self._v_buf.fill(0.0)
        self._buf_n = 0
        self._buf_pos = 0
        self._sum_y = 0.0
        self._sum_v = 0.0
        self._sum_yy = 0.0
        self._sum_yv = 0.0

    def _push_observation(self, y: float, v: float) -> None:
        """Append (y_t, v_t) to the T_m-window ring buffer in O(1).

        Running sums are updated incrementally; once per full window (when the
        write position wraps to 0) they are recomputed exactly from the buffer
        to prevent floating-point drift over millions of periods.
        """
        pos = self._buf_pos
        if self._buf_n == self.T_m:
            old_y = float(self._y_buf[pos])
            old_v = float(self._v_buf[pos])
            self._sum_y -= old_y
            self._sum_v -= old_v
            self._sum_yy -= old_y * old_y
            self._sum_yv -= old_y * old_v
        else:
            self._buf_n += 1
        self._y_buf[pos] = y
        self._v_buf[pos] = v
        self._sum_y += y
        self._sum_v += v
        self._sum_yy += y * y
        self._sum_yv += y * v
        pos += 1
        if pos == self.T_m:
            pos = 0
            self._sum_y = float(self._y_buf.sum())
            self._sum_v = float(self._v_buf.sum())
            self._sum_yy = float(np.dot(self._y_buf, self._y_buf))
            self._sum_yv = float(np.dot(self._y_buf, self._v_buf))
        self._buf_pos = pos

    def _estimate_lambda(self) -> float:
        """λ̂_t from the rolling (v, y) window via the plug-in rule (eq. 4.2).

        Returns the analytical λ^N during warm-up (fewer than ``T_m``
        observations), when ``market_maker == "fixed_lambda"``, or when the
        window has zero variance in y. Note: at ξ = 500, θ = 0.1 the weight of
        the learned slope γ̂ in λ̂ is θ / (θ + ξ²) ≈ 4·10⁻⁷, so λ̂ ≈ 1/ξ.
        """
        if self.market_maker == "fixed_lambda" or self._buf_n < self.T_m:
            return self.lambda_N
        n = float(self._buf_n)
        s_yy = self._sum_yy - self._sum_y * self._sum_y / n
        if s_yy <= 0.0:
            return self.lambda_N
        gamma_hat = (self._sum_yv - self._sum_y * self._sum_v / n) / s_yy
        return (self.theta * gamma_hat + self.xi) / (self.theta + self.xi * self.xi)
