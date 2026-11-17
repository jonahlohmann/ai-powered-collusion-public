"""Discretization utilities for continuous state/action spaces.

The Q-learning agents in Dou et al. (2025) operate on discretised grid:
- Fundamental value v  →  n_v bins  (non-uniform, Φ⁻¹-based)
- Action (order flow) x  →  n_x bins  (benchmark-scaled, symmetric): either
  one global quantity grid or one quantity grid per value state
  (``build_action_table``)
- Price p  →  n_p bins  (benchmark-scaled, symmetric around v̄)

Grid bounds are anchored to the closed-form Nash benchmark χ^N (and λ^N
for prices), pulled from ``simulation.benchmarks``.

Public API:
    build_v_grid(n_v, v_bar, sigma_v)                       -> np.ndarray
    build_action_grid(n_x, n_agents, xi, sigma_u, theta,
                      sigma_v, n_v, v_bar, iota)            -> np.ndarray
    build_chi_grid(n_x, n_agents, xi, sigma_u, theta,
                   sigma_v, iota)                           -> np.ndarray
    build_action_table(mode, n_x, v_grid, n_agents, xi,
                       sigma_u, theta, sigma_v, v_bar, iota) -> np.ndarray
    build_price_grid(n_p, n_agents, xi, sigma_u, theta,
                     sigma_v, v_bar, n_v, iota)             -> np.ndarray
    continuous_to_bin(value, grid)                          -> int
    bin_to_continuous(bin_index, grid)                      -> float

References:
    Paper §4.2 (Discretisation of State and Action Spaces).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ai_trading.simulation import benchmarks

_DEFAULT_IOTA: float = 0.1


def build_v_grid(
    n_v: int,
    v_bar: float = 0.0,
    sigma_v: float = 1.0,
) -> np.ndarray:
    """Quantile-spaced grid for the fundamental value v.

    Uses the equal-probability midpoint quantiles from DGJ §4.2:
    ``v_k = v̄ + σ_v · Φ⁻¹((2k − 1) / (2 n_v))`` for k = 1, …, n_v.
    With n_v = 10, the discretised standard deviation is ≈ 0.938.

    Args:
        n_v: Number of grid points (must be ≥ 1).
        v_bar: Mean of the fundamental value distribution.
        sigma_v: Standard deviation of the fundamental value.

    Returns:
        Array of shape ``(n_v,)`` sorted in ascending order.
    """
    if n_v < 1:
        raise ValueError(f"n_v must be ≥ 1, got {n_v}.")
    k = np.arange(1, n_v + 1, dtype=np.float64)
    quantiles = (2.0 * k - 1.0) / (2.0 * n_v)
    return (v_bar + sigma_v * norm.ppf(quantiles)).astype(np.float64, copy=False)


def build_action_grid(
    n_x: int,
    n_agents: int,
    xi: float,
    sigma_u: float,
    theta: float,
    sigma_v: float = 1.0,
    n_v: int = 10,
    v_bar: float = 0.0,
    iota: float = _DEFAULT_IOTA,
) -> np.ndarray:
    """Uniformly-spaced action grid anchored on Nash/cartel benchmarks.

    Per DGJ §4.2, the grid spans from the most-negative Nash action
    (with iota padding beyond) to the most-positive Nash action (with
    iota padding beyond), where the padding equals ``iota·(χ^N − χ^M)``
    scaled by the extreme |dv|:

        dv_max  = v_grid[-1] − v̄
        dv_min  = v_grid[0]  − v̄
        pad     = ι · (χ^N − χ^M)
        x_max   = χ^N · dv_max + pad · dv_max
        x_min   = χ^N · dv_min − pad · |dv_min|
        grid    = np.linspace(x_min, x_max, n_x)

    The grid contains 0 as the midpoint when ``n_x`` is odd and the
    v-grid is symmetric around v̄ (the common case); for even ``n_x``
    the two points nearest 0 straddle it (uniform spacing preserved).

    Args:
        n_x: Number of grid points (must be ≥ 2).
        n_agents: Number of strategic informed speculators (DGJ's symbol I).
        xi: Demand elasticity of information-insensitive investors.
        sigma_u: Standard deviation of noise trader order flow.
        theta: Market-maker inventory penalty.
        sigma_v: Standard deviation of fundamental value.
        n_v: Number of v-grid points (drives dv_max, dv_min via the
            extreme quantiles of the v-grid). Default 10 matches the
            DGJ's baseline.
        v_bar: Mean of the fundamental value; offsets dv computation.
        iota: Padding factor on the Nash–cartel aggressiveness gap.
            Default 0.1 per DGJ §4.2.

    Returns:
        Array of shape ``(n_x,)`` sorted in ascending order.
    """
    if n_x < 2:
        raise ValueError(f"n_x must be ≥ 2, got {n_x}.")
    chi_n = benchmarks.x_N(n_agents, xi, sigma_u, theta, sigma_v)
    chi_m = benchmarks.x_M(n_agents, xi, sigma_u, theta, sigma_v)
    v_grid = build_v_grid(n_v=n_v, v_bar=v_bar, sigma_v=sigma_v)
    dv_max = float(v_grid[-1] - v_bar)
    dv_min = float(v_grid[0] - v_bar)
    pad = iota * (chi_n - chi_m)
    x_grid_max = chi_n * dv_max + pad * dv_max
    x_grid_min = chi_n * dv_min - pad * abs(dv_min)
    return np.linspace(x_grid_min, x_grid_max, n_x, dtype=np.float64)


def build_chi_grid(
    n_x: int,
    n_agents: int,
    xi: float,
    sigma_u: float,
    theta: float,
    sigma_v: float = 1.0,
    iota: float = _DEFAULT_IOTA,
) -> np.ndarray:
    """Uniformly-spaced grid of aggressiveness levels χ (order flow per unit of v − v̄).

    Paper §4.2 discretises the interval
    ``[x^M − ι(x^N − x^M), x^N + ι(x^N − x^M)]``, where x^N = χ^N·(v − v̄) and
    x^M = χ^M·(v − v̄) depend on v. Dividing by (v − v̄) gives the same
    interval in aggressiveness units:

        grid = np.linspace(χ^M − ι(χ^N − χ^M), χ^N + ι(χ^N − χ^M), n_x)

    At the thesis parameters (I = 2, ξ = 500, σ_u = 1, θ = 0.1, n_x = 15) the
    grid runs from 120.83 to 170.83 in steps of 3.57.

    Args:
        n_x: Number of grid points (must be ≥ 2).
        n_agents: Number of strategic informed speculators (DGJ's symbol I).
        xi: Demand elasticity of information-insensitive investors.
        sigma_u: Standard deviation of noise trader order flow.
        theta: Market-maker inventory penalty.
        sigma_v: Standard deviation of fundamental value.
        iota: Padding factor on the Nash–cartel aggressiveness gap.

    Returns:
        Array of shape ``(n_x,)`` sorted in ascending order.
    """
    if n_x < 2:
        raise ValueError(f"n_x must be ≥ 2, got {n_x}.")
    chi_n = benchmarks.x_N(n_agents, xi, sigma_u, theta, sigma_v)
    chi_m = benchmarks.x_M(n_agents, xi, sigma_u, theta, sigma_v)
    pad = iota * (chi_n - chi_m)
    return np.linspace(chi_m - pad, chi_n + pad, n_x, dtype=np.float64)


def build_action_table(
    mode: str,
    n_x: int,
    v_grid: np.ndarray,
    n_agents: int,
    xi: float,
    sigma_u: float,
    theta: float,
    sigma_v: float = 1.0,
    v_bar: float = 0.0,
    iota: float = _DEFAULT_IOTA,
) -> np.ndarray:
    """Order quantity of action index a at value index k, shape ``(n_v, n_x)``.

    Two modes:

    - ``"global"``: every row is the one global quantity grid of
      :func:`build_action_grid` (the same quantities at every v).
    - ``"per_v"``: row k is ``build_chi_grid(...) · (v_grid[k] − v̄)``. Action
      index a then means aggressiveness level a at every v: for v < v̄ the
      quantities are negative and the highest index is still the most
      aggressive trade. At v = v̄ the row is all zeros (every action is the
      same trade).

    The largest quantity is the same in both modes: (χ^N + ι(χ^N − χ^M)) times
    the extreme |v − v̄| (218.93 at the thesis parameters).

    Args:
        mode: ``"global"`` or ``"per_v"``.
        n_x: Number of actions (must be ≥ 2).
        v_grid: Fundamental-value grid, shape ``(n_v,)`` (from
            :func:`build_v_grid` with the same ``v_bar`` and ``sigma_v``).
        n_agents: Number of strategic informed speculators (DGJ's symbol I).
        xi: Demand elasticity of information-insensitive investors.
        sigma_u: Standard deviation of noise trader order flow.
        theta: Market-maker inventory penalty.
        sigma_v: Standard deviation of fundamental value.
        v_bar: Mean of the fundamental value.
        iota: Padding factor on the Nash–cartel aggressiveness gap.

    Returns:
        Array of shape ``(n_v, n_x)``, dtype ``float64``.

    Raises:
        ValueError: If ``mode`` is neither ``"global"`` nor ``"per_v"``.
    """
    v_arr = np.asarray(v_grid, dtype=np.float64)
    if mode == "global":
        grid = build_action_grid(
            n_x=n_x,
            n_agents=n_agents,
            xi=xi,
            sigma_u=sigma_u,
            theta=theta,
            sigma_v=sigma_v,
            n_v=len(v_arr),
            v_bar=v_bar,
            iota=iota,
        )
        return np.tile(grid, (len(v_arr), 1))
    if mode == "per_v":
        chi_grid = build_chi_grid(
            n_x=n_x,
            n_agents=n_agents,
            xi=xi,
            sigma_u=sigma_u,
            theta=theta,
            sigma_v=sigma_v,
            iota=iota,
        )
        return np.outer(v_arr - v_bar, chi_grid)
    raise ValueError(f"action_grid must be 'global' or 'per_v', got {mode!r}.")


def build_price_grid(
    n_p: int,
    n_agents: int,
    xi: float,
    sigma_u: float,
    theta: float,
    sigma_v: float = 1.0,
    v_bar: float = 0.0,
    n_v: int = 10,
    iota: float = _DEFAULT_IOTA,
) -> np.ndarray:
    """Uniformly-spaced price grid per DGJ §4.2.

    DGJ set

        p_H = v̄ + λ^N · (I · max{x^M, x^N} + 1.96 · σ_u)
        p_L = v̄ + λ^N · (I · min{x^M, x^N} − 1.96 · σ_u)

    where x^N, x^M are the Nash / cartel order flows evaluated at the extreme
    v-grid points (x = χ · (v − v̄)), then discretises
    ``[p_L − ι(p_H − p_L), p_H + ι(p_H − p_L)]`` into ``n_p`` points.
    The grid is used only to index the lagged-price state; rewards use the
    raw clearing price (see ``MarketEnv``).

    Args:
        n_p: Number of grid points (must be ≥ 2).
        n_agents: Number of strategic informed speculators (DGJ's symbol I).
        xi: Demand elasticity of information-insensitive investors.
        sigma_u: Standard deviation of noise trader order flow.
        theta: Market-maker inventory penalty.
        sigma_v: Standard deviation of fundamental value.
        v_bar: Mean of the fundamental value; the grid is symmetric around it.
        n_v: Number of v-grid points (sets the extreme |v − v̄|).
        iota: Padding factor on the interval width (DGJ ι = 0.1).

    Returns:
        Array of shape ``(n_p,)`` sorted in ascending order.
    """
    if n_p < 2:
        raise ValueError(f"n_p must be ≥ 2, got {n_p}.")
    chi_n = benchmarks.x_N(n_agents, xi, sigma_u, theta, sigma_v)
    chi_m = benchmarks.x_M(n_agents, xi, sigma_u, theta, sigma_v)
    lam_n = benchmarks.lambda_N(n_agents, xi, sigma_u, theta, sigma_v)
    v_grid = build_v_grid(n_v=n_v, v_bar=v_bar, sigma_v=sigma_v)
    chi_max = max(chi_n, chi_m)
    dv_max = float(v_grid[-1] - v_bar)
    dv_min = float(v_grid[0] - v_bar)
    p_high = v_bar + lam_n * (n_agents * chi_max * dv_max + 1.96 * sigma_u)
    p_low = v_bar + lam_n * (n_agents * chi_max * dv_min - 1.96 * sigma_u)
    pad = iota * (p_high - p_low)
    return np.linspace(p_low - pad, p_high + pad, n_p, dtype=np.float64)


def continuous_to_bin(value: float, grid: np.ndarray) -> int:
    """Index of the nearest grid point to ``value``.

    Out-of-range values are mapped to the nearest endpoint (0 if below
    ``grid[0]``, ``len(grid) - 1`` if above ``grid[-1]``).

    Args:
        value: Continuous value to bin.
        grid: 1D array of grid points (assumed sorted, but ``argmin`` does
            not require sorting).

    Returns:
        Integer index in ``[0, len(grid))``.
    """
    return int(np.argmin(np.abs(np.asarray(grid, dtype=np.float64) - value)))


def bin_to_continuous(bin_index: int, grid: np.ndarray) -> float:
    """Continuous value corresponding to ``bin_index``.

    Args:
        bin_index: Index into ``grid``.
        grid: 1D array of grid points.

    Returns:
        ``grid[bin_index]`` as a Python float.

    Raises:
        IndexError: If ``bin_index`` is outside ``[0, len(grid))``.
    """
    if bin_index < 0 or bin_index >= len(grid):
        raise IndexError(f"bin_index {bin_index} out of range [0, {len(grid)}).")
    return float(grid[bin_index])
