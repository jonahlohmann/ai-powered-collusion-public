"""Closed-form analytical benchmarks for Nash and cartel equilibria.

Derived from DGJ's Online Appendix 2.1 (Dou, Goldstein & Ji, 2025).
These values are needed by three other modules:
    - env/discretization.py  — action and price grid bounds depend on λ^N,
                               x^N, x^M (DGJ §4.2).
    - agents/q_learning.py   — initial Q-matrix uses λ^N (DGJ §4.2).
    - simulation/metrics.py  — Δ^C = (π^C − π^N) / (π^M − π^N) uses π^N
                               and π^M (DGJ eq. 3.6).

All functions are pure (no side-effects, no state). Pass model parameters
directly; do not read from a YAML inside these functions.

Both equilibria are characterised by Kyle-style fixed points in which
informed speculators take the market-maker's price-impact coefficient λ
as parametric (DGJ §3.2, Online Appendix Propositions IA.1
and IA.2):

    Nash    : χ^N · (I + 1) · λ(χ^N) = 1
    Cartel  : χ^M · (2 I)   · λ(χ^M) = 1

with the shared market-maker pricing rule

    λ(χ) = (θ γ(χ) + ξ) / (θ + ξ²),
    γ(χ) = I χ / ((I χ)² + (σ_u / σ_v)²).

Per-period profits follow from Π(χ, λ) = (1 − λ I χ) χ σ_v².

:func:`realised_benchmark_profit` evaluates the same benchmarks period by
period on a realised (v_t, u_t) path (Online Appendix eqs IA.4.2–IA.4.3); the
runner uses it for the per-path Δ^C of eq. IA.4.1.

References:
    Paper §3.2; Online Appendix 2.1, Propositions IA.1 and IA.2.
"""

from __future__ import annotations

import logging
from functools import cache

import numpy as np
from scipy.optimize import brentq

logger = logging.getLogger(__name__)


def _profit_variance(sigma_v: float, v_grid: np.ndarray | None) -> float:
    """Variance scaling used in π = (1 − I·λ·χ)·χ·Var.

    When ``v_grid`` is None, returns ``sigma_v**2`` (continuous-Gaussian
    benchmark, matching DGJ §3.2). When ``v_grid`` is provided, returns
    ``np.var(v_grid)`` — the empirical variance of the discretised
    v-grid that the simulation actually samples from. Use the latter to
    make Δ^C compare against the best-achievable profit on the grid
    rather than against the continuous closed form.
    """
    if v_grid is None:
        return float(sigma_v) ** 2
    return float(np.var(np.asarray(v_grid, dtype=np.float64)))


_CHI_BRACKET: tuple[float, float] = (1e-12, 1e6)


def _lambda_of_chi(
    chi: float,
    n_agents: int,
    xi: float,
    sigma_u: float,
    theta: float,
    sigma_v: float,
) -> float:
    """Market-maker price impact at speculator aggressiveness ``chi``."""
    s2 = (sigma_u / sigma_v) ** 2
    ic = n_agents * chi
    gamma = ic / (ic * ic + s2)
    return (theta * gamma + xi) / (theta + xi * xi)


@cache
def _solve_chi(
    multiplier: int,
    n_agents: int,
    xi: float,
    sigma_u: float,
    theta: float,
    sigma_v: float,
) -> float:
    """Solve ``multiplier · χ · λ(χ) = 1`` via Brent's method.

    ``multiplier`` is ``I + 1`` for Nash and ``2 I`` for the cartel.
    """

    def residual(chi: float) -> float:
        return (
            multiplier
            * chi
            * _lambda_of_chi(chi, n_agents, xi, sigma_u, theta, sigma_v)
            - 1.0
        )

    a, b = _CHI_BRACKET
    fa, fb = residual(a), residual(b)
    if fa * fb > 0:
        raise ValueError(
            f"χ residual has no sign change on [{a}, {b}]: "
            f"f(a)={fa}, f(b)={fb}; check parameters "
            f"(n_agents={n_agents}, xi={xi}, sigma_u={sigma_u}, "
            f"theta={theta}, sigma_v={sigma_v})."
        )
    return float(brentq(residual, a, b, xtol=1e-14, rtol=1e-12))


def lambda_N(
    n_agents: int,
    xi: float,
    sigma_u: float,
    theta: float,
    sigma_v: float = 1.0,
) -> float:
    """Price-impact coefficient in the non-collusive Nash equilibrium.

    Args:
        n_agents: Number of strategic informed speculators (DGJ's symbol I).
        xi: Demand elasticity of information-insensitive investors.
        sigma_u: Standard deviation of noise trader order flow.
        theta: Market-maker inventory penalty.
        sigma_v: Standard deviation of fundamental value (default 1.0).

    Returns:
        λ^N — the equilibrium price-impact coefficient such that
        p^N(y) = v̄ + λ^N · y.

    References:
        Online Appendix Proposition IA.1.
    """
    chi = _solve_chi(n_agents + 1, n_agents, xi, sigma_u, theta, sigma_v)
    return _lambda_of_chi(chi, n_agents, xi, sigma_u, theta, sigma_v)


def lambda_M(
    n_agents: int,
    xi: float,
    sigma_u: float,
    theta: float,
    sigma_v: float = 1.0,
) -> float:
    """Price-impact coefficient in the perfect-cartel benchmark.

    Args:
        n_agents: Number of strategic informed speculators (DGJ's symbol I).
        xi: Demand elasticity of information-insensitive investors.
        sigma_u: Standard deviation of noise trader order flow.
        theta: Market-maker inventory penalty.
        sigma_v: Standard deviation of fundamental value (default 1.0).

    Returns:
        λ^M — the cartel price-impact coefficient such that
        p^M(y) = v̄ + λ^M · y.

    References:
        Online Appendix Proposition IA.2.
    """
    chi = _solve_chi(2 * n_agents, n_agents, xi, sigma_u, theta, sigma_v)
    return _lambda_of_chi(chi, n_agents, xi, sigma_u, theta, sigma_v)


def x_N(
    n_agents: int,
    xi: float,
    sigma_u: float,
    theta: float,
    sigma_v: float = 1.0,
) -> float:
    """Nash-equilibrium aggressiveness coefficient χ^N.

    The order flow itself is x^N(v) = χ^N · (v − v̄); this function returns
    the aggressiveness coefficient χ^N, not the order flow.

    Args:
        n_agents: Number of strategic informed speculators (DGJ's symbol I).
        xi: Demand elasticity of information-insensitive investors.
        sigma_u: Standard deviation of noise trader order flow.
        theta: Market-maker inventory penalty.
        sigma_v: Standard deviation of fundamental value (default 1.0).

    Returns:
        χ^N — the Nash aggressiveness coefficient.

    References:
        Paper §3.2 (Benchmark I); Online Appendix Proposition IA.1.
    """
    return _solve_chi(n_agents + 1, n_agents, xi, sigma_u, theta, sigma_v)


def x_M(
    n_agents: int,
    xi: float,
    sigma_u: float,
    theta: float,
    sigma_v: float = 1.0,
) -> float:
    """Cartel aggressiveness coefficient χ^M.

    The order flow itself is x^M(v) = χ^M · (v − v̄); this function returns
    the aggressiveness coefficient χ^M (χ^M ≤ χ^N, with equality only when
    I = 1, since a single speculator IS the cartel).

    Args:
        n_agents: Number of strategic informed speculators (DGJ's symbol I).
        xi: Demand elasticity of information-insensitive investors.
        sigma_u: Standard deviation of noise trader order flow.
        theta: Market-maker inventory penalty.
        sigma_v: Standard deviation of fundamental value (default 1.0).

    Returns:
        χ^M — the cartel aggressiveness coefficient.

    References:
        Paper §3.2 (Benchmark II); Online Appendix Proposition IA.2.
    """
    return _solve_chi(2 * n_agents, n_agents, xi, sigma_u, theta, sigma_v)


def pi_N(
    n_agents: int,
    xi: float,
    sigma_u: float,
    theta: float,
    sigma_v: float = 1.0,
    v_grid: np.ndarray | None = None,
) -> float:
    """Expected per-period profit per speculator in the Nash equilibrium.

    Used as the lower normalisation anchor in
    Δ^C = (π^C − π^N) / (π^M − π^N).

    Args:
        n_agents: Number of strategic informed speculators (DGJ's symbol I).
        xi: Demand elasticity of information-insensitive investors.
        sigma_u: Standard deviation of noise trader order flow.
        theta: Market-maker inventory penalty.
        sigma_v: Standard deviation of fundamental value (default 1.0).
        v_grid: Optional discretised v-grid the simulation samples from.
            When provided, the variance factor in π = (1 − I·λ·χ)·χ·Var
            uses ``np.var(v_grid)`` instead of ``σ_v²``, so Δ^C compares
            against the best-achievable profit on this grid rather than
            against the continuous closed form. Both ``pi_N`` and
            ``pi_M`` must use the same v-grid for the comparison to be
            self-consistent.

    Returns:
        π^N — the per-speculator expected profit.

    References:
        Paper eq. 3.6; Online Appendix Proposition IA.1.
    """
    chi = _solve_chi(n_agents + 1, n_agents, xi, sigma_u, theta, sigma_v)
    lam = _lambda_of_chi(chi, n_agents, xi, sigma_u, theta, sigma_v)
    variance = _profit_variance(sigma_v, v_grid)
    return float((1.0 - lam * n_agents * chi) * chi * variance)


def pi_M(
    n_agents: int,
    xi: float,
    sigma_u: float,
    theta: float,
    sigma_v: float = 1.0,
    v_grid: np.ndarray | None = None,
) -> float:
    """Expected per-period profit per speculator in the perfect-cartel benchmark.

    Used as the upper normalisation anchor in
    Δ^C = (π^C − π^N) / (π^M − π^N).

    Args:
        n_agents: Number of strategic informed speculators (DGJ's symbol I).
        xi: Demand elasticity of information-insensitive investors.
        sigma_u: Standard deviation of noise trader order flow.
        theta: Market-maker inventory penalty.
        sigma_v: Standard deviation of fundamental value (default 1.0).
        v_grid: Optional discretised v-grid (see :func:`pi_N`). When
            supplied, the variance factor uses ``np.var(v_grid)`` instead
            of ``σ_v²``. Use the same v-grid for ``pi_N`` and ``pi_M``.

    Returns:
        π^M — the per-speculator expected profit.

    References:
        Paper eq. 3.6; Online Appendix Proposition IA.2.
    """
    chi = _solve_chi(2 * n_agents, n_agents, xi, sigma_u, theta, sigma_v)
    lam = _lambda_of_chi(chi, n_agents, xi, sigma_u, theta, sigma_v)
    variance = _profit_variance(sigma_v, v_grid)
    return float((1.0 - lam * n_agents * chi) * chi * variance)


def realised_benchmark_profit(
    v: np.ndarray,
    u: np.ndarray,
    chi: float,
    lam: float,
    n_agents: int,
    v_bar: float = 0.0,
) -> np.ndarray:
    """Per-period benchmark profit of one speculator on a realised (v, u) path.

    All I speculators play x = χ·(v − v̄) and the price is
    p = v̄ + λ·(I·x + u) (Online Appendix eqs IA.4.2–IA.4.3). Pass
    (χ^N, λ^N) for the Nash benchmark and (χ^M, λ^M) for the cartel.

    Args:
        v: Realised fundamental values, shape (T,).
        u: Realised noise order flow, shape (T,).
        chi: Aggressiveness coefficient of the benchmark.
        lam: Price-impact coefficient of the benchmark.
        n_agents: Number of speculators (DGJ's symbol I).
        v_bar: Mean of the fundamental value.

    Returns:
        Array of shape (T,) with each period's per-speculator profit.
    """
    v_arr = np.asarray(v, dtype=np.float64)
    u_arr = np.asarray(u, dtype=np.float64)
    x = chi * (v_arr - v_bar)
    price = v_bar + lam * (n_agents * x + u_arr)
    return (v_arr - price) * x


def grid_stage_game(
    action_table: np.ndarray,
    v_grid: np.ndarray,
    lambda_n: float,
    pi_n: float,
    pi_m: float,
    v_bar: float = 0.0,
) -> dict[str, object]:
    """Two-player one-shot game on the discrete action grid, per value state.

    For each value index k the payoff of a speculator playing action i
    against a rival playing j, at the Nash price impact λ^N, is

        P[i, j] = (v_k − v̄ − λ^N · (x_i + x_j)) · x_i,   x = action_table[k].

    A pair (i, j) is a pure Nash equilibrium when neither player gains more
    than 1e-12 by a unilateral change. The returned Δ^C values normalise the
    mean over k of the per-k payoff by (π^M − π^N), as in DGJ eq. 3.6.
    Every value state is weighted equally, matching the uniform v-grid
    sampling of the simulation.

    Args:
        action_table: Order quantity of action a at value index k, shape
            ``(n_v, n_x)``. For the global grid, tile the 1-D action grid.
        v_grid: Fundamental-value grid, shape ``(n_v,)``.
        lambda_n: Nash price-impact coefficient λ^N.
        pi_n: Per-period Nash profit (grid-corrected).
        pi_m: Per-period cartel profit (grid-corrected).
        v_bar: Mean of the fundamental value.

    Returns:
        Mapping with keys ``nash_bins`` (list over k of the list of
        equilibrium pairs ``(i, j)``), ``delta_c_grid_nash`` (Δ^C of the
        mean equilibrium payoff, averaged over each k's equilibria),
        ``best_symmetric_bins`` (list over k of ``argmax_i P[i, i]``) and
        ``delta_c_grid_best_symmetric``.
    """
    table = np.asarray(action_table, dtype=np.float64)
    v_arr = np.asarray(v_grid, dtype=np.float64)
    if table.ndim != 2 or table.shape[0] != v_arr.shape[0]:
        raise ValueError(
            f"action_table has shape {table.shape}; expected (n_v, n_x) with "
            f"n_v = {v_arr.shape[0]}."
        )
    tol = 1e-12
    nash_bins: list[list[tuple[int, int]]] = []
    nash_payoffs: list[float] = []
    best_bins: list[int] = []
    best_payoffs: list[float] = []
    for k in range(table.shape[0]):
        x = table[k]
        payoff = (v_arr[k] - v_bar - lambda_n * (x[:, None] + x[None, :])) * x[:, None]
        best_reply = payoff.max(axis=0)  # best_reply[j] = max_i' P[i', j]
        is_br = payoff >= best_reply[None, :] - tol  # is_br[i, j]: i best vs j
        pairs = [
            (int(i), int(j))
            for i in range(len(x))
            for j in range(len(x))
            if is_br[i, j] and is_br[j, i]
        ]
        nash_bins.append(pairs)
        if pairs:
            nash_payoffs.append(
                float(np.mean([(payoff[i, j] + payoff[j, i]) / 2 for i, j in pairs]))
            )
        else:
            nash_payoffs.append(float("nan"))
        diag = np.diag(payoff)
        b = int(np.argmax(diag))
        best_bins.append(b)
        best_payoffs.append(float(diag[b]))
    denom = pi_m - pi_n
    return {
        "nash_bins": nash_bins,
        "delta_c_grid_nash": (float(np.mean(nash_payoffs)) - pi_n) / denom,
        "best_symmetric_bins": best_bins,
        "delta_c_grid_best_symmetric": (float(np.mean(best_payoffs)) - pi_n) / denom,
    }


def compute_all_benchmarks(
    n_agents: int,
    xi: float,
    sigma_u: float,
    theta: float,
    sigma_v: float = 1.0,
    v_grid: np.ndarray | None = None,
) -> dict[str, float]:
    """Return both equilibria's aggressiveness, price impact, and profit.

    Args:
        n_agents: Number of strategic informed speculators (DGJ's symbol I).
        xi: Demand elasticity of information-insensitive investors.
        sigma_u: Standard deviation of noise trader order flow.
        theta: Market-maker inventory penalty.
        sigma_v: Standard deviation of fundamental value (default 1.0).
        v_grid: Optional discretised v-grid (see :func:`pi_N`). When
            supplied, ``pi_N`` and ``pi_M`` use ``np.var(v_grid)`` in
            place of ``σ_v²`` so that Δ^C measures performance against
            the best-achievable profit on this grid. ``chi_*`` and
            ``lambda_*`` are unaffected — those come from the
            continuous-distribution Kyle fixed point that the
            market-maker's rolling regression converges to anyway.

    Returns:
        Mapping with keys ``chi_N``, ``lambda_N``, ``pi_N``, ``chi_M``,
        ``lambda_M``, ``pi_M``.
    """
    return {
        "chi_N": x_N(n_agents, xi, sigma_u, theta, sigma_v),
        "lambda_N": lambda_N(n_agents, xi, sigma_u, theta, sigma_v),
        "pi_N": pi_N(n_agents, xi, sigma_u, theta, sigma_v, v_grid=v_grid),
        "chi_M": x_M(n_agents, xi, sigma_u, theta, sigma_v),
        "lambda_M": lambda_M(n_agents, xi, sigma_u, theta, sigma_v),
        "pi_M": pi_M(n_agents, xi, sigma_u, theta, sigma_v, v_grid=v_grid),
    }
