"""Simulation outcome metrics.

Pure functions for the four post-convergence summaries used in DGJ:

- :func:`collusion_capacity` — Δ^C, DGJ Definition 3.1 (eq. 3.6).
- :func:`price_informativeness` — I^C, Online Appendix eq. IA.4.5.
- :func:`market_liquidity` — L^C, Online Appendix eq. IA.4.6.
- :func:`mispricing` — E^C_t, Online Appendix eq. IA.4.7.

Each function takes pre-computed summary statistics — average per-period
realised profit, the fitted aggressiveness χ̂^C, the market maker's λ̂, the
empirical σ̂_v, etc. — and returns a single ``float``. The simulation
runner extracts these statistics from logged trajectories and calls the
functions; metrics here remain decoupled from any logging schema.

Sign and magnitude conventions follow DGJ exactly:
- Δ^C = 0 in the Nash equilibrium and Δ^C = 1 in the perfect cartel; values
  outside [0, 1] are kept (not clamped) so unusual outcomes are visible.
- I^C, L^C, E^C are non-negative.

References:
    Paper §3.2 (Definitions 3.1 and 3.4); Online Appendix §4.1
    (eqs. IA.4.5–IA.4.7).
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def collusion_capacity(pi_C: float, pi_N: float, pi_M: float) -> float:
    """Collusion capacity Δ^C (DGJ Definition 3.1, eq. 3.6).

    Δ^C = (π^C − π^N) / (π^M − π^N).

    Δ^C = 0 in the Nash equilibrium and Δ^C = 1 in the perfect cartel.
    Values outside [0, 1] are returned unchanged (not clamped) so unusual
    outcomes — e.g., super-cartel profits or below-Nash performance — are
    visible to the caller.

    Args:
        pi_C: Average per-period per-agent profit realised in the
            simulation (post-convergence).
        pi_N: Analytical per-agent Nash profit (e.g.,
            :func:`ai_trading.simulation.benchmarks.pi_N`).
        pi_M: Analytical per-agent cartel profit (e.g.,
            :func:`ai_trading.simulation.benchmarks.pi_M`).

    Returns:
        Δ^C as a ``float``. If ``pi_M == pi_N`` (degenerate denominator),
        returns ``0.0`` and logs a warning.
    """
    denominator = pi_M - pi_N
    if denominator == 0.0:
        logger.warning("collusion_capacity: pi_M == pi_N (=%s); returning 0.0.", pi_N)
        return 0.0
    return float((pi_C - pi_N) / denominator)


def price_informativeness(
    chi_C: float, sigma_v_hat: float, sigma_u: float, n_agents: int
) -> float:
    """Price informativeness I^C (Online Appendix eq. IA.4.5).

    I^C = (I · χ̂^C)² · (σ̂_v / σ_u)².

    Args:
        chi_C: Estimated aggressiveness χ̂^C of the converged strategy
            (slope of agents' average order flow on ``v_t − v̄``).
        sigma_v_hat: Empirical standard deviation of v under the discrete
            grid (≈ 0.938 for n_v = 10 with σ_v = 1.0).
        sigma_u: Noise trader order-flow standard deviation.
        n_agents: Number of informed speculators (DGJ's symbol I).

    Returns:
        Non-negative ``float``.
    """
    ratio = sigma_v_hat / sigma_u
    return float((n_agents * chi_C) ** 2 * ratio * ratio)


def market_liquidity(lambda_hat: float, xi: float) -> float:
    """Market liquidity L^C (Online Appendix eq. IA.4.6).

    L^C = 1 / |1 − ξ · λ̂|.

    Args:
        lambda_hat: Market maker's price-impact estimate λ̂ (e.g., an
            entry of ``info["lambda_hat"]`` from
            :meth:`ai_trading.env.market.MarketEnv.step`).
        xi: Demand elasticity of information-insensitive investors.

    Returns:
        Positive ``float``. If ``xi * lambda_hat == 1`` exactly (zero
        denominator), returns ``inf`` and logs a warning.
    """
    denominator = 1.0 - xi * lambda_hat
    if denominator == 0.0:
        logger.warning(
            "market_liquidity: xi * lambda_hat == 1 (xi=%s, lambda_hat=%s); "
            "returning inf.",
            xi,
            lambda_hat,
        )
        return math.inf
    return float(1.0 / abs(denominator))


def mispricing(
    lambda_hat: float,
    chi_C: float,
    n_agents: int,
    v_t: float,
    v_bar: float,
) -> float:
    """Per-period mispricing E^C_t (Online Appendix eq. IA.4.7).

    E^C_t = (1 − λ̂ · I · χ̂^C) · |v_t − v̄|.

    Caller averages over the post-convergence window. The factor
    ``1 − λ̂ · I · χ̂^C`` measures the share of the value innovation that
    survives in the price residual; multiplying by ``|v_t − v̄|`` converts
    that share into the period's price-vs-value gap.

    Args:
        lambda_hat: Market maker's λ̂ for that period.
        chi_C: Estimated aggressiveness χ̂^C of the converged strategy.
        n_agents: Number of informed speculators (DGJ's symbol I).
        v_t: Fundamental value that period (continuous, not a bin index).
        v_bar: Long-run mean of the fundamental value.

    Returns:
        Non-negative ``float`` — the absolute factor times an absolute
        deviation.
    """
    factor = abs(1.0 - lambda_hat * n_agents * chi_C)
    return float(factor * abs(v_t - v_bar))
