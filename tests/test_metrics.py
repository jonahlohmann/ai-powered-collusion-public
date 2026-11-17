"""Tests for simulation outcome metrics (simulation/metrics.py)."""

from __future__ import annotations

import logging
import math

import pytest

from ai_trading.simulation import benchmarks, metrics


@pytest.fixture
def low_noise() -> dict[str, float]:
    """DGJ's baseline (I=2, ξ=500, σ_u=0.1, θ=0.1, σ_v=1.0)."""
    return {
        "n_agents": 2,
        "xi": 500.0,
        "sigma_u": 0.1,
        "theta": 0.1,
        "sigma_v": 1.0,
    }


@pytest.fixture
def high_noise() -> dict[str, float]:
    return {
        "n_agents": 2,
        "xi": 500.0,
        "sigma_u": 100.0,
        "theta": 0.1,
        "sigma_v": 1.0,
    }


def test_collusion_capacity_at_nash_is_zero(low_noise: dict[str, float]) -> None:
    pi_n = benchmarks.pi_N(**low_noise)
    pi_m = benchmarks.pi_M(**low_noise)
    assert metrics.collusion_capacity(pi_n, pi_n, pi_m) == 0.0


def test_collusion_capacity_at_cartel_is_one(low_noise: dict[str, float]) -> None:
    pi_n = benchmarks.pi_N(**low_noise)
    pi_m = benchmarks.pi_M(**low_noise)
    assert metrics.collusion_capacity(pi_m, pi_n, pi_m) == pytest.approx(1.0, rel=1e-12)


def test_collusion_capacity_above_cartel_not_clamped(
    low_noise: dict[str, float],
) -> None:
    pi_n = benchmarks.pi_N(**low_noise)
    pi_m = benchmarks.pi_M(**low_noise)
    pi_c = pi_m + (pi_m - pi_n)
    assert metrics.collusion_capacity(pi_c, pi_n, pi_m) == pytest.approx(2.0, rel=1e-12)


def test_collusion_capacity_below_nash_is_negative(
    low_noise: dict[str, float],
) -> None:
    pi_n = benchmarks.pi_N(**low_noise)
    pi_m = benchmarks.pi_M(**low_noise)
    pi_c = pi_n - (pi_m - pi_n)
    assert metrics.collusion_capacity(pi_c, pi_n, pi_m) == pytest.approx(
        -1.0, rel=1e-12
    )


def test_collusion_capacity_returns_zero_when_denominator_is_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="ai_trading.simulation.metrics"):
        result = metrics.collusion_capacity(pi_C=2.0, pi_N=1.0, pi_M=1.0)
    assert result == 0.0
    assert any("pi_M == pi_N" in r.message for r in caplog.records)


def test_collusion_capacity_returns_float(low_noise: dict[str, float]) -> None:
    pi_n = benchmarks.pi_N(**low_noise)
    pi_m = benchmarks.pi_M(**low_noise)
    out = metrics.collusion_capacity(pi_n, pi_n, pi_m)
    assert type(out) is float


@pytest.mark.parametrize("fixture_name", ["low_noise", "high_noise"])
def test_price_informativeness_matches_nash_analytical(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """At χ̂^C = χ^N: I^C = (I·χ^N)² · (σ_v/σ_u)²."""
    params = request.getfixturevalue(fixture_name)
    chi_n = benchmarks.x_N(**params)
    expected = (params["n_agents"] * chi_n) ** 2 * (
        params["sigma_v"] / params["sigma_u"]
    ) ** 2
    got = metrics.price_informativeness(
        chi_C=chi_n,
        sigma_v_hat=params["sigma_v"],
        sigma_u=params["sigma_u"],
        n_agents=int(params["n_agents"]),
    )
    assert got == pytest.approx(expected, rel=1e-12)


def test_price_informativeness_zero_when_chi_is_zero(
    low_noise: dict[str, float],
) -> None:
    got = metrics.price_informativeness(
        chi_C=0.0,
        sigma_v_hat=low_noise["sigma_v"],
        sigma_u=low_noise["sigma_u"],
        n_agents=int(low_noise["n_agents"]),
    )
    assert got == 0.0


def test_price_informativeness_returns_float(low_noise: dict[str, float]) -> None:
    out = metrics.price_informativeness(
        chi_C=0.5,
        sigma_v_hat=low_noise["sigma_v"],
        sigma_u=low_noise["sigma_u"],
        n_agents=int(low_noise["n_agents"]),
    )
    assert type(out) is float


@pytest.mark.parametrize("fixture_name", ["low_noise", "high_noise"])
def test_market_liquidity_matches_nash_analytical(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    params = request.getfixturevalue(fixture_name)
    lam_n = benchmarks.lambda_N(**params)
    expected = 1.0 / abs(1.0 - params["xi"] * lam_n)
    got = metrics.market_liquidity(lambda_hat=lam_n, xi=params["xi"])
    assert got == pytest.approx(expected, rel=1e-12)


def test_market_liquidity_returns_inf_when_denominator_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    xi = 2.0
    lam = 0.5
    with caplog.at_level(logging.WARNING, logger="ai_trading.simulation.metrics"):
        out = metrics.market_liquidity(lambda_hat=lam, xi=xi)
    assert out == math.inf
    assert any("lambda_hat" in r.message for r in caplog.records)


def test_market_liquidity_returns_float(low_noise: dict[str, float]) -> None:
    lam = benchmarks.lambda_N(**low_noise)
    out = metrics.market_liquidity(lambda_hat=lam, xi=low_noise["xi"])
    assert type(out) is float


def test_mispricing_zero_when_v_equals_v_bar() -> None:
    out = metrics.mispricing(
        lambda_hat=0.123, chi_C=0.456, n_agents=2, v_t=1.5, v_bar=1.5
    )
    assert out == 0.0


def test_mispricing_zero_when_lambda_I_chi_equals_one() -> None:
    n_agents = 2
    chi_C = 0.25
    lambda_hat = 1.0 / (n_agents * chi_C)
    out = metrics.mispricing(
        lambda_hat=lambda_hat,
        chi_C=chi_C,
        n_agents=n_agents,
        v_t=2.0,
        v_bar=0.0,
    )
    assert out == pytest.approx(0.0, abs=1e-12)


def test_mispricing_non_negative_across_inputs() -> None:
    cases = [
        (0.1, 0.2, 2, 1.5, 0.0),
        (10.0, 0.3, 2, -2.0, 0.0),
        (0.5, -0.4, 2, 1.0, 0.0),
        (1.5, 0.4, 3, 2.5, 1.0),
    ]
    for lam, chi, n, v, v_bar in cases:
        out = metrics.mispricing(
            lambda_hat=lam, chi_C=chi, n_agents=n, v_t=v, v_bar=v_bar
        )
        assert out >= 0.0


def test_mispricing_at_nash_matches_formula(low_noise: dict[str, float]) -> None:
    chi_n = benchmarks.x_N(**low_noise)
    lam_n = benchmarks.lambda_N(**low_noise)
    v_t = 1.7
    v_bar = 0.0
    expected = abs(1.0 - lam_n * low_noise["n_agents"] * chi_n) * abs(v_t - v_bar)
    got = metrics.mispricing(
        lambda_hat=lam_n,
        chi_C=chi_n,
        n_agents=int(low_noise["n_agents"]),
        v_t=v_t,
        v_bar=v_bar,
    )
    assert got == pytest.approx(expected, rel=1e-12)


def test_mispricing_returns_float() -> None:
    out = metrics.mispricing(lambda_hat=0.3, chi_C=0.4, n_agents=2, v_t=1.0, v_bar=0.0)
    assert type(out) is float
