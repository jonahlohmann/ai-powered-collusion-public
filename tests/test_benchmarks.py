"""Tests for closed-form Nash and cartel benchmarks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ai_trading.simulation import benchmarks


@pytest.fixture
def baseline() -> dict[str, float]:
    """DGJ's baseline calibration (I=2, ξ=500, σ_u=0.1, θ=0.1, σ_v=1.0)."""
    return {
        "n_agents": 2,
        "xi": 500.0,
        "sigma_u": 0.1,
        "theta": 0.1,
        "sigma_v": 1.0,
    }


def _assert_orderings(params: dict[str, float]) -> None:
    chi_n = benchmarks.x_N(**params)
    chi_m = benchmarks.x_M(**params)
    lam_n = benchmarks.lambda_N(**params)
    lam_m = benchmarks.lambda_M(**params)
    pi_n = benchmarks.pi_N(**params)
    pi_m = benchmarks.pi_M(**params)

    assert chi_n > chi_m > 0, f"χ^N > χ^M > 0 violated: {chi_n=}, {chi_m=}"
    assert lam_n > 0
    assert lam_m > 0
    assert pi_m > pi_n > 0, f"π^M > π^N > 0 violated: {pi_n=}, {pi_m=}"


def test_baseline_orderings_low_noise(baseline: dict[str, float]) -> None:
    _assert_orderings(baseline)


def test_baseline_orderings_high_noise(baseline: dict[str, float]) -> None:
    params = {**baseline, "sigma_u": 100.0}
    _assert_orderings(params)


def test_compute_all_matches_individual_calls(baseline: dict[str, float]) -> None:
    bundle = benchmarks.compute_all_benchmarks(**baseline)
    expected_keys = {"chi_N", "lambda_N", "pi_N", "chi_M", "lambda_M", "pi_M"}
    assert set(bundle.keys()) == expected_keys

    assert bundle["chi_N"] == benchmarks.x_N(**baseline)
    assert bundle["chi_M"] == benchmarks.x_M(**baseline)
    assert bundle["lambda_N"] == benchmarks.lambda_N(**baseline)
    assert bundle["lambda_M"] == benchmarks.lambda_M(**baseline)
    assert bundle["pi_N"] == benchmarks.pi_N(**baseline)
    assert bundle["pi_M"] == benchmarks.pi_M(**baseline)


def test_single_speculator_collapses_to_cartel() -> None:
    params = {
        "n_agents": 1,
        "xi": 500.0,
        "sigma_u": 0.1,
        "theta": 0.1,
        "sigma_v": 1.0,
    }
    assert math.isclose(
        benchmarks.x_N(**params), benchmarks.x_M(**params), abs_tol=1e-8
    )
    assert math.isclose(
        benchmarks.pi_N(**params), benchmarks.pi_M(**params), abs_tol=1e-8
    )
    assert math.isclose(
        benchmarks.lambda_N(**params),
        benchmarks.lambda_M(**params),
        abs_tol=1e-8,
    )


def test_profit_identity_holds_for_nash(baseline: dict[str, float]) -> None:
    """π^N = σ_v² / ((I+1)² λ^N), per Proposition IA.1."""
    lam_n = benchmarks.lambda_N(**baseline)
    pi_n = benchmarks.pi_N(**baseline)
    expected = baseline["sigma_v"] ** 2 / ((baseline["n_agents"] + 1) ** 2 * lam_n)
    assert math.isclose(pi_n, expected, rel_tol=1e-10)


def test_profit_identity_holds_for_cartel(baseline: dict[str, float]) -> None:
    """π^M = σ_v² / (4 I λ^M), per Proposition IA.2."""
    lam_m = benchmarks.lambda_M(**baseline)
    pi_m = benchmarks.pi_M(**baseline)
    expected = baseline["sigma_v"] ** 2 / (4 * baseline["n_agents"] * lam_m)
    assert math.isclose(pi_m, expected, rel_tol=1e-10)


def test_fixed_point_equation_holds_nash(baseline: dict[str, float]) -> None:
    """χ^N · (I+1) · λ^N = 1."""
    chi_n = benchmarks.x_N(**baseline)
    lam_n = benchmarks.lambda_N(**baseline)
    assert math.isclose((baseline["n_agents"] + 1) * chi_n * lam_n, 1.0, rel_tol=1e-10)


def test_fixed_point_equation_holds_cartel(baseline: dict[str, float]) -> None:
    """χ^M · 2I · λ^M = 1."""
    chi_m = benchmarks.x_M(**baseline)
    lam_m = benchmarks.lambda_M(**baseline)
    assert math.isclose(2 * baseline["n_agents"] * chi_m * lam_m, 1.0, rel_tol=1e-10)


@pytest.mark.parametrize("sigma_u", [0.1, 1.0, 10.0, 100.0])
def test_orderings_across_sigma_u_sweep(
    baseline: dict[str, float], sigma_u: float
) -> None:
    _assert_orderings({**baseline, "sigma_u": sigma_u})


def test_realised_benchmark_profit_single_period() -> None:
    out = benchmarks.realised_benchmark_profit(
        np.array([1.5]), np.array([0.2]), chi=10.0, lam=0.01, n_agents=2, v_bar=0.5
    )
    x = 10.0 * 1.0
    price = 0.5 + 0.01 * (2 * x + 0.2)
    assert out[0] == pytest.approx((1.5 - price) * x, rel=1e-12)


def test_realised_benchmark_profit_mean_equals_grid_corrected_pi() -> None:
    """With v uniform on the grid and u = 0 the path mean equals Fix-A π."""
    from ai_trading.env.discretization import build_v_grid

    v_grid = build_v_grid(5)
    args = {"n_agents": 2, "xi": 500.0, "sigma_u": 1.0, "theta": 0.1, "sigma_v": 1.0}
    for chi_fn, lam_fn, pi_fn in (
        (benchmarks.x_N, benchmarks.lambda_N, benchmarks.pi_N),
        (benchmarks.x_M, benchmarks.lambda_M, benchmarks.pi_M),
    ):
        path = benchmarks.realised_benchmark_profit(
            v_grid, np.zeros_like(v_grid), chi_fn(**args), lam_fn(**args), 2
        )
        assert float(path.mean()) == pytest.approx(
            pi_fn(**args, v_grid=v_grid), rel=1e-12
        )


def test_grid_stage_game_thesis_parameters() -> None:
    """Nash and best symmetric outcome of the one-shot game on the global grid."""
    from ai_trading.env import discretization

    params = {"n_agents": 2, "xi": 500.0, "sigma_u": 1.0, "theta": 0.1, "sigma_v": 1.0}
    v_grid = discretization.build_v_grid(n_v=5)
    action_grid = discretization.build_action_grid(n_x=15, n_v=5, **params)
    bench = benchmarks.compute_all_benchmarks(**params, v_grid=v_grid)
    out = benchmarks.grid_stage_game(
        np.tile(action_grid, (5, 1)),
        v_grid,
        bench["lambda_N"],
        bench["pi_N"],
        bench["pi_M"],
    )
    assert out["nash_bins"] == [
        [(0, 0)],
        [(4, 4)],
        [(6, 8), (7, 7), (8, 6)],
        [(10, 10)],
        [(14, 14)],
    ]
    assert out["delta_c_grid_nash"] == pytest.approx(-0.277, abs=1e-3)
    assert out["best_symmetric_bins"] == [2, 5, 7, 9, 12]
    assert out["delta_c_grid_best_symmetric"] == pytest.approx(0.993, abs=1e-3)


def test_grid_stage_game_per_v_table_thesis_parameters() -> None:
    """One-shot game on the per-v action table: Nash at bin 13 for v ≠ v̄."""
    from ai_trading.env import discretization

    params = {"n_agents": 2, "xi": 500.0, "sigma_u": 1.0, "theta": 0.1, "sigma_v": 1.0}
    v_grid = discretization.build_v_grid(n_v=5)
    table = discretization.build_action_table("per_v", n_x=15, v_grid=v_grid, **params)
    bench = benchmarks.compute_all_benchmarks(**params, v_grid=v_grid)
    out = benchmarks.grid_stage_game(
        table, v_grid, bench["lambda_N"], bench["pi_N"], bench["pi_M"]
    )
    for k in (0, 1, 3, 4):
        assert out["nash_bins"][k] == [(13, 13)]
    # v = v̄: every pair trades zero, so every pair is an equilibrium with
    # zero profit. Nothing is excluded from the averages over v.
    assert len(out["nash_bins"][2]) == 15 * 15
    assert out["delta_c_grid_nash"] == pytest.approx(-0.029, abs=1e-3)
    assert [out["best_symmetric_bins"][k] for k in (0, 1, 3, 4)] == [1, 1, 1, 1]
    assert out["delta_c_grid_best_symmetric"] == pytest.approx(1.000, abs=2e-3)
