"""Tests for grid builders and bin↔continuous conversions."""

from __future__ import annotations

import numpy as np
import pytest

from ai_trading.env import discretization
from ai_trading.simulation import benchmarks


@pytest.fixture
def market_params() -> dict[str, float]:
    """DGJ's baseline (I=2, ξ=500, σ_u=0.1, θ=0.1, σ_v=1.0)."""
    return {
        "n_agents": 2,
        "xi": 500.0,
        "sigma_u": 0.1,
        "theta": 0.1,
        "sigma_v": 1.0,
    }


def test_v_grid_length() -> None:
    grid = discretization.build_v_grid(10)
    assert grid.shape == (10,)
    assert grid.dtype == np.float64


def test_v_grid_sorted_ascending() -> None:
    grid = discretization.build_v_grid(10)
    assert np.all(np.diff(grid) > 0)


def test_v_grid_symmetric_around_v_bar() -> None:
    grid = discretization.build_v_grid(10, v_bar=0.0, sigma_v=1.0)
    assert np.allclose(grid, -grid[::-1], atol=1e-12)


def test_v_grid_discretized_std_close_to_paper() -> None:
    """DGJ §4.2: with n_v = 10, discretised σ_v ≈ 0.938."""
    grid = discretization.build_v_grid(10, v_bar=0.0, sigma_v=1.0)
    assert abs(grid.std() - 0.938) < 0.01


def test_v_grid_shifts_with_v_bar() -> None:
    grid_a = discretization.build_v_grid(10, v_bar=0.0, sigma_v=1.0)
    grid_b = discretization.build_v_grid(10, v_bar=2.5, sigma_v=1.0)
    assert np.allclose(grid_b - 2.5, grid_a, atol=1e-12)


def test_v_grid_scales_with_sigma_v() -> None:
    grid_a = discretization.build_v_grid(10, sigma_v=1.0)
    grid_b = discretization.build_v_grid(10, sigma_v=2.0)
    assert np.allclose(grid_b, 2.0 * grid_a, atol=1e-12)


def test_v_grid_rejects_zero_n_v() -> None:
    with pytest.raises(ValueError):
        discretization.build_v_grid(0)


def test_action_grid_odd_n_x_contains_zero(market_params: dict[str, float]) -> None:
    grid = discretization.build_action_grid(n_x=15, **market_params)
    assert grid.shape == (15,)
    assert grid[7] == pytest.approx(0.0, abs=1e-12)


def test_action_grid_symmetric(market_params: dict[str, float]) -> None:
    n_x = 15
    grid = discretization.build_action_grid(n_x=n_x, **market_params)
    assert np.allclose(grid, -grid[::-1], atol=1e-12)


def test_action_grid_uniform_spacing(market_params: dict[str, float]) -> None:
    grid = discretization.build_action_grid(n_x=15, **market_params)
    diffs = np.diff(grid)
    assert np.allclose(diffs, diffs[0], atol=1e-12)


def test_action_grid_even_n_x_is_uniform_around_zero(
    market_params: dict[str, float],
) -> None:
    grid = discretization.build_action_grid(n_x=10, **market_params)
    assert grid.shape == (10,)
    assert 0.0 not in grid
    assert np.allclose(grid, -grid[::-1], atol=1e-12)


def test_action_grid_bounds_match_chi_N_chi_M(
    market_params: dict[str, float],
) -> None:
    """Bounds follow the formula of DGJ §4.2 with iota padding."""
    chi_n = benchmarks.x_N(**market_params)
    chi_m = benchmarks.x_M(**market_params)
    iota = 0.1
    n_v = 5
    v_bar = 0.0
    v_grid = discretization.build_v_grid(
        n_v=n_v, v_bar=v_bar, sigma_v=market_params["sigma_v"]
    )
    dv_max = float(v_grid[-1] - v_bar)
    dv_min = float(v_grid[0] - v_bar)
    pad = iota * (chi_n - chi_m)
    expected_max = chi_n * dv_max + pad * dv_max
    expected_min = chi_n * dv_min - pad * abs(dv_min)
    grid = discretization.build_action_grid(
        n_x=15, n_v=n_v, v_bar=v_bar, iota=iota, **market_params
    )
    assert grid[-1] == pytest.approx(expected_max, rel=1e-12)
    assert grid[0] == pytest.approx(expected_min, rel=1e-12)


def test_price_grid_symmetric_around_v_bar(market_params: dict[str, float]) -> None:
    v_bar = 1.5
    grid = discretization.build_price_grid(n_p=31, v_bar=v_bar, **market_params)
    assert grid.shape == (31,)
    assert np.allclose(grid - v_bar, -(grid - v_bar)[::-1], atol=1e-12)


def test_price_grid_uniform_spacing(market_params: dict[str, float]) -> None:
    grid = discretization.build_price_grid(n_p=31, **market_params)
    diffs = np.diff(grid)
    assert np.allclose(diffs, diffs[0], atol=1e-12)


def test_price_grid_bounds_match_paper_formula(
    market_params: dict[str, float],
) -> None:
    """DGJ §4.2 bounds: p_H/p_L from χ^N at extreme v ± 1.96σ_u, ι-padded."""
    n_v, v_bar, iota = 10, 0.0, 0.1
    chi_n = benchmarks.x_N(**market_params)
    lam_n = benchmarks.lambda_N(**market_params)
    v_grid = discretization.build_v_grid(n_v=n_v, v_bar=v_bar)
    i = market_params["n_agents"]
    su = market_params["sigma_u"]
    p_high = v_bar + lam_n * (i * chi_n * float(v_grid[-1]) + 1.96 * su)
    p_low = v_bar + lam_n * (i * chi_n * float(v_grid[0]) - 1.96 * su)
    pad = iota * (p_high - p_low)
    grid = discretization.build_price_grid(
        n_p=31, n_v=n_v, v_bar=v_bar, iota=iota, **market_params
    )
    assert grid[0] == pytest.approx(p_low - pad, rel=1e-12)
    assert grid[-1] == pytest.approx(p_high + pad, rel=1e-12)


def test_price_grid_thesis_parameters_hard_values() -> None:
    """Regression values for the thesis calibration (n_v=5, σ_u=1, ξ=500)."""
    grid = discretization.build_price_grid(
        n_p=31, n_agents=2, xi=500.0, sigma_u=1.0, theta=0.1, n_v=5, iota=0.1
    )
    assert grid[0] == pytest.approx(-1.029945, abs=1e-6)
    assert grid[-1] == pytest.approx(1.029945, abs=1e-6)
    assert grid[1] - grid[0] == pytest.approx(0.068663, abs=1e-6)


def test_continuous_to_bin_roundtrip_on_grid_points() -> None:
    grid = np.linspace(-3.0, 3.0, 21, dtype=np.float64)
    for k in range(len(grid)):
        assert discretization.continuous_to_bin(grid[k], grid) == k


def test_continuous_to_bin_clips_below() -> None:
    grid = np.linspace(-3.0, 3.0, 21, dtype=np.float64)
    assert discretization.continuous_to_bin(-1e6, grid) == 0


def test_continuous_to_bin_clips_above() -> None:
    grid = np.linspace(-3.0, 3.0, 21, dtype=np.float64)
    assert discretization.continuous_to_bin(1e6, grid) == len(grid) - 1


def test_continuous_to_bin_picks_nearer_neighbour() -> None:
    grid = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    assert discretization.continuous_to_bin(0.4, grid) == 1
    assert discretization.continuous_to_bin(0.6, grid) == 2


def test_bin_to_continuous_returns_grid_value() -> None:
    grid = np.linspace(-3.0, 3.0, 21, dtype=np.float64)
    for k in range(len(grid)):
        assert discretization.bin_to_continuous(k, grid) == grid[k]


def test_bin_to_continuous_rejects_out_of_range() -> None:
    grid = np.linspace(-3.0, 3.0, 21, dtype=np.float64)
    with pytest.raises(IndexError):
        discretization.bin_to_continuous(-1, grid)
    with pytest.raises(IndexError):
        discretization.bin_to_continuous(len(grid), grid)


@pytest.mark.parametrize("sigma_u", [0.1, 1.0, 10.0, 100.0])
def test_grids_remain_sorted_across_sigma_u_sweep(
    market_params: dict[str, float], sigma_u: float
) -> None:
    params = {**market_params, "sigma_u": sigma_u}
    action_grid = discretization.build_action_grid(n_x=15, **params)
    price_grid = discretization.build_price_grid(n_p=31, **params)
    assert np.all(np.diff(action_grid) > 0)
    assert np.all(np.diff(price_grid) > 0)


# ---- chi grid and action table ----------------------------------------------

_THESIS = {"n_agents": 2, "xi": 500.0, "sigma_u": 1.0, "theta": 0.1, "sigma_v": 1.0}


def test_chi_grid_thesis_parameters() -> None:
    """χ^M − ι(χ^N − χ^M) = 120.83 … χ^N + ι(χ^N − χ^M) = 170.83, 15 points."""
    grid = discretization.build_chi_grid(n_x=15, **_THESIS)
    assert grid.shape == (15,)
    assert grid.dtype == np.float64
    assert grid[0] == pytest.approx(120.8333, abs=1e-3)
    assert grid[-1] == pytest.approx(170.8333, abs=1e-3)
    assert np.allclose(np.diff(grid), 3.5714, atol=1e-3)


def test_chi_grid_bounds_and_benchmark_levels() -> None:
    grid = discretization.build_chi_grid(n_x=15, iota=0.1, **_THESIS)
    chi_n = benchmarks.x_N(**_THESIS)
    chi_m = benchmarks.x_M(**_THESIS)
    assert grid[0] == pytest.approx(chi_m - 0.1 * (chi_n - chi_m), rel=1e-12)
    assert grid[-1] == pytest.approx(chi_n + 0.1 * (chi_n - chi_m), rel=1e-12)
    # Nearest levels: cartel at index 1, Nash at index 13.
    assert int(np.argmin(np.abs(grid - chi_m))) == 1
    assert int(np.argmin(np.abs(grid - chi_n))) == 13


def test_chi_grid_rejects_small_n_x() -> None:
    with pytest.raises(ValueError):
        discretization.build_chi_grid(n_x=1, **_THESIS)


def test_action_table_per_v_rows_thesis_parameters() -> None:
    v_grid = discretization.build_v_grid(n_v=5)
    table = discretization.build_action_table("per_v", n_x=15, v_grid=v_grid, **_THESIS)
    assert table.shape == (5, 15)
    assert table.dtype == np.float64
    assert v_grid[4] == pytest.approx(1.2816, abs=1e-4)
    # v = +1.2816: from 154.854 to 218.932, ascending.
    assert table[4, 0] == pytest.approx(154.854, abs=1e-3)
    assert table[4, -1] == pytest.approx(218.932, abs=1e-3)
    assert np.all(np.diff(table[4]) > 0)
    # v = −1.2816 is its negative; index 14 is still the largest trade in size.
    np.testing.assert_allclose(table[0], -table[4], rtol=1e-12)
    assert np.all(np.diff(np.abs(table[0])) > 0)
    # v = v̄: every action trades zero.
    assert np.all(table[2] == 0.0)
    # Row k is the chi grid scaled by v_k − v̄.
    chi = discretization.build_chi_grid(n_x=15, **_THESIS)
    np.testing.assert_allclose(table[3], chi * v_grid[3], rtol=1e-12)


def test_action_table_per_v_respects_v_bar() -> None:
    shifted = discretization.build_action_table(
        "per_v",
        n_x=15,
        v_grid=discretization.build_v_grid(n_v=5, v_bar=2.0),
        v_bar=2.0,
        **_THESIS,
    )
    centred = discretization.build_action_table(
        "per_v", n_x=15, v_grid=discretization.build_v_grid(n_v=5), **_THESIS
    )
    np.testing.assert_allclose(shifted, centred, atol=1e-9)


def test_action_table_global_equals_tiled_grid() -> None:
    v_grid = discretization.build_v_grid(n_v=5)
    table = discretization.build_action_table(
        "global", n_x=15, v_grid=v_grid, **_THESIS
    )
    grid = discretization.build_action_grid(n_x=15, n_v=5, **_THESIS)
    assert table.shape == (5, 15)
    assert np.array_equal(table, np.tile(grid, (5, 1)))


def test_action_table_modes_share_the_largest_quantity() -> None:
    v_grid = discretization.build_v_grid(n_v=5)
    per_v = discretization.build_action_table("per_v", n_x=15, v_grid=v_grid, **_THESIS)
    glob = discretization.build_action_table("global", n_x=15, v_grid=v_grid, **_THESIS)
    assert np.abs(per_v).max() == pytest.approx(np.abs(glob).max(), rel=1e-12)
    assert np.abs(per_v).max() == pytest.approx(218.93, abs=1e-2)


def test_action_table_rejects_unknown_mode() -> None:
    v_grid = discretization.build_v_grid(n_v=5)
    with pytest.raises(ValueError, match="action_grid"):
        discretization.build_action_table("per_p", n_x=15, v_grid=v_grid, **_THESIS)
