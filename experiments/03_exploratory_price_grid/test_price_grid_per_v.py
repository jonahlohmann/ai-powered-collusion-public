"""Tests for the per-value price grid of the exploratory run (DGJ, p. 23, footnote 25)."""

from __future__ import annotations

import numpy as np
import pytest

from ai_trading.env.discretization import build_price_table, continuous_to_bin
from ai_trading.env.market import MarketEnv

BASE = {
    "n_agents": 2,
    "xi": 500,
    "sigma_u": 0.1,
    "rho": 0.95,
    "theta": 0.1,
    "n_v": 5,
    "n_x": 15,
    "n_p": 31,
    "T_m": 10000,
    "iota": 0.1,
    "action_grid": "per_v",
    "reward_price": "raw",
    "market_maker": "rolling_ols",
    "seed": 7,
}


def _env(**kw: object) -> MarketEnv:
    return MarketEnv({**BASE, **kw})


def test_price_grid_default_is_global() -> None:
    env = _env()
    assert env.price_grid_mode == "global"
    assert env.price_table is None


def test_price_grid_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        _env(price_grid="finer")


def test_price_table_shape_and_rows_ascending() -> None:
    env = _env(price_grid="per_v")
    assert env.price_table is not None
    assert env.price_table.shape == (5, 31)
    assert np.all(np.diff(env.price_table, axis=1) > 0)


def test_price_table_one_action_level_is_about_one_grid_step() -> None:
    """The property of DGJ's footnote 25, at every value that differs from v̄."""
    env = _env(price_grid="per_v")
    for k, v_k in enumerate(env.v_grid):
        if abs(v_k - env.v_bar) < 1e-12:
            continue
        x_step = abs(env.action_table[k, 1] - env.action_table[k, 0])
        p_step = env.price_table[k, 1] - env.price_table[k, 0]
        ratio = env.lambda_N * x_step / p_step
        assert 0.9 < ratio < 1.1, (k, ratio)


def test_price_table_covers_every_action_pair() -> None:
    """Noise-free prices of all action pairs lie inside the grid of their value."""
    env = _env(price_grid="per_v")
    for k, v_k in enumerate(env.v_grid):
        if abs(v_k - env.v_bar) < 1e-12:
            continue
        x = env.action_table[k]
        prices = env.v_bar + env.lambda_N * (x[:, None] + x[None, :])
        assert prices.min() >= env.price_table[k, 0] - 1e-12
        assert prices.max() <= env.price_table[k, -1] + 1e-12
        bins = {env._price_bin(float(p), k)[0] for p in prices.ravel()}
        assert len(bins) >= 25, (k, len(bins))


def test_price_bin_per_v_matches_nearest_point() -> None:
    env = _env(price_grid="per_v")
    rng = np.random.default_rng(0)
    for k, v_k in enumerate(env.v_grid):
        if abs(v_k - env.v_bar) < 1e-12:
            continue
        row = env.price_table[k]
        for p in rng.uniform(row[0] - 0.01, row[-1] + 0.01, size=300):
            assert env._price_bin(float(p), k)[0] == continuous_to_bin(float(p), row)


def test_price_bin_at_mean_value_is_centre() -> None:
    env = _env(price_grid="per_v")
    k = int(np.argmin(np.abs(env.v_grid - env.v_bar)))
    assert env._price_bin(env.v_bar + 0.003, k)[0] == 15
    assert env._price_bin(env.v_bar - 0.5, k)[0] == 15


def test_step_state_uses_grid_of_the_period_value() -> None:
    env = _env(price_grid="per_v")
    env.reset()
    for _ in range(200):
        k = env._v_curr_bin
        states, _, _, info = env.step({0: 3, 1: 9})
        assert states[0][1] == k
        assert states[0][0] == env._price_bin(info["price_raw"], k)[0]
        assert states[0][0] == info["price_bin"]


def test_global_mode_unchanged_by_the_switch() -> None:
    """Same seed, same actions: 'global' reproduces the nearest-point bin of price_grid."""
    env = _env()
    env.reset()
    for _ in range(200):
        _, _, _, info = env.step({0: 5, 1: 11})
        assert info["price_bin"] == continuous_to_bin(info["price_raw"], env.price_grid)
        assert info["price"] == info["price_raw"]


def test_build_price_table_one_step_ratio_at_paper_grid() -> None:
    from ai_trading.env.discretization import build_chi_grid, build_v_grid

    v_grid = build_v_grid(n_v=10, v_bar=0.0, sigma_v=1.0)
    table = build_price_table(31, v_grid, 2, 500, 0.1, 0.1, 1.0, 0.0, 0.1)
    chi = build_chi_grid(15, 2, 500, 0.1, 0.1, 1.0, 0.1)
    for k, v_k in enumerate(v_grid):
        ratio = 0.002 * (chi[1] - chi[0]) * abs(v_k) / (table[k, 1] - table[k, 0])
        assert 0.85 < ratio < 1.1, (k, ratio)
