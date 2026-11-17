"""Tests for MarketEnv (env/market.py)."""

from __future__ import annotations

import numpy as np
import pytest

from ai_trading.env import discretization
from ai_trading.env.market import MarketEnv
from ai_trading.simulation import benchmarks


def _base_config(**overrides: object) -> dict[str, object]:
    """DGJ's baseline (low-noise corner) with overridable fields."""
    cfg: dict[str, object] = {
        "n_agents": 2,
        "xi": 500.0,
        "sigma_u": 0.1,
        "theta": 0.1,
        "rho": 0.95,
        "n_v": 10,
        "n_x": 15,
        "n_p": 31,
        "T_m": 200,
        "seed": 42,
        "sigma_v": 1.0,
        "v_bar": 0.0,
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def env() -> MarketEnv:
    return MarketEnv(_base_config())


def _random_actions(env: MarketEnv, rng: np.random.Generator) -> dict[int, int]:
    n_x = len(env.action_grid)
    return {i: int(rng.integers(0, n_x)) for i in range(env.n_agents)}


def test_reset_returns_dict_with_n_agents_entries(env: MarketEnv) -> None:
    states = env.reset()
    assert isinstance(states, dict)
    assert set(states.keys()) == set(range(env.n_agents))


def test_reset_state_shape_and_dtype(env: MarketEnv) -> None:
    states = env.reset()
    for state in states.values():
        assert isinstance(state, np.ndarray)
        assert state.shape == (3,)
        assert state.dtype == np.int64


def test_reset_state_indices_in_range(env: MarketEnv) -> None:
    states = env.reset()
    n_p = len(env.price_grid)
    n_v = len(env.v_grid)
    for state in states.values():
        p_bin, v_prev_bin, v_curr_bin = state
        assert 0 <= int(p_bin) < n_p
        assert 0 <= int(v_prev_bin) < n_v
        assert 0 <= int(v_curr_bin) < n_v


def test_step_returns_correct_types(env: MarketEnv) -> None:
    env.reset()
    rng = np.random.default_rng(0)
    states, rewards, done, info = env.step(_random_actions(env, rng))

    assert isinstance(states, dict)
    assert isinstance(rewards, dict)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert done is False

    assert set(states.keys()) == set(range(env.n_agents))
    assert set(rewards.keys()) == set(range(env.n_agents))

    for state in states.values():
        assert state.shape == (3,)
        assert state.dtype == np.int64
    for r in rewards.values():
        assert isinstance(r, float)


def test_step_states_are_valid_bin_indices(env: MarketEnv) -> None:
    env.reset()
    rng = np.random.default_rng(1)
    n_p = len(env.price_grid)
    n_v = len(env.v_grid)
    for _ in range(50):
        states, _, _, _ = env.step(_random_actions(env, rng))
        for state in states.values():
            p_bin, v_curr_bin, v_next_bin = state
            assert 0 <= int(p_bin) < n_p
            assert 0 <= int(v_curr_bin) < n_v
            assert 0 <= int(v_next_bin) < n_v


def test_step_info_dict_has_required_keys(env: MarketEnv) -> None:
    env.reset()
    rng = np.random.default_rng(3)
    _, _, _, info = env.step(_random_actions(env, rng))
    required = {"price", "order_flow", "noise", "v_t", "lambda_hat", "z_t"}
    assert required.issubset(info.keys())


def test_step_z_t_identity(env: MarketEnv) -> None:
    env.reset()
    rng = np.random.default_rng(4)
    for _ in range(20):
        _, _, _, info = env.step(_random_actions(env, rng))
        expected = -env.xi * (info["price"] - env.v_bar)
        assert info["z_t"] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_step_no_noise_order_flow_equals_sum_of_actions() -> None:
    cfg = _base_config(sigma_u=0.0)
    env = MarketEnv(cfg)
    env.reset()
    rng = np.random.default_rng(5)
    for _ in range(30):
        actions = _random_actions(env, rng)
        x_sum = sum(
            discretization.bin_to_continuous(actions[i], env.action_grid)
            for i in range(env.n_agents)
        )
        _, _, _, info = env.step(actions)
        assert info["noise"] == pytest.approx(0.0, abs=1e-12)
        assert info["order_flow"] == pytest.approx(x_sum, rel=1e-12, abs=1e-12)


def test_step_reward_sign_when_buying_at_a_discount() -> None:
    """If x > 0 and v > p, profit must be positive."""
    cfg = _base_config(sigma_u=0.0)
    env = MarketEnv(cfg)
    env.reset()
    rng = np.random.default_rng(6)

    found = False
    for _ in range(200):
        actions = _random_actions(env, rng)
        _, rewards, _, info = env.step(actions)
        for i in range(env.n_agents):
            x_i = discretization.bin_to_continuous(actions[i], env.action_grid)
            if x_i > 0 and info["v_t"] > info["price"]:
                assert rewards[i] > 0
                found = True
            if x_i < 0 and info["v_t"] < info["price"]:
                assert rewards[i] > 0
                found = True
    assert found, "No (x>0, v>p) or (x<0, v<p) case was triggered in 200 steps."


def test_reward_matches_definition(env: MarketEnv) -> None:
    env.reset()
    rng = np.random.default_rng(7)
    for _ in range(20):
        actions = _random_actions(env, rng)
        _, rewards, _, info = env.step(actions)
        for i in range(env.n_agents):
            x_i = discretization.bin_to_continuous(actions[i], env.action_grid)
            expected = (info["v_t"] - info["price"]) * x_i
            assert rewards[i] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_warmup_uses_analytical_lambda_N(env: MarketEnv) -> None:
    env.reset()
    rng = np.random.default_rng(8)
    _, _, _, info = env.step(_random_actions(env, rng))
    assert info["lambda_hat"] == pytest.approx(env.lambda_N, rel=1e-12)


def test_lambda_hat_converges_to_nash_after_warmup() -> None:
    """After T_m Nash-strategy steps, λ̂ is within 20% of analytical λ^N.

    Weak at ξ=500 (γ̂ weight ≈ 4·10⁻⁷); the regression itself is verified by
    test_rolling_ols_matches_batch_regression.
    """
    cfg = _base_config(T_m=2000, seed=123)
    env = MarketEnv(cfg)
    env.reset()

    chi_n = benchmarks.x_N(
        n_agents=env.n_agents,
        xi=env.xi,
        sigma_u=env.sigma_u,
        theta=env.theta,
        sigma_v=env.sigma_v,
    )

    last_info: dict[str, object] = {}
    for _ in range(env.T_m + 1):
        v_t = env._v_curr
        x_target = chi_n * (v_t - env.v_bar)
        action = discretization.continuous_to_bin(x_target, env.action_grid)
        actions = {i: action for i in range(env.n_agents)}
        _, _, _, last_info = env.step(actions)

    lambda_hat = float(last_info["lambda_hat"])
    assert lambda_hat == pytest.approx(env.lambda_N, rel=0.2)


def test_reset_clears_buffers_between_episodes() -> None:
    cfg = _base_config(T_m=10)
    env = MarketEnv(cfg)
    env.reset()
    rng = np.random.default_rng(9)
    for _ in range(20):
        env.step(_random_actions(env, rng))

    env.reset()
    _, _, _, info = env.step(_random_actions(env, rng))
    assert info["lambda_hat"] == pytest.approx(env.lambda_N, rel=1e-12)


def test_state_transitions_chain_correctly(env: MarketEnv) -> None:
    """state_{t+1}[1] (v_t bin) equals state_t[2] (v_t bin) — the v carried forward."""
    states = env.reset()
    rng = np.random.default_rng(10)
    prev_v_curr_bin = int(states[0][2])
    for _ in range(20):
        next_states, _, _, _ = env.step(_random_actions(env, rng))
        next_state = next_states[0]
        assert int(next_state[1]) == prev_v_curr_bin
        prev_v_curr_bin = int(next_state[2])


def test_step_u_override_uses_exact_value(env: MarketEnv) -> None:
    """step(actions, u_override=X) bypasses the RNG and uses X as u_t."""
    env.reset()
    rng = np.random.default_rng(11)
    for shock in (5.0, -3.5, 0.0):
        _, _, _, info = env.step(_random_actions(env, rng), u_override=shock)
        assert info["noise"] == pytest.approx(shock, rel=1e-12, abs=1e-12)
        x_sum = info["order_flow"] - info["noise"]
        assert info["order_flow"] == pytest.approx(x_sum + shock, rel=1e-12, abs=1e-12)


def test_step_price_is_raw_clearing_price(env: MarketEnv) -> None:
    env.reset()
    rng = np.random.default_rng(2)
    for _ in range(50):
        _, _, _, info = env.step(_random_actions(env, rng))
        expected = env.v_bar + info["lambda_hat"] * info["order_flow"]
        assert info["price"] == pytest.approx(expected, rel=1e-12, abs=1e-12)
        assert info["price_raw"] == info["price"]
        assert info["price_bin"] == discretization.continuous_to_bin(
            info["price"], env.price_grid
        )
        assert info["price_snapped"] == float(env.price_grid[info["price_bin"]])


def test_next_state_price_component_is_price_bin(env: MarketEnv) -> None:
    env.reset()
    rng = np.random.default_rng(12)
    for _ in range(20):
        states, _, _, info = env.step(_random_actions(env, rng))
        assert int(states[0][0]) == info["price_bin"]


def test_snapped_reward_mode_uses_grid_price() -> None:
    env = MarketEnv(_base_config(reward_price="snapped"))
    env.reset()
    rng = np.random.default_rng(13)
    for _ in range(20):
        actions = _random_actions(env, rng)
        _, rewards, _, info = env.step(actions)
        assert info["price"] == info["price_snapped"]
        x0 = discretization.bin_to_continuous(actions[0], env.action_grid)
        assert rewards[0] == pytest.approx(
            (info["v_t"] - info["price_snapped"]) * x0, rel=1e-12, abs=1e-12
        )


def test_invalid_reward_price_and_market_maker_raise() -> None:
    with pytest.raises(ValueError, match="reward_price"):
        MarketEnv(_base_config(reward_price="rounded"))
    with pytest.raises(ValueError, match="market_maker"):
        MarketEnv(_base_config(market_maker="magic"))


def test_rolling_ols_matches_batch_regression() -> None:
    """Incremental ring-buffer OLS equals a batch OLS on the last T_m periods.

    Uses ξ=0.5, θ=1 so the learned slope γ̂ materially affects λ̂.
    """
    t_m = 50
    cfg = _base_config(xi=0.5, theta=1.0, sigma_u=1.0, n_v=5, T_m=t_m, seed=3)
    env = MarketEnv(cfg)
    env.reset()
    rng = np.random.default_rng(0)
    ys: list[float] = []
    vs: list[float] = []
    for t in range(8 * t_m + 7):  # several wraps incl. the exact refresh
        _, _, _, info = env.step(_random_actions(env, rng))
        if t >= t_m:
            y = np.asarray(ys[-t_m:])
            v = np.asarray(vs[-t_m:])
            yc = y - y.mean()
            gamma = float(np.dot(yc, v - v.mean()) / np.dot(yc, yc))
            expected = (env.theta * gamma + env.xi) / (env.theta + env.xi**2)
            assert info["lambda_hat"] == pytest.approx(expected, rel=1e-9)
        else:
            assert info["lambda_hat"] == pytest.approx(env.lambda_N, rel=1e-12)
        ys.append(info["order_flow"])
        vs.append(info["v_t"])


def test_fixed_lambda_market_maker_always_returns_lambda_n() -> None:
    env = MarketEnv(_base_config(market_maker="fixed_lambda", T_m=10))
    env.reset()
    rng = np.random.default_rng(14)
    for _ in range(50):
        _, _, _, info = env.step(_random_actions(env, rng))
        assert info["lambda_hat"] == env.lambda_N


def test_step_u_shock_is_additive_and_keeps_rng_aligned() -> None:
    a = MarketEnv(_base_config())
    b = MarketEnv(_base_config())
    a.reset()
    b.reset()
    acts = {0: 3, 1: 4}
    _, _, _, ia = a.step(acts, u_shock=5.0)
    _, _, _, ib = b.step(acts)
    assert ia["noise"] - ib["noise"] == pytest.approx(5.0, rel=1e-12)
    assert ia["u_shock"] == 5.0 and ib["u_shock"] == 0.0
    for _ in range(10):
        _, _, _, ia = a.step(acts)
        _, _, _, ib = b.step(acts)
        assert ia["v_t"] == ib["v_t"]
        assert ia["noise"] == ib["noise"]


def test_v_current_is_value_traded_next(env: MarketEnv) -> None:
    env.reset()
    rng = np.random.default_rng(15)
    for _ in range(10):
        v_before = env.v_current
        _, _, _, info = env.step(_random_actions(env, rng))
        assert info["v_t"] == v_before


def test_rng_seed_sequence_overrides_integer_seed() -> None:
    e1 = MarketEnv(_base_config(seed=1, rng_seed_sequence=np.random.SeedSequence(7)))
    e2 = MarketEnv(_base_config(seed=2, rng_seed_sequence=np.random.SeedSequence(7)))
    assert e1.rng.random() == e2.rng.random()


def test_symmetric_grid_nash_is_stage_game_equilibrium_under_raw_price() -> None:
    """At every v, the nearest-grid Nash action is a best response to itself.

    With rewards on the rounded price (``reward_price: snapped``) this failed at
    the extreme v (best response to bin 14 was bin 13), which produced a
    spurious asymmetric 'collusion' in an earlier version of the code.
    """
    cfg = _base_config(sigma_u=1.0, n_v=5, n_x=15, n_p=31, market_maker="fixed_lambda")
    env = MarketEnv(cfg)
    env.reset()
    chi_n = benchmarks.x_N(n_agents=2, xi=500.0, sigma_u=1.0, theta=0.1, sigma_v=1.0)
    n_x = len(env.action_grid)
    for v_bin, v in enumerate(env.v_grid):
        nash_bin = discretization.continuous_to_bin(
            chi_n * (float(v) - env.v_bar), env.action_grid
        )
        payoff = np.empty(n_x)
        for a in range(n_x):
            env._v_curr = float(v)
            env._v_curr_bin = v_bin
            _, rewards, _, _ = env.step({0: a, 1: nash_bin}, u_override=0.0)
            payoff[a] = rewards[0]
        assert int(np.argmax(payoff)) == nash_bin, (
            f"v={v:+.3f}: best response to Nash bin {nash_bin} is "
            f"{int(np.argmax(payoff))}"
        )


# ---- action table and per-v action grid --------------------------------------


def _thesis_config(**overrides: object) -> dict[str, object]:
    return _base_config(sigma_u=1.0, n_v=5, n_x=15, n_p=31, **overrides)


def test_default_action_mode_is_global_with_tiled_table(env: MarketEnv) -> None:
    assert env.action_mode == "global"
    n_v, n_x = len(env.v_grid), len(env.action_grid)
    assert env.action_table.shape == (n_v, n_x)
    assert np.array_equal(env.action_table, np.tile(env.action_grid, (n_v, 1)))
    expected = discretization.build_action_grid(
        n_x=15, n_agents=2, xi=500.0, sigma_u=0.1, theta=0.1, sigma_v=1.0, n_v=10
    )
    assert np.array_equal(env.action_grid, expected)


def test_action_grid_key_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="action_grid"):
        MarketEnv(_base_config(action_grid="per_p"))


def test_step_reports_quantities_and_v_bin_on_the_global_grid(env: MarketEnv) -> None:
    rng = np.random.default_rng(3)
    states = env.reset()
    for _ in range(50):
        v_bin = int(states[0][2])
        actions = _random_actions(env, rng)
        states, _rewards, _, info = env.step(actions)
        assert info["v_bin"] == v_bin
        assert info["x"] == [
            float(env.action_grid[actions[i]]) for i in range(env.n_agents)
        ]


def test_step_rejects_out_of_range_action(env: MarketEnv) -> None:
    env.reset()
    n_x = len(env.action_grid)
    with pytest.raises(IndexError):
        env.step({0: n_x, 1: 0})
    with pytest.raises(IndexError):
        env.step({0: 0, 1: -1})


def test_per_v_env_exposes_chi_grid_and_table() -> None:
    env = MarketEnv(_thesis_config(action_grid="per_v"))
    assert env.action_mode == "per_v"
    params = {"n_agents": 2, "xi": 500.0, "sigma_u": 1.0, "theta": 0.1, "sigma_v": 1.0}
    np.testing.assert_array_equal(
        env.action_grid, discretization.build_chi_grid(n_x=15, **params)
    )
    np.testing.assert_array_equal(
        env.action_table,
        discretization.build_action_table("per_v", n_x=15, v_grid=env.v_grid, **params),
    )
    # The price grid does not depend on the action-grid mode.
    np.testing.assert_array_equal(
        env.price_grid, MarketEnv(_thesis_config()).price_grid
    )


def test_per_v_step_trades_the_action_table() -> None:
    env = MarketEnv(_thesis_config(action_grid="per_v"))
    rng = np.random.default_rng(7)
    states = env.reset()
    seen_v_bins: set[int] = set()
    for _ in range(300):
        v_bin = int(states[0][2])
        seen_v_bins.add(v_bin)
        actions = _random_actions(env, rng)
        states, rewards, _, info = env.step(actions)
        x = [float(env.action_table[v_bin, actions[i]]) for i in range(env.n_agents)]
        assert info["v_bin"] == v_bin
        assert info["x"] == x
        assert info["order_flow"] == pytest.approx(
            sum(x) + info["noise"], rel=1e-12, abs=1e-12
        )
        for i in range(env.n_agents):
            assert rewards[i] == pytest.approx(
                (info["v_t"] - info["price"]) * x[i], rel=1e-12, abs=1e-12
            )
        # Same sign as v − v̄; zero trade at v = v̄ whatever the action.
        dv = info["v_t"] - env.v_bar
        assert all(q * dv >= 0.0 for q in x)
        if dv == 0.0:
            assert x == [0.0, 0.0] and rewards == {0: 0.0, 1: 0.0}
    assert seen_v_bins == set(range(5))


def test_per_v_higher_index_is_more_aggressive_at_every_v() -> None:
    env = MarketEnv(_thesis_config(action_grid="per_v"))
    for k, v in enumerate(env.v_grid):
        size = np.abs(env.action_table[k])
        if v == env.v_bar:
            assert np.all(size == 0.0)
        else:
            assert np.all(np.diff(size) > 0)


def test_per_v_nash_level_is_a_best_response_to_itself() -> None:
    """On the per-v grid, bin 13 (nearest to χ^N) is a best reply to bin 13."""
    env = MarketEnv(_thesis_config(action_grid="per_v", market_maker="fixed_lambda"))
    env.reset()
    n_x = env.action_table.shape[1]
    for v_bin, v in enumerate(env.v_grid):
        if v == env.v_bar:
            continue
        payoff = np.empty(n_x)
        for a in range(n_x):
            env._v_curr = float(v)
            env._v_curr_bin = v_bin
            _, rewards, _, _ = env.step({0: a, 1: 13}, u_override=0.0)
            payoff[a] = rewards[0]
        assert int(np.argmax(payoff)) == 13, f"v={v:+.3f}"


# ---- state_price switch (R1.4, R1.5: no lagged price in the state) ----------


def _run_actions(env: MarketEnv, n: int, seed: int) -> list[tuple]:
    """Reset, then n steps with a fixed random action sequence; collect everything."""
    out: list[tuple] = []
    states = env.reset()
    out.append(("reset", {i: s.copy() for i, s in states.items()}))
    rng = np.random.default_rng(seed)
    for _ in range(n):
        states, rewards, done, info = env.step(_random_actions(env, rng))
        out.append((states, rewards, done, info))
    return out


def test_state_price_defaults_to_lagged() -> None:
    default = MarketEnv(_base_config())
    lagged = MarketEnv(_base_config(state_price="lagged"))
    assert default.state_price == "lagged"
    a = _run_actions(default, 50, seed=3)
    b = _run_actions(lagged, 50, seed=3)
    for (sa, *ra), (sb, *rb) in zip(a[1:], b[1:], strict=True):
        for i in sa:
            np.testing.assert_array_equal(sa[i], sb[i])
        assert ra == rb
        assert int(sa[0][0]) == ra[2]["price_bin"]


def test_state_price_none_zeroes_price_component_only() -> None:
    env = MarketEnv(_base_config(state_price="none"))
    states = env.reset()
    for s in states.values():
        assert int(s[0]) == 0
    rng = np.random.default_rng(12)
    bins = set()
    for _ in range(200):
        states, _, _, info = env.step(_random_actions(env, rng))
        for s in states.values():
            assert int(s[0]) == 0
        bins.add(info["price_bin"])
    assert len(bins) > 1  # the true price bin still varies


def test_state_price_none_leaves_prices_rewards_and_values_identical() -> None:
    lagged = _run_actions(MarketEnv(_base_config(state_price="lagged")), 300, seed=9)
    none = _run_actions(MarketEnv(_base_config(state_price="none")), 300, seed=9)
    for i in lagged[0][1]:
        np.testing.assert_array_equal(lagged[0][1][i][1:], none[0][1][i][1:])
    for (s_l, r_l, d_l, info_l), (s_n, r_n, d_n, info_n) in zip(
        lagged[1:], none[1:], strict=True
    ):
        assert r_l == r_n
        assert d_l == d_n
        assert info_l == info_n
        for i in s_l:
            np.testing.assert_array_equal(s_l[i][1:], s_n[i][1:])
            assert int(s_n[i][0]) == 0


def test_invalid_state_price_raises() -> None:
    with pytest.raises(ValueError, match="state_price"):
        MarketEnv(_base_config(state_price="current"))
