"""Tests for the simulation runner (simulation/runner.py)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from ai_trading.agents.q_learning import QLearningAgent
from ai_trading.env import discretization
from ai_trading.simulation import benchmarks, runner

# ---- helpers --------------------------------------------------------------


def _write_shared_params(path: Path, **overrides: Any) -> None:
    cfg: dict[str, Any] = {
        "n_agents": 2,
        "xi": 500,
        "rho": 0.95,
        "theta": 0.1,
        "T_m": 50,
        "seed": 42,
        "n_v": 3,
        "n_x": 5,
        "n_p": 11,
        "alpha": 0.1,
        "beta": 1.0e-4,
        "convergence_criterion": 200,
        "max_periods": 2000,
        "n_sim": 2,
        "sigma_u": [0.1],
    }
    cfg.update(overrides)
    path.write_text(yaml.safe_dump(cfg))


def _write_experiment(path: Path, shared_rel: str, **overrides: Any) -> None:
    cfg: dict[str, Any] = {
        "experiment": {"name": "smoke"},
        "params": shared_rel,
        "agents": [{"type": "qlearner"}, {"type": "qlearner"}],
        "sweep": {"sigma_u": [0.1]},
        "simulation": {
            "n_sim": 2,
            "convergence_periods": 200,
            "max_periods": 2000,
            "eval_periods": 100,
            "check_every": 50,
            "seed_base": 42,
        },
        "irf": {
            "enabled": True,
            "pre_periods": 3,
            "post_periods": 4,
            "n_trials": 2,
            "noise": {"units": "price_bins", "magnitudes": [1.0]},
            "deviation": {"enabled": True, "bins": [1], "agents": [0, 1]},
        },
        "runner": {"log_every": 1000},
        "output": {"directory": "results/smoke"},
    }
    for key, value in overrides.items():
        cfg[key] = value
    path.write_text(yaml.safe_dump(cfg))


# ---- load_config ----------------------------------------------------------


def test_load_config_merges_shared_and_experiment(tmp_path: Path) -> None:
    shared = tmp_path / "shared.yaml"
    experiment = tmp_path / "exp.yaml"
    _write_shared_params(shared)
    _write_experiment(experiment, shared_rel="shared.yaml")

    cfg = runner.load_config(experiment)

    assert cfg["n_agents"] == 2
    assert cfg["alpha"] == 0.1
    assert cfg["agents"] == [{"type": "qlearner"}, {"type": "qlearner"}]
    assert cfg["sweep"]["sigma_u"] == [0.1]
    assert cfg["simulation"]["n_sim"] == 2


def test_load_config_overrides_replace_shared_values(tmp_path: Path) -> None:
    shared = tmp_path / "shared.yaml"
    experiment = tmp_path / "exp.yaml"
    _write_shared_params(shared)
    _write_experiment(experiment, shared_rel="shared.yaml")

    # Patch in an overrides block — load_config should apply it on top.
    exp_dict = yaml.safe_load(experiment.read_text())
    exp_dict["overrides"] = {"n_v": 7, "alpha": 0.05}
    experiment.write_text(yaml.safe_dump(exp_dict))

    cfg = runner.load_config(experiment)
    assert cfg["n_v"] == 7
    assert cfg["alpha"] == 0.05
    # Non-overridden keys still come from shared.
    assert cfg["n_x"] == 5


def test_load_config_rejects_missing_params_key(tmp_path: Path) -> None:
    experiment = tmp_path / "exp.yaml"
    experiment.write_text(yaml.safe_dump({"experiment": {"name": "x"}}))
    with pytest.raises(ValueError, match="params"):
        runner.load_config(experiment)


# ---- AGENT_REGISTRY -------------------------------------------------------


def test_agent_registry_contains_qlearner() -> None:
    assert "qlearner" in runner.AGENT_REGISTRY
    assert runner.AGENT_REGISTRY["qlearner"] is QLearningAgent


def test_make_agent_produces_qlearner() -> None:
    params = {"n_agents": 2, "xi": 500.0, "sigma_u": 0.1, "theta": 0.1, "sigma_v": 1.0}
    bench = benchmarks.compute_all_benchmarks(**params)
    grids = {
        "v_grid": discretization.build_v_grid(n_v=3),
        "action_grid": discretization.build_action_grid(n_x=5, **params),
        "price_grid": discretization.build_price_grid(n_p=11, **params),
    }
    base_cfg = {
        "alpha": 0.1,
        "beta": 1e-4,
        "rho": 0.95,
        "n_agents": 2,
        "n_v": 3,
        "n_x": 5,
        "n_p": 11,
        "seed": 42,
        **bench,
    }
    agent = runner._make_agent(
        {"type": "qlearner"}, agent_id=1, base_config=base_cfg, grids=grids
    )
    assert isinstance(agent, QLearningAgent)
    assert agent.agent_id == 1


def test_make_agent_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown agent type"):
        runner._make_agent({"type": "ghost"}, agent_id=0, base_config={}, grids={})


# ---- convergence checker --------------------------------------------------


class _FakeAgent:
    def __init__(self, q: np.ndarray) -> None:
        self.Q = q

    def greedy_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=-1)


def test_convergence_checker_three_unchanged_checks() -> None:
    """3 consecutive identical snapshots ⇒ converged when 2·check_every ≥ conv."""
    q = np.zeros((2, 2, 2, 3), dtype=np.float64)
    q[..., 1] = 1.0  # uniform argmax = 1
    agents = [_FakeAgent(q.copy()), _FakeAgent(q.copy())]

    check_every = 10
    convergence_periods = 20

    converged, snaps, unchanged = runner._check_convergence_step(
        agents, None, 0, check_every, convergence_periods
    )
    assert converged is False
    assert unchanged == 0
    assert snaps is not None

    converged, snaps, unchanged = runner._check_convergence_step(
        agents, snaps, unchanged, check_every, convergence_periods
    )
    assert converged is False
    assert unchanged == 1

    converged, snaps, unchanged = runner._check_convergence_step(
        agents, snaps, unchanged, check_every, convergence_periods
    )
    assert converged is True
    assert unchanged == 2


def test_convergence_checker_resets_on_change() -> None:
    q = np.zeros((2, 2, 2, 3), dtype=np.float64)
    q[..., 1] = 1.0
    agents = [_FakeAgent(q.copy())]
    _, snaps, unchanged = runner._check_convergence_step(agents, None, 0, 10, 20)

    # Same → unchanged = 1
    _, snaps, unchanged = runner._check_convergence_step(
        agents, snaps, unchanged, 10, 20
    )
    assert unchanged == 1

    # Change argmax: now uniformly action 2.
    agents[0].Q[..., :] = 0.0
    agents[0].Q[..., 2] = 1.0
    _, snaps, unchanged = runner._check_convergence_step(
        agents, snaps, unchanged, 10, 20
    )
    assert unchanged == 0


def test_convergence_checker_skips_agents_without_q() -> None:
    class NoQAgent:
        def greedy_policy(self) -> None:
            return None

    converged, snaps, unchanged = runner._check_convergence_step(
        [NoQAgent()], None, 0, 10, 20
    )
    assert converged is False
    assert snaps is None
    assert unchanged == 0


# ---- integration ----------------------------------------------------------


def test_run_experiment_smoke(tmp_path: Path) -> None:
    """Debug-scale end-to-end run produces well-formed Parquet output."""
    shared = tmp_path / "shared.yaml"
    experiment = tmp_path / "exp.yaml"
    _write_shared_params(shared)
    _write_experiment(experiment, shared_rel="shared.yaml")

    # Point output at tmp_path/results.
    exp_dict = yaml.safe_load(experiment.read_text())
    exp_dict["output"]["directory"] = str(tmp_path / "results")
    experiment.write_text(yaml.safe_dump(exp_dict))

    start = time.perf_counter()
    artefacts = runner.run_experiment(experiment)
    elapsed = time.perf_counter() - start
    assert elapsed < 60.0, f"Smoke test ran for {elapsed:.1f}s (limit 60s)"

    assert "sessions" in artefacts
    sessions_df = pd.read_parquet(artefacts["sessions"])
    assert len(sessions_df) == 2  # n_sim=2 × 1 sigma_u
    assert {
        "session_id",
        "sigma_u",
        "xi",
        "agent_types",
        "converged",
        "periods_to_convergence",
        "delta_c",
        "chi_hat",
        "lambda_hat",
        "price_informativeness",
        "market_liquidity",
        "mispricing",
        "seed",
        "avg_profit_agent_0",
        "avg_profit_agent_1",
        "delta_c_expected",
        "delta_c_agent_0",
        "delta_c_agent_1",
        "pi_N_path",
        "pi_M_path",
        "profit_gain",
        "chi_hat_agent_0",
        "chi_hat_agent_1",
        "wall_clock_seconds",
        "periods_per_second",
        "code_version",
        "reward_price",
        "market_maker",
        "eval_greedy",
        "eval_periods",
    }.issubset(sessions_df.columns)

    for col in ("delta_c", "avg_profit_agent_0", "avg_profit_agent_1"):
        assert np.all(np.isfinite(sessions_df[col].to_numpy()))
    assert np.all(sessions_df["lambda_hat"].to_numpy() > 0)

    assert "periods" in artefacts
    periods_df = pd.read_parquet(artefacts["periods"])
    assert "eval" in set(periods_df["phase"])
    # IRF was enabled in the smoke config.
    assert "irf" in set(periods_df["phase"])
    for col in (
        "v_t",
        "p_t",
        "u_t",
        "y_t",
        "p_bin",
        "u_shock",
        "irf_type",
        "irf_path",
        "deviating_agent",
        "n_visits_agent_0",
        "lambda_hat",
    ):
        assert col in periods_df.columns
    assert "metadata" in artefacts


def test_run_experiment_writes_only_sessions_when_no_period_rows(
    tmp_path: Path,
) -> None:
    """If eval_periods=0 and IRF is disabled, no periods.parquet is written."""
    shared = tmp_path / "shared.yaml"
    experiment = tmp_path / "exp.yaml"
    _write_shared_params(shared)
    _write_experiment(experiment, shared_rel="shared.yaml")

    exp_dict = yaml.safe_load(experiment.read_text())
    exp_dict["output"]["directory"] = str(tmp_path / "results")
    exp_dict["simulation"]["eval_periods"] = 0
    exp_dict["irf"]["enabled"] = False
    experiment.write_text(yaml.safe_dump(exp_dict))

    artefacts = runner.run_experiment(experiment)
    assert "sessions" in artefacts
    assert "periods" not in artefacts


def test_save_qtables_persists_arrays_per_session(tmp_path: Path) -> None:
    """With save_qtables=true, qtables/session_NNN/ files exist for each session."""
    shared = tmp_path / "shared.yaml"
    experiment = tmp_path / "exp.yaml"
    _write_shared_params(shared)
    _write_experiment(experiment, shared_rel="shared.yaml")

    output_dir = tmp_path / "results"
    exp_dict = yaml.safe_load(experiment.read_text())
    exp_dict["output"]["directory"] = str(output_dir)
    exp_dict["irf"]["enabled"] = False
    exp_dict["overrides"] = {"save_qtables": True}
    experiment.write_text(yaml.safe_dump(exp_dict))

    runner.run_experiment(experiment)

    n_v = 3
    n_x = 5
    n_p = 11
    for session_id in range(2):  # n_sim=2 from _write_experiment
        sdir = output_dir / "qtables" / f"session_{session_id:03d}"
        assert sdir.is_dir(), f"missing {sdir}"
        for name, shape in (
            ("agent_0.npy", (n_p, n_v, n_v, n_x)),
            ("agent_1.npy", (n_p, n_v, n_v, n_x)),
            ("v_grid.npy", (n_v,)),
            ("action_grid.npy", (n_x,)),
            ("price_grid.npy", (n_p,)),
        ):
            f = sdir / name
            assert f.is_file(), f"missing {f}"
            arr = np.load(f)
            assert arr.shape == shape, f"{f} has shape {arr.shape}, expected {shape}"
            assert arr.dtype == np.float64


def test_save_qtables_off_by_default(tmp_path: Path) -> None:
    """No qtables directory is created when save_qtables is absent."""
    shared = tmp_path / "shared.yaml"
    experiment = tmp_path / "exp.yaml"
    _write_shared_params(shared)
    _write_experiment(experiment, shared_rel="shared.yaml")

    output_dir = tmp_path / "results"
    exp_dict = yaml.safe_load(experiment.read_text())
    exp_dict["output"]["directory"] = str(output_dir)
    exp_dict["irf"]["enabled"] = False
    experiment.write_text(yaml.safe_dump(exp_dict))

    runner.run_experiment(experiment)
    assert not (output_dir / "qtables").exists()


def _irf_pairs(periods_df: pd.DataFrame) -> pd.DataFrame:
    irf = periods_df[periods_df["phase"] == "irf"]
    keys = [
        "session_id",
        "sigma_u",
        "irf_type",
        "shock_magnitude",
        "deviating_agent",
        "irf_trial",
        "irf_period",
    ]
    s = irf[irf["irf_path"] == "shock"]
    b = irf[irf["irf_path"] == "base"]
    return s.merge(b, on=keys, suffixes=("_s", "_b"), validate="one_to_one")


def _smoke_run(tmp_path: Path, **sim_overrides: Any) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)  # callers may pass a subdirectory
    shared = tmp_path / "shared.yaml"
    experiment = tmp_path / "exp.yaml"
    _write_shared_params(shared)
    _write_experiment(experiment, shared_rel="shared.yaml")
    exp_dict = yaml.safe_load(experiment.read_text())
    exp_dict["output"]["directory"] = str(tmp_path / "results")
    exp_dict["simulation"].update(sim_overrides)
    experiment.write_text(yaml.safe_dump(exp_dict))
    return runner.run_experiment(experiment)


_SMOKE_PARAMS = {
    "n_agents": 2,
    "xi": 500.0,
    "sigma_u": 0.1,
    "theta": 0.1,
    "sigma_v": 1.0,
}


def test_irf_counterfactual_paths_share_random_draws(tmp_path: Path) -> None:
    periods_df = pd.read_parquet(_smoke_run(tmp_path)["periods"])
    pairs = _irf_pairs(periods_df)
    assert len(pairs) > 0
    irf_paths = set(periods_df.loc[periods_df.phase == "irf", "irf_path"])
    assert irf_paths == {"shock", "base"}
    assert pairs["irf_period"].min() == 3
    assert pairs["irf_period"].max() == 3 + 4 - 1
    assert np.array_equal(pairs["v_t_s"].to_numpy(), pairs["v_t_b"].to_numpy())
    later = pairs[pairs["irf_period"] > 3]
    assert np.allclose(later["u_t_s"], later["u_t_b"])
    assert (pairs["u_shock_b"] == 0.0).all()


def test_irf_noise_shock_is_adverse_and_bin_scaled(tmp_path: Path) -> None:
    periods_df = pd.read_parquet(_smoke_run(tmp_path)["periods"])
    pairs = _irf_pairs(periods_df)
    shock = pairs[(pairs["irf_type"] == "noise") & (pairs["irf_period"] == 3)]
    assert len(shock) > 0
    grid = discretization.build_price_grid(n_p=11, n_v=3, **_SMOKE_PARAMS)
    lam_n = benchmarks.lambda_N(**_SMOKE_PARAMS)
    expected = (grid[1] - grid[0]) / lam_n * shock["shock_magnitude"]
    assert np.allclose(np.abs(shock["u_shock_s"]), expected, rtol=1e-9)
    assert (np.sign(shock["u_shock_s"]) == np.sign(shock["v_t_s"])).all()  # v̄ = 0


def test_irf_deviation_moves_deviator_in_aggressive_direction(tmp_path: Path) -> None:
    periods_df = pd.read_parquet(_smoke_run(tmp_path)["periods"])
    pairs = _irf_pairs(periods_df)
    dev = pairs[(pairs["irf_type"] == "deviation") & (pairs["irf_period"] == 3)]
    assert len(dev) > 0
    action_grid = discretization.build_action_grid(n_x=5, n_v=3, **_SMOKE_PARAMS)
    spacing = float(np.diff(action_grid)[0])
    for _, row in dev.iterrows():
        j = int(row["deviating_agent"])
        d = (row[f"x_agent_{j}_s"] - row[f"x_agent_{j}_b"]) * np.sign(row["v_t_s"])
        assert -1e-9 <= d <= spacing * row["shock_magnitude"] + 1e-9


def test_legacy_irf_schema_raises(tmp_path: Path) -> None:
    shared = tmp_path / "shared.yaml"
    experiment = tmp_path / "exp.yaml"
    _write_shared_params(shared)
    _write_experiment(experiment, shared_rel="shared.yaml")
    exp_dict = yaml.safe_load(experiment.read_text())
    exp_dict["output"]["directory"] = str(tmp_path / "results")
    exp_dict["irf"] = {
        "enabled": True,
        "shock_periods": 5,
        "shock_magnitudes": [1.0],
        "n_trials": 2,
    }
    experiment.write_text(yaml.safe_dump(exp_dict))
    with pytest.raises(ValueError, match="pre-Phase-0"):
        runner.run_experiment(experiment)


def test_save_eval_periods_limits_saved_rows(tmp_path: Path) -> None:
    periods_df = pd.read_parquet(
        _smoke_run(tmp_path, eval_periods=100, save_eval_periods=40)["periods"]
    )
    counts = periods_df[periods_df.phase == "eval"].groupby("session_id").size()
    assert (counts == 40).all()


def test_run_experiment_is_deterministic(tmp_path: Path) -> None:
    a = _smoke_run(tmp_path / "a")
    b = _smoke_run(tmp_path / "b")
    drop = ["wall_clock_seconds", "periods_per_second"]
    sa = pd.read_parquet(a["sessions"]).drop(columns=drop)
    sb = pd.read_parquet(b["sessions"]).drop(columns=drop)
    pd.testing.assert_frame_equal(sa, sb)
    pd.testing.assert_frame_equal(
        pd.read_parquet(a["periods"]), pd.read_parquet(b["periods"])
    )


def test_run_experiment_cli_overrides(tmp_path: Path) -> None:
    shared = tmp_path / "shared.yaml"
    experiment = tmp_path / "exp.yaml"
    _write_shared_params(shared)
    _write_experiment(experiment, shared_rel="shared.yaml")
    out = tmp_path / "override_out"
    artefacts = runner.run_experiment(
        experiment, session_id_start=5, n_sim=1, output_dir=out
    )
    sessions_df = pd.read_parquet(artefacts["sessions"])
    assert sessions_df["session_id"].tolist() == [5]
    assert sessions_df["seed"].tolist() == [42 + 5]
    assert (out / "run_metadata.json").is_file()


def test_spawned_streams_do_not_overlap_adjacent_sessions() -> None:
    env_42, agents_42 = runner._spawn_seed_sequences(42, 2)
    env_43, _ = runner._spawn_seed_sequences(43, 2)
    assert not np.allclose(
        np.random.default_rng(agents_42[0]).random(8),
        np.random.default_rng(env_43).random(8),
    )
    draws = {tuple(np.random.default_rng(s).random(4)) for s in [env_42, *agents_42]}
    assert len(draws) == 3


def test_run_experiment_max_periods_override(tmp_path: Path) -> None:
    shared = tmp_path / "shared.yaml"
    experiment = tmp_path / "exp.yaml"
    _write_shared_params(shared)
    _write_experiment(experiment, shared_rel="shared.yaml")
    exp_dict = yaml.safe_load(experiment.read_text())
    exp_dict["simulation"]["convergence_periods"] = 10**9  # never converges
    exp_dict["simulation"]["max_periods_by_sigma_u"] = {0.1: 5000}
    exp_dict["irf"]["enabled"] = False
    experiment.write_text(yaml.safe_dump(exp_dict))
    out = tmp_path / "override_out"
    artefacts = runner.run_experiment(
        experiment, n_sim=1, output_dir=out, max_periods=300
    )
    sessions_df = pd.read_parquet(artefacts["sessions"])
    assert sessions_df["periods_to_convergence"].tolist() == [300]
    assert not sessions_df["converged"].any()
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["cli_overrides"]["max_periods"] == 300
    assert meta["resolved_config"]["simulation"]["max_periods"] == 300
    assert "max_periods_by_sigma_u" not in meta["resolved_config"]["simulation"]


# ---- training diagnostics ---------------------------------------------------


def test_stability_stats_hand_example() -> None:
    visits = np.array([500, 300, 150, 50])
    flipped = [np.array([True, False, True, True])]
    flip, flip_w, n_onpath = runner._stability_stats(flipped, visits, 1000, 0.1)
    assert n_onpath == 3
    assert flip == pytest.approx(2 / 3)
    assert flip_w == pytest.approx(650 / 950)


def test_stability_stats_no_onpath_state_is_nan() -> None:
    flip, flip_w, n_onpath = runner._stability_stats(
        [np.ones(3, dtype=bool)], np.zeros(3, dtype=np.int64), 1000, 0.1
    )
    assert math.isnan(flip) and math.isnan(flip_w) and n_onpath == 0


_DEBUG_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "01_replicate_baseline"
    / "config_debug.yaml"
)


def _debug_run(out: Path, diagnostics: dict[str, Any] | None) -> dict[str, Path]:
    """config_debug.yaml, one session at σ_u = 0.1, no IRF, Q-tables saved."""
    out.mkdir(parents=True, exist_ok=True)
    exp = yaml.safe_load(_DEBUG_CONFIG.read_text())
    exp["params"] = str((_DEBUG_CONFIG.parent / exp["params"]).resolve())
    exp["sweep"]["sigma_u"] = [0.1]
    exp["simulation"]["n_sim"] = 1
    exp["irf"]["enabled"] = False
    exp["output"]["directory"] = str(out / "results")
    exp["overrides"] = {"save_qtables": True}
    if diagnostics is not None:
        exp["simulation"]["diagnostics"] = diagnostics
    path = out / "exp.yaml"
    path.write_text(yaml.safe_dump(exp))
    return runner.run_experiment(path)


_DIAG = {
    "enabled": True,
    "block_periods": 2000,
    "trajectory_eval_periods": 500,
    "onpath_min_share": 0.001,
}


def test_diagnostics_smoke_writes_trajectory(tmp_path: Path) -> None:
    artefacts = _debug_run(tmp_path, _DIAG)
    sessions_df = pd.read_parquet(artefacts["sessions"])
    row = sessions_df.iloc[0]
    assert "trajectory" in artefacts
    traj = pd.read_parquet(artefacts["trajectory"])
    n_blocks = int(row["periods_to_convergence"]) // 2000
    if row["converged"] and int(row["periods_to_convergence"]) % 2000 == 0:
        n_blocks -= 1  # block processing is skipped when the check stops training
    assert len(traj) == n_blocks
    assert traj["period"].tolist() == [2000 * (k + 1) for k in range(n_blocks)]
    expected_cols = {
        "session_id",
        "sigma_u",
        "period",
        "delta_c",
        "delta_c_agent_0",
        "delta_c_agent_1",
        "chi_hat_agent_0",
        "chi_hat_agent_1",
        "block_flip_share",
        "block_flip_share_weighted",
        "block_n_onpath",
    }
    assert expected_cols.issubset(traj.columns)
    assert 0.0 <= row["stability_flip_share"] <= 1.0
    assert 0.0 <= row["stability_flip_share_weighted"] <= 1.0
    assert row["stability_n_onpath"] > 0
    assert row["stability_window_periods"] > 0
    assert isinstance(row["settled"], (bool, np.bool_))
    assert bool(row["diagnostics_enabled"]) is True


def test_diagnostics_disabled_defaults(tmp_path: Path) -> None:
    artefacts = _debug_run(tmp_path, None)
    assert "trajectory" not in artefacts
    row = pd.read_parquet(artefacts["sessions"]).iloc[0]
    assert math.isnan(row["stability_flip_share"])
    assert math.isnan(row["stability_flip_share_weighted"])
    assert row["stability_n_onpath"] == 0
    assert row["stability_window_periods"] == 0
    assert bool(row["settled"]) is False
    assert bool(row["diagnostics_enabled"]) is False


def test_diagnostics_do_not_change_training(tmp_path: Path) -> None:
    on = _debug_run(tmp_path / "on", _DIAG)
    off = _debug_run(tmp_path / "off", {**_DIAG, "enabled": False})
    s_on = pd.read_parquet(on["sessions"])
    s_off = pd.read_parquet(off["sessions"])
    assert s_on["periods_to_convergence"].tolist() == (
        s_off["periods_to_convergence"].tolist()
    )
    assert s_on["delta_c"].tolist() == s_off["delta_c"].tolist()
    for i in range(2):
        q_on = np.load(
            tmp_path / "on" / "results" / "qtables" / "session_000" / f"agent_{i}.npy"
        )
        q_off = np.load(
            tmp_path / "off" / "results" / "qtables" / "session_000" / f"agent_{i}.npy"
        )
        assert np.array_equal(q_on, q_off)


# ---- per-v action grid, degenerate states, DGJ-style switches ----------------

_THESIS_PARAMS = {
    "n_agents": 2,
    "xi": 500.0,
    "sigma_u": 1.0,
    "theta": 0.1,
    "sigma_v": 1.0,
}
_PER_V_KEYS = {"action_grid": "per_v", "exploration_counter": "v", "q_init": "paper"}


def test_degenerate_v_mask_marks_only_the_zero_row_of_the_per_v_table() -> None:
    v_grid = discretization.build_v_grid(n_v=5)
    per_v = discretization.build_action_table(
        "per_v", n_x=15, v_grid=v_grid, **_THESIS_PARAMS
    )
    glob = discretization.build_action_table(
        "global", n_x=15, v_grid=v_grid, **_THESIS_PARAMS
    )
    assert runner._degenerate_v_mask(glob) is None
    mask = runner._degenerate_v_mask(per_v)
    assert mask is not None
    assert mask.tolist() == [False, False, True, False, False]


def test_policy_snapshot_masks_degenerate_states_only() -> None:
    q = np.random.default_rng(0).normal(size=(4, 3, 3, 5))
    agent = _FakeAgent(q.copy())
    full = np.argmax(q, axis=-1)
    assert np.array_equal(runner._policy_snapshot(agent), full)
    assert np.array_equal(runner._policy_snapshot(agent, None), full)
    snap = runner._policy_snapshot(agent, np.array([False, True, False]))
    assert np.all(snap[:, :, 1] == 0)
    assert np.array_equal(snap[:, :, [0, 2]], full[:, :, [0, 2]])
    assert np.array_equal(agent.Q, q)  # the agent is not modified


def test_convergence_check_ignores_greedy_changes_in_degenerate_states() -> None:
    """A changed greedy index at v = v̄ neither resets the counter nor is a flip."""
    q = np.zeros((2, 3, 3, 4), dtype=np.float64)
    q[..., 1] = 1.0  # greedy action 1 everywhere
    mask = np.array([False, True, False])  # value index 1 is v̄ (all actions equal)

    agents = [_FakeAgent(q.copy()), _FakeAgent(q.copy())]
    _, snaps, unchanged = runner._check_convergence_step(agents, None, 0, 10, 20, mask)
    assert unchanged == 0

    # The greedy index changes in a state with v_t = v̄ (last state axis = 1).
    agents[0].Q[0, 2, 1, 3] = 5.0
    converged, snaps2, unchanged = runner._check_convergence_step(
        agents, snaps, unchanged, 10, 20, mask
    )
    assert unchanged == 1 and converged is False  # counter not reset
    assert not (snaps2[0] != snaps[0]).any()  # and nothing to count as a flip

    agents[1].Q[1, 0, 1, 0] = 7.0  # again at v̄, other agent
    converged, snaps3, unchanged = runner._check_convergence_step(
        agents, snaps2, unchanged, 10, 20, mask
    )
    assert unchanged == 2 and converged is True

    # A change at a non-degenerate v does reset the counter and is one flip.
    agents[1].Q[1, 0, 2, 0] = 7.0
    converged, snaps4, unchanged = runner._check_convergence_step(
        agents, snaps3, unchanged, 10, 20, mask
    )
    assert unchanged == 0 and converged is False
    assert int((snaps4[1] != snaps3[1]).sum()) == 1

    # Without the mask (global grid) the same change at index 1 resets.
    agents = [_FakeAgent(q.copy()), _FakeAgent(q.copy())]
    _, snaps, unchanged = runner._check_convergence_step(agents, None, 0, 10, 20)
    _, snaps, unchanged = runner._check_convergence_step(agents, snaps, 0, 10, 20)
    assert unchanged == 1
    agents[0].Q[0, 2, 1, 3] = 5.0
    _, snaps, unchanged = runner._check_convergence_step(
        agents, snaps, unchanged, 10, 20
    )
    assert unchanged == 0


def _per_v_config() -> dict[str, Any]:
    """config_debug.yaml (n_v=3, n_x=5, n_p=11) with the three per-v keys set."""
    config = runner.load_config(_DEBUG_CONFIG)
    config.update(_PER_V_KEYS)
    return config


def test_build_session_per_v_passes_the_action_table_to_agents() -> None:
    env, agents, _bench = runner.build_session(_per_v_config(), 0.1, seed=1)
    assert env.action_mode == "per_v"
    for agent in agents:
        assert isinstance(agent, QLearningAgent)
        assert agent.q_init == "paper" and agent.exploration_counter == "v"
        np.testing.assert_array_equal(agent.action_table, env.action_table)
        np.testing.assert_array_equal(agent.action_grid, env.action_grid)
        assert np.all(agent.Q[:, :, 1, :] == 0.0)  # v = v̄ row of DGJ's init
        assert np.all(agent.Q[:, :, 2, :] > 0.0)


def _irf_deviation_rows(greedy_action: int, bins: int) -> tuple[pd.DataFrame, Any]:
    """Deviation IRF rows at the shock period; both agents always play one action."""
    config = _per_v_config()
    config["irf"] = {
        "enabled": True,
        "pre_periods": 1,
        "post_periods": 2,
        "n_trials": 40,
        "deviation": {"enabled": True, "bins": [bins], "agents": [0]},
    }
    env, agents, _bench = runner.build_session(config, 0.1, seed=5)
    for agent in agents:
        assert isinstance(agent, QLearningAgent)
        agent.Q[...] = 0.0
        agent.Q[..., greedy_action] = 1.0
    state = env.reset()
    rows, _state, _offset = runner._run_irf(config, env, agents, state, 0.1, 0, 0)
    df = pd.DataFrame(rows)
    df = df[df["irf_period"] == 1]
    shock = df[df["irf_path"] == "shock"].set_index("irf_trial")
    base = df[df["irf_path"] == "base"].set_index("irf_trial")
    return shock.join(base, lsuffix="_s", rsuffix="_b"), env


def test_irf_deviation_on_per_v_grid_raises_aggressiveness_for_both_signs() -> None:
    """Per-v grid: the shift is +m bins at every v, so |x| grows also for v < v̄."""
    pairs, env = _irf_deviation_rows(greedy_action=1, bins=2)
    assert len(pairs) > 0
    v = pairs["v_t_s"].to_numpy()
    assert (v < 0).any() and (v > 0).any()  # v̄ = 0; trials never start at v̄
    k = np.abs(v[:, None] - env.v_grid[None, :]).argmin(axis=1)
    np.testing.assert_allclose(pairs["x_agent_0_b"], env.action_table[k, 1])
    np.testing.assert_allclose(pairs["x_agent_0_s"], env.action_table[k, 3])
    np.testing.assert_allclose(pairs["x_agent_1_s"], pairs["x_agent_1_b"])
    bigger = np.abs(pairs["x_agent_0_s"]) > np.abs(pairs["x_agent_0_b"])
    assert bigger[v < 0].all() and bigger[v > 0].all()


def test_irf_deviation_on_per_v_grid_is_clipped_at_the_top_level() -> None:
    pairs, env = _irf_deviation_rows(greedy_action=3, bins=2)
    v = pairs["v_t_s"].to_numpy()
    k = np.abs(v[:, None] - env.v_grid[None, :]).argmin(axis=1)
    np.testing.assert_allclose(pairs["x_agent_0_s"], env.action_table[k, 4])  # n_x − 1


def _per_v_debug_run(out: Path) -> dict[str, Path]:
    """Debug-scale run with the three per-v keys, IRF, diagnostics and Q-tables."""
    out.mkdir(parents=True, exist_ok=True)
    exp = yaml.safe_load(_DEBUG_CONFIG.read_text())
    exp["params"] = str((_DEBUG_CONFIG.parent / exp["params"]).resolve())
    exp["sweep"]["sigma_u"] = [0.1]
    exp["simulation"]["n_sim"] = 1
    exp["simulation"]["max_periods"] = 30000
    exp["simulation"]["eval_periods"] = 2000
    exp["simulation"]["diagnostics"] = _DIAG
    exp["output"]["directory"] = str(out / "results")
    exp["overrides"] = {"save_qtables": True, **_PER_V_KEYS}
    path = out / "exp.yaml"
    path.write_text(yaml.safe_dump(exp))
    return runner.run_experiment(path)


def test_per_v_debug_run_end_to_end(tmp_path: Path) -> None:
    """Smoke run of a debug-scale per-v config (params_debug + the three keys)."""
    artefacts = _per_v_debug_run(tmp_path / "a")
    params = {**_THESIS_PARAMS, "sigma_u": 0.1}
    chi = discretization.build_chi_grid(n_x=5, **params)
    v_grid = discretization.build_v_grid(n_v=3)
    table = discretization.build_action_table("per_v", n_x=5, v_grid=v_grid, **params)

    row = pd.read_parquet(artefacts["sessions"]).iloc[0]
    for col in ("delta_c", "delta_c_expected", "chi_hat", "avg_profit_agent_0"):
        assert math.isfinite(row[col])
    for i in range(2):  # greedy play: χ̂ is an average of grid levels
        assert chi[0] - 1e-6 <= row[f"chi_hat_agent_{i}"] <= chi[-1] + 1e-6
    assert 0.0 <= row["stability_flip_share"] <= 1.0
    assert 1 <= row["stability_n_onpath_degenerate"] <= row["stability_n_onpath"]
    assert "trajectory" in artefacts

    # Every traded quantity is an entry of the action table at that v.
    periods_df = pd.read_parquet(artefacts["periods"])
    assert {"eval", "irf"} <= set(periods_df["phase"])
    k = np.abs(periods_df["v_t"].to_numpy()[:, None] - v_grid[None, :]).argmin(axis=1)
    for i in range(2):
        x = periods_df[f"x_agent_{i}"].to_numpy()
        assert np.isclose(x[:, None], table[k]).any(axis=1).all()
        assert np.all(x[k == 1] == 0.0)  # v = v̄

    # Deviation IRF: the deviator never trades less (in size) on the shock path.
    pairs = _irf_pairs(periods_df)
    dev = pairs[(pairs["irf_type"] == "deviation") & (pairs["irf_period"] == 3)]
    assert len(dev) > 0
    for _, r in dev.iterrows():
        j = int(r["deviating_agent"])
        assert abs(r[f"x_agent_{j}_s"]) >= abs(r[f"x_agent_{j}_b"]) - 1e-9

    qdir = tmp_path / "a" / "results" / "qtables" / "session_000"
    np.testing.assert_array_equal(np.load(qdir / "action_table.npy"), table)
    np.testing.assert_array_equal(np.load(qdir / "action_grid.npy"), chi)
    assert np.load(qdir / "agent_0.npy").shape == (11, 3, 3, 5)

    # Same seed, same result.
    again = _per_v_debug_run(tmp_path / "b")
    drop = ["wall_clock_seconds", "periods_per_second"]
    pd.testing.assert_frame_equal(
        pd.read_parquet(artefacts["sessions"]).drop(columns=drop),
        pd.read_parquet(again["sessions"]).drop(columns=drop),
    )


def test_per_v_training_loop_matches_an_independent_implementation() -> None:
    """End-to-end check of the DGJ-style path against a second implementation.

    The loop below is written from DGJ's text (§4.2: per-v interval, initial
    Q-matrix as the explicit sum over the opponent's actions; eq. 4.3: ε from
    the visits of the current v) and shares only the benchmarks, the v grid
    and the price grid with the package. λ is fixed at λ^N in both.
    """
    config = _per_v_config()
    config.update({"market_maker": "fixed_lambda", "beta": 1.0e-3})
    seed, periods, sigma_u = 11, 20_000, 0.1
    env, agents, bench = runner.build_session(config, sigma_u, seed)
    state = env.reset()
    first_state = tuple(int(z) for z in state[0])
    x_repo = np.empty((periods, 2))
    for t in range(periods):
        actions = {i: a.act(state[i]) for i, a in enumerate(agents)}
        nxt, rewards, done, info = env.step(actions)
        for i, a in enumerate(agents):
            a.update(runner.Transition(state[i], actions[i], rewards[i], nxt[i], done))
        x_repo[t] = info["x"]
        state = nxt

    n_v, n_x, n_p = 3, 5, 11
    rho, alpha, beta, iota = 0.95, float(config["alpha"]), 1.0e-3, 0.1
    chi_n, chi_m, lam = bench["chi_N"], bench["chi_M"], bench["lambda_N"]
    v_grid = [float(v) for v in env.v_grid]
    lo, hi = chi_m - iota * (chi_n - chi_m), chi_n + iota * (chi_n - chi_m)
    levels = [lo + j * (hi - lo) / (n_x - 1) for j in range(n_x)]
    qty = [[levels[j] * v_grid[k] for j in range(n_x)] for k in range(n_v)]
    q0 = [
        [
            sum((v_grid[k] - lam * (qty[k][j] + xo)) * qty[k][j] for xo in qty[k])
            / ((1 - rho) * n_x)
            for j in range(n_x)
        ]
        for k in range(n_v)
    ]
    tables = [np.empty((n_p, n_v, n_v, n_x)) for _ in range(2)]
    for q in tables:
        for k in range(n_v):
            q[:, :, k, :] = q0[k]
    visits = [[0] * n_v for _ in range(2)]
    children = np.random.SeedSequence(seed).spawn(3)
    rng_env = np.random.default_rng(children[0])
    rng_agents = [np.random.default_rng(c) for c in children[1:]]
    vp, vc = int(rng_env.integers(0, n_v)), int(rng_env.integers(0, n_v))
    pb = int(np.argmin(np.abs(env.price_grid - 0.0)))
    assert (pb, vp, vc) == first_state

    x_own = np.empty((periods, 2))
    n_greedy = 0
    for t in range(periods):
        acts = []
        for i in range(2):
            if rng_agents[i].random() < math.exp(-beta * visits[i][vc]):
                acts.append(int(rng_agents[i].integers(0, n_x)))
            else:
                acts.append(int(np.argmax(tables[i][pb, vp, vc])))
                n_greedy += 1
            visits[i][vc] += 1
        xs = [qty[vc][a] for a in acts]
        price = lam * (sum(xs) + float(rng_env.normal(0.0, sigma_u)))
        pb_next = int(np.argmin(np.abs(env.price_grid - price)))
        v_next = int(rng_env.integers(0, n_v))
        for i in range(2):
            target = (v_grid[vc] - price) * xs[i] + rho * float(
                tables[i][pb_next, vc, v_next].max()
            )
            cell = (pb, vp, vc, acts[i])
            tables[i][cell] = (1 - alpha) * tables[i][cell] + alpha * target
        x_own[t] = xs
        pb, vp, vc = pb_next, vc, v_next

    assert n_greedy > periods  # the greedy branch is well exercised
    np.testing.assert_allclose(x_repo, x_own, rtol=1e-12, atol=1e-9)
    for i in range(2):
        np.testing.assert_allclose(agents[i].Q, tables[i], rtol=1e-9, atol=1e-7)
        assert agents[i].N_v.tolist() == visits[i]
    assert (pb, vp, vc) == tuple(int(z) for z in state[0])


def test_global_grid_run_saves_tiled_action_table(tmp_path: Path) -> None:
    """Default (global) grid: action_table.npy is the tiled 1-D grid; no masking."""
    artefacts = _debug_run(tmp_path, _DIAG)
    qdir = tmp_path / "results" / "qtables" / "session_000"
    grid = np.load(qdir / "action_grid.npy")
    np.testing.assert_array_equal(
        np.load(qdir / "action_table.npy"), np.tile(grid, (3, 1))
    )
    row = pd.read_parquet(artefacts["sessions"]).iloc[0]
    assert row["stability_n_onpath_degenerate"] == 0


# ---- settled flag over non-degenerate states ---------------------------------


def test_stability_stats_hand_example_with_exclude() -> None:
    """Hand example with state 1 excluded: on-path = states 0 and 2."""
    visits = np.array([500, 300, 150, 50])
    flipped = [np.array([True, False, True, True])]
    exclude = np.array([False, True, False, False])
    flip, flip_w, n_onpath = runner._stability_stats(
        flipped, visits, 1000, 0.1, exclude=exclude
    )
    assert n_onpath == 2
    assert flip == pytest.approx(1.0)
    assert flip_w == pytest.approx(650 / 650)
    # exclude=None is the default behaviour.
    assert runner._stability_stats(flipped, visits, 1000, 0.1, exclude=None) == (
        runner._stability_stats(flipped, visits, 1000, 0.1)
    )


def test_degenerate_state_mask_uses_the_current_value_index() -> None:
    assert runner._degenerate_state_mask(None, 4, 3) is None
    mask = runner._degenerate_state_mask(np.array([False, True, False]), 4, 3)
    assert mask.shape == (4 * 3 * 3,)
    for p in range(4):
        for vp in range(3):
            for v in range(3):
                assert mask[p * 9 + vp * 3 + v] == (v == 1)


def _per_v_diag_run(out: Path, settled_max: float = 0.10) -> dict[str, Path]:
    """Debug-scale per-v run with diagnostics; no IRF (only the flags matter)."""
    out.mkdir(parents=True, exist_ok=True)
    exp = yaml.safe_load(_DEBUG_CONFIG.read_text())
    exp["params"] = str((_DEBUG_CONFIG.parent / exp["params"]).resolve())
    exp["sweep"]["sigma_u"] = [0.1]
    exp["simulation"]["n_sim"] = 1
    exp["simulation"]["max_periods"] = 30000
    exp["simulation"]["eval_periods"] = 500
    exp["simulation"]["diagnostics"] = {
        **_DIAG,
        "settled_max_flip_share": float(settled_max),
    }
    exp["irf"]["enabled"] = False
    exp["output"]["directory"] = str(out / "results")
    exp["overrides"] = dict(_PER_V_KEYS)
    path = out / "exp.yaml"
    path.write_text(yaml.safe_dump(exp))
    return runner.run_experiment(path)


def test_per_v_settled_flag_uses_the_non_degenerate_share(tmp_path: Path) -> None:
    artefacts = _per_v_diag_run(tmp_path / "a")
    row = pd.read_parquet(artefacts["sessions"]).iloc[0]
    plain, nd = row["stability_flip_share"], row["stability_flip_share_nd"]
    n_on, n_deg = row["stability_n_onpath"], row["stability_n_onpath_degenerate"]
    assert n_deg >= 1  # the v = v̄ states are on-path
    # Degenerate states never flip, so leaving them out can only raise the share,
    # by exactly the rescaling documented in docs/parameters.md.
    assert nd >= plain
    assert nd == pytest.approx(plain * n_on / (n_on - n_deg))
    assert (
        row["stability_flip_share_weighted_nd"] >= row["stability_flip_share_weighted"]
    )
    assert bool(row["settled"]) == bool(nd <= 0.10)
    traj = pd.read_parquet(artefacts["trajectory"])
    assert {"block_flip_share_nd", "block_flip_share_weighted_nd"} <= set(traj.columns)
    both = traj.dropna(subset=["block_flip_share", "block_flip_share_nd"])
    assert (both["block_flip_share_nd"] >= both["block_flip_share"]).all()

    # A threshold between the two shares: the plain share would call the session
    # settled, the share over non-degenerate states must not. Training does not
    # depend on the threshold.
    assert plain < nd, "test needs flips in the last window"
    mid = (plain + nd) / 2
    row_mid = pd.read_parquet(
        _per_v_diag_run(tmp_path / "b", settled_max=mid)["sessions"]
    ).iloc[0]
    assert row_mid["stability_flip_share_nd"] == nd
    assert plain <= mid
    assert bool(row_mid["settled"]) is False


def test_global_grid_nd_columns_equal_the_plain_columns(tmp_path: Path) -> None:
    """Global grid: the _nd columns equal the plain columns, the flag unchanged."""
    artefacts = _debug_run(tmp_path, _DIAG)
    row = pd.read_parquet(artefacts["sessions"]).iloc[0]
    assert row["stability_flip_share_nd"] == row["stability_flip_share"]
    assert (
        row["stability_flip_share_weighted_nd"] == row["stability_flip_share_weighted"]
    )
    assert bool(row["settled"]) == bool(row["stability_flip_share"] <= 0.10)
    traj = pd.read_parquet(artefacts["trajectory"])
    pd.testing.assert_series_equal(
        traj["block_flip_share_nd"], traj["block_flip_share"], check_names=False
    )
    pd.testing.assert_series_equal(
        traj["block_flip_share_weighted_nd"],
        traj["block_flip_share_weighted"],
        check_names=False,
    )


def test_nd_columns_are_nan_without_diagnostics(tmp_path: Path) -> None:
    row = pd.read_parquet(_debug_run(tmp_path, None)["sessions"]).iloc[0]
    assert math.isnan(row["stability_flip_share_nd"])
    assert math.isnan(row["stability_flip_share_weighted_nd"])
    assert bool(row["settled"]) is False


# ---- mixed populations (actor-critic, fixed policy) --------------------------


def _mixed_debug_run(out: Path, agents: list[dict[str, Any]]) -> dict[str, Path]:
    """Debug-scale per-v run with the given agent list, diagnostics, IRF, saved tables."""
    out.mkdir(parents=True, exist_ok=True)
    exp = yaml.safe_load(_DEBUG_CONFIG.read_text())
    exp["params"] = str((_DEBUG_CONFIG.parent / exp["params"]).resolve())
    exp["agents"] = agents
    exp["sweep"]["sigma_u"] = [0.1]
    exp["simulation"]["n_sim"] = 1
    exp["simulation"]["max_periods"] = 30000
    exp["simulation"]["eval_periods"] = 2000
    exp["simulation"]["diagnostics"] = _DIAG
    exp["output"]["directory"] = str(out / "results")
    exp["overrides"] = {"save_qtables": True, **_PER_V_KEYS}
    path = out / "exp.yaml"
    path.write_text(yaml.safe_dump(exp))
    return runner.run_experiment(path)


def test_agent_registry_contains_the_wp4_agents() -> None:
    assert {"qlearner", "actor_critic", "fixed_chi"} <= set(runner.AGENT_REGISTRY)


def test_q_ac_debug_run_end_to_end(tmp_path: Path) -> None:
    """Q–AC smoke run: concentration column, stability numbers, saved tables."""
    artefacts = _mixed_debug_run(
        tmp_path / "qa", [{"type": "qlearner"}, {"type": "actor_critic"}]
    )
    row = pd.read_parquet(artefacts["sessions"]).iloc[0]
    assert row["agent_types"] == "qlearner_actor_critic"
    assert math.isnan(row["policy_concentration_agent_0"])
    assert 0.0 < row["policy_concentration_agent_1"] <= 1.0
    for col in (
        "delta_c",
        "chi_hat_agent_1",
        "stability_flip_share",
        "stability_flip_share_nd",
        "stability_flip_share_weighted",
    ):
        assert math.isfinite(row[col]), col
    assert row["stability_n_onpath"] > 0
    assert isinstance(bool(row["converged"]), bool)
    assert "trajectory" in artefacts
    periods_df = pd.read_parquet(artefacts["periods"])
    assert {"eval", "irf"} <= set(periods_df["phase"])
    assert (periods_df["n_visits_agent_1"] >= 0).all()

    qdir = tmp_path / "qa" / "results" / "qtables" / "session_000"
    assert (qdir / "agent_0.npy").exists()
    assert not (qdir / "agent_1.npy").exists()
    assert np.load(qdir / "agent_1_prefs.npy").shape == (11, 3, 3, 5)
    assert np.load(qdir / "agent_1_values.npy").shape == (11, 3, 3)
    assert np.all(np.load(qdir / "agent_1_prefs.npy").max(axis=-1) == 0.0)


def test_fixed_opponent_debug_run_converges_on_the_fixed_policy(tmp_path: Path) -> None:
    """A learner against fixed_chi: the fixed agent saves nothing and never flips."""
    artefacts = _mixed_debug_run(
        tmp_path / "br",
        [{"type": "actor_critic"}, {"type": "fixed_chi", "config": {"chi": "nash"}}],
    )
    row = pd.read_parquet(artefacts["sessions"]).iloc[0]
    assert row["agent_types"] == "actor_critic_fixed_chi"
    assert math.isnan(row["policy_concentration_agent_1"])
    assert math.isfinite(row["policy_concentration_agent_0"])
    assert math.isfinite(row["delta_c_agent_1"])
    qdir = tmp_path / "br" / "results" / "qtables" / "session_000"
    assert sorted(p.name for p in qdir.glob("agent_*")) == [
        "agent_0_prefs.npy",
        "agent_0_values.npy",
    ]


# ---- state_price switch (R1.4, R1.5) -----------------------------------------


def test_state_price_none_smoke_leaves_price_slices_untouched(tmp_path: Path) -> None:
    """Debug scale, state_price none: finite metrics; price index != 0 never written."""
    out = tmp_path / "noprice"
    out.mkdir(parents=True, exist_ok=True)
    exp = yaml.safe_load(_DEBUG_CONFIG.read_text())
    exp["params"] = str((_DEBUG_CONFIG.parent / exp["params"]).resolve())
    exp["sweep"]["sigma_u"] = [0.1]
    exp["simulation"]["n_sim"] = 1
    exp["irf"]["enabled"] = False
    exp["output"]["directory"] = str(out / "results")
    exp["overrides"] = {"save_qtables": True, "state_price": "none", **_PER_V_KEYS}
    path = out / "exp.yaml"
    path.write_text(yaml.safe_dump(exp))
    artefacts = runner.run_experiment(path)

    row = pd.read_parquet(artefacts["sessions"]).iloc[0]
    for col in ("delta_c", "profit_gain", "chi_hat", "lambda_hat"):
        assert math.isfinite(float(row[col]))

    config = runner.load_config(path)
    env, agents, _bench = runner.build_session(config, 0.1, seed=42)
    assert env.state_price == "none"
    for i, agent in enumerate(agents):
        assert isinstance(agent, QLearningAgent)
        q0 = agent.Q.copy()
        q = np.load(out / "results" / "qtables" / "session_000" / f"agent_{i}.npy")
        assert q.shape == q0.shape
        np.testing.assert_array_equal(q[1:], q0[1:])
        assert not np.array_equal(q[0], q0[0])  # index 0 is where learning happens
