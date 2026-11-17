"""Tests for scripts/irf_replay.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ai_trading.simulation import runner
from ai_trading.viz import figures as viz

ROOT = Path(__file__).resolve().parents[1]
DEBUG_CFG = ROOT / "experiments/01_replicate_baseline/config_debug.yaml"
PER_V = {"action_grid": "per_v", "exploration_counter": "v", "q_init": "paper"}


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ir = _load("irf_replay")
dt = _load("deviation_test")


def _debug_session(out: Path) -> tuple[Path, dict[str, Any]]:
    """Debug per-v run with the IRF on and saved tables; returns (run dir, config)."""
    exp = yaml.safe_load(DEBUG_CFG.read_text())
    exp["params"] = str((DEBUG_CFG.parent / exp["params"]).resolve())
    exp["sweep"]["sigma_u"] = [0.1]
    exp["simulation"]["n_sim"] = 1
    exp["simulation"]["max_periods"] = 20000
    exp["simulation"]["eval_periods"] = 2000
    exp["irf"]["n_trials"] = 5
    exp["output"]["directory"] = str(out / "results")
    exp["overrides"] = {"save_qtables": True, **PER_V}
    path = out / "exp.yaml"
    path.write_text(yaml.safe_dump(exp))
    runner.run_experiment(path)
    return out / "results", runner.load_config(path)


def test_replay_matches_the_runner_sign_pattern_and_schema(tmp_path: Path) -> None:
    run_dir, config = _debug_session(tmp_path)
    sid, qdir, periods_file = ir.find_sessions(run_dir)[0]
    env, agents, _ = runner.build_session(config, 0.1, seed=42 + sid + ir.SEED_OFFSET)
    q_init = dt.restore_agents(agents, qdir)
    visited = ir.visited_masks(agents, q_init)
    assert all(m.shape == (11, 3, 3) for m in visited)
    assert all(m.any() for m in visited)
    noise = [float(m) for m in config["irf"]["noise"]["magnitudes"]]
    rows = ir.replay_session(
        env,
        agents,
        visited,
        sid,
        0.1,
        noise=noise,
        deviation=[1],
        trials=5,
        pre=3,
        post=4,
    )
    rep = pd.DataFrame(rows)
    assert set(rep["irf_type"]) == {"noise", "deviation"}
    assert set(rep["irf_period"]) == {3, 4, 5, 6}
    assert set(rep["irf_path"]) == {"shock", "base"}
    # Same schema as the runner's IRF rows (minus nothing the pairing needs).
    runner_periods = pd.read_parquet(periods_file)
    assert set(viz.IRF_PAIR_KEYS) <= set(rep.columns)
    for col in (
        "p_bin",
        "x_agent_0",
        "profit_agent_1",
        "n_visits_agent_0",
        "v_t",
        "p_t",
    ):
        assert col in rep.columns and col in runner_periods.columns

    # Sign pattern: an adverse noise shock moves the price away from v̄ in the direction of
    # v_t − v̄ at the shock period, in the replay as in the runner's own IRF.
    ev = runner_periods[runner_periods["phase"] == "eval"]
    both = pd.concat([ev, rep], ignore_index=True)
    pairs_rep = viz.irf_pairs(both, "noise", 0.0, 2)
    pairs_run = viz.irf_pairs(runner_periods, "noise", 0.0, 2)
    at_rep = pairs_rep[pairs_rep["irf_period"] == 3]
    at_run = pairs_run[pairs_run["irf_period"] == 3]
    assert (at_rep["dp"] > 0).all() and (at_run["dp"] > 0).all()
    assert np.sign(at_rep["dp"].mean()) == np.sign(at_run["dp"].mean())
    # The deviator trades more aggressively at the shock period (per-v grid: +1 level).
    dev = viz.irf_pairs(both, "deviation", 0.0, 2)
    eff = viz.effective_deviation_trials(dev, 3)
    at = eff[eff["irf_period"] == 3]
    for j in (0, 1):
        blk = at[at["deviating_agent"] == j]
        assert (blk[f"x_agent_{j}_s"].abs() >= blk[f"x_agent_{j}_b"].abs()).all()
