"""Tests for the parameter table (scripts/make_thesis_outputs.py: t2_rows, make_t2)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_thesis_outputs.py"
# Config of the baseline R1 (the baseline specification).
_M2 = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "01_replicate_baseline"
    / "config_r1_baseline.yaml"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("make_thesis_outputs", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_thesis_outputs"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_main(mod):
    import yaml

    cfg = yaml.safe_load(_M2.read_text())
    shared = Path(_M2).parent.parent / "shared" / "params_baseline.yaml"
    merged = {**yaml.safe_load(shared.read_text()), **cfg}
    merged.setdefault("max_periods", 100_000_000)
    return mod.Run(
        key="main",
        path=Path("."),
        sessions=pd.DataFrame({"session_id": range(30)}),
        periods=pd.DataFrame(),
        trajectory=pd.DataFrame(),
        metadata={"resolved_config": merged},
    )


def test_t2_every_row_has_a_nonempty_reason() -> None:
    mod = _load_script()
    rows = mod.t2_rows(_fake_main(mod))
    assert len(rows) == len(mod.PAPER_VALUES) + 5
    for label, _paper, _thesis, reason in rows:
        assert reason.strip(), f"empty reason for {label}"
    assert all(reason.strip() for *_, reason in mod.PAPER_VALUES)


def test_t2_design_rows_read_the_config() -> None:
    mod = _load_script()
    rows = {r[0]: r for r in mod.t2_rows(_fake_main(mod))}
    assert rows["Exploration counter"][2] == "per $v$"
    assert rows["Q-value initialisation"][2] == "Section~4.2"
    assert rows["Action grid layout"][2] == "per-$v$ levels"
    assert rows["Maximum periods"][2] == r"$1\times10^{8}$"
    assert rows["Entropy decay"][2] == r"$=\beta$"


def test_make_t2_writes_midrule_before_actor_critic_block(tmp_path: Path) -> None:
    mod = _load_script()
    out = mod.Outputs(tmp_path / "f", tmp_path / "t")
    mod.make_t2(out, _fake_main(mod))
    text = (tmp_path / "t" / "tab_parameters.tex").read_text()
    assert text.count("\\midrule") == 2
    assert text.index("\\midrule", text.index("Sessions")) < text.index(
        "Actor step size"
    )


def _write_qa_metadata(qa_dir: Path, agent_cfg: dict | None) -> None:
    entry = {"type": "actor_critic"}
    if agent_cfg is not None:
        entry["config"] = agent_cfg
    meta = {"resolved_config": {"agents": [{"type": "qlearner"}, entry]}}
    (qa_dir / "s0").mkdir(parents=True)
    (qa_dir / "s0" / "run_metadata.json").write_text(json.dumps(meta))


def test_actor_critic_params_read_the_run_config(tmp_path: Path) -> None:
    mod = _load_script()
    qa = tmp_path / "qa"
    _write_qa_metadata(qa, {"alpha_actor": 0.05, "entropy_coef": 0.2})
    ac = mod._actor_critic_params(qa)
    assert ac["alpha_actor"] == 0.05
    assert ac["entropy_coef"] == 0.2
    assert ac["alpha_critic"] == 0.05  # not overridden: code default
    assert ac["decay_is_beta"] and ac["scale_is_pi_m"]


def test_actor_critic_params_default_without_overrides(tmp_path: Path) -> None:
    mod = _load_script()
    qa = tmp_path / "qa"
    _write_qa_metadata(qa, None)
    with_run = mod._actor_critic_params(qa)
    without_run = mod._actor_critic_params(tmp_path / "missing")
    assert with_run == without_run
    assert (with_run["alpha_actor"], with_run["entropy_coef"]) == (0.1, 0.1)
