"""Tests for scripts/qtable_trigger_test.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "qtable_trigger_test", ROOT / "scripts/qtable_trigger_test.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tt = _load_script()
V_GRID = np.array([-1.0, 0.0, 1.0])


def test_deviation_cell_moves_away_from_the_centre_in_the_direction_of_v_prev() -> None:
    assert tt.deviation_cell(5, 2, 0, V_GRID, 0.0, 11) == (6, 2, 0)  # v_prev > v_bar
    assert tt.deviation_cell(5, 0, 2, V_GRID, 0.0, 11) == (4, 0, 2)  # v_prev < v_bar
    assert tt.deviation_cell(5, 1, 2, V_GRID, 0.0, 11) is None  # v_prev = v_bar
    assert tt.deviation_cell(10, 2, 0, V_GRID, 0.0, 11) is None  # out of range (top)
    assert tt.deviation_cell(0, 0, 2, V_GRID, 0.0, 11) is None  # out of range (bottom)


def test_eval_states_builds_full_states_from_consecutive_rows() -> None:
    rows = []
    for k, (pb, v, n0) in enumerate(
        [(5, 1.0, 10), (6, -1.0, 20), (5, 1.0, 30), (6, -1.0, 40)]
    ):
        rows.append(
            {
                "period": k,
                "phase": "eval",
                "irf_type": "",
                "p_bin": pb,
                "v_t": v,
                "n_visits_agent_0": n0,
                "n_visits_agent_1": 1,
            }
        )
    st = tt.eval_states(pd.DataFrame(rows), V_GRID, 2)
    # states: (5, 2, 0) ×2 and (6, 0, 2) ×1
    assert len(st) == 2
    a = st.set_index(["p_bin", "v_prev_bin", "v_bin"])
    assert a.loc[(5, 2, 0), "visits"] == 2 and a.loc[(6, 0, 2), "visits"] == 1
    assert a.loc[(5, 2, 0), "train_visits_agent_0"] == 40  # max over its occurrences


def test_classify_cells_labels_and_untouched_flag() -> None:
    n_p, n_v, n_x = 11, 3, 3
    chi = np.array([100.0, 150.0, 200.0])
    table = chi[None, :] * V_GRID[:, None]
    init = np.zeros((n_p, n_v, n_v, n_x))
    saved = init.copy()
    # on-path cell (5, 2, 0): greedy 1 (150); deviation cell (6, 2, 0): greedy 2 (200) → punish
    saved[5, 2, 0] = [0.0, 1.0, 0.0]
    saved[6, 2, 0] = [0.0, 0.0, 1.0]
    # on-path cell (6, 0, 2): greedy 2; deviation cell (5, 0, 2): greedy 0 (100) → forgive
    saved[6, 0, 2] = [0.0, 0.0, 1.0]
    saved[5, 0, 2] = [1.0, 0.0, 0.0]
    # on-path cell (4, 2, 2): greedy 1; deviation cell (5, 2, 2) untouched → argmax 0 → forgive, untouched
    saved[4, 2, 2] = [0.0, 1.0, 0.0]
    states = pd.DataFrame(
        {
            "p_bin": [5, 6, 4, 5],
            "v_prev_bin": [2, 0, 2, 1],
            "v_bin": [0, 2, 2, 1],
            "visits": [3, 1, 2, 7],
            "train_visits_agent_0": [50, 500, 20, 1],
        }
    )
    cells = tt.classify_cells(
        states,
        np.argmax(saved, axis=-1),
        saved,
        init,
        table,
        V_GRID,
        0.0,
        states["train_visits_agent_0"].to_numpy(),
    )
    # the (5, 1, 1) state has v = v_bar (degenerate) and is skipped
    assert len(cells) == 3
    by = cells.set_index(["p_bin", "v_prev_bin", "v_bin"])
    assert by.loc[(5, 2, 0), "label"] == "punish" and not by.loc[(5, 2, 0), "untouched"]
    assert (
        by.loc[(5, 2, 0), "chi_diff"] == 50.0
    )  # (200 - 150) / (−1) … sign: x = χ·v with v = −1
    assert by.loc[(6, 0, 2), "label"] == "forgive"
    assert by.loc[(4, 2, 2), "untouched"]
    # the deviation cell of (5, 2, 0) is (6, 2, 0), which is not on-path here → visits unknown
    assert by.loc[(5, 2, 0), "dev_train_visits"] == -1
    shares = tt.weighted_shares(cells[~cells["untouched"]])
    assert (
        shares["punish"] == 3 / 4
        and shares["forgive"] == 1 / 4
        and shares["same"] == 0.0
    )
