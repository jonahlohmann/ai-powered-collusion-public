"""Tests for the pure helpers of viz/figures.py."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ai_trading.viz import figures as viz


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic() -> None:
    rng = np.random.default_rng(1)
    v = rng.normal(0.3, 0.1, size=30)
    lo, hi = viz.bootstrap_ci(v)
    assert lo < v.mean() < hi
    assert (lo, hi) == viz.bootstrap_ci(v)
    assert hi - lo < 0.2
    assert viz.bootstrap_ci([1.0]) == (math.nan, math.nan) or all(
        math.isnan(x) for x in viz.bootstrap_ci([1.0])
    )


def _periods_with_irf() -> pd.DataFrame:
    """Two sessions, eval rows and one deviation IRF spec with two trials each.

    Trial 1 of session 0 is clipped (deviator's quantity unchanged at the shock
    period) and must be dropped by ``effective_deviation_trials``. Trial 1 of
    session 1 leads to a never-visited state and must be dropped by
    ``visited_trials``.
    """
    rows = []
    for sid in (0, 1):
        for k in range(4):  # eval rows: x = 10, profit = 5, p - v_bar = 0.5, v = 1
            rows.append(
                {
                    "session_id": sid,
                    "sigma_u": 1.0,
                    "period": k,
                    "v_t": 1.0,
                    "p_t": 0.5,
                    "p_bin": 15 + k % 2,
                    "phase": "eval",
                    "irf_type": "",
                    "irf_path": "",
                    "irf_trial": -1,
                    "irf_period": -1,
                    "shock_magnitude": math.nan,
                    "deviating_agent": -1,
                    "x_agent_0": 10.0,
                    "profit_agent_0": 5.0,
                    "n_visits_agent_0": 3,
                    "x_agent_1": 10.0,
                    "profit_agent_1": 5.0,
                    "n_visits_agent_1": 3,
                }
            )
        for trial in (0, 1):
            for t in (3, 4):
                for path in ("shock", "base"):
                    dev_moved = not (sid == 0 and trial == 1)
                    x0 = 12.0 if (path == "shock" and t == 3 and dev_moved) else 10.0
                    x1 = 11.0 if (path == "shock" and t == 4) else 10.0
                    visits = (
                        0
                        if (sid == 1 and trial == 1 and t == 4 and path == "shock")
                        else 2
                    )
                    rows.append(
                        {
                            "session_id": sid,
                            "sigma_u": 1.0,
                            "period": 100 + trial * 10 + t,
                            "v_t": 1.0,
                            "p_t": 0.5 + (0.1 if path == "shock" else 0.0),
                            "p_bin": 15,
                            "phase": "irf",
                            "irf_type": "deviation",
                            "irf_path": path,
                            "irf_trial": trial,
                            "irf_period": t,
                            "shock_magnitude": 4.0,
                            "deviating_agent": 0,
                            "x_agent_0": x0,
                            "profit_agent_0": 5.0 + (1.0 if path == "shock" else 0.0),
                            "n_visits_agent_0": visits,
                            "x_agent_1": x1,
                            "profit_agent_1": 5.0,
                            "n_visits_agent_1": 2,
                        }
                    )
    return pd.DataFrame(rows)


def test_irf_pairs_normalise_by_the_eval_mean() -> None:
    p = _periods_with_irf()
    pairs = viz.irf_pairs(p, "deviation", 0.0, 2)
    assert viz.n_trials(pairs) == 4
    at4 = pairs[pairs["irf_period"] == 4]
    np.testing.assert_allclose(at4["dx_1"], 0.1)  # (11 - 10) / 10
    np.testing.assert_allclose(at4["dprof_0"], 0.2)  # (6 - 5) / 5
    np.testing.assert_allclose(at4["dp"], 0.2)  # 0.1 / 0.5


def test_effective_and_visited_filters_drop_the_right_trials() -> None:
    p = _periods_with_irf()
    pairs = viz.irf_pairs(p, "deviation", 0.0, 2)
    eff = viz.effective_deviation_trials(pairs, pre=3)
    assert viz.n_trials(eff) == 3
    assert not ((eff["session_id"] == 0) & (eff["irf_trial"] == 1)).any()
    vis = viz.visited_trials(eff, pre=3, n_agents=2)
    assert viz.n_trials(vis) == 2
    assert set(zip(vis["session_id"], vis["irf_trial"])) == {(0, 0), (1, 0)}


def test_irf_profile_uses_sessions_as_the_unit() -> None:
    p = _periods_with_irf()
    pairs = viz.irf_pairs(p, "deviation", 0.0, 2)
    prof = viz.irf_profile(pairs, "dx_1", [3, 4])
    assert list(prof["irf_period"]) == [3, 4]
    assert prof.loc[1, "mean"] == pytest.approx(0.1)
    assert prof.loc[1, "n_sessions"] == 2
    assert prof.loc[1, "lo"] <= 0.1 <= prof.loc[1, "hi"]


def test_eval_state_visits_and_policy_by_price_bin() -> None:
    v_grid = np.array([-1.0, 0.0, 1.0])
    rows = []
    # session 0: p_bin 2 then 3, v alternates 1.0 / -1.0
    for k, (pb, v) in enumerate([(2, 1.0), (3, -1.0), (2, 1.0), (3, -1.0)]):
        rows.append(
            {
                "session_id": 0,
                "period": k,
                "v_t": v,
                "p_bin": pb,
                "phase": "eval",
                "irf_type": "",
            }
        )
    visits = viz.eval_state_visits(pd.DataFrame(rows), v_grid, 0.0)
    # states: (2, v=1 -> prev bin 2, v bin 0), (3, 0, 2), (2, 2, 0)
    assert visits["visits"].sum() == 3
    assert set(map(tuple, visits[["p_bin", "v_prev_bin", "v_bin"]].to_numpy())) == {
        (2, 2, 0),
        (3, 0, 2),
    }
    chi = np.array([100.0, 150.0, 200.0])
    table = chi[None, :] * v_grid[:, None]
    greedy = {
        0: [np.full((4, 3, 3), 1), np.full((4, 3, 3), 2)]
    }  # agents play 150 and 200
    pol = viz.policy_by_price_bin(visits, greedy, table, v_grid, 0.0, min_share=0.0)
    assert set(pol["abs_dv"]) == {1.0}
    np.testing.assert_allclose(pol["chi"], 175.0)


def test_macro_names_are_letters_only() -> None:
    # The keys are arbitrary examples with digits, dots and underscores.
    assert viz.macro_name("m1") == "MOne"
    assert viz.macro_name("alpha_het") == "AlphaHet"
    assert viz.macro_name("0.5") == "ZeroFive"
    for name in ("m2_beta5e7", "phase0", "0p125"):
        assert viz.macro_name(name).isalpha()


def test_booktabs_fragment_and_fmt() -> None:
    frag = viz.booktabs_fragment(["A", "B"], [["x", "1.00"]])
    assert frag.startswith("\\begin{tabular}{lr}") and "\\bottomrule" in frag
    assert viz.fmt(1234.5, 1, thousands=True) == "1,234.5"
    assert viz.fmt(float("nan")) == ""
    assert viz.fmt(0.3456, 2) == "0.35"
