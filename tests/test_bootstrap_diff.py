"""Tests for the difference CI helper (scripts/make_thesis_outputs.py: bootstrap_diff_ci)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_thesis_outputs.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("make_thesis_outputs", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_thesis_outputs"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_bootstrap_diff_ci_point_estimate_and_bracket() -> None:
    mod = _load_script()
    rng = np.random.default_rng(1)
    a = rng.normal(0.34, 0.05, size=30)
    b = rng.normal(0.14, 0.05, size=30)
    diff, lo, hi = mod.bootstrap_diff_ci(a, b)
    assert diff == a.mean() - b.mean()
    assert lo < diff < hi
    assert lo > 0.0  # a clear difference excludes zero
    # Normal approximation for independent samples: half-width about 1.96 * SE.
    se = math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    assert abs((hi - lo) / 2 - 1.96 * se) < 0.3 * 1.96 * se


def test_bootstrap_diff_ci_is_deterministic_and_antisymmetric() -> None:
    mod = _load_script()
    a = np.array([0.1, 0.2, 0.3, 0.4])
    b = np.array([0.0, 0.1, 0.1, 0.2, 0.3])
    assert mod.bootstrap_diff_ci(a, b) == mod.bootstrap_diff_ci(a, b)
    d1, lo1, hi1 = mod.bootstrap_diff_ci(a, a)
    assert d1 == 0.0 and lo1 < 0.0 < hi1


def test_bootstrap_diff_ci_needs_two_values_per_sample() -> None:
    mod = _load_script()
    diff, lo, hi = mod.bootstrap_diff_ci([0.5], [0.1, 0.2, np.nan])
    assert math.isclose(diff, 0.35)
    assert math.isnan(lo) and math.isnan(hi)
