"""Thesis figures and table helpers.

Pure helpers (bootstrap CI, IRF pair aggregation, policy aggregation, LaTeX
fragments) and one function per thesis figure. The script
``scripts/make_thesis_outputs.py`` loads the runs, calls these functions and
writes ``outputs/figures/*.pdf`` and ``outputs/tables/*.tex``. Nothing here reads
files or interprets results.

Style (``set_thesis_style``): 15 cm text width, serif 10 pt, greyscale-safe
(black, mid-grey, one accent colour; series also differ by marker and line
style), thin axes without top and right spines, light horizontal grid, vector
PDF. Every Δ^C axis carries the reference lines 0 "Nash", 1 "cartel" and the
grid Nash equilibrium (:func:`add_delta_c_lines`).

IRF conventions follow ``scripts/baseline_report.py``: a trial is the pair of
a shock path and its counterfactual base path under common random numbers;
responses are shock − base, sign-adjusted by sgn(v_t − v̄) and expressed in
percent of the session's eval-window mean (order flow, profit) so that
sessions are comparable. A deviation trial is *effective* when the deviator's
traded quantity differs between the two paths at the shock period (otherwise
the forced shift was clipped at the grid edge). A trial is *visited* when, at
the first post-shock period, both agents had visited the post-shock state in
training (``n_visits_agent_i > 0``); in a never-visited state the greedy
action is the argmax of the initial Q-values, not a learned response.
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

CM = 1.0 / 2.54
TEXT_WIDTH_CM = 15.0
BLACK = "#000000"
GREY = "#808080"
LIGHT_GREY = "#c8c8c8"
ACCENT = "#b2182b"
MARKERS = ("o", "s", "^", "D", "v")
LINESTYLES = ("-", "--", "-.", ":")

IRF_PAIR_KEYS = [
    "session_id",
    "sigma_u",
    "irf_type",
    "irf_trial",
    "irf_period",
    "shock_magnitude",
    "deviating_agent",
]


# ---- style -----------------------------------------------------------------------


def set_thesis_style() -> None:
    """Configure matplotlib once for all thesis figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "TeX Gyre Termes",
                "Times",
                "DejaVu Serif",
            ],
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": LIGHT_GREY,
            "grid.linewidth": 0.4,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "legend.frameon": False,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "pdf.fonttype": 42,
            "savefig.format": "pdf",
            "figure.dpi": 100,
            "mathtext.fontset": "stix",
        }
    )


def new_figure(
    height_cm: float, ncols: int = 1, nrows: int = 1, **kw: Any
) -> tuple[Figure, Any]:
    """Figure of the text width (15 cm) and the given height."""
    return plt.subplots(
        nrows, ncols, figsize=(TEXT_WIDTH_CM * CM, height_cm * CM), **kw
    )


def add_delta_c_lines(
    ax: Axes, grid_nash: float | None, orientation: str = "h", label_x: float = 1.0
) -> None:
    """Reference lines 0 "Nash", 1 "cartel" and the grid Nash Δ^C, labelled directly.

    Args:
        ax: Target axes.
        grid_nash: Δ^C of the grid Nash equilibrium (None to omit).
        orientation: ``"h"`` when Δ^C is on the y-axis, ``"v"`` when on the x-axis.
        label_x: Axes-fraction position of the text labels along the other axis.
    """
    refs = [(0.0, "Nash", BLACK, "-", 0.0), (1.0, "cartel", BLACK, "--", 0.0)]
    if grid_nash is not None and math.isfinite(grid_nash):
        # Offset the label when the grid Nash line is close to the Nash line.
        shift = -7.0 if abs(grid_nash) < 0.08 else 0.0
        refs.append((float(grid_nash), "grid Nash", GREY, ":", shift))
    for value, text, colour, ls, shift in refs:
        if orientation == "h":
            ax.axhline(value, color=colour, linestyle=ls, linewidth=0.7, zorder=1)
            ax.annotate(
                text,
                xy=(label_x, value),
                xycoords=("axes fraction", "data"),
                xytext=(2, shift),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
                color=colour,
                annotation_clip=False,
            )
        else:
            ax.axvline(value, color=colour, linestyle=ls, linewidth=0.7, zorder=1)
            ax.annotate(
                text,
                xy=(value, label_x),
                xycoords=("data", "axes fraction"),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=colour,
                annotation_clip=False,
            )


# ---- pure helpers ----------------------------------------------------------------


def bootstrap_ci(
    values: Any, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean; ``(nan, nan)`` below two values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def eval_baselines(periods: pd.DataFrame, v_bar: float, n_agents: int) -> pd.DataFrame:
    """Per-session eval means of sign-adjusted order flow, profit and price."""
    ev = periods[periods["phase"] == "eval"].copy()
    sgn = np.sign(ev["v_t"].to_numpy() - v_bar)
    agg: dict[str, Any] = {}
    for i in range(n_agents):
        ev[f"xt_{i}"] = ev[f"x_agent_{i}"].to_numpy() * sgn
        agg[f"xt_{i}"] = (f"xt_{i}", "mean")
        agg[f"prof_{i}"] = (f"profit_agent_{i}", "mean")
    ev["pt"] = (ev["p_t"].to_numpy() - v_bar) * sgn
    agg["pt"] = ("pt", "mean")
    return ev.groupby(["session_id", "sigma_u"]).agg(**agg).reset_index()


def irf_pairs(
    periods: pd.DataFrame, irf_type: str, v_bar: float, n_agents: int
) -> pd.DataFrame:
    """Shock/base pairs with normalised responses ``dx_i``, ``dprof_i``, ``dp``.

    ``dx_i`` is (x_shock − x_base)·sgn(v − v̄) divided by the session's eval
    mean of the sign-adjusted order flow; ``dprof_i`` the profit difference
    divided by the eval mean profit; ``dp`` the price difference, sign-adjusted,
    divided by the eval mean of the sign-adjusted price deviation. Values are
    fractions (multiply by 100 for percent).
    """
    irf = periods[(periods["phase"] == "irf") & (periods["irf_type"] == irf_type)]
    if irf.empty:
        return irf.copy()
    s = irf[irf["irf_path"] == "shock"]
    b = irf[irf["irf_path"] == "base"]
    pairs = s.merge(b, on=IRF_PAIR_KEYS, suffixes=("_s", "_b"))
    base = eval_baselines(periods, v_bar, n_agents)
    pairs = pairs.merge(base, on=["session_id", "sigma_u"], how="left")
    sgn = np.sign(pairs["v_t_s"].to_numpy() - v_bar)
    for i in range(n_agents):
        pairs[f"dx_{i}"] = (
            (pairs[f"x_agent_{i}_s"] - pairs[f"x_agent_{i}_b"]) * sgn / pairs[f"xt_{i}"]
        )
        pairs[f"dprof_{i}"] = (
            pairs[f"profit_agent_{i}_s"] - pairs[f"profit_agent_{i}_b"]
        ) / pairs[f"prof_{i}"]
    pairs["dp"] = (pairs["p_t_s"] - pairs["p_t_b"]) * sgn / pairs["pt"]
    return pairs


def _trial_keys() -> list[str]:
    return [k for k in IRF_PAIR_KEYS if k != "irf_period"]


def effective_deviation_trials(pairs: pd.DataFrame, pre: int) -> pd.DataFrame:
    """Keep deviation trials in which the deviator's quantity moved at the shock period."""
    if pairs.empty:
        return pairs
    at = pairs[pairs["irf_period"] == pre]
    keep = []
    for j, blk in at.groupby("deviating_agent"):
        j = int(j)
        moved = blk[blk[f"x_agent_{j}_s"] != blk[f"x_agent_{j}_b"]]
        keep.append(moved[_trial_keys()])
    if not keep:
        return pairs.iloc[0:0]
    return pairs.merge(pd.concat(keep), on=_trial_keys(), how="inner")


def visited_trials(pairs: pd.DataFrame, pre: int, n_agents: int) -> pd.DataFrame:
    """Keep trials whose post-shock state (t = pre + 1) both agents had visited in training."""
    if pairs.empty:
        return pairs
    nxt = pairs[pairs["irf_period"] == pre + 1]
    ok = np.ones(len(nxt), dtype=bool)
    for i in range(n_agents):
        ok &= nxt[f"n_visits_agent_{i}_s"].to_numpy() > 0
    return pairs.merge(nxt.loc[ok, _trial_keys()], on=_trial_keys(), how="inner")


def n_trials(pairs: pd.DataFrame) -> int:
    """Number of distinct trials in a pairs frame."""
    return 0 if pairs.empty else int(pairs[_trial_keys()].drop_duplicates().shape[0])


def irf_profile(pairs: pd.DataFrame, column: str, periods: list[int]) -> pd.DataFrame:
    """Session-mean response per IRF period with a bootstrap 95% band.

    Returns a frame with columns ``irf_period``, ``mean``, ``lo``, ``hi``,
    ``n_sessions``; the band is over session means (sessions are the unit).
    """
    rows = []
    for t in periods:
        blk = pairs[pairs["irf_period"] == t]
        per = blk.groupby("session_id")[column].mean().to_numpy()
        lo, hi = bootstrap_ci(per)
        rows.append(
            {
                "irf_period": t,
                "mean": float(np.mean(per)) if per.size else math.nan,
                "lo": lo,
                "hi": hi,
                "n_sessions": int(per.size),
            }
        )
    return pd.DataFrame(rows)


def eval_state_visits(
    periods: pd.DataFrame, v_grid: np.ndarray, v_bar: float
) -> pd.DataFrame:
    """Visit counts of full states (p_bin, v_prev_bin, v_bin) in the eval rows.

    The state the agents act in at eval row k is (p_bin of row k − 1, value
    bin of row k − 1, value bin of row k) within the same session, so the
    first row of every session is dropped. Returns one row per visited state
    and session with the count.
    """
    ev = periods[(periods["phase"] == "eval") & (periods["irf_type"] == "")]
    ev = ev.sort_values(["session_id", "period"])
    out = []
    v_arr = np.asarray(v_grid, dtype=np.float64)
    for sid, blk in ev.groupby("session_id"):
        v_bin = np.abs(blk["v_t"].to_numpy()[:, None] - v_arr[None, :]).argmin(axis=1)
        p_bin = blk["p_bin"].to_numpy()
        st = pd.DataFrame(
            {
                "session_id": int(sid),
                "p_bin": p_bin[:-1],
                "v_prev_bin": v_bin[:-1],
                "v_bin": v_bin[1:],
            }
        )
        out.append(
            st.groupby(["session_id", "p_bin", "v_prev_bin", "v_bin"])
            .size()
            .rename("visits")
            .reset_index()
        )
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def policy_by_price_bin(
    visits: pd.DataFrame,
    greedy: dict[int, list[np.ndarray]],
    action_table: np.ndarray,
    v_grid: np.ndarray,
    v_bar: float,
    min_share: float = 0.001,
) -> pd.DataFrame:
    """Visit-weighted mean greedy aggressiveness χ per lagged-price bin and |v − v̄|.

    Args:
        visits: Output of :func:`eval_state_visits`.
        greedy: Per session id, the list of greedy-policy arrays
            ``(n_p, n_v, n_v)`` of the agents (from the saved tables).
        action_table: ``(n_v, n_x)`` quantities.
        v_grid: Value grid. ``v_bar``: its mean.
        min_share: On-path threshold as a share of the session's eval visits.

    Returns:
        One row per session, |v − v̄| level and p_bin with ``chi`` (mean over
        agents of the visit-weighted mean χ of the greedy action) and
        ``visits``. States with v = v̄ are skipped (χ undefined there).
    """
    v_arr = np.asarray(v_grid, dtype=np.float64)
    table = np.asarray(action_table, dtype=np.float64)
    rows = []
    for sid, blk in visits.groupby("session_id"):
        if int(sid) not in greedy:
            continue
        total = float(blk["visits"].sum())
        blk = blk[blk["visits"] >= min_share * total]
        pols = greedy[int(sid)]
        for _, r in blk.iterrows():
            p, vp, v = int(r["p_bin"]), int(r["v_prev_bin"]), int(r["v_bin"])
            dv = float(v_arr[v] - v_bar)
            if abs(dv) < 1e-12:
                continue
            chis = [float(table[v, int(pol[p, vp, v])] / dv) for pol in pols]
            rows.append(
                {
                    "session_id": int(sid),
                    "abs_dv": round(abs(dv), 3),
                    "p_bin": p,
                    "visits": float(r["visits"]),
                    "chi_w": float(np.mean(chis)) * float(r["visits"]),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["session_id", "abs_dv", "p_bin", "chi", "visits"])
    df = pd.DataFrame(rows)
    g = df.groupby(["session_id", "abs_dv", "p_bin"]).agg(
        chi_w=("chi_w", "sum"), visits=("visits", "sum")
    )
    g["chi"] = g["chi_w"] / g["visits"]
    return g.drop(columns="chi_w").reset_index()


# ---- LaTeX fragments ---------------------------------------------------------------


def fmt(x: Any, decimals: int = 2, thousands: bool = False) -> str:
    """Number formatting for tables: decimal point, optional thousands comma, '' for NaN."""
    if x is None:
        return ""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(xf):
        return ""
    if thousands:
        return f"{xf:,.{decimals}f}"
    return f"{xf:.{decimals}f}"


def tex_escape(text: str) -> str:
    """Escape the characters LaTeX treats specially in table text."""
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def booktabs_fragment(
    header: list[str], rows: list[list[str]], align: str | None = None
) -> str:
    """A booktabs tabular fragment (no table environment, no caption)."""
    n = len(header)
    align = align or ("l" + "r" * (n - 1))
    lines = [
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def macro_name(*parts: str) -> str:
    """Letters-only macro name from parts (digits and symbols are spelled out or dropped)."""
    digits = {
        "0": "Zero",
        "1": "One",
        "2": "Two",
        "3": "Three",
        "4": "Four",
        "5": "Five",
        "6": "Six",
        "7": "Seven",
        "8": "Eight",
        "9": "Nine",
    }
    out = []
    for part in parts:
        for token in str(part).replace("-", "_").split("_"):
            if not token:
                continue
            word = "".join(digits.get(ch, ch) for ch in token)
            word = "".join(ch for ch in word if ch.isalpha())
            if word:
                out.append(word[0].upper() + word[1:])
    return "".join(out)


# ---- figures -------------------------------------------------------------------------


def _delta_c_axis(ax: Axes, grid_nash: float | None, values: Any) -> None:
    v = np.asarray(values, dtype=np.float64)
    lo = min(-0.1, float(np.nanmin(v)) - 0.05 if v.size else -0.1)
    if grid_nash is not None and math.isfinite(grid_nash):
        lo = min(lo, grid_nash - 0.05)
    hi = max(1.05, float(np.nanmax(v)) + 0.05 if v.size else 1.05)
    ax.set_ylim(lo, hi)


def fig_learning_curves(
    trajectory: pd.DataFrame, sessions: pd.DataFrame, grid_nash: float | None
) -> Figure:
    """F2: Δ^C learning curves, one grey line per session, median in black, convergence markers."""
    fig, ax = new_figure(7.5)
    conv = sessions.set_index("session_id")["converged"].astype(bool)
    stop = sessions.set_index("session_id")["periods_to_convergence"]
    curves = []
    for sid, blk in trajectory.groupby("session_id"):
        blk = blk.sort_values("period")
        x = blk["period"].to_numpy() / 1e6
        y = blk["delta_c"].to_numpy()
        ax.plot(x, y, color=GREY, linewidth=0.5, alpha=0.7, zorder=2)
        curves.append(blk.set_index("period")["delta_c"])
        if int(sid) in conv.index and bool(conv.loc[int(sid)]):
            ax.plot(
                [x[-1]],
                [y[-1]],
                marker="o",
                markersize=3,
                color=ACCENT,
                linestyle="none",
                zorder=4,
            )
    if curves:
        wide = pd.concat(curves, axis=1)
        med = wide.median(axis=1).dropna()
        ax.plot(
            med.index.to_numpy() / 1e6,
            med.to_numpy(),
            color=BLACK,
            linewidth=1.4,
            zorder=3,
        )
    add_delta_c_lines(ax, grid_nash)
    _delta_c_axis(ax, grid_nash, trajectory["delta_c"].to_numpy())
    ax.set_xlabel("Training periods (millions)")
    ax.set_ylabel(r"Collusion capacity $\Delta^C$")
    ax.set_xlim(left=0, right=float(stop.max()) / 1e6 * 1.02)
    handles = [
        plt.Line2D([], [], color=GREY, linewidth=0.5, label="session"),
        plt.Line2D([], [], color=BLACK, linewidth=1.4, label="median across sessions"),
        plt.Line2D(
            [],
            [],
            color=ACCENT,
            marker="o",
            linestyle="none",
            markersize=3,
            label="converged",
        ),
    ]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    return fig


def _strip_box(
    ax: Axes,
    groups: list[tuple[str, np.ndarray, np.ndarray | None]],
    seed: int = 0,
) -> None:
    """Strip plot with box per group; hollow markers where the mask is False."""
    rng = np.random.default_rng(seed)
    positions = np.arange(len(groups))
    data = [g[1][np.isfinite(g[1])] for g in groups]
    ax.boxplot(
        [d if d.size else [np.nan] for d in data],
        positions=positions,
        widths=0.5,
        showfliers=False,
        medianprops={"color": BLACK, "linewidth": 1.0},
        boxprops={"color": GREY, "linewidth": 0.7},
        whiskerprops={"color": GREY, "linewidth": 0.7},
        capprops={"color": GREY, "linewidth": 0.7},
        zorder=2,
    )
    for pos, (_label, values, filled) in zip(positions, groups):
        values = np.asarray(values, dtype=np.float64)
        filled = (
            np.ones(values.shape, dtype=bool)
            if filled is None
            else np.asarray(filled, bool)
        )
        jitter = rng.uniform(-0.15, 0.15, size=values.size)
        ok = np.isfinite(values)
        ax.scatter(
            pos + jitter[ok & filled],
            values[ok & filled],
            s=14,
            facecolor=ACCENT,
            edgecolor=ACCENT,
            linewidth=0.6,
            zorder=3,
        )
        ax.scatter(
            pos + jitter[ok & ~filled],
            values[ok & ~filled],
            s=14,
            facecolor="white",
            edgecolor=ACCENT,
            linewidth=0.6,
            zorder=3,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([g[0] for g in groups])


def fig_delta_c_distribution(
    runs: list[tuple[str, pd.DataFrame]], grid_nash: float | None
) -> Figure:
    """F3: per-session Δ^C by run (strip + box); unsettled sessions as hollow markers."""
    fig, ax = new_figure(7.0)
    groups = []
    for label, s in runs:
        groups.append(
            (
                label,
                s["delta_c"].to_numpy(dtype=float),
                s["settled"].astype(bool).to_numpy(),
            )
        )
    _strip_box(ax, groups)
    add_delta_c_lines(ax, grid_nash)
    all_vals = np.concatenate([g[1] for g in groups]) if groups else np.array([0.0])
    _delta_c_axis(ax, grid_nash, all_vals)
    ax.set_ylabel(r"Collusion capacity $\Delta^C$ per session")
    ax.set_xlabel("")
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=ACCENT,
            markeredgecolor=ACCENT,
            label="settled session",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=ACCENT,
            label="not settled",
        ),
    ]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    return fig


def fig_policy(
    policy: pd.DataFrame,
    chi_n: float,
    chi_m: float,
    price_grid: np.ndarray,
    v_bar: float,
) -> Figure:
    """F4: mean greedy aggressiveness χ by lagged-price bin, one panel per |v − v̄| level.

    Mean across sessions (line with markers) and interquartile band; χ^N and
    χ^M as reference lines. Only price bins visited in at least one session.
    """
    levels = sorted(policy["abs_dv"].unique())
    fig, axes = new_figure(6.5, ncols=max(len(levels), 1), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, lv in zip(axes, levels):
        blk = policy[policy["abs_dv"] == lv]
        g = blk.groupby("p_bin")["chi"]
        stat = pd.DataFrame(
            {
                "mean": g.mean(),
                "q1": g.quantile(0.25),
                "q3": g.quantile(0.75),
                "n": g.count(),
            }
        )
        x = np.asarray(price_grid, dtype=np.float64)[stat.index.to_numpy()] - v_bar
        ax.fill_between(
            x,
            stat["q1"],
            stat["q3"],
            color=LIGHT_GREY,
            alpha=0.8,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            x,
            stat["mean"],
            color=BLACK,
            marker="o",
            markersize=3,
            linewidth=1.0,
            zorder=3,
        )
        ax.axhline(chi_n, color=GREY, linestyle=":", linewidth=0.8)
        ax.axhline(chi_m, color=GREY, linestyle="--", linewidth=0.8)
        ax.annotate(
            r"$\chi^N$",
            xy=(1.0, chi_n),
            xycoords=("axes fraction", "data"),
            xytext=(2, 0),
            textcoords="offset points",
            fontsize=8,
            va="center",
            color=GREY,
            annotation_clip=False,
        )
        ax.annotate(
            r"$\chi^M$",
            xy=(1.0, chi_m),
            xycoords=("axes fraction", "data"),
            xytext=(2, 0),
            textcoords="offset points",
            fontsize=8,
            va="center",
            color=GREY,
            annotation_clip=False,
        )
        ax.set_title(rf"$|v_t - \bar v| = {lv:g}$", fontsize=9)
        ax.set_xlabel(r"Lagged price $p_{t-1} - \bar v$")
    axes[0].set_ylabel(r"Greedy aggressiveness $\hat\chi$")
    handles = [
        plt.Line2D(
            [], [], color=BLACK, marker="o", markersize=3, label="mean across sessions"
        ),
        matplotlib.patches.Patch(color=LIGHT_GREY, label="interquartile range"),
    ]
    axes[-1].legend(handles=handles, loc="lower right")
    fig.tight_layout()
    return fig


def fig_irf_deviation(
    pairs_by_size: dict[float, pd.DataFrame], pre: int, post: int
) -> Figure:
    """F5: deviation IRF, shock − base, t = pre to pre + post − 1, one row per deviation size.

    ``pairs_by_size`` maps the deviation size (bins) to the pairs of the
    kept trials; the deviator's and the non-deviator's columns are relabelled
    ``dx_dev``/``dx_oth`` and ``dprof_dev``/``dprof_oth`` by the caller.
    """
    sizes = sorted(pairs_by_size)
    fig, axes = new_figure(
        4.2 * len(sizes) + 1.0, ncols=2, nrows=len(sizes), squeeze=False
    )
    periods = list(range(pre, pre + post))
    for row, size in enumerate(sizes):
        pairs = pairs_by_size[size]
        for col, (kind, label) in enumerate(
            (
                ("dx", "Order flow (% of eval mean)"),
                ("dprof", "Profit (% of eval mean)"),
            )
        ):
            ax = axes[row, col]
            for who, colour, ls, mk, name in (
                ("dev", BLACK, "-", "o", "deviator"),
                ("oth", ACCENT, "--", "s", "non-deviator"),
            ):
                prof = irf_profile(pairs, f"{kind}_{who}", periods)
                x = prof["irf_period"].to_numpy()
                ax.fill_between(
                    x,
                    100 * prof["lo"],
                    100 * prof["hi"],
                    color=colour,
                    alpha=0.15,
                    linewidth=0,
                )
                ax.plot(
                    x,
                    100 * prof["mean"],
                    color=colour,
                    linestyle=ls,
                    marker=mk,
                    markersize=3,
                    label=name,
                )
            ax.axhline(0.0, color=GREY, linewidth=0.6)
            ax.axvline(pre, color=LIGHT_GREY, linewidth=0.6)
            ax.set_xticks(periods)
            ax.set_xlabel(f"Period $t$ (deviation at $t = {pre}$)")
            ax.set_ylabel(label)
            ax.set_title(
                f"Deviation of {size:g} levels, {n_trials(pairs)} trials", fontsize=9
            )
            if row == 0 and col == 0:
                ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_irf_noise(
    pairs_by_size: dict[float, pd.DataFrame], pre: int, post: int, n_agents: int
) -> Figure:
    """F5b: noise-shock IRF (visited-state trials), one panel per shock size."""
    sizes = sorted(pairs_by_size)
    fig, axes = new_figure(6.0, ncols=max(len(sizes), 1), sharey=True, squeeze=False)
    periods = list(range(pre, pre + post))
    for ax, size in zip(axes[0], sizes):
        pairs = pairs_by_size[size]
        series = [
            (
                f"dx_{i}",
                BLACK if i == 0 else ACCENT,
                LINESTYLES[i % 4],
                MARKERS[i],
                f"order flow agent {i}",
            )
            for i in range(n_agents)
        ]
        series.append(("dp", GREY, ":", "^", "price"))
        for col, colour, ls, mk, name in series:
            prof = irf_profile(pairs, col, periods)
            x = prof["irf_period"].to_numpy()
            ax.fill_between(
                x,
                100 * prof["lo"],
                100 * prof["hi"],
                color=colour,
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(
                x,
                100 * prof["mean"],
                color=colour,
                linestyle=ls,
                marker=mk,
                markersize=3,
                label=name,
            )
        ax.axhline(0.0, color=GREY, linewidth=0.6)
        ax.set_xticks(periods)
        ax.set_xlabel(f"Period $t$ (shock at $t = {pre}$)")
        ax.set_title(f"Shock {size:g} price bins, {n_trials(pairs)} trials", fontsize=9)
    axes[0, 0].set_ylabel("Response (% of eval mean)")
    axes[0, 0].legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_populations(
    groups: list[tuple[str, np.ndarray, np.ndarray | None]], grid_nash: float | None
) -> Figure:
    """F6: Δ^C by population (session means) and per agent for Q–AC; strip + box."""
    fig, ax = new_figure(7.0)
    _strip_box(ax, groups)
    add_delta_c_lines(ax, grid_nash)
    all_vals = (
        np.concatenate([np.asarray(g[1], float) for g in groups])
        if groups
        else np.array([0.0])
    )
    _delta_c_axis(ax, grid_nash, all_vals)
    ax.set_ylabel(r"Collusion capacity $\Delta^C$")
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    return fig


def fig_populations_irf(bars: list[tuple[str, float, float, float, int]]) -> Figure:
    """F7: non-deviator's order-flow response at t = pre + 1 by population.

    ``bars`` holds (label, mean, lo, hi, n_sessions) in percent.
    """
    fig, ax = new_figure(6.0)
    x = np.arange(len(bars))
    means = np.array([b[1] for b in bars])
    lo = np.array([b[2] for b in bars])
    hi = np.array([b[3] for b in bars])
    ax.errorbar(
        x,
        means,
        yerr=np.vstack([means - lo, hi - means]),
        fmt="o",
        color=BLACK,
        ecolor=GREY,
        elinewidth=0.8,
        capsize=3,
        markersize=4,
    )
    ax.axhline(0.0, color=GREY, linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b[0]}\n(n = {b[4]})" for b in bars], fontsize=8)
    ax.set_ylabel("Non-deviator order flow at $t = 4$\n(% of eval mean, shock − base)")
    fig.tight_layout()
    return fig
