"""Static comparison plots for sweeps."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .viz import GRID, INK_2, SERIES, plt

METRIC_LABELS = {
    "loc_score": ("Localisation score", "mass on true positions (0-1)"),
    "found_frac": ("Targets found", "fraction"),
    "track_err_m": ("Mean track error", "metres"),
    "staleness_s": ("Time since target last seen", "seconds"),
    "coverage": ("Area covered", "fraction"),
    "map_divergence": ("Belief divergence between agents", "0 = identical maps"),
    "kbytes": ("Data sent, all agents", "kB"),
    "replans": ("Cumulative replans", "count"),
    "alive": ("Agents alive", "count"),
}

DOT_KEYS = [
    ("loc_score_mean", "Localisation score (higher is better)"),
    ("track_err_m_mean", "Track error, m (lower is better)"),
    ("found_final", "Targets found by end (higher is better)"),
    ("kbytes_per_agent_min", "kB per agent-minute"),
]


def _color(i):
    return SERIES[i] if i < len(SERIES) else INK_2


def _style(ax):
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.tick_params(labelsize=8)


def sweep_plots(spec, variants, results, out_dir: Path):
    out = []
    keys = spec.get("plots", ["loc_score", "found_frac", "track_err_m", "kbytes"])
    n = len(keys)
    cols = min(n, 2)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.4 * rows + 0.8), squeeze=False)
    for ax, key in zip(axes.ravel(), keys):
        title, unit = METRIC_LABELS.get(key, (key, ""))
        for i, v in enumerate(variants):
            runs = [r[3] for r in results if r[0] == v]
            t = np.asarray(runs[0]["t"]) / 60.0
            Y = np.array([r[key] for r in runs], dtype=float)
            with np.errstate(all="ignore"):
                mean = np.nanmean(Y, axis=0)
                se = np.nanstd(Y, axis=0, ddof=1) / np.sqrt(np.isfinite(Y).sum(0).clip(1)) if len(Y) > 1 else 0 * mean
            ax.fill_between(t, mean - se, mean + se, color=_color(i), alpha=0.15, lw=0)
            ax.plot(t, mean, color=_color(i), lw=2.0, label=v)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(unit, fontsize=8, color=INK_2)
        ax.set_xlabel("mission time (minutes)", fontsize=8, color=INK_2)
        _style(ax)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=min(len(variants), 4), frameon=False, fontsize=9,
               bbox_to_anchor=(0.01, 1.0))
    fig.suptitle(f"{spec['name']}: mean ± 1 s.e. over {len(spec['seeds'])} seeds", x=0.01, ha="left",
                 y=0.955, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    p = out_dir / "timeseries.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    out.append(p)

    fig, axes = plt.subplots(1, len(DOT_KEYS), figsize=(3.6 * len(DOT_KEYS), 0.55 * len(variants) + 1.6), sharey=True)
    y = np.arange(len(variants))[::-1]
    for ax, (key, title) in zip(axes, DOT_KEYS):
        for i, v in enumerate(variants):
            x = np.array([r[2][key] for r in results if r[0] == v], dtype=float)
            x = x[np.isfinite(x)]
            if len(x) == 0:
                continue
            ci = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
            ax.scatter(x, np.full(len(x), y[i]) + 0.18, s=10, color=_color(i), alpha=0.35, lw=0)
            ax.errorbar(x.mean(), y[i], xerr=ci, fmt="o", ms=8, color=_color(i), ecolor=_color(i),
                        elinewidth=2, capsize=0, mec="#fcfcfb", mew=1.5)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(True, axis="x", color=GRID, lw=0.6)
        ax.tick_params(labelsize=8)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(variants, fontsize=9, color=INK_2)
    axes[0].tick_params(axis="y", length=0)
    fig.suptitle(f"{spec['name']}: mean and 95% CI (small dots: individual seeds)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "summary.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    out.append(p)
    return out

