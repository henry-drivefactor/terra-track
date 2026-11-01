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
    "belief_uncertainty": ("Total belief uncertainty", "nats (per agent)"),
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



def audit_figure(run: dict, out: Path) -> Path:
    """Effort, naive, corrected, prior and true activity maps side by side."""
    from . import audit
    from .viz import base_map
    res = audit.compute(run["audit"], run["world"]["cell"])
    M = res["maps"]
    ext = run["world"]["extent"]
    vmax = float(np.quantile(np.concatenate([M[k].ravel() for k in ("corrected", "prior", "truth")]), 0.99))
    panels = [
        ("effort", "Observation effort", "looks per cell", None),
        ("naive", "Naive: detection counts", f"shape error vs truth {res['tv_naive']:.2f}", vmax),
        ("corrected", "Effort-corrected", f"shape error vs truth {res['tv_corrected']:.2f}", vmax),
        ("prior", "Rules-based prior only", f"shape error vs truth {res['tv_prior']:.2f}", vmax),
        ("truth", "True vehicle occupancy", "time-averaged", vmax),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.2))
    bg = base_map(run["world"], faint=True)
    from .viz import _belief_cmap
    from matplotlib.colors import PowerNorm
    cmap = _belief_cmap()
    for ax, (key, title, sub, vm) in zip(axes, panels):
        ax.imshow(bg, origin="lower", extent=ext)
        v = vm if vm is not None else float(M[key].max()) or 1.0
        ax.imshow(M[key], origin="lower", extent=ext, cmap=cmap, norm=PowerNorm(0.5, 0, v), interpolation="nearest")
        ax.set_title(title, loc="left", fontsize=10)
        ax.text(0, -0.05, sub, transform=ax.transAxes, fontsize=8.5, color=INK_2, va="top")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Observation-effort bias audit (agent A{run['audit']['agent']}'s records, "
                 f"{res['block_m']:.0f} m blocks). Rank correlation of detections with effort: "
                 f"{res['rho_detections_effort']:.2f}; shape error = total-variation distance, 0 = identical",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


SCALING_LABELS = {
    "belief_uncertainty_mean": ("Belief uncertainty (lower is better)", "nats per agent, mission mean"),
    "belief_uncertainty_final": ("Belief uncertainty at end (lower is better)", "nats per agent"),
    "loc_score_mean": ("Localisation score (higher is better)", "mass on true positions, mission mean"),
    "found_final": ("Targets found by end (higher is better)", "fraction"),
    "track_err_m_mean": ("Track error (lower is better)", "metres"),
    "coverage_final": ("Area covered by end", "fraction"),
    "kbytes_per_agent_min": ("Data per agent", "kB per agent-minute"),
}


def scaling_plot(spec, meta, results, out_dir: Path):
    """Each summary metric against the grid's x value, one line per series (mean and 95% CI)."""
    grid = spec["grid"]
    keys = spec.get("scaling_metrics", ["belief_uncertainty_mean", "loc_score_mean", "found_final", "track_err_m_mean"])
    series = list(grid["series"])
    cols = min(len(keys), 2)
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.6 * rows + 0.8), squeeze=False)
    for ax, key in zip(axes.ravel(), keys):
        for i, sname in enumerate(series):
            xs, ms, cis = [], [], []
            for xv in grid["x"]["values"]:
                vals = np.array([r[2][key] for r in results if meta[r[0]] == (sname, xv)], dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                xs.append(xv)
                ms.append(vals.mean())
                cis.append(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            xs, ms, cis = map(np.array, (xs, ms, cis))
            ax.fill_between(xs, ms - cis, ms + cis, color=_color(i), alpha=0.15, lw=0)
            ax.plot(xs, ms, color=_color(i), lw=2.0, marker="o", ms=6, mec="#fcfcfb", mew=1.5, label=sname)
            if len(xs):
                ax.annotate(sname, (xs[-1], ms[-1]), xytext=(6, 0), textcoords="offset points",
                            fontsize=8.5, color=INK_2, va="center")
        title, unit = SCALING_LABELS.get(key, (key, ""))
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(unit, fontsize=8, color=INK_2)
        ax.set_xlabel(grid["x"].get("label", grid["x"]["key"]), fontsize=8, color=INK_2)
        ax.set_xticks(grid["x"]["values"])
        ax.margins(x=0.12)
        _style(ax)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=len(series), frameon=False, fontsize=9, bbox_to_anchor=(0.01, 1.0))
    fig.suptitle(f"{spec['name']}: mean and 95% CI over {len(spec['seeds'])} seeds", x=0.01, ha="left",
                 y=0.955, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    p = out_dir / "scaling.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return [p]
