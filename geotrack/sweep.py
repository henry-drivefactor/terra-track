"""Monte Carlo sweeps: variants x seeds, run in parallel, summarised as CSV, markdown and plots."""
from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from .config import load_scenario

SUMMARY_KEYS = [
    ("loc_score_mean", "localisation score (mean)", "{:.3f}"),
    ("belief_uncertainty_mean", "belief uncertainty, nats (mean)", "{:.0f}"),
    ("found_final", "targets found (final)", "{:.2f}"),
    ("t_found_75_s", "time to find 75% (s)", "{:.0f}"),
    ("track_err_m_mean", "track error (m)", "{:.0f}"),
    ("staleness_s_mean", "time since last seen (s)", "{:.0f}"),
    ("coverage_final", "area covered (final)", "{:.2f}"),
    ("map_divergence_mean", "belief divergence", "{:.3f}"),
    ("kbytes_per_agent_min", "kB / agent / min", "{:.2f}"),
    ("replans_total", "replans", "{:.0f}"),
    ("audit_tv_naive", "activity map error, naive", "{:.3f}"),
    ("audit_tv_corrected", "activity map error, effort-corrected", "{:.3f}"),
]


def _job(args):
    from .sim import Simulation
    scenario, overrides, variant, seed = args
    cfg = load_scenario(scenario, {**overrides, "seed": seed, "sim.frame_every": 0})
    sim = Simulation(cfg, record_frames=False)
    summary = sim.run()
    return variant, seed, summary, {k: list(map(float, v)) for k, v in sim.metrics.items()}


def run_sweep(path: str, workers: int = 0, out: str | None = None) -> Path:
    with open(path) as f:
        spec = yaml.safe_load(f)
    name = spec["name"]
    out_dir = Path(out or f"runs/sweeps/{name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = spec.get("overrides", {})
    grid = spec.get("grid")
    if grid:
        # Cross product: one variant per (series, x value).
        xk = grid["x"]["key"]
        variants, meta = {}, {}
        for sname, sov in grid["series"].items():
            for xv in grid["x"]["values"]:
                v = f"{sname} | {xv}"
                variants[v] = {**(sov or {}), xk: xv}
                meta[v] = (sname, xv)
    else:
        variants = spec["variants"]
    seeds = spec["seeds"]
    jobs = [(spec["scenario"], {**base, **(ov or {})}, v, s) for v, ov in variants.items() for s in seeds]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"sweep {name}: {len(variants)} variants x {len(seeds)} seeds = {len(jobs)} runs on {workers} workers")
    results = []
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            v, s, summ, _ = results[-1]
            print(f"  [{i}/{len(jobs)}] {v} seed {s}: loc {summ['loc_score_mean']:.3f} found {summ['found_final']:.2f}",
                  flush=True)
    order = {v: i for i, v in enumerate(variants)}
    results.sort(key=lambda r: (order[r[0]], r[1]))

    with open(out_dir / "results.csv", "w", newline="") as f:
        keys = list(results[0][2])
        w = csv.writer(f)
        w.writerow(["variant"] + keys)
        for v, _, summ, _ in results:
            w.writerow([v] + [summ[k] for k in keys])
    with open(out_dir / "timeseries.csv", "w", newline="") as f:
        mkeys = list(results[0][3])
        w = csv.writer(f)
        w.writerow(["variant", "seed"] + mkeys)
        for v, s, _, m in results:
            for row in zip(*(m[k] for k in mkeys)):
                w.writerow([v, s] + list(row))

    table = _markdown(spec, variants, results)
    (out_dir / "summary.md").write_text(table)
    print("\n" + table)
    from .plots import scaling_plot, sweep_plots
    plots = scaling_plot(spec, meta, results, out_dir) if grid else sweep_plots(spec, list(variants), results, out_dir)
    for p in plots:
        print(f"plot -> {p}")
    return out_dir


def _markdown(spec, variants, results) -> str:
    lines = [f"# {spec['name']}", "", spec.get("description", "").strip(), "",
             f"Scenario `{spec['scenario']}`, {len(spec['seeds'])} seeds per variant. "
             "Mean ± standard error over seeds.", ""]
    hdr = "| variant | " + " | ".join(lbl for _, lbl, _ in SUMMARY_KEYS) + " |"
    lines += [hdr, "|" + "---|" * (len(SUMMARY_KEYS) + 1)]
    for v in variants:
        rows = [r[2] for r in results if r[0] == v]
        cells = []
        for k, _, fmt in SUMMARY_KEYS:
            x = np.array([r[k] for r in rows], dtype=float)
            ok = x[np.isfinite(x)]
            if len(ok) == 0:
                cells.append("–")
                continue
            se = ok.std(ddof=1) / np.sqrt(len(ok)) if len(ok) > 1 else 0.0
            miss = f" ({len(x) - len(ok)} n/a)" if len(ok) < len(x) else ""
            cells.append(f"{fmt.format(ok.mean())} ± {fmt.format(se)}{miss}")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
