"""Observation-effort bias audit.

Agents do not look everywhere equally: they loiter over tracks and favour roads.
A map built from detection counts alone therefore shows where the swarm looked as
much as where vehicles were. Because every record carries its footprint, effort
is known exactly, and activity can be estimated as detections per look:

    naive      = detections, rescaled to the expected vehicle count
    corrected  = (detections + k * prior) / (effort + k)     vehicles per cell-look,
                 shrunk towards the prior where effort is thin
    truth      = time-averaged true vehicle occupancy

All three are in vehicles per cell, aggregated to blocks, and compared with the
truth by total-variation distance of their spatial shape (0 = same, 1 = disjoint).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

BLOCK_M = 500.0
SHRINK_LOOKS = 8.0   # prior weight, in full looks at a block (best of 0.25-32 on S1 seeds)


def _blocks(a, f):
    ny, nx = a.shape
    return a[: ny // f * f, : nx // f * f].reshape(ny // f, f, nx // f, f).sum(axis=(1, 3))


def _tv(p, q):
    p, q = p / max(p.sum(), 1e-12), q / max(q.sum(), 1e-12)
    return float(0.5 * np.abs(p - q).sum())


def compute(inp: dict, cell: float, block_m: float = BLOCK_M) -> dict:
    f = max(1, int(round(block_m / cell)))
    area = f * f
    E, D = _blocks(inp["effort"], f), _blocks(inp["dets"], f)
    T = _blocks(inp["true_occ"], f) / max(inp["steps"], 1) / area
    P = _blocks(inp["prior"], f) / area
    n = inp["n_targets"]
    naive = D / D.sum() * n / area if D.sum() > 0 else np.zeros_like(D)
    k = SHRINK_LOOKS * area
    corrected = (D + k * P) / (E + k)
    rho = lambda a, b: float(spearmanr(a.ravel(), b.ravel()).statistic) if a.std() > 0 and b.std() > 0 else float("nan")
    return {
        "maps": {"effort": E / area, "naive": naive, "corrected": corrected, "prior": P, "truth": T},
        "tv_naive": _tv(naive, T),
        "tv_corrected": _tv(corrected, T),
        "tv_prior": _tv(P, T),
        "rho_detections_effort": rho(D, E),
        "rho_truth_effort": rho(T, E),
        "block_m": block_m,
    }
