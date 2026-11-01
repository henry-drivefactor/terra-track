"""Scenario configuration: defaults, YAML loading and dotted overrides."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "name": "unnamed",
    "seed": 0,
    "sim": {
        "dt_s": 5.0,
        "duration_s": 2400.0,
        "metrics_every": 6,      # steps between metric samples
        "frame_every": 2,        # steps between recorded animation frames (0 = no frames)
    },
    "world": {
        "size_m": 10000.0,
        "cell_m": 50.0,
        "relief_m": 300.0,
        "n_rivers": 1,
        "forest_fraction": 0.25,
        "forest_factor": 0.45,   # off-road passability multiplier in forest
        "max_slope": 0.45,       # slope (rise/run) at which ground becomes impassable
        "n_towns": 7,
        "n_exits": 3,
        "road_loops": 0.35,      # fraction of extra (non-tree) Delaunay edges turned into roads
        "ridge_walls": [],
        "wetness": 0.0,          # ground wetness, 0 = dry .. 1 = saturated (off-road only)       # optional [{pos, width, height_m, gaps: [..], gap_width}]
    },
    "targets": {
        "road": 6,               # vehicles that drive between road nodes
        "offroad": 2,            # vehicles that take least-time cross-country routes
        "static": 1,             # parked vehicles
        "road_speed": 12.0,      # m/s
        "offroad_speed": 5.0,    # m/s on perfectly passable ground
        "pause_s": [30.0, 240.0],
        "min_trip_m": 2500.0,
    },
    "agents": {
        "count": 4,
        "speed": 25.0,           # m/s, constant airspeed
        "min_turn_radius": 150.0,
        "footprint_m": 250.0,    # sensor ground footprint radius
        "pd": 1.0,               # detection probability inside the footprint
        "base": [0.08, 0.08],    # launch point as a fraction of the area
    },
    "comms": {
        "range_m": 4000.0,
        "loss": 0.1,             # per-packet, per-link drop probability
        "heartbeat_timeout_s": 60.0,
        "max_bytes_per_packet": 0,  # 0 = unlimited
        "resend_after_steps": 3,
    },
    "belief": {
        "model": "terrain",      # terrain | isotropic
        "expected_targets": None,  # None = true target count
        "assumed_wetness": None,   # wetness the belief assumes; None = the true value
        "road_weight": 1.0,
        "offroad_weight": 0.04,
        "persistence_s": 20.0,   # heading persistence used to set belief spread rate
        "isotropic_speed": 7.0,  # m/s assumed by the terrain-agnostic model
        "snapshot_every": 6,
        "snapshot_keep": 40,
        "particles": 600,
        "turn_concentration": 2.5,
        "stopped_fraction": 0.1,
    },
    "planner": {
        "type": "coordinated",   # coordinated | greedy | lawnmower
        "replan": "event",       # open_loop | per_leg | event
        "horizon_s": 300.0,
        "leg_s": 60.0,
        "commit_legs": 5,        # how much of a route teammates treat as committed (open_loop: all of it)
        "objective": "entropy",  # entropy (reduce total belief uncertainty) | detections
        "w_search": 1.0,
        "w_track": 1.0,
        "discount_per_min": 0.9,
        "n_goals": 6,
        "plan_particles": 96,
        "eval_stride": 2,
        "overlap_threshold": 0.25,
        "min_event_gap_s": 15.0,
    },
    "events": [],                # [{type: agent_loss, agent: 2, t_s: 900}, {type: blackout, t_s: .., until_s: ..}]
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, key: str, value) -> None:
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    if isinstance(value, dict) and isinstance(d.get(parts[-1]), dict):
        d[parts[-1]] = deep_merge(d[parts[-1]], value)
    else:
        d[parts[-1]] = value


def load_scenario(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    for k, v in (overrides or {}).items():
        set_dotted(cfg, k, v)
    return cfg


def parse_override(text: str) -> tuple[str, object]:
    """'planner.type=greedy' -> ('planner.type', 'greedy') with YAML value parsing."""
    key, _, raw = text.partition("=")
    return key.strip(), yaml.safe_load(raw)
