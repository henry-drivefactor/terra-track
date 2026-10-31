"""Simulation loop, ground-truth metrics and run recording."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

from . import audit
from .agent import Agent
from .belief import SearchModel
from .comms import Network
from .planner import InfoPlanner, LawnmowerPlanner
from .targets import TargetSim
from .tracking import TrackMotion
from .world import World

LOC_RADIUS_M = 250.0


def _downsample(a: np.ndarray, f: int = 2) -> np.ndarray:
    """Block-mean a (ny, nx) array by f for compact frame storage."""
    ny, nx = a.shape
    if ny % f or nx % f:
        return a
    return a.reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))


def build_planner(world, cfg, n_agents):
    kind = cfg["planner"]["type"]
    if kind == "coordinated":
        return InfoPlanner(world, cfg)
    if kind == "greedy":
        return InfoPlanner(world, cfg, greedy=True)
    if kind == "lawnmower":
        return LawnmowerPlanner(world, cfg, n_agents)
    raise ValueError(f"unknown planner type {kind!r}")


class Simulation:
    def __init__(self, cfg: dict, record_frames: bool = True):
        self.cfg = cfg
        seed = cfg["seed"]
        self.dt = cfg["sim"]["dt_s"]
        self.n_steps = int(round(cfg["sim"]["duration_s"] / self.dt))
        self.world = World(cfg, np.random.default_rng([seed, 1]))
        self.targets = TargetSim(self.world, cfg, np.random.default_rng([seed, 2]))
        self.net = Network(cfg, np.random.default_rng([seed, 3]))
        n_t = len(self.targets.targets)
        terrain = cfg["belief"]["model"] == "terrain"
        self.search_model = SearchModel(self.world, cfg, n_t)
        self.track_motion = TrackMotion(self.world, cfg, terrain)

        n = cfg["agents"]["count"]
        g = self.world.grid
        bx, by = cfg["agents"]["base"][0] * g.size_x, cfg["agents"]["base"][1] * g.size_y
        h0 = np.arctan2(g.size_y / 2 - by, g.size_x / 2 - bx)
        self.agents = []
        for i in range(n):
            off = (i - (n - 1) / 2) * 150.0
            pose = (bx - off * np.sin(h0), by + off * np.cos(h0), h0 + (i - (n - 1) / 2) * 0.3)
            self.agents.append(Agent(i, n, cfg, self.search_model, self.track_motion, pose))
        self.planner = build_planner(self.world, cfg, n)
        p = cfg["planner"]
        self.replan_mode = "open_loop" if p["type"] == "lawnmower" else p["replan"]
        self.leg = max(1, int(round(p["leg_s"] / self.dt)))
        self.min_event_gap = int(round(p["min_event_gap_s"] / self.dt))
        if self.replan_mode != "open_loop":
            for a in self.agents:
                a.commit_steps = int(p["commit_legs"]) * self.leg
        self.losses = {int(round(e["t_s"] / self.dt)): e["agent"] for e in cfg["events"] if e["type"] == "agent_loss"}

        self.record_frames = record_frames and cfg["sim"]["frame_every"] > 0
        self.frames: list[dict] = []
        self.metrics: dict[str, list] = {}
        self.events: list[tuple[float, str]] = []
        self.found_step = np.full(n_t, -1)
        self.last_seen = np.full(n_t, -1)
        self.observed = np.zeros(g.N, bool)
        self.true_occ = np.zeros(g.N)   # vehicle-steps per cell, for the bias audit
        self.replans = 0
        self.event_replans = 0

        # Pre-launch: all agents are together, so they plan in turn with full knowledge.
        for a in self.agents:
            for b in self.agents:
                a.team[b.id] = {"status": (0, b.x, b.y, b.h, 0)}
        for a in self.agents:
            a.set_plan(self.planner.plan(a, 0), 0)
            for b in self.agents:
                if b is not a:
                    b.team[a.id]["plan"] = a.team[a.id]["plan"]
            self.replans += 1

    def log(self, step, text):
        self.events.append((step * self.dt, text))

    # ------------------------------------------------------------------- run
    def run(self, progress: bool = False) -> dict:
        t0 = time.perf_counter()
        for step in range(1, self.n_steps + 1):
            self.step(step)
            if progress and step % 60 == 0:
                print(f"  t={step * self.dt:6.0f}s  found {int((self.found_step >= 0).sum())}/{len(self.found_step)}"
                      f"  ({time.perf_counter() - t0:.1f}s wall)", flush=True)
        self.runtime = time.perf_counter() - t0
        return self.summary()

    def step(self, step: int) -> None:
        now = step * self.dt
        if step in self.losses:
            a = self.agents[self.losses[step]]
            if a.alive:
                a.alive = False
                self.log(step, f"A{a.id} LOST")
        self.targets.step(now - self.dt, self.dt)
        tpos, th, ts = self.targets.positions(), self.targets.headings(), self.targets.speeds()
        np.add.at(self.true_occ, self.world.grid.index(tpos[:, 0], tpos[:, 1]), 1.0)
        self.steps_run = step
        alive = [a for a in self.agents if a.alive]
        for a in alive:
            a.move()
            for k in a.sense(step, tpos, th, ts):
                if self.found_step[k] < 0:
                    self.found_step[k] = step
                    self.log(step, f"A{a.id} found T{k}")
                self.last_seen[k] = step
            self.observed[self.world.grid.disc_cells(a.x, a.y, a.R)] = True
        if self.net.blacked_out(now) and not self.net.blacked_out(now - self.dt):
            self.log(step, "COMMS BLACKOUT")
        if not self.net.blacked_out(now) and self.net.blacked_out(now - self.dt):
            self.log(step, "comms restored")
        links = self.net.exchange(self.agents, [a.make_packet(step) for a in alive], now)
        for a in alive:
            created = a.update_belief(step)
            events = a.check_events(step, created, self.cfg["planner"]["overlap_threshold"])
            reason = None
            if a.route_exhausted():
                reason = "route complete"
            elif self.replan_mode in ("per_leg", "event"):
                stagger = (a.id * self.leg) // len(self.agents) if a.n_plans == 1 else 0
                if step - a.plan_step >= self.leg + stagger:
                    reason = "leg complete"
            if (self.replan_mode == "event" and events and reason is None
                    and step - a.last_event_replan >= self.min_event_gap):
                reason = "; ".join(events)
                a.last_event_replan = step
                self.event_replans += 1
                self.log(step, f"A{a.id} replan: {reason}")
            if reason:
                a.set_plan(self.planner.plan(a, step), step)
                self.replans += 1
        if step % self.cfg["sim"]["metrics_every"] == 0:
            self._metrics(step)
        if self.record_frames and step % self.cfg["sim"]["frame_every"] == 0:
            self._frame(step, tpos, links)

    # --------------------------------------------------------------- metrics
    def _metrics(self, step: int) -> None:
        g = self.world.grid
        tpos = self.targets.positions()
        alive = [a for a in self.agents if a.alive]
        loc, err = [], []
        for k, (tx, ty) in enumerate(tpos):
            near = g.disc_cells(tx, ty, LOC_RADIUS_M)
            lk, ek = [], []
            for a in alive:
                tr = a.tracks.tracks.get(k)
                if tr is not None:
                    d = np.hypot(g.cx[tr.cells] - tx, g.cy[tr.cells] - ty)
                    lk.append(float((d <= LOC_RADIUS_M).mean()))
                    ek.append(float(d.mean()))
                else:
                    lk.append(min(1.0, float(a.search.lam[near].sum())))
            loc.append(np.mean(lk) if lk else 0.0)
            if ek:
                err.append(np.mean(ek))
        div = []
        for i, a in enumerate(alive):
            for b in alive[i + 1:]:
                div.append(0.5 * np.abs(a.search.lam - b.search.lam).sum() / max(a.search.lam.sum(), 1e-9))
        m = {
            "t": step * self.dt,
            "found_frac": float((self.found_step >= 0).mean()),
            "loc_score": float(np.mean(loc)),
            "track_err_m": float(np.mean(err)) if err else np.nan,
            "staleness_s": float(np.mean(np.where(self.last_seen >= 0, (step - self.last_seen) * self.dt, step * self.dt))),
            "coverage": float(self.observed.mean()),
            "expected_undiscovered": float(np.mean([a.search.expected_undiscovered for a in alive])) if alive else np.nan,
            "true_undiscovered": int((self.found_step < 0).sum()),
            "map_divergence": float(np.mean(div)) if div else 0.0,
            "kbytes": self.net.bytes / 1000.0,
            "packets": self.net.packets,
            "replans": self.replans,
            "alive": len(alive),
        }
        for k, v in m.items():
            self.metrics.setdefault(k, []).append(v)

    def _frame(self, step: int, tpos, links) -> None:
        g = self.world.grid
        fr = {
            "t": step * self.dt,
            "agents": np.array([[a.x, a.y, a.h, a.alive] for a in self.agents]),
            "routes": [a.route[a.route_i:a.route_i + 96:4, :2].astype(np.float32) if a.alive and a.route is not None
                       else np.empty((0, 2), np.float32) for a in self.agents],
            "targets": tpos.astype(np.float32),
            "found": self.found_step >= 0,
            "links": list(links),
            "lam": [_downsample(a.search.lam.reshape(g.ny, g.nx)).astype(np.float16) for a in self.agents],
            "tracks": [{tid: tr.xy(g)[:80].astype(np.float32) for tid, tr in a.tracks.tracks.items()}
                       for a in self.agents],
            "metrics": {k: v[-1] for k, v in self.metrics.items()} if self.metrics else {},
        }
        self.frames.append(fr)

    # --------------------------------------------------------------- summary
    def summary(self) -> dict:
        m = {k: np.asarray(v, dtype=float) for k, v in self.metrics.items()}
        n_t = len(self.found_step)
        found_t = np.sort(self.found_step[self.found_step >= 0] * self.dt)

        def t_frac(f):
            k = int(np.ceil(f * n_t))
            return float(found_t[k - 1]) if 0 < k <= len(found_t) else float("nan")
        half = len(m["t"]) // 2
        aud = audit.compute(self.audit_inputs(), self.world.grid.cell)
        return {
            "name": self.cfg["name"],
            "seed": self.cfg["seed"],
            "planner": self.cfg["planner"]["type"],
            "replan": self.replan_mode,
            "belief_model": self.cfg["belief"]["model"],
            "agents": self.cfg["agents"]["count"],
            "targets": n_t,
            "found_final": float(m["found_frac"][-1]),
            "t_found_50_s": t_frac(0.5),
            "t_found_75_s": t_frac(0.75),
            "t_found_all_s": t_frac(1.0),
            "loc_score_mean": float(np.mean(m["loc_score"])),
            "loc_score_2nd_half": float(np.mean(m["loc_score"][half:])),
            "track_err_m_mean": float(np.nanmean(m["track_err_m"])) if np.isfinite(m["track_err_m"]).any() else float("nan"),
            "staleness_s_mean": float(np.mean(m["staleness_s"])),
            "coverage_final": float(m["coverage"][-1]),
            "map_divergence_mean": float(np.mean(m["map_divergence"])),
            "kbytes_total": float(m["kbytes"][-1]),
            "kbytes_per_agent_min": float(m["kbytes"][-1] / self.cfg["agents"]["count"] / (m["t"][-1] / 60.0)),
            "packets_total": int(m["packets"][-1]),
            "replans_total": self.replans,
            "event_replans": self.event_replans,
            "audit_tv_naive": aud["tv_naive"],
            "audit_tv_corrected": aud["tv_corrected"],
            "audit_tv_prior": aud["tv_prior"],
            "audit_rho_detections_effort": aud["rho_detections_effort"],
            "runtime_s": round(getattr(self, "runtime", float("nan")), 2),
        }

    def audit_inputs(self) -> dict:
        """Effort and detections from the best-informed surviving agent, plus truth, for the bias audit."""
        alive = [a for a in self.agents if a.alive] or self.agents
        a = max(alive, key=lambda a: len(a.store.records))
        g = self.world.grid
        return {"agent": a.id, "effort": a.effort.reshape(g.ny, g.nx), "dets": a.det_count.reshape(g.ny, g.nx),
                "true_occ": self.true_occ.reshape(g.ny, g.nx), "steps": getattr(self, "steps_run", 0),
                "prior": self.search_model.prior.reshape(g.ny, g.nx), "n_targets": len(self.found_step)}

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        w = self.world
        payload = {
            "cfg": self.cfg,
            "world": {"elev": w.elev, "water": w.water, "forest": w.forest, "road": w.road,
                      "passability": w.passability, "towns": w.towns, "wetness": w.wetness,
                      "boggy": w.lowland & (w.passability < w.passability_for(0.0)), "exits": w.exits,
                      "extent": w.grid.extent, "cell": w.grid.cell},
            "prior": self.search_model.prior.reshape(w.grid.ny, w.grid.nx),
            "frames": self.frames,
            "metrics": {k: np.asarray(v) for k, v in self.metrics.items()},
            "events": self.events,
            "summary": self.summary(),
            "audit": self.audit_inputs(),
        }
        with open(out / "run.pkl", "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(out / "summary.json", "w") as f:
            json.dump(payload["summary"], f, indent=2)
        keys = list(self.metrics)
        with open(out / "metrics.csv", "w") as f:
            f.write(",".join(keys) + "\n")
            for row in zip(*(self.metrics[k] for k in keys)):
                f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) + "\n")
        return out
