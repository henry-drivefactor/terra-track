"""Terrain-constrained tracking of discovered targets with grid particle filters.

Each particle is (cell, heading, speed factor). Motion is a correlated random walk
on the 8-connected grid: particles prefer to keep their heading, move at the
ground speed of the cell they are in, stay on roads when on a road, and can never
enter water or impassable ground. Between sightings the cloud therefore spreads
along roads and through passes rather than in circles. Footprints that saw
nothing prune particles (negative information), which resolves branches - "we
looked down the north road and it was empty, so it went south".

All randomness is seeded from (target, sighting step, step), so agents holding the
same records hold the same particle clouds, whatever order the records arrived in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world import DIR_LEN, World


class TrackMotion:
    def __init__(self, world: World, cfg: dict, terrain: bool):
        b = cfg["belief"]
        g = self.grid = world.grid
        self.dt = cfg["sim"]["dt_s"]
        if terrain:
            mob = world.mobility.ravel().astype(float)
            road = world.road.ravel()
            self.prior = world.prior_weights(b)
        else:
            mob = np.full(g.N, float(b["isotropic_speed"]))
            road = np.zeros(g.N, bool)
            self.prior = np.ones(g.N)
        self.speed = mob
        self.max_speed = float(mob.max())
        valid = g.nbr >= 0
        nb = np.where(valid, g.nbr, 0)
        base = np.where(valid, mob[nb], 0.0) / self.max_speed
        if terrain:
            on, nroad = road[:, None], np.where(valid, road[nb], False)
            base = base * np.where(on & ~nroad, 0.03, 1.0) * np.where(on & nroad, 4.0, 1.0)
        self.base = base
        ang = np.arange(8) * np.pi / 4
        self.turn = np.exp(b["turn_concentration"] * np.cos(ang[None, :] - ang[:, None]))  # [heading, dir]
        self.n_sub = max(1, int(np.ceil(self.max_speed * self.dt / g.cell)))
        self.p_scale = self.dt / (g.cell * self.n_sub)

    def step(self, cells, heads, sf, rng):
        nbr = self.grid.nbr
        n = len(cells)
        for _ in range(self.n_sub):
            w = self.base[cells] * self.turn[heads]
            tot = w.sum(1)
            u = rng.random(n) * tot
            k = np.minimum((np.cumsum(w, 1) < u[:, None]).sum(1), 7)
            p = self.speed[cells] * sf * self.p_scale / DIR_LEN[k]
            move = (rng.random(n) < p) & (tot > 0)
            cells = np.where(move, nbr[cells, k], cells)
            heads = np.where(move, k, heads)
        return cells, heads


@dataclass
class Sighting:
    step: int
    x: float
    y: float
    heading: float
    speed: float


class Track:
    def __init__(self, tid: int):
        self.tid = tid
        self.sight: Sighting | None = None
        self.cells = self.heads = self.sf = None
        self.t = -1
        self.snaps: dict = {}
        self.lost = False
        self.reset = False
        self.dirty_from: int | None = None

    def xy(self, grid):
        return np.column_stack([grid.cx[self.cells], grid.cy[self.cells]])

    def entropy(self) -> float:
        _, counts = np.unique(self.cells, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log(p)).sum())


class TrackSet:
    def __init__(self, motion: TrackMotion, cfg: dict, seed: int):
        b = cfg["belief"]
        self.motion, self.grid = motion, motion.grid
        self.P = int(b["particles"])
        self.stopped_fraction = b["stopped_fraction"]
        self.keep = int(b["snapshot_keep"])
        self.pd = cfg["agents"]["pd"]
        self.seed = seed
        self.tracks: dict[int, Track] = {}
        self._masks: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ public
    def ingest(self, new_records, store, t_now: int) -> list[int]:
        """Fold new records in and advance every track to t_now. Returns ids of new tracks."""
        created = []
        for o in sorted(new_records, key=lambda o: o.step):
            for d in o.dets:
                tr = self.tracks.get(d.tid)
                if tr is None:
                    tr = self.tracks[d.tid] = Track(d.tid)
                    created.append(d.tid)
                if tr.sight is None or o.step > tr.sight.step:
                    tr.sight = Sighting(o.step, d.x, d.y, d.heading, d.speed)
                    tr.reset = True
        for tr in self.tracks.values():
            if tr.reset:
                continue
            late = [o.step for o in new_records if tr.sight.step < o.step <= tr.t]
            if late:
                tr.dirty_from = min(late) if tr.dirty_from is None else min(tr.dirty_from, min(late))
        self._masks.clear()
        for tr in self.tracks.values():
            self._advance(tr, store, t_now)
        return created

    def forecast(self, tr: Track, n_steps: int, stride: int, n_particles: int, seed) -> np.ndarray:
        """Unpruned forward prediction: (n_particles, n_steps // stride, 2) positions."""
        rng = np.random.default_rng(seed)
        cells, heads, sf = tr.cells[:n_particles], tr.heads[:n_particles], tr.sf[:n_particles]
        out = np.empty((len(cells), n_steps // stride, 2))
        for s in range(n_steps // stride * stride):
            cells, heads = self.motion.step(cells, heads, sf, rng)
            if (s + 1) % stride == 0:
                out[:, s // stride, 0] = self.grid.cx[cells]
                out[:, s // stride, 1] = self.grid.cy[cells]
        return out

    # ---------------------------------------------------------------- internals
    def _init_from_sighting(self, tr: Track) -> None:
        s, g, m = tr.sight, self.grid, self.motion
        rng = np.random.default_rng([self.seed, tr.tid, s.step, 0])
        c = int(g.index(s.x, s.y))
        tr.cells = np.full(self.P, c, dtype=np.int64)
        h = int(np.round(s.heading / (np.pi / 4))) % 8
        tr.heads = (h + rng.choice([-1, 0, 0, 1], self.P)) % 8
        moving = s.speed > 0.5
        base_sf = s.speed / max(m.speed[c], 0.5) if moving else 0.7
        tr.sf = np.clip(base_sf * rng.uniform(0.8, 1.2, self.P), 0.0, 1.5)
        stopped = rng.random(self.P) < (self.stopped_fraction if moving else 0.6)
        tr.sf[stopped] = 0.0
        tr.t = s.step
        tr.snaps = {s.step: (tr.cells, tr.heads, tr.sf)}
        tr.lost = False

    def _mask(self, store, step: int):
        if step not in self._masks:
            fps = store.by_step.get(step, ())
            if not fps:
                self._masks[step] = None
            else:
                mask = np.zeros(self.grid.N, bool)
                for o in fps:
                    mask[self.grid.disc_cells(o.x, o.y, o.radius)] = True
                self._masks[step] = mask
        return self._masks[step]

    def _advance(self, tr: Track, store, t_now: int) -> None:
        if tr.reset:
            self._init_from_sighting(tr)
            tr.reset, tr.dirty_from = False, None
        elif tr.dirty_from is not None:
            base = tr.dirty_from - 1
            if base in tr.snaps:
                tr.cells, tr.heads, tr.sf = tr.snaps[base]
                tr.snaps = {s: v for s, v in tr.snaps.items() if s <= base}
                tr.t = base
            else:
                self._init_from_sighting(tr)
            tr.dirty_from = None
        while tr.t < t_now:
            tr.t += 1
            self._step(tr, store)
            tr.snaps[tr.t] = (tr.cells, tr.heads, tr.sf)
        for s in [s for s in tr.snaps if s < tr.t - self.keep]:
            del tr.snaps[s]

    def _step(self, tr: Track, store) -> None:
        rng = np.random.default_rng([self.seed, tr.tid, tr.sight.step, tr.t])
        cells, heads = self.motion.step(tr.cells, tr.heads, tr.sf, rng)
        sf = tr.sf
        mask = self._mask(store, tr.t)
        if mask is not None:
            alive = ~mask[cells]
            if self.pd < 1.0:
                alive |= rng.random(len(cells)) >= self.pd
            if not alive.all():
                if alive.any():
                    idx = rng.choice(np.flatnonzero(alive), self.P)
                    cells, heads, sf = cells[idx], heads[idx], sf[idx]
                else:
                    cells, heads, sf = self._reseed(tr, mask, rng)
        tr.cells, tr.heads, tr.sf = cells, heads, sf

    def _reseed(self, tr: Track, mask, rng):
        """Every hypothesis was ruled out: spread over unobserved reachable ground."""
        g, m = self.grid, self.motion
        reach = m.max_speed * (tr.t - tr.sight.step) * m.dt
        cand = g.disc_cells(tr.sight.x, tr.sight.y, max(reach, 2 * g.cell))
        cand = cand[(m.speed[cand] > 0) & ~mask[cand]]
        if cand.size == 0:
            cand = np.flatnonzero((m.speed > 0) & ~mask)
        w = m.prior[cand] + 1e-12
        cells = rng.choice(cand, self.P, p=w / w.sum())
        tr.lost = True
        return cells, rng.integers(0, 8, self.P), np.clip(rng.uniform(0.3, 1.1, self.P), 0, 1.5)
