"""Ground-truth vehicles. Their behaviour is rule-based and independent of the belief model."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csgraph

from .world import World, path_from_pred


class Target:
    def __init__(self, tid: int, kind: str, x: float, y: float):
        self.id, self.kind = tid, kind
        self.x, self.y = x, y
        self.heading, self.speed = 0.0, 0.0
        self.path = None           # (n, 2) waypoints
        self.wp = 0                # index of the next waypoint
        self.pause_until = 0.0
        self.speed_factor = 1.0


class TargetSim:
    def __init__(self, world: World, cfg: dict, rng: np.random.Generator):
        self.world, self.rng = world, rng
        self.tc = cfg["targets"]
        g = world.grid
        self._node_cells = [int(g.index(*p)) for p in world.nodes]
        self._pred_cache: dict[int, np.ndarray] = {}
        road_cells = np.flatnonzero(world.road.ravel())
        pas_cells = np.flatnonzero((world.passability.ravel() > 0.3) & ~world.road.ravel())
        self.targets: list[Target] = []
        for kind in ("road", "offroad", "static"):
            pool = road_cells if kind == "road" else pas_cells
            for _ in range(int(self.tc[kind])):
                c = int(rng.choice(pool))
                t = Target(len(self.targets), kind, g.cx[c], g.cy[c])
                t.speed_factor = rng.uniform(0.75, 1.05)
                t.heading = rng.uniform(-np.pi, np.pi)
                # Stagger departures so vehicles are not synchronised.
                t.pause_until = rng.uniform(0, self.tc["pause_s"][1]) if kind != "static" else np.inf
                self.targets.append(t)

    # ------------------------------------------------------------------ routes
    def _road_pred(self, node_cell: int) -> np.ndarray:
        if node_cell not in self._pred_cache:
            _, pred = csgraph.dijkstra(self.world.road_graph, indices=node_cell, return_predecessors=True)
            self._pred_cache[node_cell] = pred
        return self._pred_cache[node_cell]

    def _new_route(self, t: Target) -> None:
        g, rng = self.world.grid, self.rng
        here = int(g.index(t.x, t.y))
        for _ in range(20):
            if t.kind == "road":
                dst = int(rng.choice(self._node_cells))
                path = path_from_pred(self._road_pred(dst), here, dst)
            else:
                cand = int(rng.integers(g.N))
                if self.world.mobility.flat[cand] <= 0:
                    continue
                dst = cand
                _, pred = csgraph.dijkstra(self.world.xc_graph, indices=dst, return_predecessors=True)
                path = path_from_pred(pred, here, dst)
            if path and len(path) > 1 and np.hypot(g.cx[dst] - t.x, g.cy[dst] - t.y) >= self.tc["min_trip_m"]:
                t.path = np.column_stack([g.cx[path], g.cy[path]])
                t.wp = 1
                return
        t.path = None

    # -------------------------------------------------------------------- step
    def step(self, now: float, dt: float) -> None:
        g = self.world.grid
        for t in self.targets:
            if t.kind == "static" or now < t.pause_until:
                t.speed = 0.0
                continue
            if t.path is None:
                self._new_route(t)
                if t.path is None:
                    t.pause_until = now + 60.0
                    continue
            remaining = dt
            while remaining > 1e-6 and t.path is not None:
                c = int(g.index(t.x, t.y))
                v = (self.tc["road_speed"] if t.kind == "road" else max(self.world.mobility.flat[c], 1.0)) * t.speed_factor
                tx, ty = t.path[t.wp]
                d = np.hypot(tx - t.x, ty - t.y)
                step = v * remaining
                if d > 1e-9:
                    t.heading = np.arctan2(ty - t.y, tx - t.x)
                if step < d:
                    t.x += (tx - t.x) * step / d
                    t.y += (ty - t.y) * step / d
                    remaining = 0.0
                else:
                    t.x, t.y = tx, ty
                    remaining -= d / v
                    t.wp += 1
                    if t.wp >= len(t.path):
                        t.path = None
                        lo, hi = self.tc["pause_s"]
                        t.pause_until = now + dt + self.rng.uniform(lo, hi)
                t.speed = v

    def positions(self) -> np.ndarray:
        return np.array([[t.x, t.y] for t in self.targets]).reshape(-1, 2)

    def headings(self) -> np.ndarray:
        return np.array([t.heading for t in self.targets])

    def speeds(self) -> np.ndarray:
        return np.array([t.speed for t in self.targets])
