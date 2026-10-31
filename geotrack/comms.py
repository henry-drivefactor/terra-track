"""Lossy, range-limited broadcast mesh with byte accounting."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PACKET_HEADER_BYTES = 8
STATUS_BYTES = 16        # per agent: step, x, y, heading, plan version
PLAN_HEADER_BYTES = 12
PLAN_WAYPOINT_BYTES = 8  # committed routes are sent as a waypoint every 30 s


def plan_bytes(n_steps: int, dt: float) -> int:
    return PLAN_HEADER_BYTES + PLAN_WAYPOINT_BYTES * int(np.ceil(n_steps * dt / 30.0))


@dataclass
class Packet:
    sender: int
    step: int
    contiguous: np.ndarray
    status: dict                 # agent -> (step, x, y, h, plan_version)
    plans: dict = field(default_factory=dict)    # agent -> (version, plan_step, route)
    records: list = field(default_factory=list)
    nbytes: int = 0


class Network:
    def __init__(self, cfg: dict, rng: np.random.Generator):
        c = cfg["comms"]
        self.range = c["range_m"]
        self.loss = c["loss"]
        self.rng = rng
        self.dt = cfg["sim"]["dt_s"]
        self.blackouts = [(e["t_s"], e["until_s"]) for e in cfg["events"] if e["type"] == "blackout"]
        self.bytes = 0
        self.packets = 0
        self.deliveries = 0

    def blacked_out(self, now: float) -> bool:
        return any(a <= now < b for a, b in self.blackouts)

    def exchange(self, agents, packets, now: float):
        """Broadcast each packet to every alive agent in range; returns the in-range link list."""
        alive = [a for a in agents if a.alive]
        links = []
        pos = {a.id: (a.x, a.y) for a in alive}
        dark = self.blacked_out(now)
        for i, a in enumerate(alive):
            for b in alive[i + 1:]:
                if np.hypot(pos[a.id][0] - pos[b.id][0], pos[a.id][1] - pos[b.id][1]) <= self.range and not dark:
                    links.append((a.id, b.id))
        nbrs = {a.id: [] for a in alive}
        for i, j in links:
            nbrs[i].append(j)
            nbrs[j].append(i)
        by_id = {a.id: a for a in alive}
        for p in packets:
            self.bytes += p.nbytes
            self.packets += 1
            for j in nbrs.get(p.sender, []):
                if self.rng.random() >= self.loss:
                    by_id[j].receive(p)
                    self.deliveries += 1
        return links
