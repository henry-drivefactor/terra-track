"""Observation records: the only thing agents exchange about what they have seen."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Bytes on the wire, used for bandwidth accounting.
OBS_HEADER_BYTES = 16   # origin, seq, step, x, y, radius
DET_BYTES = 12          # target id, x, y, heading, speed


@dataclass(frozen=True)
class Detection:
    tid: int
    x: float
    y: float
    heading: float
    speed: float


@dataclass(frozen=True)
class Obs:
    origin: int
    seq: int
    step: int
    x: float
    y: float
    radius: float
    dets: tuple = ()

    @property
    def key(self):
        return (self.origin, self.seq)

    @property
    def nbytes(self) -> int:
        return OBS_HEADER_BYTES + DET_BYTES * len(self.dets)


@dataclass
class ObsStore:
    """Per-agent set of known observation records, indexed by step."""
    n_agents: int
    records: dict = field(default_factory=dict)
    by_step: dict = field(default_factory=dict)
    contiguous: np.ndarray = None    # per origin: count of records 0..k-1 all present

    def __post_init__(self):
        self.contiguous = np.zeros(self.n_agents, dtype=np.int64)
        self.max_seq = np.full(self.n_agents, -1, dtype=np.int64)

    def add(self, obs: Obs) -> bool:
        if obs.key in self.records:
            return False
        self.records[obs.key] = obs
        self.by_step.setdefault(obs.step, []).append(obs)
        o = obs.origin
        self.max_seq[o] = max(self.max_seq[o], obs.seq)
        while (o, int(self.contiguous[o])) in self.records:
            self.contiguous[o] += 1
        return True

    def missing_for(self, peer_contiguous: np.ndarray):
        """Records this store holds beyond a peer's advertised contiguous prefix.

        A peer with gaps may be offered records it already has; receivers de-duplicate.
        """
        out = []
        for o in range(self.n_agents):
            for s in range(int(peer_contiguous[o]), int(self.max_seq[o]) + 1):
                r = self.records.get((o, s))
                if r is not None:
                    out.append(r)
        return out
