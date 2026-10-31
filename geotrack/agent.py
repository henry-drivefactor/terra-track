"""A swarm agent: fixed-wing platform, perfect footprint sensor, private belief, gossip comms."""
from __future__ import annotations

import numpy as np

from .belief import SearchMap
from .comms import PACKET_HEADER_BYTES, STATUS_BYTES, Packet, plan_bytes
from .observations import Detection, Obs, ObsStore
from .tracking import TrackSet


class Agent:
    def __init__(self, aid: int, n_agents: int, cfg: dict, search_model, track_motion, pose):
        a, c = cfg["agents"], cfg["comms"]
        self.id, self.n = aid, n_agents
        self.cfg = cfg
        self.dt = cfg["sim"]["dt_s"]
        self.alive = True
        self.x, self.y, self.h = pose
        self.R = a["footprint_m"]
        self.pd = a["pd"]
        self.rng = np.random.default_rng([cfg["seed"], 100 + aid])
        self.store = ObsStore(n_agents)
        self.search = SearchMap(search_model, self.pd, int(cfg["belief"]["snapshot_keep"]))
        self.tracks = TrackSet(track_motion, cfg, seed=cfg["seed"])
        self.seq = 0

        self.route = None
        self.route_i = 0
        self.plan_step = 0
        self.plan_version = 0
        self.n_plans = 0

        # Gossip state: what this agent knows about teammates, and what it thinks peers know.
        self.team: dict[int, dict] = {aid: {"status": (0, self.x, self.y, self.h, 0)}}
        self.peer_contig: dict[int, np.ndarray] = {}
        self.peer_pv: dict[int, dict] = {}
        self.last_heard: dict[int, int] = {}
        self.sent: dict = {}
        self.new_records: list = []
        self.received_plans: list = []
        self.known_at_plan: dict[int, int] = {}
        self.stale: set[int] = set()
        self.timeout_steps = int(round(c["heartbeat_timeout_s"] / self.dt))
        self.resend_after = int(c["resend_after_steps"])
        self.max_bytes = int(c["max_bytes_per_packet"])
        self.last_event_replan = -10 ** 9

    # ------------------------------------------------------------ flight/sense
    def set_plan(self, route, step: int) -> None:
        self.route, self.route_i, self.plan_step = route, 0, step
        self.plan_version += 1
        self.n_plans += 1
        self.team[self.id]["plan"] = (self.plan_version, step, route)
        self.known_at_plan = {a: tm["plan"][0] for a, tm in self.team.items() if "plan" in tm}

    def move(self) -> None:
        if self.route is not None and self.route_i < len(self.route):
            self.x, self.y, self.h = self.route[self.route_i]
            self.route_i += 1

    def route_exhausted(self) -> bool:
        return self.route is None or self.route_i >= len(self.route)

    def sense(self, step: int, tpos, theads, tspeeds) -> list[int]:
        d = np.hypot(tpos[:, 0] - self.x, tpos[:, 1] - self.y)
        hit = np.flatnonzero((d <= self.R) & (self.rng.random(len(d)) < self.pd))
        dets = tuple(Detection(int(k), float(tpos[k, 0]), float(tpos[k, 1]), float(theads[k]), float(tspeeds[k]))
                     for k in hit)
        obs = Obs(self.id, self.seq, step, float(self.x), float(self.y), self.R, dets)
        self.seq += 1
        self.store.add(obs)
        self.new_records.append(obs)
        self.team[self.id]["status"] = (step, self.x, self.y, self.h, self.plan_version)
        return [int(k) for k in hit]

    # ------------------------------------------------------------------ comms
    def make_packet(self, step: int) -> Packet:
        status = {a: tm["status"] for a, tm in self.team.items()}
        nbytes = PACKET_HEADER_BYTES + 2 * self.n + STATUS_BYTES * len(status)
        peers = [p for p, s in self.last_heard.items() if step - s <= 3]
        plans, plan_cost = {}, []
        for a, tm in self.team.items():
            if "plan" not in tm:
                continue
            v, ps, route = tm["plan"]
            want = [p for p in peers if self.peer_pv.get(p, {}).get(a, 0) < v
                    and step - self.sent.get((p, "p", a, v), -10 ** 9) > self.resend_after]
            if want:
                plans[a] = tm["plan"]
                plan_cost.append((a, v, plan_bytes(len(route) - (step - ps), self.dt), want))
        recs, rec_peers = {}, {}
        for p in peers:
            for r in self.store.missing_for(self.peer_contig[p]):
                if step - self.sent.get((p, r.key), -10 ** 9) > self.resend_after:
                    recs[r.key] = r
                    rec_peers.setdefault(r.key, []).append(p)
        budget = self.max_bytes - nbytes if self.max_bytes > 0 else float("inf")
        for a, v, cost, want in plan_cost:
            if cost > budget:
                del plans[a]
                continue
            budget -= cost
            nbytes += cost
            for p in want:
                self.sent[(p, "p", a, v)] = step
        chosen = []
        for r in sorted(recs.values(), key=lambda r: -r.step):   # newest first
            if r.nbytes > budget:
                break
            budget -= r.nbytes
            nbytes += r.nbytes
            chosen.append(r)
            for p in rec_peers[r.key]:
                self.sent[(p, r.key)] = step
        return Packet(self.id, step, self.store.contiguous.copy(), status, plans, chosen, nbytes)

    def receive(self, pkt: Packet) -> None:
        s = pkt.sender
        self.last_heard[s] = pkt.step
        self.peer_contig[s] = pkt.contiguous
        self.peer_pv[s] = {a: st[4] for a, st in pkt.status.items()}
        for a, st in pkt.status.items():
            if a != self.id and (a not in self.team or self.team[a]["status"][0] < st[0]):
                self.team.setdefault(a, {})["status"] = st
        for a, plan in pkt.plans.items():
            if a != self.id and plan[0] > self.team.get(a, {}).get("plan", (0,))[0]:
                self.team.setdefault(a, {})["plan"] = plan
                self.received_plans.append(a)
        for r in pkt.records:
            if self.store.add(r):
                self.new_records.append(r)

    # ----------------------------------------------------------------- belief
    def update_belief(self, step: int) -> list[int]:
        new, self.new_records = self.new_records, []
        self.search.ingest(new, self.store, step)
        return self.tracks.ingest(new, self.store, step)

    def committed_routes(self, step: int) -> list[np.ndarray]:
        """Remaining committed legs of teammates believed to be alive."""
        out = []
        for a, tm in self.team.items():
            if a == self.id or a in self.stale or "plan" not in tm:
                continue
            _, ps, route = tm["plan"]
            rem = route[step - ps:]
            if len(rem):
                out.append(rem)
        return out

    def check_events(self, step: int, created: list[int], overlap_threshold: float) -> list[str]:
        events = []
        if created:
            events.append(f"new target {created}")
        for a, tm in self.team.items():
            if a == self.id:
                continue
            age = step - tm["status"][0]
            if age > self.timeout_steps and a not in self.stale:
                self.stale.add(a)
                events.append(f"lost contact with agent {a}")
            elif age <= self.timeout_steps and a in self.stale:
                self.stale.discard(a)
        for a in dict.fromkeys(self.received_plans):
            v, ps, route = self.team[a]["plan"]
            if self.known_at_plan.get(a, 0) >= v:
                continue
            # The earlier plan (ties: lower id) has right of way; the later planner yields.
            if (ps < self.plan_step or (ps == self.plan_step and a < self.id)) and self.route is not None:
                if _overlap(self.route[self.route_i:], route[step - ps:], self.R) > overlap_threshold:
                    events.append(f"plan conflict with agent {a}")
        self.received_plans.clear()
        return events


def _overlap(mine, theirs, R) -> float:
    if len(mine) == 0 or len(theirs) == 0:
        return 0.0
    a = {(int(x // R), int(y // R)) for x, y in mine[::2, :2]}
    b = {(int(x // R), int(y // R)) for x, y in theirs[::2, :2]}
    return len(a & b) / len(a)
