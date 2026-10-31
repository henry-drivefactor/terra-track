"""Search belief: intensity of not-yet-discovered targets on the terrain grid.

The prediction step is a terrain-constrained advection-diffusion, written as a
Markov transition on grid cells:

    P(i -> j) = r_ij * min(1, pi_j / pi_i)

r_ij is a symmetric rate set by how fast vehicles can move between the two cells
(roads fast, rough ground slow, water and cliffs never). pi is the rules-based
prior. This is a Metropolis construction, so pi is the stationary distribution:
searched-and-emptied ground refills from its surroundings, along roads and
passable ground, back towards the prior. Viewed continuously this is a
Fokker-Planck (advection-diffusion) equation with drift up the gradient of
log(pi). The same linear operator would give the mean dynamics of a
spatio-temporal GP with a physics-derived kernel; this filter propagates only the
mean intensity (no covariance), so it is a grid Bayes filter, not a GP.

The update step is exact for perfect sensing: inside a sensor footprint any
target would have been seen, so the undiscovered intensity there drops to
(1 - pd) of its value (zero when pd = 1). Detected targets leave this map and
are carried by tracks instead.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .world import DIR_LEN, World

_BASE_RATE = 0.15


class SearchModel:
    """Shared, read-only prediction operator for one belief model (terrain or isotropic)."""

    def __init__(self, world: World, cfg: dict, n_targets: int):
        b = cfg["belief"]
        g = world.grid
        self.grid = g
        self.terrain = b["model"] == "terrain"
        if self.terrain:
            wet = world.assumed_wetness(b)
            pi = world.prior_weights(b, wet)
            m = world.mobility_for(wet).ravel() / world.road_speed
            v_ref, var_per_sub = world.road_speed, 0.3   # spread along a 1-D road
        else:
            pi = np.ones(g.N)
            m = np.ones(g.N)
            v_ref, var_per_sub = b["isotropic_speed"], 0.6
        expected = b["expected_targets"] if b["expected_targets"] is not None else n_targets
        self.prior = pi / pi.sum() * expected

        rows, cols, vals = [], [], []
        out = np.zeros(g.N)
        for k in range(8):
            j = g.nbr[:, k]
            i = np.nonzero(j >= 0)[0]
            j = j[i]
            r = _BASE_RATE * (1.0 if DIR_LEN[k] == 1 else 0.5) * np.minimum(m[i], m[j])
            with np.errstate(divide="ignore", invalid="ignore"):
                acc = np.where(pi[i] > 0, np.minimum(1.0, pi[j] / pi[i]), 0.0)
            p = r * acc
            keep = p > 0
            rows.append(i[keep]); cols.append(j[keep]); vals.append(p[keep])
            np.add.at(out, i[keep], p[keep])
        rows.append(np.arange(g.N)); cols.append(np.arange(g.N)); vals.append(1.0 - out)
        P = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(g.N, g.N))
        self.PT = P.T.tocsr()

        dt = cfg["sim"]["dt_s"]
        var_cells = v_ref ** 2 * b["persistence_s"] * dt / g.cell ** 2
        self.substeps = max(1, int(np.ceil(var_cells / var_per_sub)))

    def predict(self, lam: np.ndarray) -> np.ndarray:
        for _ in range(self.substeps):
            lam = self.PT @ lam
        return lam


class SearchMap:
    """One agent's undiscovered-target intensity, kept consistent with late-arriving records.

    The map at step t is a deterministic function of the set of footprints known
    for steps <= t, regardless of arrival order: late records trigger a rewind to
    the last snapshot before them and a replay. Two agents holding the same records
    therefore hold identical maps.
    """

    def __init__(self, model: SearchModel, pd: float, keep: int):
        self.model, self.pd, self.keep = model, pd, keep
        self.lam = model.prior.copy()
        self.t = 0
        self.snaps = {0: self.lam.copy()}
        self.approx_updates = 0   # late records older than the snapshot window

    def _apply(self, lam, footprints):
        g = self.model.grid
        for o in footprints:
            lam[g.disc_cells(o.x, o.y, o.radius)] *= (1.0 - self.pd)
        return lam

    def ingest(self, new_records, store, t_now: int) -> None:
        oldest = min(self.snaps)
        late = [o for o in new_records if oldest < o.step <= self.t]
        too_old = [o for o in new_records if o.step <= oldest and o.step <= self.t]
        if late:
            base = max(s for s in self.snaps if s < min(o.step for o in late))
            self.lam = self.snaps[base].copy()
            for s in [s for s in self.snaps if s > base]:
                del self.snaps[s]
            self.t = base
        while self.t < t_now:
            self.t += 1
            self.lam = self._apply(self.model.predict(self.lam), store.by_step.get(self.t, ()))
            self.snaps[self.t] = self.lam.copy()
        if too_old:
            # Older than every snapshot: apply at the current time (approximate).
            self._apply(self.lam, too_old)
            self.approx_updates += len(too_old)
        for s in [s for s in self.snaps if s < self.t - self.keep]:
            del self.snaps[s]

    @property
    def expected_undiscovered(self) -> float:
        return float(self.lam.sum())
