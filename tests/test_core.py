import numpy as np
import pytest

from geotrack.belief import SearchMap, SearchModel
from geotrack.config import load_scenario
from geotrack.observations import Detection, Obs, ObsStore
from geotrack.planner import Kinematics
from geotrack.sim import Simulation
from geotrack.tracking import TrackMotion, TrackSet
from geotrack.world import World


@pytest.fixture(scope="module")
def small():
    cfg = load_scenario(overrides={
        "world.size_m": 3000.0, "world.n_towns": 3, "world.n_exits": 1,
        "sim.duration_s": 300.0, "sim.frame_every": 0,
        "targets.road": 2, "targets.offroad": 1, "targets.static": 0,
        "agents.count": 3, "belief.particles": 200,
    })
    world = World(cfg, np.random.default_rng(0))
    return cfg, world


@pytest.mark.parametrize("model", ["terrain", "isotropic"])
def test_prior_is_stationary_and_mass_conserved(small, model):
    cfg, world = small
    cfg = {**cfg, "belief": {**cfg["belief"], "model": model}}
    m = SearchModel(world, cfg, n_targets=3)
    lam = m.predict(m.prior)
    assert lam.sum() == pytest.approx(3.0)
    assert np.abs(lam - m.prior).max() < 1e-12
    bumped = m.prior.copy()
    bumped[: world.grid.N // 2] = 0.0
    assert m.predict(bumped).sum() == pytest.approx(bumped.sum())


def test_footprint_clears_intensity(small):
    cfg, world = small
    m = SearchModel(world, cfg, 3)
    sm = SearchMap(m, pd=1.0, keep=10)
    store = ObsStore(1)
    o = Obs(0, 0, 1, 1500.0, 1500.0, 250.0)
    store.add(o)
    sm.ingest([o], store, 1)
    assert sm.lam[world.grid.disc_cells(1500.0, 1500.0, 250.0)].max() == 0.0
    assert sm.expected_undiscovered < 3.0


def _records():
    rng = np.random.default_rng(1)
    recs = []
    for step in range(1, 13):
        for origin in range(2):
            recs.append(Obs(origin, step - 1, step, *rng.uniform(300, 2700, 2), 250.0))
    # One sighting so the tracker has something to do.
    recs[4] = Obs(0, 2, 3, 1500.0, 1500.0, 250.0, (Detection(0, 1500.0, 1500.0, 0.5, 8.0),))
    return recs


def _replay(cfg, world, arrivals):
    m = SearchModel(world, cfg, 3)
    tm = TrackMotion(world, cfg, terrain=True)
    sm, ts, store = SearchMap(m, 1.0, 30), TrackSet(tm, cfg, seed=3), ObsStore(2)
    for step, batch in arrivals:
        new = [r for r in batch if store.add(r)]
        sm.ingest(new, store, step)
        ts.ingest(new, store, step)
    return sm, ts


def test_fusion_is_order_independent(small):
    """Late records rewind and replay, so arrival order does not change the belief."""
    cfg, world = small
    recs = _records()
    in_order = [(s, [r for r in recs if r.step == s]) for s in range(1, 13)]
    # Agent 1's records arrive three steps late.
    late = [(s, [r for r in recs if (r.origin == 0 and r.step == s) or (r.origin == 1 and r.step == s - 3)])
            for s in range(1, 13)] + [(12, [r for r in recs if r.origin == 1 and r.step > 9])]
    a, ta = _replay(cfg, world, in_order)
    b, tb = _replay(cfg, world, late)
    assert np.array_equal(a.lam, b.lam)
    assert np.array_equal(ta.tracks[0].cells, tb.tracks[0].cells)


def test_kinematics_respect_turn_limit(small):
    cfg, world = small
    kin = Kinematics(world, cfg)
    rates = np.array([[10.0] * 20, [-10.0] * 20, [0.0] * 20])   # demands beyond the limit
    out = kin.rollout(1500.0, 1500.0, 0.0, 20, rates=rates)
    dh = np.abs(np.angle(np.exp(1j * np.diff(out[:, :, 2], axis=1))))
    assert dh.max() <= kin.omega * kin.dt + 1e-9
    step = np.hypot(*np.diff(out[2, :, :2], axis=0).T)
    # Constant airspeed: chord length equals v*dt on straight legs, slightly less in turns.
    assert step.max() <= kin.v * kin.dt + 1e-6 and step.min() > 0.9 * kin.v * kin.dt


def test_lossless_comms_converge(small):
    cfg, _ = small
    cfg = {**cfg, "comms": {**cfg["comms"], "loss": 0.0, "range_m": 1e6}}
    sim = Simulation(cfg, record_frames=False)
    sim.run()
    n = [len(a.store.records) for a in sim.agents]
    # Everyone holds everyone's records, give or take the last step in flight.
    assert max(n) - min(n) <= len(sim.agents)
    assert sim.summary()["map_divergence_mean"] < 0.01
