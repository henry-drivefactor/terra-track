# geo-trackbot

A proof-of-concept simulation of **decentralised active search and terrain-constrained
tracking**. A small swarm of fixed-wing agents (3–10) searches an area for ground
vehicles, then keeps tracking them as they move. Each agent:

- holds its own belief about where vehicles are,
- fuses what it and its teammates observe,
- plans its own route to maximise expected information gain,
- coordinates only by sharing compact observation records and the route legs it has committed to.

There is no central controller.

Everything is in Python (numpy, scipy, matplotlib). There's no flight-dynamics
simulator and no autopilot integration. Sensing is idealised: anything inside an
agent's ground footprint is detected perfectly. The point of the POC is the
search, fusion, tracking and planning logic.

```
pip install -r requirements.txt
python -m geotrack run scenarios/s1_valley_search.yaml --animate     # run + MP4
python -m geotrack sweep sweeps/planners.yaml                        # Monte Carlo comparison
python -m pytest -q
```

---

## How it works

Each simulation step (default 5 s) runs this loop:

```
targets move ─► agents fly committed route ─► sense footprint ─► broadcast packet
     ▲                                                               │
     │        replan if leg finished or an event fired ◄── fuse ◄────┘
     └──────────────── (belief: search map + tracks) ◄───────────────┘
```

### 1. World (`geotrack/world.py`)
The world is a procedurally generated area, 10 km square at 50 m cells by default. It contains:
- fractal and ridged-noise terrain, plus optional *ridge walls* with passes,
- rivers,
- forest,
- a road network routed between towns and map exits along least-cost paths.

From slope and land cover the world derives **passability** (0 means impassable) and
**ground speed** per cell: fast on roads, slow in forest and on steep ground, zero
in water or on cliffs.

### 2. Targets (`geotrack/targets.py`)
Ground-truth vehicles follow simple rules:
- *road* vehicles drive shortest road routes between towns and exits, pausing at each,
- *off-road* vehicles take least-time cross-country routes, using roads where that is quicker,
- *static* vehicles stay parked.

Their motion is generated independently of the belief model, so the belief is never
tested against itself.

### 3. Sensing (`geotrack/agent.py`)
Each agent carries a circular ground footprint (250 m radius by default). Every
target inside it is detected with its exact position, heading, speed and ID.
Every step produces one **observation record**: footprint centre, radius, step and
detections, about 16 bytes plus 12 per detection. Detection probability `pd` is a
parameter, 1.0 by default.

### 4. Belief part 1: search map (`geotrack/belief.py`)
The search map holds the intensity of **not-yet-discovered** targets per cell, i.e.
the expected count.

- **Prior:** rules-based. Roads carry the most weight, then passable open ground;
  water and cliffs carry none.
- **Prediction:** a terrain-constrained advection–diffusion, implemented as a
  Metropolis Markov transition on grid cells:
  `P(i→j) = r_ij · min(1, π_j / π_i)`. Here `r_ij` is set by the ground speed of both
  cells and `π` is the prior.
  - The prior is therefore the stationary distribution. Ground that has been
    searched and emptied refills from its surroundings, along roads and through
    passes, back towards the prior.
  - In continuous form this is a Fokker–Planck equation with drift up `∇log π`: a
    linear Gauss–Markov system equivalent to a spatio-temporal Gaussian process
    with a physics-derived kernel, solved on the grid.
- **Update:** exact negative information. Inside a footprint any target would have
  been seen, so the intensity there is multiplied by `(1 − pd)`. Discovered targets
  leave this map and become tracks.

### 5. Belief part 2: tracks (`geotrack/tracking.py`)
Each discovered target gets a grid particle filter (600 particles).

- Particles follow a correlated random walk on the 8-connected grid. They keep
  their heading, move at the cell's ground speed, stay on roads when on a road, and
  can never enter impassable cells.
- So between sightings the cloud spreads *along roads and through passes*, not in
  circles.
- Footprints that saw nothing prune particles. This resolves branches: "we looked
  down the north road and it was empty, so it went south".
- If every hypothesis is ruled out, the track re-seeds over unobserved reachable
  ground.

### 6. Order-independent fusion
Records from teammates arrive late, out of order, or not at all. Both the search map
and the tracks keep per-step snapshots. A late record triggers a rewind to the step
before it and a replay.

All randomness is seeded from (target, sighting step, step). The belief is
therefore a deterministic function of the *set* of records held, whatever order
they arrived in: two agents holding the same records hold identical beliefs.
`tests/test_core.py` checks this.

### 7. Planning (`geotrack/planner.py`)
At each replan the agent generates candidate routes by rolling out fixed-wing
kinematics (constant airspeed, turn-rate limit set by a 150 m minimum turn radius).
There are four kinds of candidate:
- a 5×5 fan of two-segment turn manoeuvres,
- pursuit of the highest-value regions of the belief,
- pursuit of each track's predicted position,
- the remainder of the current route.

Each candidate is scored in **nats of expected information gain** over a 5-minute horizon:

```
  w_search · E[new detections along the route] · H(where an undiscovered target is)
+ w_track  · Σ_tracks  P(re-observe the target) · H(track forecast at that time)
```

Entropies are measured over footprint-sized bins, so the two terms are in the same
units, and gains are discounted by time.

### 8. Decentralised coordination
Before scoring, an agent removes from its belief everything its teammates' **committed
legs** are expected to observe:
- intensity under their committed footprints,
- track particles they will fly over.

Only the next two legs (2 minutes) count as committed, because teammates replan
after that.

When an agent receives a plan it didn't know about when it planned, and the two
overlap, the earlier plan has right of way (ties go to the lower ID) and the later
planner yields and replans. There is no central allocator.

### 9. Replanning modes
| mode | when an agent replans |
|---|---|
| `open_loop` | only when its 5-minute route runs out |
| `per_leg` | every leg (1 minute, staggered across agents) |
| `event` | every leg, **and** immediately on events: a new target, lost contact with a teammate (heartbeat timeout), or a plan conflict |

### 10. Comms (`geotrack/comms.py`)
The radio model is a range-limited broadcast mesh with per-link packet loss and
optional blackouts. Every step each agent broadcasts one packet containing:
- a header,
- a version vector (which records it holds),
- a status table (last known position and plan version of every agent),
- whatever records or committed plans its recently heard neighbours lack.

Items not acknowledged within a few steps are re-sent. Status and records relay
multi-hop. Every byte is counted.

### 11. Metrics (`geotrack/sim.py`)
Every metric is measured against **ground truth**, identically for every planner:

| metric | meaning |
|---|---|
| `loc_score` | headline metric: mean over all targets of the belief mass within 250 m of the true position (track particles if found, search intensity if not). 1 = every target pinned down |
| `found_frac` | fraction of targets seen at least once |
| `track_err_m` | mean distance of track particles from the true position |
| `staleness_s` | mean time since each target was last seen |
| `coverage` | fraction of the area observed at least once |
| `map_divergence` | mean pairwise L1 difference between agents' search maps (0 = identical) |
| `kbytes`, `packets`, `replans` | cumulative communication and planning load |

---

## Scenarios

| file | what it shows |
|---|---|
| `s1_valley_search.yaml` | 4 agents, 9 vehicles (6 road, 2 off-road, 1 parked) on ridged terrain with a river and road network. Baseline for comparisons. |
| `s2_degraded_swarm.yaml` | 6 agents, 50% packet loss, 3 km radio range, a 2-minute blackout, agents A1 and A4 lost mid-mission. Beliefs drift apart and re-converge; survivors re-spread once the lost agents' commitments time out. |
| `s3_terrain_funnel.yaml` | A steep ridge crossable only at two passes and by road; most vehicles drive cross-country. Shows what the terrain constraint buys. |

Override anything from the command line:

```
python -m geotrack run scenarios/s1_valley_search.yaml --seed 3 \
    --set planner.type=greedy agents.count=6 comms.loss=0.4 --animate
```

All parameters and their defaults are in `geotrack/config.py`.

## Visualisation

`run --animate`, or `animate <run_dir>`, renders an MP4 (`--format gif` for GIF):

- **left:** terrain, roads, towns, true vehicles (filled once found), agents with footprints,
  committed routes, trails and live comms links.
- **right:** one agent's belief (`--agent N`). Blue is the expected number of
  undiscovered targets per cell; orange is track particles; × marks true
  positions for reference. Next to it are live metrics and the event log.
- **bottom:** localisation and found fraction, track error and data sent over time.

`--layout all` shows every agent's belief side by side, which is the best way to see
divergence under comms loss (S2). `--snapshot [frame]` writes a single PNG.

## Experiments

```
python -m geotrack sweep sweeps/planners.yaml       # coordinated vs isotropic vs greedy vs lawnmower
python -m geotrack sweep sweeps/terrain_model.yaml  # terrain-constrained vs isotropic belief (S3)
python -m geotrack sweep sweeps/replanning.yaml     # open-loop vs per-leg vs event-triggered
python -m geotrack sweep sweeps/resilience.yaml     # packet loss and agent loss (S2)
```

Each sweep writes `runs/sweeps/<name>/` containing:
- `results.csv`: one row per run,
- `timeseries.csv`,
- `summary.md`: mean ± s.e. per variant,
- `timeseries.png` and `summary.png`.

RESULTS_PLACEHOLDER

## Assumptions and limitations

- **Perfect sensing.** Detection inside the footprint is certain and gives true
  identity, so there is no data association, no false alarms and no confidence
  model. `agents.pd` (detection probability) is threaded through the belief
  updates, but nothing else about imperfect sensing is modelled.
- **No line-of-sight.** Terrain constrains where vehicles can go, not what agents can see.
- **Closed population.** No vehicles enter or leave the area. The search map is a
  Poisson intensity whose total is the expected number of undiscovered targets.
- **Non-evasive targets.** Vehicles do not react to the agents.
- **Kinematic agents.** Constant speed and altitude, turn-rate-limited 2-D flight,
  no collisions, no endurance limit.
- **The search map ignores its own motion inside the planning horizon.** Candidate
  routes are scored against the current intensity (tracks *are* forecast forward).
  This is a deliberate cost trade-off.
- **Snapshot window.** Records older than the rewind window (default 40 steps,
  about 3 minutes) are applied at the current time, approximately; these are counted
  in `SearchMap.approx_updates`.

## Layout

```
geotrack/
  config.py        defaults, YAML loading, dotted overrides
  world.py         terrain, land cover, roads, passability, grid helpers
  targets.py       ground-truth vehicle behaviour
  observations.py  observation records and per-agent record store
  belief.py        search map: advection-diffusion prediction, negative-information update, rewind
  tracking.py      terrain-constrained particle tracks
  planner.py       kinematic rollouts, information-gain scoring, greedy and lawnmower baselines
  agent.py         platform, sensing, gossip, events
  comms.py         lossy range-limited broadcast mesh
  sim.py           main loop, metrics, recording
  viz.py           animation and snapshots
  plots.py         sweep comparison plots
  sweep.py         Monte Carlo runner
scenarios/         scenario YAMLs
sweeps/            experiment YAMLs
tests/             unit and integration tests
```
