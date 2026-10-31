"""Synthetic world: terrain, land cover, road network and ground mobility.

Grid convention: arrays are (ny, nx), row = y index, col = x index, origin at the
bottom-left. Flat cell index = iy * nx + ix. Positions are metres; cell centres sit
at ((ix + 0.5) * cell, (iy + 0.5) * cell).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import Delaunay

# 8-connected moves as (dx, dy); index k points at angle k * 45 degrees.
DIRS = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)])
DIR_LEN = np.hypot(DIRS[:, 0], DIRS[:, 1])


class Grid:
    def __init__(self, nx: int, ny: int, cell: float):
        self.nx, self.ny, self.cell = nx, ny, cell
        self.N = nx * ny
        self.size_x, self.size_y = nx * cell, ny * cell
        iy, ix = np.divmod(np.arange(self.N), nx)
        self.cx = (ix + 0.5) * cell
        self.cy = (iy + 0.5) * cell
        self.nbr = np.full((self.N, 8), -1, dtype=np.int64)
        for k, (dx, dy) in enumerate(DIRS):
            jx, jy = ix + dx, iy + dy
            ok = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
            self.nbr[ok, k] = jy[ok] * nx + jx[ok]

    @property
    def extent(self):
        return (0.0, self.size_x, 0.0, self.size_y)

    def index(self, x, y):
        ix = np.clip((np.asarray(x) / self.cell).astype(np.int64), 0, self.nx - 1)
        iy = np.clip((np.asarray(y) / self.cell).astype(np.int64), 0, self.ny - 1)
        return iy * self.nx + ix

    def inside(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        return (x >= 0) & (x < self.size_x) & (y >= 0) & (y < self.size_y)

    def disc_offsets(self, radius: float):
        """(dy, dx) offsets of cells whose centres lie within radius of a cell centre."""
        r = int(np.ceil(radius / self.cell))
        dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
        keep = (dx * dx + dy * dy) * self.cell ** 2 <= radius ** 2
        return dy[keep], dx[keep]

    def disc_cells(self, x: float, y: float, radius: float):
        """Flat indices of cells whose centres lie within radius of (x, y)."""
        r = int(np.ceil(radius / self.cell)) + 1
        ix0, iy0 = int(x // self.cell), int(y // self.cell)
        ix = np.arange(max(ix0 - r, 0), min(ix0 + r + 1, self.nx))
        iy = np.arange(max(iy0 - r, 0), min(iy0 + r + 1, self.ny))
        if ix.size == 0 or iy.size == 0:
            return np.empty(0, dtype=np.int64)
        gy, gx = np.meshgrid(iy, ix, indexing="ij")
        d2 = ((gx + 0.5) * self.cell - x) ** 2 + ((gy + 0.5) * self.cell - y) ** 2
        keep = d2 <= radius ** 2
        return (gy[keep] * self.nx + gx[keep]).astype(np.int64)

    def graph(self, cost: np.ndarray) -> sparse.csr_matrix:
        """8-connected graph; edge weight = mean cell cost x step length. inf cost blocks."""
        c = cost.ravel()
        rows, cols, w = [], [], []
        for k in range(8):
            j = self.nbr[:, k]
            i = np.nonzero(j >= 0)[0]
            j = j[i]
            ww = 0.5 * (c[i] + c[j]) * DIR_LEN[k] * self.cell
            fin = np.isfinite(ww)
            rows.append(i[fin]); cols.append(j[fin]); w.append(ww[fin])
        return sparse.csr_matrix(
            (np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(self.N, self.N))


def path_from_pred(pred: np.ndarray, src: int, root: int):
    """Walk a predecessor array (rooted at `root`) from src back to the root."""
    if src == root:
        return [src]
    if pred[src] < 0:
        return None
    path = [src]
    while path[-1] != root:
        path.append(int(pred[path[-1]]))
    return path


def _noise(rng, shape, sigma):
    n = ndimage.gaussian_filter(rng.standard_normal(shape), max(sigma, 0.5), mode="reflect")
    return (n - n.mean()) / (n.std() + 1e-12)


def fractal(rng, shape, base_sigma, octaves=5, persistence=0.5):
    out = np.zeros(shape)
    amp, sigma = 1.0, base_sigma
    for _ in range(octaves):
        out += amp * _noise(rng, shape, sigma)
        amp *= persistence
        sigma /= 2
    return out / out.std()


def _norm01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-12)


class World:
    def __init__(self, cfg: dict, rng: np.random.Generator):
        w = cfg["world"]
        self.cfg = cfg
        n = int(round(w["size_m"] / w["cell_m"]))
        self.grid = g = Grid(n, n, w["cell_m"])
        shape = (g.ny, g.nx)

        # Elevation: smooth hills plus ridged noise for ridgelines and valleys.
        base = _norm01(fractal(rng, shape, g.nx / 6))
        ridged = (1.0 - np.abs(fractal(rng, shape, g.nx / 5, octaves=4)) / 2.5).clip(0, 1) ** 2
        elev = w["relief_m"] * (0.55 * base + 0.45 * _norm01(ridged))
        yy, xx = np.mgrid[0:g.ny, 0:g.nx]
        yy, xx = yy / (g.ny - 1), xx / (g.nx - 1)
        for r in w.get("ridge_walls") or []:
            band = np.exp(-(((yy if r.get("axis", "x") == "x" else xx) - r["pos"]) / r.get("width", 0.04)) ** 2)
            along = xx if r.get("axis", "x") == "x" else yy
            gap = np.zeros(shape)
            for gp in r.get("gaps", []):
                gap = np.maximum(gap, np.exp(-((along - gp) / r.get("gap_width", 0.035)) ** 2))
            elev += r.get("height_m", 250.0) * band * (1.0 - gap)

        # Rivers: least-elevation paths edge to edge.
        water = np.zeros(shape, bool)
        for i in range(w["n_rivers"]):
            cost = np.exp(4.0 * _norm01(elev)) + 0.2 * rng.random(shape)
            G = g.graph(cost)
            if i % 2 == 0:
                a = (int(rng.integers(g.ny // 5, 4 * g.ny // 5)), 0)
                b = (int(rng.integers(g.ny // 5, 4 * g.ny // 5)), g.nx - 1)
            else:
                a = (0, int(rng.integers(g.nx // 5, 4 * g.nx // 5)))
                b = (g.ny - 1, int(rng.integers(g.nx // 5, 4 * g.nx // 5)))
            src, dst = a[0] * g.nx + a[1], b[0] * g.nx + b[1]
            _, pred = csgraph.dijkstra(G, indices=dst, return_predecessors=True)
            for c in path_from_pred(pred, src, dst) or []:
                water.flat[c] = True
        water = ndimage.binary_dilation(water, iterations=1)
        elev = np.where(water, elev - 8.0, elev)
        self.elev, self.water = elev, water

        gy, gx = np.gradient(elev, g.cell)
        self.slope = np.hypot(gx, gy)
        forest_field = fractal(rng, shape, g.nx / 12)
        self.forest = (forest_field > np.quantile(forest_field, 1 - w["forest_fraction"])) & ~water
        pas = np.clip(1.0 - (self.slope / w["max_slope"]) ** 1.5, 0.0, 1.0)
        pas = pas * np.where(self.forest, w["forest_factor"], 1.0)
        pas[water] = 0.0
        pas[pas < 0.08] = 0.0
        self.passability = pas

        self._build_roads(rng, w)

        t = cfg["targets"]
        self.road_speed = t["road_speed"]
        self.mobility = np.where(self.road, t["road_speed"], t["offroad_speed"] * self.passability)
        self.road_graph = g.graph(np.where(self.road, 1.0, np.inf))
        with np.errstate(divide="ignore"):
            self.xc_graph = g.graph(np.where(self.mobility > 0, 1.0 / np.maximum(self.mobility, 1e-9), np.inf))

    # ------------------------------------------------------------------ roads
    def _build_roads(self, rng, w):
        g = self.grid
        margin = int(0.06 * g.nx)
        ok = self.passability > 0.6
        ok[:margin, :] = ok[-margin:, :] = ok[:, :margin] = ok[:, -margin:] = False
        cand = np.flatnonzero(ok.ravel())
        min_sep = 0.6 * g.size_x / np.sqrt(max(w["n_towns"], 1))
        towns = []
        for c in rng.permutation(cand):
            p = np.array([g.cx[c], g.cy[c]])
            if all(np.hypot(*(p - q)) > min_sep for q in towns):
                towns.append(p)
            if len(towns) == w["n_towns"]:
                break
        exits = []
        for _ in range(w["n_exits"]):
            for _try in range(200):
                side = rng.integers(4)
                u = rng.uniform(0.15, 0.85)
                ix, iy = [(int(u * (g.nx - 1)), 0), (int(u * (g.nx - 1)), g.ny - 1),
                          (0, int(u * (g.ny - 1))), (g.nx - 1, int(u * (g.ny - 1)))][side]
                if self.passability[iy, ix] > 0.3:
                    exits.append(np.array([g.cx[iy * g.nx + ix], g.cy[iy * g.nx + ix]]))
                    break
        self.towns = np.array(towns)
        self.exits = np.array(exits).reshape(-1, 2)

        pts = self.towns
        edges = set()
        if len(pts) >= 3:
            tri = Delaunay(pts)
            for s in tri.simplices:
                for a, b in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2])):
                    edges.add((min(a, b), max(a, b)))
        elif len(pts) == 2:
            edges.add((0, 1))
        edges = sorted(edges, key=lambda e: np.hypot(*(pts[e[0]] - pts[e[1]])))
        # Minimum spanning tree via Kruskal, then a share of the remaining short edges as loops.
        parent = list(range(len(pts)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        tree, rest = [], []
        for a, b in edges:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
                tree.append((a, b))
            else:
                rest.append((a, b))
        chosen = [(pts[a], pts[b]) for a, b in tree + rest[: int(round(w["road_loops"] * len(rest)))]]
        for e in self.exits:
            nearest = pts[np.argmin(np.hypot(*(pts - e).T))]
            chosen.append((e, nearest))

        road = np.zeros((g.ny, g.nx), bool)
        build = 1.0 + (self.slope / 0.08) ** 2 + np.where(self.forest, 1.0, 0.0)
        build = build + np.where(self.water, 40.0, 0.0) + np.where((self.passability == 0) & ~self.water, 60.0, 0.0)
        for a, b in chosen:
            cost = np.where(road, 0.25, build)
            G = g.graph(cost)
            src, dst = int(g.index(*a)), int(g.index(*b))
            _, pred = csgraph.dijkstra(G, indices=dst, return_predecessors=True)
            for c in path_from_pred(pred, src, dst) or []:
                road.flat[c] = True
        self.road = road
        self.nodes = np.vstack([self.towns, self.exits]) if len(self.exits) else self.towns

    # ------------------------------------------------------------- belief prior
    def prior_weights(self, belief_cfg: dict) -> np.ndarray:
        """Rules-based prior: vehicles concentrate on roads, then on passable open ground."""
        return np.where(self.road, belief_cfg["road_weight"],
                        belief_cfg["offroad_weight"] * self.passability).ravel()
