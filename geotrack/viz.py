"""Animation and static plots of recorded runs (matplotlib only)."""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LightSource, ListedColormap, PowerNorm  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

# Palette roles (light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TRACK = "#eb6834"
WATER = np.array([0.62, 0.77, 0.96])
BOG = np.array([0.66, 0.58, 0.47])
FOREST = np.array([0.80, 0.88, 0.76])
ROAD = np.array([0.32, 0.32, 0.31])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
})


def agent_color(i: int) -> str:
    return SERIES[i] if i < len(SERIES) else INK_2


def load_run(path: str | Path) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "run.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)


def base_map(world: dict, faint: bool = False) -> np.ndarray:
    """Hillshaded terrain with water, forest and roads, as an RGB image."""
    ls = LightSource(azdeg=315, altdeg=40)
    # hillshade assumes row 0 is the top of the image; ours is the bottom.
    shade = ls.hillshade(world["elev"][::-1], vert_exag=3.0, dx=world["cell"], dy=world["cell"])[::-1]
    grey = 0.80 + 0.20 * shade if faint else 0.62 + 0.38 * shade
    rgb = np.repeat(grey[..., None], 3, axis=2)
    if not faint:
        rgb = np.where(world["forest"][..., None], rgb * FOREST / FOREST.max(), rgb)
    if "boggy" in world and world["boggy"].any():
        rgb = np.where(world["boggy"][..., None], 0.5 * rgb + 0.5 * BOG, rgb)
    rgb = np.where(world["water"][..., None], WATER, rgb)
    if not faint:
        rgb = np.where(world["road"][..., None], ROAD, rgb)
    else:
        rgb = np.where(world["road"][..., None], np.array([0.6, 0.6, 0.58]), rgb)
    return np.clip(rgb, 0, 1)


def _belief_cmap():
    base = matplotlib.colors.to_rgba_array(BLUE_RAMP)
    n = 256
    xs = np.linspace(0, 1, n)
    rgb = np.stack([np.interp(xs, np.linspace(0, 1, len(base)), base[:, c]) for c in range(3)], axis=1)
    alpha = np.clip(xs * 1.6, 0, 0.92)
    return ListedColormap(np.column_stack([rgb, alpha]))


def _tri(x, y, h, size):
    pts = np.array([[1.0, 0.0], [-0.6, 0.55], [-0.35, 0.0], [-0.6, -0.55]]) * size
    c, s = np.cos(h), np.sin(h)
    return pts @ np.array([[c, s], [-s, c]]) + [x, y]


class Animator:
    def __init__(self, run: dict, agent: int = 0, layout: str = "single"):
        self.run, self.agent, self.layout = run, agent, layout
        self.cfg = run["cfg"]
        self.frames = run["frames"]
        w = run["world"]
        self.extent = w["extent"]
        self.size = self.extent[1]
        self.R = self.cfg["agents"]["footprint_m"]
        self.n = self.cfg["agents"]["count"]
        self.cmap = _belief_cmap()
        prior = run["prior"]
        f = prior.shape[0] // self.frames[0]["lam"][0].shape[0]
        vmax = float(prior.reshape(prior.shape[0] // f, f, prior.shape[1] // f, f).mean((1, 3)).max())
        vmax = max(vmax, 4.0 * float(prior.mean()))   # keep a flat (isotropic) prior mid-scale, not saturated
        self.norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)
        self.trail = int(120 / (self.cfg["sim"]["dt_s"] * self.cfg["sim"]["frame_every"]))
        self.metric_t = run["metrics"]["t"]

        if layout == "all":
            self._setup_all(w)
        else:
            self._setup_single(w)

    # --------------------------------------------------------------- layouts
    def _setup_single(self, w):
        self.fig = plt.figure(figsize=(16, 9), dpi=100)
        self.ax_w = self.fig.add_axes([0.015, 0.06, 0.50, 0.86])
        self.ax_b = self.fig.add_axes([0.53, 0.37, 0.30, 0.55])
        self.ax_txt = self.fig.add_axes([0.845, 0.37, 0.15, 0.55])
        self.ax_txt.axis("off")
        self.ax_m = [self.fig.add_axes([0.545 + i * 0.155, 0.07, 0.125, 0.21]) for i in range(3)]
        self._world_panel(self.ax_w, w)
        self.b_panels = [self._belief_panel(self.ax_b, w, self.agent, big=True)]
        self._metric_panels()
        self.legend_text = self.ax_txt.text(0.0, 1.0, "", va="top", ha="left", fontsize=8.5, color=INK_2,
                                            family="monospace", transform=self.ax_txt.transAxes)

    def _setup_all(self, w):
        self.fig = plt.figure(figsize=(16, 9), dpi=100)
        self.ax_w = self.fig.add_axes([0.015, 0.06, 0.50, 0.86])
        self._world_panel(self.ax_w, w)
        cols = 3 if self.n > 4 else 2
        rows = int(np.ceil(self.n / cols))
        cw, ch = 0.46 / cols, 0.84 / rows
        side = min(cw, ch * 9 / 16)
        self.b_panels = []
        for i in range(self.n):
            r, c = divmod(i, cols)
            ax = self.fig.add_axes([0.53 + c * cw, 0.92 - (r + 1) * ch + 0.02, side * 0.93, ch - 0.06])
            self.b_panels.append(self._belief_panel(ax, w, i, big=False))
        self.ax_m = []
        self.cursors = []
        self.legend_text = None

    def _world_panel(self, ax, w):
        ax.imshow(base_map(w), origin="lower", extent=self.extent, interpolation="bilinear")
        if len(w["towns"]):
            ax.scatter(w["towns"][:, 0], w["towns"][:, 1], s=26, marker="s", c=SURFACE, edgecolors=INK, lw=1.0, zorder=3)
        ax.set_xlim(0, self.size); ax.set_ylim(0, self.size)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(GRID)
        self.title = ax.set_title("", loc="left")
        self.link_lc = LineCollection([], colors=MUTED, linewidths=0.9, linestyles=(0, (3, 3)), zorder=4)
        ax.add_collection(self.link_lc)
        self.a_poly, self.a_fp, self.a_route, self.a_trail, self.a_lbl = [], [], [], [], []
        for i in range(self.n):
            col = agent_color(i)
            fp = Circle((0, 0), self.R, facecolor=col, alpha=0.16, edgecolor=col, lw=1.0, zorder=5)
            ax.add_patch(fp)
            route, = ax.plot([], [], color=col, lw=1.2, ls=(0, (4, 3)), zorder=6)
            trail, = ax.plot([], [], color=col, lw=2.0, alpha=0.55, zorder=6, solid_capstyle="round")
            poly = Polygon(_tri(0, 0, 0, 160), closed=True, facecolor=col, edgecolor=SURFACE, lw=1.5, zorder=8)
            ax.add_patch(poly)
            lbl = ax.text(0, 0, f"A{i}", fontsize=8.5, fontweight="bold", color=INK, zorder=9,
                          bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec=col, lw=1.0, alpha=0.9))
            self.a_fp.append(fp); self.a_route.append(route); self.a_trail.append(trail)
            self.a_poly.append(poly); self.a_lbl.append(lbl)
        self.t_found = ax.scatter([], [], s=70, marker="D", c=INK, edgecolors=SURFACE, lw=1.5, zorder=7)
        self.t_unfound = ax.scatter([], [], s=70, marker="D", facecolors="none", edgecolors=INK, lw=1.4, zorder=7)
        ax.text(0.01, -0.035, "◆ target (found)   ◇ target (not yet found)   ▲ agent + footprint   - - committed route   "
                "··· comms link   ■ town", transform=ax.transAxes, fontsize=8, color=INK_2, va="top")

    def _belief_panel(self, ax, w, i, big):
        ax.imshow(base_map(w, faint=True), origin="lower", extent=self.extent, interpolation="bilinear")
        lam0 = self.frames[0]["lam"][i].astype(float)
        im = ax.imshow(lam0, origin="lower", extent=self.extent, cmap=self.cmap, norm=self.norm,
                       interpolation="bilinear", zorder=2)
        parts = ax.scatter([], [], s=5 if big else 2, c=TRACK, alpha=0.45, lw=0, zorder=3)
        truth = ax.scatter([], [], s=30 if big else 12, marker="x", c=INK, lw=1.1, zorder=4)
        me = Polygon(_tri(0, 0, 0, 200 if big else 260), closed=True, facecolor=agent_color(i),
                     edgecolor=SURFACE, lw=1.2, zorder=5)
        ax.add_patch(me)
        ax.set_xlim(0, self.size); ax.set_ylim(0, self.size)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(agent_color(i)); s.set_linewidth(1.5)
        title = ax.set_title(f"A{i} belief", loc="left", fontsize=10 if big else 9)
        labels = [ax.text(0, 0, "", fontsize=7.5 if big else 6, color=INK, zorder=6, visible=False)
                  for _ in range(len(self.frames[0]["targets"]))]
        if big:
            ax.text(0.0, -0.03, "blue: expected undiscovered targets per cell   orange: track particles   "
                    "x: true positions (reference)", transform=ax.transAxes, fontsize=7.5, color=INK_2, va="top")
        return dict(ax=ax, i=i, im=im, parts=parts, truth=truth, me=me, title=title, labels=labels, big=big)

    def _metric_panels(self):
        m = self.run["metrics"]
        t = m["t"] / 60.0
        specs = [
            ("Targets found & localised", [("found_frac", "found", SERIES[0]), ("loc_score", "localisation", SERIES[1])],
             (0, 1.02), "fraction"),
            ("Mean track error", [("track_err_m", None, SERIES[0])], None, "metres"),
            ("Data sent (all agents)", [("kbytes", None, SERIES[0])], None, "kB"),
        ]
        self.cursors = []
        for ax, (title, series, ylim, unit) in zip(self.ax_m, specs):
            for key, label, col in series:
                ax.plot(t, m[key], color=col, lw=2.0)
                if label:
                    j = int(len(t) * (0.55 if key == "found_frac" else 0.8))
                    ax.text(t[j], m[key][j] + (0.06 if key == "found_frac" else -0.1), label, color=INK_2,
                            fontsize=7.5)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_title(title, loc="left", fontsize=8.5)
            ax.set_xlabel("minutes", fontsize=7.5)
            ax.set_ylabel(unit, fontsize=7.5)
            ax.tick_params(labelsize=7)
            ax.grid(True, axis="y", color=GRID, lw=0.6)
            self.cursors.append(ax.axvline(0, color=MUTED, lw=1.0))

    # ----------------------------------------------------------------- frame
    def draw(self, k: int):
        fr = self.frames[k]
        cfg = self.cfg
        mins, secs = divmod(int(fr["t"]), 60)
        found = int(fr["found"].sum())
        wet = self.run["world"].get("wetness", 0.0)
        ground = f"wet ground {wet:.1f}" if wet > 0 else "dry ground"
        self.title.set_text(f"{cfg['name']}  ·  {cfg['planner']['type']} planner, {cfg['planner']['replan']} replanning, "
                            f"{cfg['belief']['model']} belief, {ground}  ·  T+{mins:02d}:{secs:02d}  ·  "
                            f"found {found}/{len(fr['found'])}")
        A = fr["agents"]
        segs = [[A[i, :2], A[j, :2]] for i, j in fr["links"]]
        self.link_lc.set_segments(segs)
        k0 = max(0, k - self.trail)
        for i in range(self.n):
            x, y, h, alive = A[i]
            if alive:
                self.a_poly[i].set_xy(_tri(x, y, h, 170))
                self.a_poly[i].set_facecolor(agent_color(i))
                self.a_fp[i].center = (x, y)
                self.a_fp[i].set_visible(True)
                r = fr["routes"][i]
                self.a_route[i].set_data(r[:, 0], r[:, 1]) if len(r) else self.a_route[i].set_data([], [])
            else:
                self.a_poly[i].set_xy(_tri(x, y, h, 170))
                self.a_poly[i].set_facecolor(MUTED)
                self.a_fp[i].set_visible(False)
                self.a_route[i].set_data([], [])
            tr = np.array([f["agents"][i, :2] for f in self.frames[k0:k + 1] if f["agents"][i, 3]])
            self.a_trail[i].set_data(tr[:, 0], tr[:, 1]) if len(tr) else self.a_trail[i].set_data([], [])
            self.a_lbl[i].set_position((x + 180, y + 140))
            self.a_lbl[i].set_text(f"A{i}" if alive else f"A{i} ✕")
        T = fr["targets"]
        self.t_found.set_offsets(T[fr["found"]] if fr["found"].any() else np.empty((0, 2)))
        self.t_unfound.set_offsets(T[~fr["found"]] if (~fr["found"]).any() else np.empty((0, 2)))

        for P in self.b_panels:
            i = P["i"]
            P["im"].set_data(fr["lam"][i].astype(float))
            tracks = fr["tracks"][i]
            pts = np.vstack(list(tracks.values())) if tracks else np.empty((0, 2))
            P["parts"].set_offsets(pts)
            P["truth"].set_offsets(T)
            x, y, h, alive = A[i]
            P["me"].set_xy(_tri(x, y, h, 200 if P["big"] else 260))
            P["me"].set_facecolor(agent_color(i) if alive else MUTED)
            for tid, lbl in enumerate(P["labels"]):
                if tid in tracks and len(tracks[tid]):
                    mx, my = tracks[tid].mean(0)
                    lbl.set_position((mx + 150, my + 150)); lbl.set_text(f"T{tid}"); lbl.set_visible(True)
                else:
                    lbl.set_visible(False)
            exp_u = float(fr["lam"][i].astype(float).sum() * (self.run["prior"].size / fr["lam"][i].size))
            status = "" if alive else " (lost)"
            P["title"].set_text(f"A{i} belief{status}  ·  {len(tracks)} tracks  ·  E[undiscovered] {exp_u:.1f}"
                                if P["big"] else f"A{i}{status} · {len(tracks)} tracks · E[u] {exp_u:.1f}")
        if self.layout == "all":
            md = fr["metrics"].get("map_divergence", 0.0)
            self.title.set_text(self.title.get_text() + f"  ·  belief divergence {md:.2f}")

        for c in self.cursors:
            c.set_xdata([fr["t"] / 60.0, fr["t"] / 60.0])
        if self.legend_text is not None:
            ev = [f"{int(t) // 60:02d}:{int(t) % 60:02d} {txt}" for t, txt in self.run["events"] if t <= fr["t"]]
            ev = [e if len(e) <= 34 else e[:33] + "…" for e in ev[-16:]]
            mt = fr["metrics"]
            head = (f"found        {mt.get('found_frac', 0):.2f}\nlocalisation {mt.get('loc_score', 0):.2f}\n"
                    f"coverage     {mt.get('coverage', 0):.2f}\ndivergence   {mt.get('map_divergence', 0):.3f}\n"
                    f"data sent    {mt.get('kbytes', 0):.0f} kB\nreplans      {mt.get('replans', 0)}\n\nEVENTS\n")
            self.legend_text.set_text(head + "\n".join(ev))
        return []

    def save(self, out: str | Path, fps: int = 12, progress: bool = True):
        out = Path(out)
        anim = animation.FuncAnimation(self.fig, self.draw, frames=len(self.frames), blit=False)
        if out.suffix == ".gif":
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=4000, codec="libx264",
                                            extra_args=["-pix_fmt", "yuv420p"])
        cb = (lambda i, n: print(f"  frame {i + 1}/{n}", flush=True) if (i + 1) % 50 == 0 else None) if progress else None
        anim.save(out, writer=writer, progress_callback=cb)
        plt.close(self.fig)
        return out

    def snapshot(self, out: str | Path, k: int | None = None):
        self.draw(len(self.frames) - 1 if k is None else min(k, len(self.frames) - 1))
        self.fig.savefig(out, dpi=100)
        plt.close(self.fig)
        return out
