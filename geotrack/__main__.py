"""Command line: python -m geotrack {run, animate, sweep} ..."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_scenario, parse_override


def _overrides(items):
    return dict(parse_override(s) for s in items or [])


def cmd_run(args):
    from .sim import Simulation
    cfg = load_scenario(args.scenario, _overrides(args.set))
    if args.seed is not None:
        cfg["seed"] = args.seed
    out = Path(args.out or f"runs/{cfg['name']}_s{cfg['seed']}")
    print(f"running {cfg['name']} (seed {cfg['seed']}) -> {out}")
    sim = Simulation(cfg, record_frames=not args.no_frames)
    summary = sim.run(progress=True)
    sim.save(out)
    for k, v in summary.items():
        print(f"  {k:22s} {v}")
    if args.animate:
        args.run = str(out)
        args.output = None
        cmd_animate(args)


def cmd_animate(args):
    from .viz import Animator, load_run
    run = load_run(args.run)
    layout = getattr(args, "layout", "single")
    anim = Animator(run, agent=getattr(args, "agent", 0), layout=layout)
    base = Path(args.run) if Path(args.run).is_dir() else Path(args.run).parent
    if getattr(args, "snapshot", None) is not None:
        out = Path(args.output or base / f"snapshot_{layout}.png")
        frame = None if args.snapshot < 0 else args.snapshot
        print(f"snapshot -> {anim.snapshot(out, frame)}")
        return
    out = Path(args.output or base / f"animation_{layout}.{getattr(args, 'format', 'mp4')}")
    print(f"rendering {len(run['frames'])} frames -> {out}")
    anim.save(out, fps=getattr(args, "fps", 12))


def cmd_sweep(args):
    from .sweep import run_sweep
    run_sweep(args.sweep, workers=args.workers, out=args.out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="geotrack", description="Decentralised active search and terrain-constrained tracking")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one scenario")
    r.add_argument("scenario")
    r.add_argument("--seed", type=int)
    r.add_argument("--out")
    r.add_argument("--set", nargs="*", help="dotted overrides, e.g. planner.type=greedy comms.loss=0.5")
    r.add_argument("--no-frames", action="store_true", help="skip frame recording (faster, no animation)")
    r.add_argument("--animate", action="store_true", help="render the animation after the run")
    r.add_argument("--layout", choices=["single", "all"], default="single")
    r.add_argument("--agent", type=int, default=0)
    r.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    r.add_argument("--fps", type=int, default=12)
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("animate", help="render a recorded run")
    a.add_argument("run", help="run directory or run.pkl")
    a.add_argument("--output")
    a.add_argument("--layout", choices=["single", "all"], default="single",
                   help="single: world + one agent's belief; all: world + every agent's belief")
    a.add_argument("--agent", type=int, default=0, help="agent whose belief is shown (single layout)")
    a.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    a.add_argument("--fps", type=int, default=12)
    a.add_argument("--snapshot", type=int, nargs="?", const=-1, help="save one PNG frame instead (default: last)")
    a.set_defaults(func=cmd_animate)

    s = sub.add_parser("sweep", help="Monte Carlo comparison across variants and seeds")
    s.add_argument("sweep", help="sweep YAML")
    s.add_argument("--workers", type=int, default=0, help="processes (0 = cpu count - 1)")
    s.add_argument("--out")
    s.set_defaults(func=cmd_sweep)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
