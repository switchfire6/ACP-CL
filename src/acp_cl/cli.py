"""Command-line entry points. Config files are plain, auditable JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run_suite
from .learner import METHODS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Adaptive critical-period continual-learning experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run a paired method/seed suite")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--methods", nargs="+", choices=METHODS)
    run.add_argument("--seeds", nargs="+", type=int)
    run.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    run.add_argument("--resume", action="store_true", help="Continue completed-experience checkpoints")
    run.add_argument("--no-plots", action="store_true")
    analysis = commands.add_parser("analyze", help="Summarize already completed runs")
    analysis.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.command == "analyze":
        from .analysis import analyze
        analyze(args.directory)
        print(args.directory / "summary.md")
        return 0
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    methods = args.methods or config.get("methods", ["er", "er_recycle", "fixed", "acp"])
    seeds = args.seeds or config.get("seeds", [0])
    run_suite(config, output=args.output, seeds=seeds, methods=methods,
              device_name=args.device, resume=args.resume)
    if not args.no_plots:
        from .analysis import analyze
        analyze(args.output)
        print(f"Results: {args.output / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
