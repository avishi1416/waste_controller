"""`python -m wastectl <command>` — the single entry point (CLAUDE.md section 7)."""
import argparse
import sys

from wastectl import __version__
from wastectl.config import DEFAULT_CONFIG, load_config

# command -> TODO.md section that implements it; unbuilt commands fail loudly instead of faking output
TODO_REF = {
    "data": "step 1 Data",
    "fundamentals": "step 3 NumPy fundamentals",
    "train": "step 2/step 8/step 9/step 11 training",
    "eval": "step 5 Evaluation",
    "bench": "step 5 Evaluation",
    "infer": "step 4 Pipeline",
    "demo": "P0.5 Demo polish",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wastectl", description="Waste sorting & recycling quality controller")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("data", help="prepare datasets").add_argument("action", choices=["prepare"])
    sub.add_parser("fundamentals", help="run NumPy demos").add_argument("demo", nargs="?", default="all")
    for name in ("train", "eval"):
        sub.add_parser(name).add_argument("model", choices=["yolo", "frcnn", "cls", "lstm", "gan", "pipeline"])
    sub.add_parser("bench", help="FPS / memory benchmark")
    infer = sub.add_parser("infer", help="run the pipeline on an image or video")
    infer.add_argument("source")
    infer.add_argument("--stream", default=None, help="target stream (default: config purity.target_stream)")
    sub.add_parser("demo", help="launch the demo app")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    load_config(args.config)  # fail fast on a broken config
    print(f"wastectl {args.command}: not implemented yet (TODO.md {TODO_REF[args.command]})", file=sys.stderr)
    return 2
