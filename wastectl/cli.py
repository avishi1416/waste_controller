"""`python -m wastectl <command>` — the single entry point (CLAUDE.md section 7)."""
import argparse
import json
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

    data = sub.add_parser("data", help="download / prepare datasets, SAM pseudo-masks")
    data.add_argument("action", choices=["download", "prepare", "sam"])
    data.add_argument("--sources", nargs="+", default=["warp", "taco", "trashnet"], choices=["warp", "taco", "trashnet"])
    data.add_argument("--sam-model", default="sam2.1_b.pt")
    data.add_argument("--device", default=None)
    sub.add_parser("fundamentals", help="run NumPy demos").add_argument(
        "demo", nargs="?", default="all", choices=["all", "tensors", "neurons", "mlp", "optimizers", "svd", "learning_rules", "ablations"]
    )
    for name in ("train", "eval"):
        sp = sub.add_parser(name)
        sp.add_argument("model", choices=["yolo", "frcnn", "cls", "lstm", "gan", "pipeline"])
        sp.add_argument("--backbone", choices=["resnet18", "googlenet", "both"], default="both", help="cls only")
        sp.add_argument("--epochs", type=int)
        sp.add_argument("--batch", type=int)
        sp.add_argument("--device", default=None, help="e.g. 0 / cuda / cpu")
        sp.add_argument("--fraction", type=float, default=1.0, help="yolo: train on a fraction of the data (smoke runs)")
        sp.add_argument("--workers", type=int, default=4)
        sp.add_argument("--split", default="test", help="eval split")
    sub.add_parser("bench", help="FPS / memory benchmark")
    infer = sub.add_parser("infer", help="run the pipeline on an image or video")
    infer.add_argument("source")
    infer.add_argument("--stream", default=None, help="target stream (default: config purity.target_stream)")
    sub.add_parser("demo", help="launch the demo app")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)  # fail fast on a broken config
    if args.command == "fundamentals":
        from wastectl.fundamentals.demos import run

        out = f"{cfg['paths']['experiments']}/fundamentals"
        for name, res in run(args.demo, out, cfg["seed"]).items():
            print(f"[{name}] {res}")
        print(f"plots + results.json -> {out}/")
        return 0
    if args.command == "data":
        if args.action == "download":
            from wastectl.data.download import SOURCES

            for s in args.sources:
                SOURCES[s](cfg["paths"]["data_raw"])
        elif args.action == "prepare":
            from wastectl.data.prepare import prepare

            print(json.dumps(prepare(cfg), indent=2))
        else:
            from wastectl.data.sam import sam_refine

            sam_refine(cfg, args.sam_model, args.device)
        return 0
    if args.command in ("train", "eval") and args.model in ("yolo", "cls"):
        backbones = ["resnet18", "googlenet"] if args.backbone == "both" else [args.backbone]
        if args.model == "yolo":
            from wastectl.vision import yolo

            if args.command == "train":
                yolo.train(cfg, args.epochs, args.batch, args.device, args.fraction)
            else:
                print(json.dumps(yolo.evaluate(cfg, split=args.split, device=args.device), indent=2))
        else:
            from wastectl.vision import classifier

            for b in backbones:
                if args.command == "train":
                    classifier.train(cfg, b, args.epochs or 10, args.batch or 64, workers=args.workers, device=args.device)
                else:
                    print(json.dumps(classifier.evaluate(cfg, b, args.split, args.device), indent=2))
        return 0
    print(f"wastectl {args.command}: not implemented yet (TODO.md {TODO_REF[args.command]})", file=sys.stderr)
    return 2
