"""Append-only experiment log: one JSON line per run in experiments/experiments.jsonl (CLAUDE.md section 5)."""
import json
import platform
import subprocess
import time
from pathlib import Path


def _git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _hardware():
    hw = {"platform": platform.platform(), "cpu": platform.processor()}
    try:
        import torch

        hw["torch"] = torch.__version__
        hw["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        hw["gpu"] = None
    return hw


def log_run(cfg, kind, metrics, params=None, started=None):
    rec = {
        "run_id": f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}",
        "kind": kind,
        "git": _git_hash(),
        "seed": cfg["seed"],
        "params": params or {},
        "metrics": metrics,
        "wall_time_s": round(time.time() - started, 1) if started else None,
        "hardware": _hardware(),
    }
    path = Path(cfg["paths"]["experiments"]) / "experiments.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    return rec


def read_runs(cfg):
    path = Path(cfg["paths"]["experiments"]) / "experiments.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
