from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
REQUIRED = ("seed", "classes", "purity", "paths", "models")


def load_config(path=DEFAULT_CONFIG) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in (cfg or {})]
    if missing:
        raise ValueError(f"{path}: missing config keys {missing}")
    return cfg
