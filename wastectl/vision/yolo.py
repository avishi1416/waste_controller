"""YOLO-seg training/evaluation through Ultralytics (TODO.md step 2). Runs on the GPU laptop."""
import shutil
import time
from pathlib import Path

from wastectl.explog import log_run


def _data_yaml(cfg):
    p = Path(cfg["paths"]["data_processed"]) / "yolo" / "data.yaml"
    if not p.exists():
        raise SystemExit(f"{p} missing: run `python -m wastectl data prepare` first")
    return str(p)


def _metrics(res, names):
    m = {"box_map50": res.box.map50, "box_map50_95": res.box.map}
    if getattr(res, "seg", None) is not None:
        m |= {"mask_map50": res.seg.map50, "mask_map50_95": res.seg.map}
    m["per_class_box_ap50_95"] = {names[int(c)]: float(ap) for c, ap in zip(res.box.ap_class_index, res.box.maps[res.box.ap_class_index])}
    return {k: (float(v) if not isinstance(v, dict) else v) for k, v in m.items()}


def train(cfg, epochs=None, batch=None, device=None, fraction=1.0, weights=None, resume=False):
    from ultralytics import YOLO

    y = cfg["models"]["yolo"]
    weights = weights or y["weights"]
    run_dir = Path(cfg["paths"]["experiments"]).resolve() / "runs" / Path(weights).stem
    params = {"weights": weights, "imgsz": y["imgsz"], "epochs": epochs or y["epochs"], "batch": batch or y["batch"],
              "amp": y["amp"], "fraction": fraction, "freeze": y.get("freeze"), "workers": y.get("workers", 2)}
    started = time.time()
    if resume:  # continue an interrupted run (sleep, crash) from its last saved epoch
        last = run_dir / "weights" / "last.pt"
        if not last.exists():
            raise FileNotFoundError(f"nothing to resume: {last}")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(weights)
        model.train(data=_data_yaml(cfg), seed=cfg["seed"], deterministic=True, device=device,
                    project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
                    **{k: params[k] for k in ("imgsz", "epochs", "batch", "amp", "fraction", "freeze", "workers")})
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    out = Path(cfg["paths"]["weights"]) / f"{Path(weights).stem}_waste.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    metrics = evaluate(cfg, out, split="val", log=False)
    log_run(cfg, "train-yolo", metrics, {**params, "output": str(out)}, started)
    print(f"YOLO weights -> {out}\n{metrics}")
    return out


def evaluate(cfg, weights=None, split="test", log=True, device=None):
    from ultralytics import YOLO

    weights = weights or Path(cfg["paths"]["weights"]) / f"{Path(cfg['models']['yolo']['weights']).stem}_waste.pt"
    started = time.time()
    model = YOLO(str(weights))
    res = model.val(data=_data_yaml(cfg), split=split, imgsz=cfg["models"]["yolo"]["imgsz"], device=device,
                    project=str(Path(cfg["paths"]["experiments"]).resolve() / "runs"), name=f"val_{split}", exist_ok=True)
    metrics = {"split": split, **_metrics(res, model.names)}
    if log:
        log_run(cfg, "eval-yolo", metrics, {"weights": str(weights)}, started)
    return metrics
