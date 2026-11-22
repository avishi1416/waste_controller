"""ResNet-18 / GoogLeNet crop classifiers (torchvision, ImageNet-pretrained, fine-tuned). Runs on the GPU laptop."""
import time
from pathlib import Path

import numpy as np

from wastectl.eval.metrics import confusion_matrix, macro_f1, per_class_f1
from wastectl.explog import log_run

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def build(backbone, n_classes, pretrained=True):
    import torch.nn as nn
    from torchvision import models

    if backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif backbone == "googlenet":
        # torchvision loads the aux heads for the pretrained weights, then drops them: plain logits in train and eval
        m = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT if pretrained else None, aux_logits=False, init_weights=False)
    else:
        raise ValueError(backbone)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def samples(crops_dir, split, classes):
    """Fixed class order from config, so label ids match across splits even when a class folder is missing."""
    out = []
    for i, c in enumerate(classes):
        d = Path(crops_dir) / split / c
        if d.exists():
            out += [(p, i) for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXT]
    return out


class CropDataset:
    """Module-level (not nested) so Windows DataLoader workers can pickle it."""

    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image

        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def transforms(imgsz, train):
    from torchvision import transforms as T

    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return T.Compose([T.RandomResizedCrop(imgsz, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), T.ColorJitter(0.3, 0.3, 0.3), T.ToTensor(), norm])
    return T.Compose([T.Resize((imgsz, imgsz)), T.ToTensor(), norm])


def _loader(items, imgsz, train, batch, workers):
    import torch

    return torch.utils.data.DataLoader(CropDataset(items, transforms(imgsz, train)), batch_size=batch, shuffle=train,
                                       num_workers=workers, pin_memory=True, persistent_workers=workers > 0)


def _predict(model, loader, device):
    import torch

    model.eval()
    ys, ps = [], []
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for x, y in loader:
            ps.append(model(x.to(device)).argmax(1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def _report(cm, classes):
    return {"macro_f1": macro_f1(cm), "per_class_f1": dict(zip(classes, per_class_f1(cm).round(4).tolist())), "confusion": cm.tolist()}


def train(cfg, backbone, epochs=10, batch=64, lr=3e-4, workers=4, device=None):
    import torch

    torch.manual_seed(cfg["seed"])
    classes, imgsz = cfg["classes"], cfg["models"]["classifier"]["imgsz"]
    crops = Path(cfg["paths"]["data_processed"]) / "crops"
    tr, va = samples(crops, "train", classes), samples(crops, "val", classes)
    if not tr:
        raise SystemExit(f"no crops in {crops}/train: run `data prepare` first")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build(backbone, len(classes)).to(device)
    counts = np.bincount([y for _, y in tr], minlength=len(classes)).astype(float)
    weight = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(classes), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)  # inverse-frequency weights: WaRP/TACO are imbalanced
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    tl, vl = _loader(tr, imgsz, True, batch, workers), _loader(va, imgsz, False, batch, workers)
    started, best, out = time.time(), -1.0, Path(cfg["paths"]["weights"]) / f"cls_{backbone}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x.to(device)), y.to(device))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        f1 = macro_f1(confusion_matrix(*_predict(model, vl, device), len(classes)))
        history.append(round(f1, 4))
        print(f"[{backbone}] epoch {ep + 1}/{epochs} val macro-F1 {f1:.4f}")
        if f1 > best:
            best = f1
            torch.save({"backbone": backbone, "classes": classes, "imgsz": imgsz, "state_dict": model.state_dict()}, out)
    log_run(cfg, f"train-cls-{backbone}", {"best_val_macro_f1": best, "val_f1_history": history},
            {"epochs": epochs, "batch": batch, "lr": lr, "n_train": len(tr), "n_val": len(va), "output": str(out)}, started)
    return out


def load(path, device="cpu"):
    import torch

    ck = torch.load(path, map_location=device, weights_only=True)
    m = build(ck["backbone"], len(ck["classes"]), pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck


def evaluate(cfg, backbone, split="test", device=None):
    import torch

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ck = load(Path(cfg["paths"]["weights"]) / f"cls_{backbone}.pt", device)
    items = samples(Path(cfg["paths"]["data_processed"]) / "crops", split, ck["classes"])
    started = time.time()
    y, p = _predict(model, _loader(items, ck["imgsz"], False, 64, 4), device)
    metrics = {"split": split, "n": len(items), **_report(confusion_matrix(y, p, len(ck["classes"])), ck["classes"])}
    log_run(cfg, f"eval-cls-{backbone}", metrics, {}, started)
    return metrics
