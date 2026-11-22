"""Fetch raw datasets into data/raw/<source>/. Raw data is untrusted: only images/JSON/text are read, never executed."""
import json
import shutil
import time
import subprocess
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TACO_ANNOTATIONS = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
TRASHNET_ZIP = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
WARP_KAGGLE = "parohod/warp-waste-recycling-plant-dataset"
# img.tacodataset.org rejects Python's default "Python-urllib" agent with 403; identify honestly instead
HEADERS = {"User-Agent": "wastectl-dataset-downloader/0.1"}


def _fetch(url, dest, timeout=60, retries=3):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    tmp.replace(dest)  # never leave a half-written file under the real name
    return dest


def _safe_extract(zip_path, out):
    out = Path(out).resolve()
    with zipfile.ZipFile(zip_path) as z:
        for m in z.namelist():
            if not (out / m).resolve().is_relative_to(out):
                raise ValueError(f"unsafe path in {zip_path}: {m}")
        z.extractall(out)


def taco(raw, workers=8):
    """Annotations + 640px thumbnails (YOLO labels are normalised, so thumbnail scale doesn't matter)."""
    root = Path(raw) / "taco"
    ann = json.loads(_fetch(TACO_ANNOTATIONS, root / "annotations.json").read_text(encoding="utf-8"))
    jobs = [(im["flickr_640_url"] or im["flickr_url"], root / "images" / im["file_name"]) for im in ann["images"]]
    failed = []

    def get(job):
        try:
            _fetch(*job)
        except Exception as e:  # keep going; failures are reported, and prepare skips missing images
            failed.append((job[0], repr(e)))

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(get, jobs))
    print(f"TACO: {len(jobs) - len(failed)}/{len(jobs)} images" + (f", {len(failed)} failed" if failed else ""))
    return root


def trashnet(raw):
    root = Path(raw) / "trashnet"
    z = _fetch(TRASHNET_ZIP, root / "dataset-resized.zip")
    if not any(root.glob("**/cardboard")):
        _safe_extract(z, root)
    print(f"TrashNet: extracted to {root}")
    return root


def warp(raw):
    """Needs the Kaggle CLI (`pip install kaggle`) and ~/.kaggle/kaggle.json from the owner's account."""
    root = Path(raw) / "warp"
    if any(root.glob("**/Warp-D")):
        print(f"WaRP: already present in {root}")
        return root
    if not shutil.which("kaggle"):
        raise SystemExit("WaRP needs the Kaggle CLI: `pip install kaggle` and put kaggle.json in ~/.kaggle/")
    subprocess.run(["kaggle", "datasets", "download", "-d", WARP_KAGGLE, "-p", str(root)], check=True)
    for z in root.glob("*.zip"):
        _safe_extract(z, root)
    print(f"WaRP: extracted to {root}")
    return root


SOURCES = {"taco": taco, "trashnet": trashnet, "warp": warp}
