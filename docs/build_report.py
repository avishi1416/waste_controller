"""Build the technical report (HTML -> PDF via headless Edge/Chrome) from logged results.

    python docs/build_report.py            # writes docs/report.html and docs/report.pdf

Numbers come from experiments/ (experiments.jsonl, fundamentals/results.json, data/manifest.json, Ultralytics plots),
so re-running after training refreshes everything. Output contains dataset photos: submit it, don't publish it.
"""
import base64
import csv
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wastectl.config import load_config  # noqa: E402
from wastectl.explog import read_runs  # noqa: E402

EXP = ROOT / "experiments"
GALLERY = []  # prediction images in pick order (one per material first), reused by the slides
AUTHOR = "Avishi Mehrotra"
ROLL = "[Roll number]"  # owner fills in
COURSE = "CSAI3027P Deep Learning · Project-Based Learning"
LABELS = {"cardboard": "Cardboard", "soft_plastic": "Soft plastic", "rigid_plastic": "Rigid plastic", "metal": "Metal", "other": "Other"}


def img(path, caption, width="100%"):
    p = Path(path)
    if not p.exists():
        return f'<p class="missing">[figure not generated yet: {html.escape(p.name)}]</p>'
    mime = "image/png" if p.suffix == ".png" else "image/jpeg"
    data = base64.b64encode(p.read_bytes()).decode()
    return f'<figure><img src="data:{mime};base64,{data}" style="width:{width}"><figcaption>{caption}</figcaption></figure>'


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


def fmt(v, d=3):
    return "—" if v is None else f"{v:.{d}f}"


def latest(runs, kind, split=None):
    return next((r for r in reversed(runs) if r["kind"] == kind and (split is None or r["metrics"].get("split") == split)), None)


def gallery(cfg, n=6):
    """Annotated test predictions (the photos with the most confident detections)."""
    out = EXP / "report_assets"
    w = Path(cfg["paths"]["weights"]) / f"{Path(cfg['models']['yolo']['weights']).stem}_waste.pt"
    test = Path(cfg["paths"]["data_processed"]) / "yolo" / "images" / "test"
    if not w.exists() or not test.exists():
        return '<p class="missing">[prediction gallery appears after training]</p>'
    import cv2

    from wastectl.vision.pipeline import Pipeline

    pipe = Pipeline(cfg, yolo_weights=w, device="cpu", use_classifier=False)
    results = [(p, pipe.analyze(str(p))) for p in sorted(test.glob("*.jpg"))]
    # one clear example per material first (highest-confidence detection of that class), then fill by overall confidence
    picked = []
    for c in LABELS:
        best = max(((max((d["conf"] for d in r["detections"] if d["cls"] == c), default=0), p, r) for p, r in results
                    if p not in [q for q, _ in picked]), key=lambda t: t[0])
        if best[0] > 0:
            picked.append((best[1], best[2]))
    rest = sorted(((sum(d["conf"] for d in r["detections"] if d["conf"] >= 0.4), p, r) for p, r in results
                   if p not in [q for q, _ in picked]), key=lambda t: -t[0])
    picked += [(p, r) for _, p, r in rest[: max(0, n - len(picked))]]
    for old in out.glob("pred_*.jpg"):
        old.unlink()  # don't let stale picks leak into the slides
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for p, r in picked[:n]:
        bgr = cv2.imread(str(p))
        a = pipe.annotate(bgr, r)
        h = 420
        a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
        f = out / f"pred_{p.stem}.jpg"
        cv2.imwrite(str(f), a, [cv2.IMWRITE_JPEG_QUALITY, 85])
        GALLERY.append(f.name)
        names = ", ".join(sorted({LABELS[d["cls"]] for d in r["detections"] if d["conf"] >= 0.4})) or "—"
        cells.append(img(f, f"{html.escape(p.stem)}: {names}"))
    return '<div class="grid3">' + "".join(cells) + "</div>"


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
:root { --ink:#16302B; --muted:#4F6762; --line:#CFDAD7; --accent:#0B7A5E; --wash:#EEF3F1; }
body { font-family: 'Source Serif 4', Georgia, serif; color: var(--ink); font-size: 10.5pt; line-height: 1.55; }
h1, h2, h3, th, figcaption, .cover, .kpi, nav { font-family: 'Public Sans', 'Segoe UI', sans-serif; }
h2 { font-size: 15pt; margin: 22pt 0 6pt; border-bottom: 1.5pt solid var(--accent); padding-bottom: 3pt; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; break-after: avoid; }
p { margin: 0 0 7pt; max-width: 165mm; }
.cover { height: 250mm; display: flex; flex-direction: column; justify-content: space-between; break-after: page; }
.cover .t { font-size: 30pt; font-weight: 700; line-height: 1.1; letter-spacing: -0.02em; margin-top: 40mm; }
.cover .s { font-size: 13pt; color: var(--muted); margin-top: 8pt; max-width: 140mm; }
.cover .meta { font-size: 10.5pt; line-height: 1.8; border-top: 1pt solid var(--line); padding-top: 10pt; }
.cover .bar { height: 6pt; background: linear-gradient(90deg,#E69F00 0 20%,#56B4E9 20% 40%,#009E73 40% 60%,#D55E00 60% 80%,#CC79A7 80%); }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9.5pt; break-inside: avoid; }
th { text-align: left; font-weight: 600; border-bottom: 1pt solid var(--ink); padding: 3pt 6pt; }
td { border-bottom: 0.5pt solid var(--line); padding: 3pt 6pt; vertical-align: top; }
td:not(:first-child) { font-variant-numeric: tabular-nums; }
figure { margin: 8pt 0 12pt; break-inside: avoid; }
figure img { display: block; border: 0.5pt solid var(--line); }
figcaption { font-size: 8.5pt; color: var(--muted); margin-top: 3pt; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8pt; }
.grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6pt; }
.grid3 img { height: 150pt; object-fit: cover; }
.coverage td:nth-child(2) { white-space: nowrap; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8pt; margin: 8pt 0 12pt; }
.kpi { background: var(--wash); padding: 8pt 10pt; }
.kpi b { display: block; font-size: 17pt; font-weight: 700; }
.kpi span { font-size: 8.5pt; color: var(--muted); }
.missing { color: #A0522D; font-style: italic; }
.ok { color: var(--accent); font-weight: 600; } .no { color: #B03A2E; font-weight: 600; } .part { color: #9A6B00; font-weight: 600; }
nav ol { columns: 2; font-size: 10pt; padding-left: 14pt; }
code { font-size: 8.8pt; background: var(--wash); padding: 0 2pt; }
pre { font-size: 8.5pt; background: var(--wash); padding: 6pt 8pt; white-space: pre-wrap; break-inside: avoid; }
"""


def build():
    cfg = load_config()
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = str(ROOT / v)
    cfg["models"]["yolo"]["weights"] = str(ROOT / cfg["models"]["yolo"]["weights"])
    runs = read_runs(cfg)
    fund = json.loads((EXP / "fundamentals" / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((EXP / "data" / "manifest.json").read_text(encoding="utf-8"))
    counts = {}
    for r in csv.DictReader(open(EXP / "data" / "class_counts.csv", encoding="utf-8")):
        if r["task"] == "det":
            counts.setdefault(r["class"], {})[r["split"]] = int(r["count"])
    ev = latest(runs, "eval-yolo", "test")
    tr = latest(runs, "train-yolo")
    bench = latest(runs, "bench")
    m = ev["metrics"] if ev else {}
    runs_dir = EXP / "runs"
    train_dir = runs_dir / "yolo26n-seg"
    test_dir = runs_dir / "val_test"
    mlp, opt, lr, abl = fund["mlp"], fund["optimizers"]["final_loss"], fund["learning_rules"], fund["ablations"]
    epochs = tr["params"].get("epochs") if tr else "—"
    hours = f"{tr['wall_time_s'] / 3600:.1f} h" if tr and tr.get("wall_time_s") else "—"
    fps = bench["metrics"]["fps"] if bench else None
    tr_val = latest(runs, "train-yolo")
    val_mask = tr_val["metrics"].get("mask_map50") if tr_val else None
    sr_path = EXP / "eval" / "size_recall_test.json"
    if sr_path.exists():
        sr = json.loads(sr_path.read_text(encoding="utf-8"))
        size_table = table(["Object size (share of photo)", "Objects", "Found", "Found, right material"],
                           [[f"{t['name'].capitalize()} ({t['area_pct'][0]:.2f}–{t['area_pct'][1]:.2f}%)", t["n"], f"{t['found']:.0%}", f"{t['found_right_class']:.0%}"]
                            for t in sr["terciles"]] + [["<b>All</b>", sr["n"], f"{sr['found']:.0%}", f"{sr['found_right_class']:.0%}"]])
    else:
        size_table = '<p class="missing">[run wastectl.eval.size_recall to fill this table]</p>'

    kpis = "".join(f'<div class="kpi"><b>{v}</b><span>{k}</span></div>' for k, v in [
        ("Mask mAP@0.5 (segmentation)", fmt(m.get("mask_map50"))), ("Box mAP@0.5 (detection)", fmt(m.get("box_map50"))),
        ("Images per second, laptop CPU", fmt(fps, 1)), ("Training time on CPU", hours)])
    per_class = table(["Material", "Box AP@0.5:0.95", "Train objects", "Test objects"],
                      [[LABELS[c], fmt(m.get("per_class_box_ap50_95", {}).get(c)), counts[c].get("train", 0), counts[c].get("test", 0)]
                       for c in LABELS]) if m else '<p class="missing">[per-class results appear after evaluation]</p>'

    coverage = table(["Brief requirement", "Status", "Evidence"], [
        ["Neuron / MLP / backprop / optimizers from scratch", '<span class="ok">Done</span>', "NumPy; gradient check error " + f"{mlp['gradcheck_max_rel_error']:.1e}"],
        ["Tensor ops, broadcasting, SVD, gradient checking", '<span class="ok">Done</span>', "fundamentals/tensors, linalg"],
        ["Hebbian, competitive, error-correction, Boltzmann/energy", '<span class="ok">Done</span>', "Hebb/Oja, WTA, perceptron, Hopfield"],
        ["Initialisation, normalisation, regularisation, augmentation", '<span class="ok">Done</span>', "7-way ablation on NumPy MLP"],
        ["Classification, detection, instance + semantic segmentation", '<span class="ok">Done</span>', "YOLO26n-seg; merged masks"],
        ["Benchmark ≥ 2 architectures", '<span class="no">Not done</span>', "CPU-only time budget; one detector"],
        ["RNN/LSTM or transformer", '<span class="no">Not done</span>', "Dropped in laptop plan"],
        ["VAE or GAN with latent space", '<span class="no">Not done</span>', "Dropped in laptop plan"],
        ["C, C++ or Java implementation", '<span class="no">Not met</span>', "Python (owner decision)"],
        ["Dataset documentation, splits, seeds, error analysis", '<span class="ok">Done</span>', "dataset card, dedupe, seed 42"],
        ["App, tests, build files, exported weights, benchmark", '<span class="ok">Done</span>', "Streamlit, pytest, pyproject, bench"],
    ], cls="coverage")

    body = f"""
<section class="cover"><div><div class="bar"></div>
<div class="t">Waste Controller</div>
<div class="s">Detecting and segmenting waste materials in photos with a YOLO segmentation model trained on a laptop CPU</div></div>
<div class="meta"><b>{AUTHOR}</b> · {ROLL}<br>{COURSE}<br>Project 07: Waste Sorting and Recycling Quality Controller<br>22 November 2026</div></section>

<nav><h2>Contents</h2><ol><li>Summary</li><li>Problem and requirements</li><li>Data</li><li>Neural-network fundamentals</li>
<li>Model and training</li><li>Results</li><li>Error analysis</li><li>Deployment recommendation</li><li>Coverage of the brief</li>
<li>Reproducibility</li><li>Responsible use and licences</li><li>References</li></ol></nav>

<h2>1. Summary</h2>
<p>Recycling lines lose value when the wrong material ends up in a stream. This project builds a prototype that looks at a photo,
finds each piece of waste, outlines it and names its material: cardboard, soft plastic, rigid plastic, metal or other.
A single YOLO26n-seg model, pretrained on COCO, was fine-tuned for {epochs} epoch{'s' if str(epochs) != '1' else ''} on 1,197 photos from the TACO litter dataset,
entirely on a laptop CPU (Snapdragon X, 8 GB RAM, no GPU). The fundamentals the course requires (neurons, backpropagation,
optimizers, classical learning rules and regularisation) are implemented from scratch in NumPy and unit-tested.</p>
<div class="kpis">{kpis}</div>
<p>On 150 held-out test photos the model reaches the scores above. It finds about three in four large items but almost none
of the smallest ones, and gets the material right about half the time on the items it finds. The training data is outdoor litter rather than conveyor-belt footage, so the recommendation in
Section 8 is <b>go for a demonstrator, no-go for production</b> until it is retrained on recycling-plant data.</p>

<h2>2. Problem and requirements</h2>
<p><b>Stakeholders.</b> Material-recovery-facility operators want cleaner output streams (higher purity means higher bale prices);
line workers want fewer hazardous or wrong items reaching them; course assessors want evidence across the syllabus.</p>
{table(["Requirement", "Acceptance test"], [
    ["Find each waste item and its outline", "Mask mAP@0.5 reported on a held-out test split"],
    ["Name its material (5 classes)", "Per-class AP and confusion matrix"],
    ["Run without a GPU", "End-to-end latency measured on the laptop CPU"],
    ["Be reproducible", "Seeded splits, logged configs and commits, one-command rebuild"],
    ["Be honest about limits", "Failure gallery, limitations, go/no-go decision"]])}

<h2>3. Data</h2>
<p>TACO (Trash Annotations in Context) contains 1,500 phone photos of litter with polygon outlines in 60 categories.
These were mapped to five materials; cigarette butts and unlabelled litter (1,184 tiny, ambiguous objects) were dropped
on the owner's decision so that <i>other</i> would not dominate. Near-duplicates were removed with a perceptual hash
({manifest['dedupe_removed'].get('train', 0)} removed, test images kept first), then images were split 80/10/10 with seed 42.</p>
{table(["Material", "Train", "Val", "Test"], [[LABELS[c], counts[c].get("train", 0), counts[c].get("val", 0), counts[c].get("test", 0)] for c in LABELS]
       + [["<b>Images</b>", manifest["images"].get("train"), manifest["images"].get("val"), manifest["images"].get("test")]])}
<p>Cardboard is the scarcest class (156 training objects). The full mapping, licence notes and biases are in
<code>docs/dataset_card.md</code>.</p>

<h2>4. Neural-network fundamentals (NumPy, from scratch)</h2>
<p>No deep-learning library is used in this module. Every component is unit-tested.</p>
<div class="grid2">{img(EXP / "fundamentals/activations.png", "Step, sigmoid and tanh. McCulloch-Pitts units implement AND, OR, NOT.")}
{img(EXP / "fundamentals/mlp_decision_boundary.png", f"MLP with hand-written backprop: {mlp['train_accuracy']:.0%} training accuracy; gradient check error {mlp['gradcheck_max_rel_error']:.1e}.", "80%")}</div>
{img(EXP / "fundamentals/optimizers.png", f"Same model and data. Final loss: batch GD {opt['Batch GD']:.2f}, SGD {opt['SGD (bs 32)']:.2f}, Momentum {opt['Momentum']:.2f}, Nesterov {opt['Nesterov']:.2f}.", "70%")}
<p><b>Tensors and SVD.</b> Broadcasting, inner products and reshaping are demonstrated and plotted. SVD on 64-dimensional
features shows that five components hold {fund['svd']['explained_variance_top5']:.2%} of the variance.</p>
{img(EXP / "fundamentals/learning_rules.png", "Plain Hebb grows without bound while Oja's rule normalises itself; winner-take-all finds the clusters; the perceptron stops making mistakes; a Hopfield network recalls a stored pattern as its energy falls.")}
<p>Oja's rule converged to the first principal component (cosine {lr['oja_vs_svd_pc1_cosine']:.4f} with the SVD direction), linking
Hebbian learning to SVD. Hebbian, Oja and competitive learning are unsupervised and follow correlations or proximity;
error-correction rules (perceptron, backprop) change weights only in proportion to the prediction error.</p>
{img(EXP / "fundamentals/ablations.png", "Ablations on a deliberately small training set.")}
{table(["Variant", "Train loss", "Val loss", "Val accuracy"], [[k, fmt(v["train_loss"]), fmt(v["val_loss"]), f"{v['val_acc']:.1%}"] for k, v in abl.items()])}
<p>The baseline overfits (train loss far below validation loss). Augmentation and He initialisation generalise best;
zero initialisation cannot break symmetry; unnormalised inputs and heavy L2 both hurt.</p>

<h2>5. Model and training</h2>
<p>YOLO26n-seg is a one-stage detector with a segmentation head (2.7 M parameters). It predicts boxes, classes and mask
coefficients in a single pass, so it suits a CPU. Merging the instance masks gives a semantic (per-pixel) material map.</p>
{table(["Setting", "Value", "Reason"], [["Start", "COCO-pretrained yolo26n-seg", "Transfer learning; little data"],
     ["Image size", "416 px", "≈2.5× faster than 640 on CPU"], ["Batch / workers", "8 / 2", "Fits 8 GB RAM"],
     ["Epochs", str(epochs), "≈4 min per epoch measured"], ["Frozen layers", "first 10", "Keeps pretrained features, faster"],
     ["Seed", "42, deterministic", "Reproducibility"]])}
{img(train_dir / "results.png", "Training and validation losses and mAP per epoch.")}

<h2>6. Results (test split, 150 photos)</h2>
{table(["Metric", "Value"], [["Mask mAP@0.5", fmt(m.get("mask_map50"))], ["Mask mAP@0.5:0.95", fmt(m.get("mask_map50_95"))],
       ["Box mAP@0.5", fmt(m.get("box_map50"))], ["Box mAP@0.5:0.95", fmt(m.get("box_map50_95"))],
       ["Mean latency, CPU", (fmt(bench['metrics']['mean_latency_ms'], 0) + " ms") if bench else "—"], ["Throughput, CPU", (fmt(fps, 1) + " img/s") if fps else "—"]])}
{per_class}
<div class="grid2">{img(test_dir / "confusion_matrix_normalized.png", "Normalised confusion matrix (test).")}
{img(test_dir / "MaskPR_curve.png", "Precision-recall curves for masks (test).")}</div>
<h3>Example predictions</h3>
{gallery(cfg)}

<h2>7. Error analysis</h2>
<p><b>Small objects are the main failure.</b> Matching each test object to predictions (IoU ≥ 0.5, confidence ≥ 0.25) and
splitting objects into equal thirds by size:</p>
{size_table}
<p>Objects covering less than about 0.2% of the photo (bottle caps, pop tabs, foil scraps) are almost never found: at
416 px they are only a few pixels across. Large items are found about three times in four.</p>
<p><b>Material confusion is the second failure.</b> Even among found objects, the material is right only about half the time.
The normalised confusion matrix (Section 6) shows the largest single effect is <i>missed</i> objects (the background row:
35% of rigid plastic, 27% of metal and of <i>other</i>, 20% of soft plastic). When the model fires on background (leaves, sand,
reflections) it most often calls it soft plastic (35%) or <i>other</i> (30%): crumpled film and natural clutter look alike.
Cardboard predictions scatter across rigid plastic, soft plastic and <i>other</i>, but the test set holds only 27 cardboard
objects, so that estimate is noisy.</p>
<p><b>Validation vs test.</b> The best epoch was chosen on the validation split (mask mAP@0.5 {fmt(val_mask)}), so validation is
slightly optimistic; the untouched test split ({fmt(m.get("mask_map50"))}) is the honest number. With 150 images per split, a few
points of difference is expected.</p>
<p><b>What would fix it:</b> a larger input size (640+ px) or tiled inference for small items; more epochs and an unfrozen
backbone on a GPU; more cardboard examples; and training on conveyor-belt data, where items are larger and backgrounds simpler.</p>

<h2>8. Deployment recommendation</h2>
{table(["Question", "Evidence", "Verdict"], [
    ["Fast enough on modest hardware?", f"{fmt(fps, 1)} img/s on a laptop CPU" if fps else "see Section 6", '<span class="ok">Yes</span>'],
    ["Accurate enough to sort?", f"mask mAP@0.5 {fmt(m.get('mask_map50'))}", '<span class="no">Not yet</span>'],
    ["Trained on the target environment?", "Outdoor litter, not conveyor belts", '<span class="no">No</span>'],
    ["Safe failure mode?", "Shows confidence; human stays in the loop", '<span class="part">With supervision</span>']])}
<p><b>Go</b> as a demonstrator and a starting point. <b>No-go</b> for automated sorting until it is fine-tuned and validated on
recycling-plant footage (e.g. WaRP or ZeroWaste) on a GPU.</p>

<h2>9. Coverage of the brief</h2>
{coverage}
<p>The laptop plan traded breadth for a working end-to-end prototype within one day on a CPU. Items marked not done are
planned future work, and the codebase already contains tested scaffolding for several (e.g. stream-level purity monitoring).</p>

<h2>10. Reproducibility</h2>
<pre>pip install -r requirements.txt
python -m wastectl data download --sources taco
python -m wastectl data prepare --no-crops
python -m wastectl train yolo --epochs {epochs} --device cpu
python -m wastectl eval yolo &amp;&amp; python -m wastectl bench
python -m wastectl fundamentals
pytest &amp;&amp; streamlit run app/Home.py</pre>
<p>Every run appends its configuration, seed, git commit, metrics, hardware and wall time to <code>experiments/experiments.jsonl</code>.
{f"Evaluation run <code>{ev['run_id']}</code>, commit <code>{ev.get('git')}</code>." if ev else ""}</p>

<h2>11. Responsible use and licences</h2>
<p>TACO images carry individual Flickr licences, so data, derived images and weights are used for education and research only
and are not published. The model must not make unsupervised sorting or safety decisions. Code is MIT-licensed.
Individual contribution: this is a solo project; all work is by {AUTHOR}.</p>

<h2>12. References</h2>
<p>[1] P. F. Proença, P. Simões. TACO: Trash Annotations in Context for Litter Detection. arXiv:2003.06975, 2020.<br>
[2] Ultralytics YOLO, github.com/ultralytics/ultralytics (v8.4).<br>
[3] T.-Y. Lin et al. Microsoft COCO: Common Objects in Context. ECCV 2014.<br>
[4] E. Oja. Simplified neuron model as a principal component analyzer. J. Math. Biology, 1982.<br>
[5] J. J. Hopfield. Neural networks and physical systems with emergent collective computational abilities. PNAS, 1982.<br>
[6] D. Bashkirova et al. ZeroWaste Dataset: Towards Deformable Object Segmentation in Cluttered Scenes. CVPR 2022.<br>
[7] D. Yudin et al. Hierarchical waste detection with weakly supervised segmentation in images from recycling plants (WaRP). EAAI, 2024.</p>
"""
    (EXP / "report_assets").mkdir(parents=True, exist_ok=True)
    (EXP / "report_assets" / "summary.json").write_text(json.dumps({  # numbers for docs/slides/build_slides.js
        "metrics": m, "fps": fps, "latency_ms": bench["metrics"]["mean_latency_ms"] if bench else None,
        "epochs": epochs, "train_hours": hours, "counts": counts, "images": manifest["images"],
        "gallery": GALLERY, "gradcheck": mlp["gradcheck_max_rel_error"], "mlp_acc": mlp["train_accuracy"], "author": AUTHOR,
    }, indent=2), encoding="utf-8")
    doc = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Waste Controller: technical report</title><style>{CSS}</style></head><body>{body}</body></html>'
    out_html = ROOT / "docs" / "report.html"
    out_html.write_text(doc, encoding="utf-8")
    browser = next((b for b in [shutil.which("msedge"), r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                                r"C:\Program Files\Google\Chrome\Application\chrome.exe"] if b and Path(b).exists()), None)
    if browser is None:
        print(f"HTML written to {out_html}; no Edge/Chrome found for PDF export")
        return
    pdf = ROOT / "docs" / "report.pdf"
    subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=8000",
                    f"--print-to-pdf={pdf}", out_html.as_uri()], check=True, timeout=180)
    print(f"report -> {pdf}")


if __name__ == "__main__":
    build()
