# CLAUDE.md — Waste Sorting & Recycling Quality Controller (PBL Project 07)

Guide for any Claude/AI session working in this repo. Read this fully before touching code.
The task list lives in [TODO.md](TODO.md). Source briefs: `~/Downloads/Deep Learning - PBL.docx`
(Project 07 + common requirements + rubric) and `~/Downloads/PBL Guidelines.docx` (faculty process).

## 1. What this is

Course CSAI3027P Deep Learning, PBL **Project 07 — Waste Sorting and Recycling Quality Controller**
(industry domain: circular economy). A conveyor-belt vision system that **classifies material,
segments objects, detects instances and flags contamination**, plus an LSTM stream monitor and
an LSTM stream monitor and a GAN for rare categories, validated with **mAP, macro F1, IoU, purity rate and FPS**.

- **Owner:** solo student. Individual contribution record = one person, but still required.
- **Deadline: 22 Nov 2026 (today).** The brief assumes 8 weeks; we have one working day (8+ hours).
  Work strictly in [TODO.md](TODO.md) priority order: P0 → P1 → P2. Never start P1 while a P0 is open.
- **Fresh rewrite.** The old Python/Streamlit app in `Desktop/WASTE CONTROLLER` is **discarded** —
  do not read it, copy from it, or reuse its `best.pt`.

## 2. Decisions made by the owner (do not change without asking)

| Topic | Decision |
|---|---|
| Language | **Python only.** ⚠ The brief mandates C/C++/Java. This is a known, accepted risk against the "C C++ or Java implementation" acceptance check and the 20% fundamentals rubric line. State it honestly in the report's limitations section; do not hide it. |
| DL stack | **PyTorch + Ultralytics** (YOLO), torchvision (ResNet-18, GoogLeNet, Faster R-CNN), **NumPy for all from-scratch fundamentals**. |
| App | **CLI + Streamlit** dashboard. |
| Datasets | **WaRP, TACO, TrashNet** (ZeroWaste dropped today, see row below). Kept regardless of licence, under an **education/research use only** notice (Q12b). |
| Classes | **ZeroWaste-style 4 + `other`:** `cardboard`, `soft_plastic`, `rigid_plastic`, `metal`, `other`. Anything not fitting (glass, paper, food, textile…) → `other`. |
| Contamination / purity | **Wrong material in stream, count-based.** The user sets a target stream (e.g. `rigid_plastic`). Every detected item of any other class is a contaminant. `purity = target_items / all_items` per frame and over a sliding window. `other` is always a contaminant. |
| Detector benchmark | **YOLO (seg) vs Faster R-CNN (ResNet-50-FPN).** |
| Classifier | **ResNet-18 and GoogLeNet**, both fine-tuned on object crops and compared. |
| LSTM stream monitor | **Both heads:** (a) forecast next-window purity, (b) classify window as normal / contamination event / empty-or-jam. |
| Generative | **DCGAN only** (VAE dropped, Q26) on rare-class crops, with latent-space visualisation (interpolation + 2-D projection) and a test of whether synthetic crops improve rare-class F1. |
| Compute | Owner has **free Colab (slow)** and **a second laptop with a GPU** (specs: Q21). The dev laptop (no GPU, Windows, possibly ARM under x64 emulation) runs the app/inference on CPU. **Training runs on the GPU laptop** (RTX 4050 6 GB, i5-12450H, CUDA installed), using AMP; Colab is overflow only (Q22). |
| Git | Code on **GitHub** to sync the dev laptop ↔ GPU laptop (visibility: Q25). Never commit data or weights. |
| ZeroWaste | **Not used today** (gated access, no time). Data = WaRP + TACO + TrashNet. LSTM uses simulated streams from WaRP (Q12c). |
| Pipeline / seg / test set / tooling | Q1–Q4 accepted: two-stage YOLO-seg → crop classifier; semantic map from merged YOLO-seg masks (no U-Net); conveyor-only test set; Python 3.11 + argparse + `wastectl`. §3 items 1–7 are now decisions. |
| Purity defaults | Target stream `rigid_plastic`; alert when purity < **0.90**; sliding window **30 frames**; conf ≥ 0.25 to draw, ≥ 0.40 to count toward purity. All set in `configs/default.yaml`. |
| Class mapping | TrashNet `plastic` → `rigid_plastic`; paper → `other`; glass → `other`. |
| Models | YOLO **nano** for the app, plus **small** if time allows; images 640 (det) / 224 (cls) / 64×64 (GAN); rare classes = the 1–2 smallest by instance count (likely `metal`). |
| App pages | P0: Analyzer, Stream monitor, Evaluation. P2: Fundamentals, Model Lab. |
| Deliverables | Report delivered as **PDF**. Claude builds a **slide deck**. Everything must **look professional**: consistent typography, clean charts, cover page, numbered sections, captions. No college template. |

## 3. Calls Claude made (all confirmed by the owner, 22 Nov 2026)

1. **"YOLO with ResNet/GoogLeNet" reading:** Ultralytics YOLO has its own backbone, so the main
   pipeline is two-stage: **YOLO-seg detects + segments → ResNet-18/GoogLeNet re-classifies each crop**.
   Faster R-CNN (ResNet-50 backbone) is the benchmark rival.
2. **Semantic segmentation** = YOLO-seg instance masks merged per class into a class map. No separate U-Net.
3. **Dataset roles** (each set only where its labels allow):
   - ZeroWaste: **not used today** (gated access). Future work.
   - LSTM: simulated conveyor streams built from WaRP test images.
   - WaRP: WaRP-D detection, WaRP-C classifier crops, WaRP-S seg where present.
   - TACO: detection + seg (outdoor litter, a domain shift — training only).
   - TrashNet: classifier crops only (no boxes).
4. **Test set is conveyor-only** (WaRP test). TACO and TrashNet are never in test.
   Metrics are reported overall and per source dataset.
5. **Leakage control:** use official splits where they exist. Split WaRP by video/sequence,
   never by frame. Perceptual-hash dedupe across all sources before splitting. Seed = 42 everywhere.
6. **Python 3.11** (installed locally). The CLI uses stdlib `argparse`, not typer/click. Config is YAML.
7. Package name `wastectl`.

## 4. Repository layout (target)

```
waste_controller/
├─ CLAUDE.md  TODO.md  README.md  pyproject.toml  requirements.txt
├─ configs/
│   ├─ default.yaml          # paths, seed, target stream, thresholds, window size
│   └─ class_map.yaml        # source-dataset label → {cardboard, soft_plastic, rigid_plastic, metal, other}
├─ wastectl/
│   ├─ cli.py                # `python -m wastectl <cmd>`
│   ├─ data/                 # download notes, converters → unified YOLO + COCO format, dedupe, splits, crops
│   ├─ fundamentals/         # NumPy only: tensors, neurons, mlp, optimizers, gradcheck, svd,
│   │                        #   hebbian (+Oja), competitive, error-correction, boltzmann/hopfield, conv viz
│   ├─ vision/               # yolo.py, frcnn.py, classifier.py, pipeline.py (det→seg→cls→purity)
│   ├─ stream/               # lstm.py (forecast + anomaly heads), sequence builder
│   ├─ generative/           # gan.py, latent viz
│   ├─ eval/                 # metrics (mAP, macro F1, IoU, purity, FPS), benchmark, robustness, error analysis
│   └─ purity.py             # contamination + purity logic (single source of truth)
├─ app/                      # Streamlit: Home.py + pages/
├─ notebooks/                # optional notebooks — they import wastectl, no logic lives only here
├─ tests/                    # pytest
├─ experiments/              # experiments.jsonl log + per-run metrics/plots (small files only)
├─ weights/                  # gitignored; exported .pt
├─ data/                     # gitignored; raw/ and processed/
└─ docs/                     # dataset_card, model_card, report, architecture diagram, user_guide, slides
```

## 5. Working rules

- **Smallest working thing first.** Day-long deadline: one working YOLO run beats three half-runs.
  Start GPU-laptop training as early as possible and do CPU work (fundamentals, app, docs) while it runs.
- **Fundamentals are NumPy only.** No torch, no autograd inside `wastectl/fundamentals/`. This is the
  closest we get to the brief's "implemented directly, not delegated" requirement.
- **No logic only in notebooks.** Notebooks call `wastectl` functions so results are reproducible from the CLI.
- **Every training run** appends one line to `experiments/experiments.jsonl`
  (run id, git hash, config, seed, dataset version, metrics, wall time, hardware).
- **Never commit** data, weights or anything over 5 MB. Weights are referenced by path in config.
- **Purity logic lives only in `wastectl/purity.py`.** The app, eval and LSTM features all import it.
- **Honesty:** if a model isn't trained, the app shows "not trained" and never fakes output.
  Unreached TODO items are listed as gaps in the report, not dropped silently.
- **Tests:** `pytest` must pass before every commit. Required: gradient check on the MLP, optimizer
  sanity, purity math, class-map coverage (every source label maps), metric functions on toy inputs.
- **Commits:** small, after each TODO item, pushed to the public GitHub repo. Don't commit on a red test run.

## 6. Metrics (definitions to keep consistent)

| Metric | Definition |
|---|---|
| mAP | COCO mAP@0.5 and mAP@0.5:0.95 on the conveyor test set, per class + mean |
| Macro F1 | Unweighted mean of per-class F1 for crop classification (5 classes) |
| IoU | Mask IoU per class (semantic map) and box IoU for matched detections |
| Purity rate | `target_items / all_detected_items` per frame and over window W (config). Contamination flag when purity < threshold (config) |
| FPS | End-to-end pipeline frames/s, measured locally on the dev-laptop CPU and on the RTX 4050. Report both plus peak memory |

The evaluation also reports confusion matrices, ablations (optimizer, init, norm, regularisation,
augmentation, synthetic data on/off), robustness (blur, brightness, noise, occlusion), failure cases,
and an **evidence-based go/no-go deployment recommendation**.

## 7. Commands (target — keep in sync as they're built)

```
pip install -r requirements.txt
python -m wastectl data prepare        # convert + map classes + dedupe + split
python -m wastectl fundamentals all    # run NumPy demos, save plots to experiments/
python -m wastectl train {yolo|frcnn|cls|lstm|gan} --config configs/default.yaml
python -m wastectl eval  {yolo|frcnn|cls|lstm|pipeline}
python -m wastectl bench               # FPS / memory
python -m wastectl infer <image|video> --stream rigid_plastic
streamlit run app/Home.py
pytest
```

## 8. Rubric map (where each mark comes from)

| Weight | Criterion | Evidence in repo |
|---|---|---|
| 10 | Problem framing, dataset governance | docs/dataset_card.md, class_map.yaml, split + dedupe scripts, licences |
| 20 | NN fundamentals + tensor maths | wastectl/fundamentals + tests (⚠ Python, not C/C++/Java) |
| 15 | Optimisation, learning rules, regularisation | optimizer/ablation plots, Hebbian/competitive/Boltzmann demos |
| 20 | CNN classification, segmentation, detection | YOLO vs FRCNN, ResNet vs GoogLeNet, mAP/IoU/F1 tables |
| 15 | RNN/LSTM + GAN | stream/, generative/, latent plots |
| 10 | Software engineering, tests, performance, usability | CLI, Streamlit, pytest, bench |
| 10 | Report, demo, reproducibility, responsible use | docs/, demo video, experiments.jsonl, seeds |

## 9. Graphify

Graphify is installed (`python -m graphify`, skill `/graphify`). When `graphify-out/graph.json`
exists, query it first for structural questions (`/graphify query "..."`). Rebuild with
`/graphify . --update` after each finished P-level. Exclude `data/`, `weights/` and `.venv/`.

## 10. Open questions for the owner

Status: **OPEN** until answered. Claude must not build anything that depends on an open question.
When answered, write the answer inline, mark it **ANSWERED**, and move durable decisions into §2.
"Proposed" is Claude's suggestion, not a decision.

### Answered (22 Nov 2026), recorded in §2
- **Q1–Q8, Q10, Q11, Q13–Q15:** all proposals accepted. — **ANSWERED**
- **Q16.** Free Colab (slow) + a second laptop with a GPU. — **ANSWERED** → details in Q21–Q23
- **Q17.** Yes. — **ANSWERED**
- **Q18.** PDF. — **ANSWERED**
- **Q19.** Slides. — **ANSWERED** (format: Q24)
- **Q20.** No template; must look professional. — **ANSWERED**

### Round 2 — answers and remaining open items
- **Q9.** Google Drive has room for all four datasets. — **ANSWERED**
- **Q12.** Owner said "drop" data whose licence is non-commercial/unclear. ⚠ **Conflict found on licence check (22 Nov 2026):**
  - ZeroWaste: **CC BY-NC 4.0** (non-commercial), and the Zenodo download is **access-restricted** (request needs an academic email + citation). Approval may not arrive today.
  - WaRP: no licence found in search. Must check the Kaggle page / GitHub repo (AIRI-Institute/WaRP). Currently **unclear**.
  - TACO: annotations reportedly CC BY 4.0 (per a mirror, unconfirmed). Images carry **per-image Flickr licences**.
  - TrashNet: **MIT**. Classification only, no boxes or masks.
  Applied literally, the rule drops ZeroWaste and WaRP, leaving no conveyor data and no masks, so the project can't meet the brief.
  **Q12b decision: keep all four datasets** with an **"education/research use only"** notice. The dataset card lists each source, licence (or "unstated"), citation and access terms. The README, report, model card and app footer all state that the project and its weights are non-commercial. Never redistribute raw data or weights publicly. — **ANSWERED**
- **Q12c.** No time to request ZeroWaste access; the priority is a convincing working demo. **ZeroWaste is out for today** (listed as a gap and as future work). Consequences, applied by Claude:
  - The class scheme stays `cardboard / soft_plastic / rigid_plastic / metal / other`, now mapped from WaRP, TACO and TrashNet.
  - The conveyor-only test set = **WaRP test split** only.
  - LSTM streams are **simulated conveyor streams**: ordered sequences of WaRP test images with controlled contamination injections. They must be labelled "simulated" everywhere (report, app, slides). — **ANSWERED**
- **Q21.** GPU laptop has NVIDIA driver, CUDA and Python installed. RAM/free disk are checked at setup. — **ANSWERED**
- **Q22.** Not answered explicitly. Earlier the owner left compute to Claude ("whatever u think is best"). **Claude's call:** train everything on the GPU laptop (RTX 4050), with Colab only as overflow for one parallel job. — **CLAUDE'S CALL**
- **Q23.** Code syncs between the laptops through **GitHub**. — **ANSWERED** (visibility: Q25)
- **Q24.** Slides as **.pptx**, for a **4-minute talk** → ~6–8 slides, one idea each, demo-first. — **ANSWERED**
- **Q25.** GitHub repo is **public**. Data and weights are never committed, and the README carries the education/research-use notice. — **ANSWERED**
- **Q26.** "You can use LSTM and GAN". Read as: the advanced module is **LSTM + DCGAN**, and the **VAE is dropped**. The brief accepts VAE *or* GAN, so this is fine. **Claude's call:** demo polish goes in as P0.5, and P1 order is LSTM → GAN → Faster R-CNN. YOLO vs ResNet vs GoogLeNet already meets the brief's "at least two architectures", so Faster R-CNN is the first thing to cut if time runs out. — **ANSWERED (owner confirmed)**
- **Q27.** **MIT** licence for the code only. Datasets keep their own licences. — **ANSWERED**
- **Q21 (specs).** RTX 4050 Laptop, 6 GB VRAM, Intel i5-12450H. 6 GB fits YOLO-seg n/s at 640 (batch ~8–16), Faster R-CNN R50-FPN (batch ~2–4), classifiers, LSTM and GAN. Use AMP.
