# TODO — Project 07, due 22 Nov 2026 (today)

Rules: work top to bottom. Finish every **P0** before **P0.5**, P0.5 before **P1**, P1 before **P2**.
Tick `[x]` only when the item is done and verified (test passes / metric logged / file exists).
Time estimates are rough (8+ working hours). Anything unticked at submission goes into the report's **Gaps** section.
Decisions behind this list: [CLAUDE.md](CLAUDE.md) §2 and §10.

Legend: 💻 dev laptop (CPU) · 🎮 GPU laptop (RTX 4050 6 GB, AMP on) · ⏱ rough time

---

## P0 — Must pass the acceptance criteria (≈5–6 h)

### 0. Setup (⏱ 20 min) 💻
- [x] `.gitignore` (data/, weights/, .venv/, runs/, graphify-out/cache, *.pt), `pyproject.toml`, `requirements.txt` with pinned versions
- [x] `wastectl/` skeleton + `cli.py` (argparse subcommands from CLAUDE.md §7) + `configs/default.yaml` (target `rigid_plastic`, alert < 0.90, window 30, conf 0.25 draw / 0.40 count, seed 42)
- [x] `pytest` runs (one smoke test). First commit
- [ ] Public GitHub repo, push, clone on the GPU laptop. Check RAM/disk, `torch.cuda.is_available()`, and print the GPU name

### 1. Data (⏱ 75 min, mostly downloads) 🎮
- [ ] Download WaRP (D, C, S), TACO, TrashNet into `data/raw/`. Record URL, version, licence (or "unstated"), citation and download date for each. **ZeroWaste: not used today (gap)**
- [ ] `configs/class_map.yaml`: every source label → {cardboard, soft_plastic, rigid_plastic, metal, other}. TrashNet plastic→rigid_plastic, paper→other, glass→other. Test: no unmapped labels. List any remaining ambiguous labels in the dataset card
- [ ] Converters → unified YOLO-seg format + COCO JSON (COCO for Faster R-CNN, P1)
- [ ] Perceptual-hash dedupe across sources. Log the count removed
- [ ] Splits: official where they exist, by sequence for WaRP. **Test = WaRP test only.** Seed 42. Write the split manifest
- [ ] Crop extractor → classifier dataset (boxes + WaRP-C + TrashNet)
- [ ] Class-count table per split per source. Identify the 1–2 rarest classes (GAN target)

### 2. Start GPU training EARLY (kick off, then do §3 on the dev laptop while it runs) 🎮
- [ ] YOLO-seg **nano**, 640, AMP → `weights/yolo_seg_n.pt`. Log to experiments.jsonl
- [ ] ResNet-18 + GoogLeNet on 224 crops → weights + macro F1 (queue after YOLO, or run alongside if VRAM allows)

### 3. NumPy fundamentals (⏱ 90 min) 💻 — `wastectl/fundamentals/`
- [x] Tensor sandbox: creation, broadcasting, inner products, reshape, with a plot of each
- [x] McCulloch-Pitts neuron (AND/OR/NOT) + sigmoid neuron; step vs sigmoid vs tanh plot
- [x] MLP: forward, cross-entropy, backprop; table of parameters vs hyperparameters
- [x] Optimizers: batch GD, SGD, Momentum, Nesterov; loss curves on the same task
- [x] Numerical gradient check (test: relative error < 1e-6)
- [x] SVD: compress crop embeddings + explained-variance plot _(runs on synthetic 64-d features; swap in real crop embeddings after step 1)_
- [x] Tests: gradcheck, optimizers reduce loss, shapes

### 4. Main pipeline + purity (⏱ 60 min) 💻
- [ ] `wastectl/purity.py`: count-based purity, contamination flag, 30-frame window, conf thresholds. Unit-tested on toy detections
- [ ] `vision/pipeline.py`: YOLO-seg → crops → classifier → per-class semantic map → purity. Works on an image and on an image sequence/video
- [ ] `infer` CLI → annotated image/video + JSON

### 5. Evaluation core (⏱ 45 min) 💻/🎮
- [ ] Metrics: mAP@0.5 / 0.5:0.95, macro F1, mask IoU, purity, FPS + peak memory (CPU dev laptop and RTX 4050)
- [ ] Confusion matrices (detector + classifiers), per-source breakdown
- [ ] Benchmark table: YOLO-seg vs ResNet-18 vs GoogLeNet (this satisfies the brief's "≥ 2 architectures")
- [ ] `bench` CLI

### 6. Streamlit app, core (⏱ 45 min) 💻
- [ ] Analyzer: upload image → boxes, masks, classes, purity, contamination flag
- [ ] Stream page: image sequence/video → per-frame purity timeline
- [ ] Evaluation dashboard (reads experiments.jsonl + saved metrics)
- [ ] "Not trained" shown honestly for missing weights; education/research-use footer

### 7. Minimum docs (⏱ 30 min) 💻
- [ ] README (install, commands, results table, education-use notice, licence), dataset card, model card
- [ ] Limitations: Python not C/C++/Java, one-day timeline, ZeroWaste absent, TACO/TrashNet domain shift, simulated streams
- [ ] `/graphify . --update`, commit, push

---

## P0.5 — Demo polish: "show a good working project" (⏱ 45 min) 💻
- [ ] Professional Streamlit look: consistent theme, clear headings, metric cards, readable charts, no raw tracebacks
- [ ] Curated demo set: 6–10 WaRP test images (clean stream, contaminated stream, hard cases) + one simulated stream clip
- [ ] One-command demo: `python -m wastectl demo` or `streamlit run app/Home.py` works from a fresh clone, given the weights path
- [ ] Rehearse the 4-minute demo path end to end; fix anything slow (> 2 s per image on CPU) or flaky

---

## P1 — Advanced models: LSTM + GAN, then detector benchmark (≈2–3 h)

### 8. LSTM stream monitor 🎮/💻
- [ ] **Simulated conveyor streams** from WaRP test images: ordered sequences with controlled contamination injections + empty/jam stretches. Generator is seeded and documented. Label as **simulated** everywhere
- [ ] Per-frame features from the pipeline (class counts, purity) → 30-frame windows
- [ ] One LSTM, two heads: next-window purity forecast (MAE) + window class normal / contamination / empty-or-jam (macro F1). Plot predictions vs actual
- [ ] Hook into the Streamlit Stream page: early alert before the purity drop

### 9. DCGAN for rare classes 🎮
- [ ] DCGAN on 64×64 crops of the rarest 1–2 classes: sample grid per epoch, final samples
- [ ] Latent-space visualisation: interpolation strip + 2-D projection (PCA/t-SNE) of generated vs real embeddings
- [ ] Experiment: classifier with vs without synthetic crops → rare-class F1 delta (report honestly even if it doesn't help)

### 10. Learning rules + ablations 💻
- [ ] Hebbian, modified Hebbian (Oja), competitive (winner-take-all) demos with plots
- [ ] Error-correction (perceptron/delta) + a written comparison against Hebbian
- [ ] Boltzmann/energy-based demo (small RBM or Hopfield) with an energy plot
- [ ] NumPy MLP ablations: init (zero/Xavier/He), normalization, L2/dropout, augmentation → overfitting vs generalization curves

### 11. Faster R-CNN benchmark (first to cut) 🎮
- [ ] torchvision Faster R-CNN ResNet-50-FPN on the COCO export, small batch (2–4) + AMP → mAP on WaRP test
- [ ] Add it to the benchmark table (mAP, per-class AP, FPS CPU/GPU, size)

---

## P2 — Report, slides and extra marks (whatever time is left)
- [ ] 2D/3D convolution visualisations (ResNet filters + feature maps; 3D conv over a short frame stack)
- [ ] Robustness: blur, brightness, noise, occlusion → metric-drop table
- [ ] Error-analysis gallery: top false positives/negatives with causes
- [ ] Streamlit: Fundamentals page + Model Lab (benchmark tables, GAN latent plots, LSTM plots)
- [ ] Go/no-go deployment recommendation (latency, memory, purity accuracy, risks)
- [ ] **Technical report → PDF**, professional: cover page, numbered sections, captioned figures, tables, references, limitations, gaps
- [ ] Architecture diagram, user guide, experiment-log summary, individual contribution record (solo)
- [ ] **Slides (.pptx), 6–8 slides for a 4-min talk:** problem → data → pipeline → live demo → results → limitations/next steps
- [ ] Demo video script (owner records)
- [ ] Responsible-use + licence/ethics notes (education/research use only)
- [ ] Final `pytest`, `/graphify . --update`, final commit + push

---

## Gaps (fill at submission)
- ZeroWaste dataset not used (gated access, no time). Real conveyor video replaced by simulated streams.
- VAE not built (owner chose GAN; the brief accepts either).
- _Other unticked items move here, each with a one-line reason._
