# TODO — Project 07, laptop plan (due 22 Nov 2026)

**Active plan: CLAUDE.md, Round 5.** One model (YOLO26n-seg) trained on TACO on the OmniBook CPU.
It detects, segments and classifies waste. Work top to bottom; tick `[x]` only when verified.
Anything unticked at submission goes into the report's **Gaps** section.

---

## Done ✅
- [x] Repo, CLI (`python -m wastectl`), config, MIT licence, pinned deps, pytest suite
- [x] NumPy fundamentals: McCulloch-Pitts / sigmoid neurons, MLP + backprop, GD/SGD/Momentum/Nesterov, gradient check, SVD, tensor demos
- [x] Learning rules: Hebbian, Oja, competitive, perceptron (error-correction), Hopfield (energy-based)
- [x] MLP ablations: init, input normalisation, L2, augmentation → overfitting vs generalisation
- [x] Data pipeline: TACO download, class map (cigarette/litter dropped), YOLO-seg conversion, dhash dedupe, splits, class counts, visual label check
- [x] Purity / contamination logic (bonus readout in the app)
- [x] Inference pipeline + `infer` / `bench` CLI (logic tested with a fake detector)

## 1. Data for the laptop plan (⏱ 10 min)
- [x] Rebuild TACO as 80/10/10 train/val/test (`data prepare --no-crops`)
- [x] Test for the no-WaRP split; update class counts _(1197 / 150 / 150 images, 3 near-duplicates removed)_

## 2. Training on the OmniBook (⏱ ~1.5–3 h, runs in background)
- [x] Pretrained `weights/yolo26n-seg.pt` downloaded
- [x] 1-epoch timing run → choose the epoch count that fits _(≈4 min/epoch → 30 epochs ≈ 2 h; val mask mAP50 0.127 after 1 epoch)_
- [ ] Full training run → `weights/yolo26n-seg_waste.pt`, logged to experiments.jsonl
- [ ] Laptop stays plugged in, sleep disabled, heavy apps closed (8 GB RAM)

## 3. Evaluation (⏱ 20 min)
- [ ] Test split: box mAP@0.5 / 0.5:0.95, **mask mAP@0.5 / 0.5:0.95**, per-class AP
- [ ] Confusion matrix + PR curves (Ultralytics plots) copied to `experiments/`
- [ ] `bench`: CPU latency / FPS on the test images
- [ ] Failure gallery: what it misses (small/cluttered items) and why

## 4. Streamlit app (⏱ 45 min, built while training runs)
- [x] Upload a photo → masks + boxes + waste type + confidence, counts per class
- [x] Optional purity readout ("is this a clean plastic stream?")
- [x] Evaluation page (metrics + plots) and Fundamentals page (existing plots)
- [x] Professional look; honest "model not trained yet" state; education/research-use footer

## 5. Docs (⏱ 30 min)
- [ ] README: install, commands, results table, limitations, licence/use notice
- [x] Dataset card (TACO: source, licence, class mapping, splits, counts) + model card (YOLO26n-seg)

## 6. Demo (⏱ 20 min)
- [ ] 6–10 curated test photos that show it working (plus 1–2 honest failures)
- [ ] End-to-end rehearsal of the 4-minute demo

## 7. Report + slides (⏱ 1–1.5 h)
- [ ] Technical report → **PDF** _(builder done: `python docs/build_report.py`; final numbers + error analysis after evaluation)_: cover, problem, data, fundamentals, model, results, limitations, gaps, references
- [ ] **Slides (.pptx), 6–8 for a 4-min talk** _(8 slides built: `node docs/slides/build_slides.js`; refresh after evaluation)_: problem → data → model → live demo → results → limitations
- [x] Demo video script (owner records) → `docs/demo_script.md`

## Optional (only if time remains)
- [ ] Tiny GAN on 64×64 TACO crops (generative component, ~1 h CPU)
- [ ] Final `/graphify . --update`

---

## Gaps (dropped for the laptop plan; stated in the report)
- WaRP conveyor dataset + SAM pseudo-masks (no conveyor footage; TACO is outdoor litter)
- Faster R-CNN benchmark; ResNet-18 / GoogLeNet crop classifiers
- LSTM stream monitor; VAE/GAN (unless the optional GAN gets done)
- Implementation language is Python, not the C/C++/Java the brief requires
