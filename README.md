# Waste Controller: waste detection & segmentation prototype

Course project for **CSAI3027P Deep Learning** (PBL Project 07, *Waste Sorting and Recycling Quality Controller*).
A single **YOLO26n-seg** model, fine-tuned on waste photos, finds each piece of waste in an image, outlines it,
and names its material: **cardboard, soft plastic, rigid plastic, metal, or other**. It was trained entirely on a
laptop CPU. The repo also contains the neural-network fundamentals (neurons, backpropagation, optimizers, learning
rules), written from scratch in NumPy.

> **Education and research use only.** The model is trained on TACO, whose images carry individual Flickr
> licences. Datasets and trained weights are not distributed with this repository.

## Results (held-out test split, 150 photos)

| Metric | Value |
|---|---|
| Mask mAP@0.5 (segmentation) | **0.209** |
| Mask mAP@0.5:0.95 | 0.142 |
| Box mAP@0.5 (detection) | **0.211** |
| Box mAP@0.5:0.95 | 0.154 |
| Speed, laptop CPU (end to end) | **14.3 images/s** (70 ms mean) |
| Training | 30 epochs, 2.1 h on a Snapdragon X CPU, no GPU |

Best material: metal (box AP@0.5:0.95 0.27); weakest: *other* (0.08). Found 74% of large items but only 1% of the
smallest third, which is the main failure mode (see the error analysis in the report).

## Quick start

```bash
pip install -r requirements.txt

python -m wastectl data download --sources taco   # ~1,500 photos + annotations
python -m wastectl data prepare --no-crops        # unify labels, dedupe, split 80/10/10
python -m wastectl train yolo --device cpu        # fine-tune YOLO26n-seg (weights/yolo26n-seg.pt)
python -m wastectl eval yolo                      # test-split metrics + plots
streamlit run app/Home.py                         # the demo app
```

Other commands:

```bash
python -m wastectl infer photo.jpg                # annotated image + JSON in experiments/infer/
python -m wastectl infer folder_of_frames/        # annotated video + per-frame timeline
python -m wastectl bench                          # CPU latency / FPS
python -m wastectl fundamentals                   # NumPy demos -> experiments/fundamentals/*.png
pytest                                            # test suite
```

## How it works

```
photo ─► YOLO26n-seg ─► per object: box + outline mask + material + confidence
                     └► merged masks ─► per-pixel material map
                     └► optional: share of items belonging to a chosen recycling stream
```

| Part | Where |
|---|---|
| Data download, label mapping (60 TACO labels → 5 materials), near-duplicate removal, splits | `wastectl/data/`, `configs/class_map.yaml` |
| Detection + segmentation training and evaluation | `wastectl/vision/yolo.py` |
| Inference pipeline, overlays, video timeline, benchmark | `wastectl/vision/pipeline.py` |
| Stream share / contamination logic | `wastectl/purity.py` |
| NumPy fundamentals: MLP, backprop, optimizers, gradient check, SVD, Hebbian/Oja/competitive/perceptron/Hopfield, ablations | `wastectl/fundamentals/` |
| Demo app | `app/` |
| Experiment log (every run: config, seed, git commit, metrics, hardware) | `experiments/experiments.jsonl` |

## Data

**TACO**, Trash Annotations in Context ([paper](https://arxiv.org/abs/2003.06975), [repo](https://github.com/pedropro/TACO)):
1,500 photos of litter with polygon outlines. Labels are mapped to five materials; cigarette butts and unlabelled
litter are dropped. Near-duplicates are removed with a perceptual hash before splitting 80/10/10 (seed 42).
See [docs/dataset_card.md](docs/dataset_card.md).

## Limitations

- TACO shows outdoor litter, not a recycling conveyor belt. Expect weaker results on sorting-line footage.
- Small or cluttered items are often missed; similar-looking materials (clear plastic vs glass) get confused.
- Trained on a CPU at 416 px for a limited number of epochs. A GPU and more data would improve accuracy.
- Not built: the brief's LSTM/transformer and VAE/GAN components, the conveyor dataset (WaRP), and a
  Faster R-CNN comparison. Implemented in Python, while the brief asks for C, C++ or Java.

## Licence

Code: [MIT](LICENSE). Datasets and derived weights keep their own licences (education/research use).
