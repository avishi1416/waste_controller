# Viva notes: Waste Controller (CSAI3027P PBL, Project 07)

## 1. The project in 30 seconds
A prototype that takes a photo of waste and, for **every item** in it, gives:
1. a **box** around it (object detection),
2. its **outline** (instance segmentation; merging outlines gives a per-pixel material map = semantic segmentation),
3. its **material**: cardboard, soft plastic, rigid plastic, metal, or other (classification).

One model, **YOLO26n-seg**, does all three. It was fine-tuned on the **TACO** litter dataset, entirely on my laptop's
**CPU** (HP OmniBook 5, Snapdragon X, 8 GB RAM, no GPU). I also built the course fundamentals from scratch in NumPy.

**Why it matters:** recycling streams lose value when the wrong material gets in (contamination). A vision system
has to know what each item is *and* where it is.

## 2. Pipeline (what happens to a photo)
```
photo → resize to 416 px → YOLO26n-seg → per item: box + mask + class + confidence
                                       → masks merged → per-pixel material map
                                       → optional: share of items in a chosen stream ("purity")
```
Code: `wastectl/vision/pipeline.py`. App: `streamlit run app/Home.py`.

## 3. Data
- **TACO** (Trash Annotations in Context): 1,500 phone photos of litter, 4,784 objects, 60 categories,
  each with a **polygon outline**.
- **Mapped 60 → 5 materials** (`configs/class_map.yaml`). Paper, glass, foam and drink cartons → *other*.
  **Dropped** cigarette butts (667) and unlabelled litter (517): tiny, ambiguous, and they would dominate *other*.
- **Near-duplicate removal:** a 64-bit *difference hash* (dHash) per image; images within Hamming distance 3 are
  treated as duplicates and removed, keeping test images first, so no near-copy of a test photo is in training.
  3 removed.
- **Split** 80/10/10 → **1,197 train / 150 val / 150 test**, seed 42.
- **Imbalance:** cardboard is the rarest class (156 training objects vs 842 for *other*).
- **Checks:** EXIF orientation check (0 mismatches), and labels drawn over images to confirm they line up.

**Why TACO?** It already has real polygon masks (needed for segmentation), it's free, and it's small enough
for CPU training. **Weakness:** outdoor litter, not a conveyor belt (domain gap).

## 4. Model: YOLO26n-seg
- **One-stage detector:** predicts boxes and classes in a single forward pass (vs two-stage R-CNN, which first
  proposes regions, then classifies them, so it's slower). The "n" (nano) variant has **2.7 M parameters,
  9.1 GFLOPs**, which is why it's feasible on a CPU.
- **Segmentation head:** predicts a small set of *prototype masks* plus per-object *mask coefficients*; each
  object's mask is a weighted combination of prototypes (the YOLACT idea).
- **Transfer learning:** started from **COCO-pretrained** weights (80 everyday classes), replaced the head with
  5 classes and fine-tuned. Pretrained features (edges, textures, shapes) transfer, so far less data is needed.

### Training settings (and why)
| Setting | Value | Reason |
|---|---|---|
| Image size | 416 | ~2.5× faster than 640 on CPU |
| Batch | 8 | fits 8 GB RAM |
| Epochs | 30 | ~4.5 min/epoch measured, so ~2 h |
| Frozen layers | first 10 (backbone) | faster; keeps pretrained features; less overfitting on 1.2k images |
| Optimiser | Ultralytics auto (SGD/AdamW by schedule) | tuned defaults |
| Augmentation | mosaic, HSV colour jitter, flips, scaling | more variety from little data |
| Seed | 42, deterministic | reproducible |

**Loss terms you'll see in the log:** box loss (box position), seg loss (mask shape), cls loss (material class),
dfl loss (distribution focal loss for box edges) and a semantic loss. All should fall over epochs.

## 5. Metrics: be ready to define these
- **IoU (Intersection over Union)** = overlap area ÷ union area of prediction and ground truth (box or mask).
- **Precision** = correct detections ÷ all detections. **Recall** = correct detections ÷ all real objects.
- **AP (Average Precision)** = area under the precision-recall curve for one class.
- **mAP@0.5** = mean AP over classes, where a detection counts as correct if IoU ≥ 0.5.
- **mAP@0.5:0.95** = mAP averaged over IoU thresholds 0.5, 0.55 … 0.95 (stricter; rewards precise outlines).
- **Box mAP** measures detection; **Mask mAP** measures segmentation.
- **Macro F1** = unweighted mean of per-class F1, where F1 = 2PR/(P+R). It treats rare classes equally.
- **Purity** (bonus) = items of the target material ÷ all confidently detected items.

## 6. Results (fill in from `docs/report.pdf` after training)
| | Value |
|---|---|
| Mask mAP@0.5 (test) | _____ |
| Box mAP@0.5 (test) | _____ |
| Speed on laptop CPU | ~36 ms per image inference (measured in the timing run) |
| Progress | epoch 1: mask mAP@0.5 0.123 → epoch 2: 0.173 (validation) |

Expect roughly 0.2–0.4 mask mAP@0.5. TACO is hard: small objects, cluttered outdoor scenes.

## 7. Fundamentals I built from scratch (NumPy, no DL library)
| Concept | What I showed | Key number |
|---|---|---|
| McCulloch-Pitts neuron | threshold unit implementing AND, OR, NOT | truth tables correct |
| Sigmoid / tanh / step | activation shapes compared | — |
| MLP + backprop | forward pass, softmax cross-entropy, hand-written backward pass | 99% on 3-class spirals |
| **Gradient check** | backprop vs numerical central differences | **relative error 5.6 × 10⁻⁸** |
| Optimisers | batch GD, SGD, Momentum, Nesterov on the same model | final loss: GD 0.72, SGD 0.70, **Momentum 0.05, Nesterov 0.06** |
| SVD | compress 64-d features; explained variance | top 5 components = 99.96% |
| Hebbian | Δw = η·y·x grows without bound | norm ×10¹⁴ |
| **Oja's rule** | Δw = η·y·(x − y·w) self-normalises | converges to the **1st principal component** (cosine 0.99997 with SVD) |
| Competitive (WTA) | only the winning prototype moves | finds the 3 clusters |
| Error-correction | perceptron: update only on mistakes | 14 → 0 mistakes |
| Boltzmann / energy | Hopfield network: Hebbian storage, energy descent | recalls the stored pattern; energy −23.9 → −50.2 |
| Ablations | init, normalisation, L2, augmentation on a small set | baseline overfits (train 0.078 vs val 0.415); augmentation 89.3 → 92.5%; He init 93.0%; zero init 33% (can't break symmetry); heavy L2 underfits (72%) |

**Why momentum wins:** it accumulates a velocity, so it keeps moving through flat regions and plateaus where
plain GD/SGD with the same learning rate stalls. **Nesterov** evaluates the gradient at the look-ahead point,
which gives a small correction.

**Why zero init fails:** every hidden unit gets the same gradient, so they stay identical (the symmetry never breaks).

**Hebbian vs error-correction:** Hebbian, Oja and competitive learning are *unsupervised* (they follow input
correlations or proximity, with no target). Perceptron and backprop are *supervised* (weights move in proportion
to the prediction error).

## 8. Engineering
- Python package `wastectl` with one CLI: `python -m wastectl data|fundamentals|train|eval|infer|bench`.
- **43 automated tests** (pytest): gradient check, optimisers, learning rules, purity maths, data pipeline end to end,
  pipeline logic with a fake detector, every app page renders.
- Every run is logged to `experiments/experiments.jsonl`: config, seed, git commit, metrics, hardware, time.
- `--resume` continues training after an interruption. Public GitHub repo, MIT licence for the code.

## 9. Decisions and trade-offs (examiners love "why")
| Decision | Why | Cost |
|---|---|---|
| Laptop CPU training | only hardware available today | lower accuracy; smaller images, fewer epochs |
| YOLO-seg nano | one model covers detection + segmentation + classification; fast | less accurate than larger models |
| TACO only | real masks, ready, small | outdoor litter, not conveyor belts |
| 5 materials | matches a plastics/metal/cardboard sorting line | paper/glass lumped into *other* |
| Drop cigarettes | tiny and ambiguous; would dominate *other* | those items stay unlabelled |
| Python | fastest path to a working prototype | **the brief asks for C/C++/Java**: say so honestly |

## 10. Limitations (say these before they ask)
1. Outdoor litter, not conveyor footage → it will degrade on a real sorting line.
2. Small, occluded or cluttered items get missed; look-alike materials get confused.
3. Cardboard has the fewest examples → least reliable.
4. Not built: the brief's ≥2-architecture benchmark, LSTM/transformer, VAE/GAN, and C/C++/Java.
5. Verdict: **go as a demonstrator, no-go for production** until retrained on plant data (WaRP, ZeroWaste) on a GPU.

## 11. Likely viva questions, with short answers
- **Why YOLO and not Faster R-CNN?** One-stage, so much faster, especially on a CPU; one model gives
  boxes and masks. Faster R-CNN is two-stage: often more accurate on small objects, but too slow to train here.
- **Detection vs semantic vs instance segmentation?** Detection = boxes. Semantic = class per pixel, with no
  separation between objects. Instance = a separate mask per object. I get instance masks and merge them for semantic.
- **What is transfer learning and why use it?** Reusing weights learned on a big dataset (COCO); early layers
  learn generic features. It needs far less data and compute.
- **Why freeze the backbone?** Faster on a CPU, keeps general features, and reduces overfitting on 1.2k images.
- **How did you avoid data leakage?** Perceptual-hash near-duplicate removal before splitting, test kept first,
  fixed seed; the test set is never used for training or model selection.
- **What does mAP@0.5 = 0.3 mean?** On average across classes, the precision-recall trade-off at IoU ≥ 0.5 has
  an area of 0.3: it finds and correctly outlines many clear items but misses or misplaces a lot of small ones.
- **How would you improve it?** A GPU, 640 px, more epochs, conveyor data (WaRP/ZeroWaste), class re-balancing
  or synthetic data for cardboard, and a larger model (s/m).
- **How do you know backprop is correct?** The gradient check: analytic gradients match numerical central
  differences to 5.6 × 10⁻⁸ relative error.
- **What is overfitting and how did you show it?** Low training loss and high validation loss. With a small
  training set the baseline reached train 0.078 vs val 0.415; augmentation and good initialisation reduced the gap.
- **What's the link between Oja's rule and SVD/PCA?** Oja's rule converges to the top eigenvector of the input
  covariance, which is the first principal component (the first right-singular vector in SVD). I measured
  cosine 0.99997.
- **Why did you drop cigarette butts?** Tiny, numerous and ambiguous; they'd make *other* half the data and
  distort training. It was a deliberate, documented choice.
- **Is it ready for a real plant?** No. Accuracy and domain gap. Fast enough on modest hardware, though.
