# Model card: Waste Controller YOLO26n-seg

## What it is
- **Architecture:** Ultralytics YOLO26n-seg, a one-stage detector with an instance-segmentation head
  (2.7 M parameters, 9.1 GFLOPs at 416 px).
- **Starting point:** COCO-pretrained `yolo26n-seg.pt`.
- **Fine-tuned on:** TACO mapped to 5 materials. See [dataset_card.md](dataset_card.md).
- **Output per object:** bounding box, outline mask, material (cardboard / soft plastic / rigid plastic / metal / other)
  and confidence. Merged masks give a per-pixel material map.

## Training
| Setting | Value | Why |
|---|---|---|
| Hardware | HP OmniBook 5, Snapdragon X CPU, 8 GB RAM, no GPU | Owner's laptop |
| Image size | 416 | ~2.5× faster than 640 on CPU |
| Batch size | 8 | Fits 8 GB RAM |
| Epochs | 30 | ~4 min/epoch measured, so ~2 h |
| Frozen layers | first 10 (backbone) | Faster CPU training; keeps pretrained features |
| Workers | 2 | RAM headroom |
| Seed | 42, deterministic | Reproducibility |
| Optimiser, LR schedule, augmentation | Ultralytics defaults (auto optimiser, mosaic, HSV, flips) | Tuned defaults |

Every run is logged in `experiments/experiments.jsonl` (config, seed, git commit, metrics, hardware, wall time).

## Results
1-epoch timing run (validation split): box mAP@0.5 0.137, mask mAP@0.5 0.127; 36 ms per image inference on CPU.

Full run (test split, 150 photos): _filled in after training_.

| Metric | Test |
|---|---|
| Mask mAP@0.5 | _pending_ |
| Mask mAP@0.5:0.95 | _pending_ |
| Box mAP@0.5 | _pending_ |
| Box mAP@0.5:0.95 | _pending_ |
| End-to-end CPU latency | _pending_ |

## Intended use
- Demonstrating waste detection and segmentation for a course project; exploring what a sorting-line vision
  system would see.
- **Not for:** operational sorting decisions, safety-critical use, or commercial use (data licences are
  education/research).

## Limitations
- Trained on outdoor litter; it will degrade on conveyor-belt footage until fine-tuned on such data (WaRP, ZeroWaste).
- Small, occluded or cluttered items are often missed. Look-alike materials (clear plastic vs glass, coated paper vs
  cardboard) are confused.
- Cardboard has the fewest training examples and is the least reliable class.
- CPU-only training for 30 epochs at 416 px: accuracy is well below what a GPU run at 640 px would reach.

## Responsible use
- Show confidence with every prediction (the app does) and keep a human in the loop.
- Re-evaluate on data from the target site before relying on it. The model has never seen a recycling facility.
