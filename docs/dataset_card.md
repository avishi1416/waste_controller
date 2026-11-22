# Dataset card: TACO, mapped to five waste materials

## Source
- **TACO (Trash Annotations in Context)**, Proença & Simões, 2020. [arXiv:2003.06975](https://arxiv.org/abs/2003.06975) ·
  [github.com/pedropro/TACO](https://github.com/pedropro/TACO) · dataset DOI 10.5281/zenodo.3242156
- 1,500 photos of litter in real environments (streets, beaches, grass, indoors), mostly taken with phones.
  4,784 annotations across 60 categories, each with a polygon outline (instance segmentation) and a bounding box.
- Downloaded 22 Nov 2026: `annotations.json` from the GitHub repo, plus the 640 px images served by `img.tacodataset.org`.

## Licence and use
- Images are hosted on Flickr and each carries **its own licence**. Annotations are reported as CC BY 4.0
  (per a mirror; not confirmed on the original release).
- This project uses the data for **education and research only**. Raw images, derived images and trained weights
  are **not** published in the repository.
- Cite: Proença, P. F., & Simões, P. (2020). *TACO: Trash Annotations in Context for Litter Detection.* arXiv:2003.06975.

## Label mapping (60 → 5)
Defined in [`configs/class_map.yaml`](../configs/class_map.yaml); every TACO label must map, or preparation fails.

| Material | TACO labels |
|---|---|
| cardboard | Corrugated carton, Other carton, Egg carton, Meal carton, Pizza box, Toilet tube |
| soft_plastic | Plastic film, Other plastic wrapper, Garbage bag, Single-use carrier bag, Polypropylene bag, Crisp packet, Six pack rings, Plastic gloves |
| rigid_plastic | Clear / Other plastic bottle, Plastic bottle cap, Plastic lid, Disposable / Other plastic cup, Spread tub, Tupperware, Disposable / Other plastic container, Plastic utensils, Plastic straw, Squeezable tube |
| metal | Drink can, Food can, Aerosol, Metal bottle cap, Metal lid, Pop tab, Scrap metal, Aluminium foil, Aluminium blister pack |
| other | All glass, all paper (incl. paper cup/straw/bag), Drink carton, foam / styrofoam, Other plastic, Battery, Carded blister pack, Food waste, Rope & strings, Shoe |
| *dropped* | Cigarette (667), Unlabeled litter (517): tiny, ambiguous items that would dominate `other` |

Judgement calls: paper goes to `other` (the five classes follow a plastics/metal/cardboard sorting line);
drink cartons are composites, so `other`; "Other plastic" is ambiguous, so `other`.

## Preparation
1. **Orientation check.** Each image is EXIF-rotated and compared with the annotation's width/height;
   mismatches would be skipped (0 found).
2. **One polygon per object.** Multi-part outlines keep their largest part (a YOLO-seg format constraint).
3. **Near-duplicate removal.** 64-bit difference hash; images within Hamming distance 3 of an already-kept image
   are removed, with test images kept first. **3 removed.**
4. **Split** 80/10/10 into train/val/test, seed 42 (random by image; TACO has no sequence structure).
5. Visual check: label overlays drawn on sample images match the objects.

## Counts (objects per split)

| Material | Train | Val | Test |
|---|---:|---:|---:|
| cardboard | 156 | 27 | 23 |
| soft_plastic | 706 | 71 | 76 |
| rigid_plastic | 771 | 103 | 107 |
| metal | 424 | 59 | 67 |
| other | 842 | 96 | 67 |
| **images** | **1,197** | **150** | **150** |

Source of truth: `experiments/data/class_counts.csv` and `experiments/data/manifest.json`.

## Known biases and gaps
- **Domain:** outdoor litter, not a recycling conveyor belt (different lighting, backgrounds, clutter and scale).
- **Imbalance:** cardboard is under-represented (156 training objects vs 842 for `other`).
- **Geography and capture:** mostly European photos taken by volunteers with phones.
- **Not used today:** WaRP (industrial conveyor) and ZeroWaste (recycling plant) would close the domain gap.
  See CLAUDE.md, Round 5.
