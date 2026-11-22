# Demo script: 4 minutes

**Before you start:** laptop plugged in, notifications off, the app running (`streamlit run app/Home.py`) in one browser tab,
and `docs/slides/waste_controller.pptx` open in PowerPoint. The speaker notes on each slide match this script.

| Time | Slide / screen | Say (roughly) | Do |
|---|---|---|---|
| 0:00–0:20 | 1. Title | "This is Waste Controller: a prototype that looks at a photo of waste, finds every item, outlines it, and names its material. I trained it entirely on this laptop, without a GPU." | — |
| 0:20–0:50 | 2. Problem | "Recycling streams lose value when the wrong material gets in. A sorting system has to know what each item is and exactly where it is. So one model does three jobs: detection, segmentation and material classification." | — |
| 0:50–1:20 | 3. Data | "I used TACO: 1,500 real photos with hand-drawn outlines. I mapped 60 labels to five materials, removed near-duplicates so test photos can't leak into training, and split 80/10/10. Cardboard is the rarest class." | Point at the cardboard bar |
| 1:20–1:50 | 4. Model | "The model is YOLO26n-seg, a small network that predicts boxes, materials and outlines in one pass, so it runs on a CPU. I fine-tuned COCO-pretrained weights at 416 pixels: about four minutes per epoch on this laptop." | — |
| 1:50–2:50 | **Live app** | "Here it is running. I'll pick a test photo it never saw… each item gets an outline in its material colour, a box and a confidence. The counts update, and this readout shows what share of the items belong to the plastic stream." | Switch to the browser → *Waste Analyzer* → pick a test photo → move the confidence slider once → pick a second photo with several items |
| 2:50–3:20 | 6. Results | "On 150 held-out photos it reaches these scores. It's strongest on common classes and weakest on cardboard, and it runs in real time on a laptop CPU." | Point at the per-material chart |
| 3:20–3:40 | 7. Fundamentals | "Behind it, I built the course fundamentals from scratch in NumPy: backprop verified with gradient checking, four optimizers, classical learning rules, and ablations showing overfitting." | — |
| 3:40–4:00 | 8. Close | "It works end to end, but it was trained on outdoor litter, so it's a demonstrator, not yet a production sorter. Next: recycling-plant data on a GPU, a stream monitor and synthetic data for rare items. Thank you." | — |

**If the live demo fails:** stay on slide 5 (example predictions) and describe those instead. It covers the same point.

**Recording tips:** use Windows + Alt + R (Xbox Game Bar) or PowerPoint's *Record* tab; speak slightly slower than feels
natural; keep the mouse still when you're not pointing at something.
