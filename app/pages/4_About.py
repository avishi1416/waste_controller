"""About: what the prototype is, the data, and its limits."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import footer, legend, page  # noqa: E402

page("About this prototype")
st.markdown("""
**What it does.** One YOLO26n-seg model, fine-tuned on waste photos, finds each item in an image, outlines it
(instance segmentation) and names its material. Merging the outlines gives a per-pixel material map
(semantic segmentation). An optional readout shows what share of the items belong to a chosen recycling stream.

**Materials it knows**
""")
legend()
st.markdown("""
**Data.** TACO (Trash Annotations in Context): 1,500 photos of litter with hand-drawn outlines, mapped from 60 labels
to the five materials above. Cigarette butts and unlabelled litter were left out. Split 80/10/10 into train,
validation and test, with near-duplicate photos removed so none leak into the test set.

**Trained on a laptop.** Fine-tuned on a CPU (no GPU) at 416-pixel resolution, starting from a model pretrained on COCO.

**Limits you should know**
- TACO shows litter outdoors, not a conveyor belt. Expect weaker results on industrial sorting-line footage.
- Small or cluttered items are often missed. Clear, medium-sized items (bottles, cans, cartons, bags) work best.
- Materials that look alike (clear plastic vs glass, coated paper vs cardboard) get confused.
- Built in Python; the course brief asks for C, C++ or Java.
""")
footer()
