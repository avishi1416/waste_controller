"""Shared Streamlit helpers: page setup, styling, cached model, material legend."""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wastectl.config import load_config  # noqa: E402

LABELS = {"cardboard": "Cardboard", "soft_plastic": "Soft plastic", "rigid_plastic": "Rigid plastic",
          "metal": "Metal", "other": "Other / non-recyclable"}
HEX = {"cardboard": "#E69F00", "soft_plastic": "#56B4E9", "rigid_plastic": "#009E73",
       "metal": "#D55E00", "other": "#CC79A7"}  # Okabe-Ito, matches the pipeline overlay

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button { font-family: 'Public Sans', system-ui, sans-serif; }
h1 { font-weight: 700; letter-spacing: -0.02em; font-size: 2.1rem; margin-bottom: 0.2rem; }
h2, h3 { font-weight: 700; letter-spacing: -0.01em; }
.lede { font-size: 1.05rem; max-width: 62ch; color: #3C5650; margin-bottom: 1.2rem; }
.bin { display: inline-flex; align-items: center; gap: .5rem; padding: .35rem .7rem .35rem .4rem;
       margin: 0 .4rem .4rem 0; border-radius: 4px; background: #FFFFFF; border: 1px solid #CFDAD7; font-size: .92rem; }
.bin i { width: .9rem; height: 1.3rem; border-radius: 2px; display: inline-block; }
.bin b { font-weight: 700; margin-left: .2rem; }
.note { font-size: .82rem; color: #5B726D; border-top: 1px solid #CFDAD7; padding-top: .6rem; margin-top: 2rem; }
.empty { padding: 2.5rem 1.5rem; border: 1px dashed #A9BDB8; border-radius: 6px; color: #3C5650; background: #FBFDFC; }
</style>
"""


def page(title, lede=None):
    st.set_page_config(page_title=f"{title} | Waste Controller", page_icon="♻️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    if lede:
        st.markdown(f'<p class="lede">{lede}</p>', unsafe_allow_html=True)


@st.cache_data
def config():
    cfg = load_config()
    for k, v in cfg["paths"].items():  # resolve relative paths against the repo, not the launch folder
        cfg["paths"][k] = str(ROOT / v)
    cfg["models"]["yolo"]["weights"] = str(ROOT / cfg["models"]["yolo"]["weights"])
    return cfg


@st.cache_resource
def pipeline():
    """Returns (Pipeline, None) or (None, reason) when the model isn't trained yet."""
    from wastectl.vision.pipeline import Pipeline

    try:
        return Pipeline(config(), device="cpu", use_classifier=False), None
    except FileNotFoundError as e:
        return None, str(e)


def legend(counts=None):
    chips = []
    for c, name in LABELS.items():
        n = f"<b>{counts.get(c, 0)}</b>" if counts is not None else ""
        chips.append(f'<span class="bin"><i style="background:{HEX[c]}"></i>{name}{n}</span>')
    st.markdown("".join(chips), unsafe_allow_html=True)


def footer():
    st.markdown(
        '<p class="note">Prototype for CSAI3027P Deep Learning PBL, Project 07. Trained on the TACO dataset; '
        "for education and research use only. Predictions can be wrong: check before acting on them.</p>",
        unsafe_allow_html=True,
    )
