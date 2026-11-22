"""Every Streamlit page renders without exceptions (headless AppTest)."""
from pathlib import Path

import pytest

from conftest import needs_torch

APP = Path(__file__).resolve().parents[1] / "app"


@needs_torch
@pytest.mark.parametrize("script", ["Home.py", "pages/2_Evaluation.py", "pages/3_Fundamentals.py", "pages/4_About.py"])
def test_page_renders(script):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP / script), default_timeout=120).run()
    assert not at.exception, at.exception
    assert at.title  # every page sets a title
