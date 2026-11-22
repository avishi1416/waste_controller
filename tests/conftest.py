import pytest

try:
    import torch  # noqa: F401

    TORCH_OK = True
except Exception:  # ImportError, or OSError when Windows Application Control blocks torch DLLs
    TORCH_OK = False

needs_torch = pytest.mark.skipif(not TORCH_OK, reason="PyTorch cannot load on this machine")
