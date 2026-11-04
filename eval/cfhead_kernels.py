"""Optional access to separately installed upstream CFHead kernels."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


def load_upstream(relative: str, module_name: str):
    root = os.environ.get("CFHEAD_REPO")
    if not root:
        raise ImportError("Set CFHEAD_REPO to a separately obtained CFHead checkout")
    path = Path(root).expanduser().resolve() / relative
    if not path.is_file():
        raise ImportError(f"CFHead dependency not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


try:
    _kernels = load_upstream("kernel/cfhead_triton.py", "_mob_cfhead_triton")
    TRITON_AVAILABLE = bool(_kernels.TRITON_AVAILABLE)
except ImportError:
    _kernels = None
    TRITON_AVAILABLE = False


def coarse_triton(lut, index_t):
    if not TRITON_AVAILABLE:
        raise RuntimeError("Upstream CFHead Triton kernels are unavailable")
    return _kernels.cfhead_coarse_forward(lut, index_t)


def fine_triton(hidden, weight, candidate_idx):
    if not TRITON_AVAILABLE:
        raise RuntimeError("Upstream CFHead Triton kernels are unavailable")
    return _kernels.cfhead_fine_forward(hidden, weight, candidate_idx)
