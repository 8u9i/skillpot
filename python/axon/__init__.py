"""Axon — Adaptive eXecutable Object Notation.

A binary-first, hardware-aligned container format for ML model weights.
"""

from axon._bindings import AxonFile, DType, load, __version__

try:
    from axon import torch_integration as torch
except ImportError:
    torch = None

__all__ = ["AxonFile", "DType", "load", "torch", "__version__"]
