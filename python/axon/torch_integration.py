"""PyTorch integration for .axon files.

Provides drop-in replacements for torch.save() and torch.load()
that use .axon as the storage backend for state dicts.

Usage:
    import axon
    import torch

    # Save a model's state_dict as .axon
    model = torch.nn.Linear(10, 10)
    axon.torch.save(model.state_dict(), "model.axon")

    # Load state_dict from .axon
    state_dict = axon.torch.load("model.axon")
    model.load_state_dict(state_dict)
"""

import io
import struct

import torch

from axon._bindings import AxonFile, DType, load, AXON_MAGIC

# ── DType mapping: PyTorch → Axon ─────────────────────────────────

_TORCH_TO_AXON = {
    torch.float32: DType.F32,
    torch.float16: DType.F16,
    torch.bfloat16: DType.BF16,
    torch.int32: DType.I32,
    torch.int64: DType.I64,
    torch.uint8: DType.U8,
    torch.int8: DType.I8,
    torch.int16: DType.I16,
    torch.float64: DType.F32,  # fallback: downcast to F32
}

_AXON_TO_TORCH = {
    DType.F32: torch.float32,
    DType.F16: torch.float16,
    DType.BF16: torch.bfloat16,
    DType.I32: torch.int32,
    DType.I64: torch.int64,
    DType.U8: torch.uint8,
    DType.I8: torch.int8,
    DType.I16: torch.int16,
}


def _check_axon_available() -> bool:
    """Check if the native axon library is available (has the axon CLI)."""
    import subprocess
    import shutil
    return shutil.which("axon") is not None or any(
        (Path(__file__).parent.parent.parent / "target" / d / "axon").exists()
        for d in ["release", "debug"]
    )


def _use_cli_fallback() -> bool:
    return not _check_axon_available()


def save(state_dict: dict, path: str, model_name: str = None) -> None:
    """Save a PyTorch state_dict as an .axon file.

    Args:
        state_dict: Ordered dict of tensor name → tensor.
        path: Output .axon file path.
        model_name: Optional model identifier stored in the manifest.
    """
    from pathlib import Path
    import subprocess
    import tempfile
    import json

    # Build manifest JSON for the axon CLI pack command
    tensors = []
    data_dir = Path(path).parent / f".axon_tmp_{Path(path).stem}"
    data_dir.mkdir(exist_ok=True)

    try:
        for name, tensor in state_dict.items():
            t = tensor.contiguous()  # ensure contiguous memory

            # Map dtype
            axon_dtype = _TORCH_TO_AXON.get(t.dtype)
            if axon_dtype is None:
                raise ValueError(f"Unsupported dtype {t.dtype} for tensor '{name}'. "
                                 f"Supported: {list(_TORCH_TO_AXON.keys())}")

            # Write raw tensor bytes
            raw_path = data_dir / name.replace('/', '_')
            raw_path.write_bytes(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.view(torch.float16).numpy().tobytes())

            tensors.append({
                "name": name,
                "dtype": axon_dtype,
                "shape": list(t.shape),
            })

        manifest = {
            "model": model_name or "",
            "tensors": tensors,
        }

        manifest_path = data_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        # Use the axon CLI to pack
        cmd = [
            "axon", "pack",
            "--manifest", str(manifest_path),
            "--data-dir", str(data_dir),
            "--output", str(Path(path).absolute()),
        ]
        if model_name:
            cmd.extend(["--model", model_name])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"axon pack failed: {result.stderr}")

    finally:
        # Clean up temp files
        import shutil
        shutil.rmtree(data_dir, ignore_errors=True)


def load_state_dict(path: str, map_location=None) -> dict:
    """Load a PyTorch state_dict from an .axon file.

    Args:
        path: Path to the .axon file.
        map_location: Torch device to load tensors to (e.g., 'cuda:0').

    Returns:
        Ordered dict of tensor name → torch.Tensor.
    """
    from pathlib import Path

    axon_file = AxonFile(path)

    state_dict = {}
    for name in axon_file.names:
        info = axon_file.info(name)
        raw_bytes = axon_file[name]

        # Map Axon dtype → Torch dtype
        torch_dtype = _AXON_TO_TORCH.get(info.dtype)
        if torch_dtype is None:
            raise ValueError(f"Unsupported axon dtype code {info.dtype} for tensor '{name}'")

        # Convert raw bytes → numpy → torch
        import numpy as np

        np_dtype = {
            DType.F32: np.float32,
            DType.F16: np.float16,
            DType.BF16: np.float16,  # stored as f16, view to bf16
            DType.I32: np.int32,
            DType.I64: np.int64,
            DType.U8: np.uint8,
            DType.I8: np.int8,
            DType.I16: np.int16,
        }.get(info.dtype, np.float32)

        arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(info.shape)
        tensor = torch.from_numpy(arr.copy())

        # Handle BF16: stored as F16 in numpy, view as BF16
        if info.dtype == DType.BF16:
            tensor = tensor.view(torch.bfloat16)

        if map_location:
            tensor = tensor.to(map_location)

        state_dict[name] = tensor

    return state_dict


def load(path: str, map_location=None):
    """Load a complete .axon file and return (state_dict, metadata).

    This is the recommended entry point for PyTorch users.

    Args:
        path: Path to the .axon file.
        map_location: Torch device to load tensors to.

    Returns:
        Tuple of (state_dict: dict, metadata: dict)
    """
    axon_file = AxonFile(path)
    state_dict = {}
    for name in axon_file.names:
        info = axon_file.info(name)
        raw_bytes = axon_file[name]

        torch_dtype = _AXON_TO_TORCH.get(info.dtype)
        if torch_dtype is None:
            raise ValueError(f"Unsupported axon dtype code {info.dtype} for tensor '{name}'")

        import numpy as np
        np_dtype = {
            DType.F32: np.float32, DType.F16: np.float16,
            DType.BF16: np.float16, DType.I32: np.int32,
            DType.I64: np.int64, DType.U8: np.uint8,
            DType.I8: np.int8, DType.I16: np.int16,
        }.get(info.dtype, np.float32)

        arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(info.shape)
        tensor = torch.from_numpy(arr.copy())

        if info.dtype == DType.BF16:
            tensor = tensor.view(torch.bfloat16)

        if map_location:
            tensor = tensor.to(map_location)

        state_dict[name] = tensor

    metadata = {
        "model": axon_file.model_name,
        "tensor_count": axon_file.tensor_count,
    }

    return state_dict, metadata


def is_axon_file(path: str) -> bool:
    """Check if a file is an .axon file by reading its magic bytes."""
    with open(path, "rb") as f:
        magic = f.read(4)
    return magic == AXON_MAGIC


# Register .axon as a known file format
# Users can do: torch.load("model.axon") and it will work if they've
# already loaded this module.
_ORIGINAL_TORCH_LOAD = torch.load


def _patched_torch_load(f, *args, **kwargs):
    """Monkey-patch torch.load to handle .axon files transparently."""
    import pathlib
    if isinstance(f, (str, pathlib.PurePath)):
        path = str(f)
        if path.endswith('.axon') or (isinstance(f, str) and is_axon_file(path)):
            sd, meta = load(path, kwargs.get('map_location'))
            return sd
    return _ORIGINAL_TORCH_LOAD(f, *args, **kwargs)
