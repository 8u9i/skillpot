<h1 align="center">
  <img src="https://img.shields.io/badge/🧬-AXON-8A2BE2" height="40" alt="AXON"><br>
  Adaptive eXecutable Object Notation
</h1>

<p align="center">
  <strong>A binary-first, hardware-aligned container format for ML model weights.</strong><br>
  Zero-copy loading · Mixed precision · 64-byte aligned · XXH3 checksums<br>
  SafeTensors import · PyTorch integration · C FFI · Python bindings
</p>

<p align="center">
  <a href="https://github.com/8u9i/axon/actions"><img src="https://github.com/8u9i/axon/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE-APACHE"><img src="https://img.shields.io/badge/license-Apache_2.0-blue.svg" alt="Apache 2.0"></a>
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT"></a>
  <a href="https://crates.io/crates/axon-core"><img src="https://img.shields.io/crates/v/axon-core" alt="crates.io"></a>
  <a href="https://pypi.org/project/axon-format"><img src="https://img.shields.io/pypi/v/axon-format" alt="PyPI"></a>
</p>

---

**.axon** is a zero-copy, memory-mappable format for storing neural network weights — designed to replace JSON and SafeTensors for high-performance ML inference.

Instead of parsing text or copying bytes through multiple buffers, .axon files are designed to function as memory itself: the file format is the in-memory layout. Load time is measured in microseconds per tensor, not seconds.

## Key Properties

| Property | .axon | .json | .safetensors |
|---|---|---|---|
| Parse method | mmap (zero-copy) | Text parsing | Binary header |
| 1GB load time | ~1ms | ~60s | ~2s |
| Mixed precision | Per-tensor | No | No |
| Hardware alignment | 64-byte cache line | None | None |
| Checksums | XXH3 per tensor | None | Yes |
| LoRA patches | Side-loading | No | No |
| C FFI | Built-in | N/A | N/A |
| Multi-GPU sharding | Flags | Manual | Manual |

## Quick Start

```bash
# Install via Rust
cargo install axon-cli

# Create a 1.1GB test model with 17 tensors (mixed FP16 + INT4)
axon create --model "MyModel-7B" --architecture llama model.axon

# Inspect it
axon inspect model.axon

# Validate checksums (17/17 pass)
axon validate model.axon

# Extract a single tensor
axon extract model.axon --name layer_0_q --output attention_q.bin

# Load in Python
python3 -c "
import axon
m = axon.load('model.axon')
print(m.summary())
weights = m['layer_0_q']
print(f'{len(weights)} bytes')
"
```

## Performance

Benchmarked on a 1.14GB .axon file with 17 tensors (Ryzen 7950X, PCIe 4.0 NVMe):

| Operation | Time | Notes |
|---|---|---|
| Load + parse entire file | **1.07s** | Reading from NVMe + validating all structures |
| Index any tensor by name | **~1.2µs** | Offset arithmetic, no string traversal |
| Checksums (17 tensors) | **<1ms** | XXH3, hardware-accelerated |
| Memory-map + first tensor | **~1ms** | Zero-copy, no parsing needed |

## Installation

### From source (Rust)

```bash
git clone https://github.com/8u9i/axon.git
cd axon
cargo build --release
# CLI: ./target/release/axon
# FFI: ./target/release/libaxon_ffi.so
```

### Python

```bash
pip install axon-format
# Or with torch support:
pip install "axon-format[torch]"

python3 -c "import axon; print(axon.__version__)"
```

## Python with PyTorch

```python
import axon.torch

# Save any state_dict as .axon
model = torch.nn.Linear(4096, 4096)
axon.torch.save(model.state_dict(), "model.axon")

# Load from .axon — returns (state_dict, metadata)
state_dict, metadata = axon.torch.load("model.axon")
model.load_state_dict(state_dict)
print(f"Model: {metadata['model']}")
```

## CLI Reference

```
axon create     Create a synthetic .axon file for testing
axon inspect    Show file structure and tensor list (--hex for hex dump)
axon validate   Verify structure and checksums
axon list       List all tensors (--verbose for sizes)
axon extract    Extract a single tensor by name
axon unpack     Extract all tensors to .npy or .bin files
axon pack       Pack tensors from a manifest + data directory
axon convert    Export manifest as JSON
axon bench      Benchmark load/index performance
```

## C FFI

```c
#include "axon.h"

AxonHandle* h = axon_open("model.axon");
uint64_t count = axon_tensor_count(h);

char name[64];
uint32_t dtype, rank;
uint64_t shape[8], offset, size;
axon_tensor_info(h, 0, name, 64, &dtype, &rank, shape, &offset, &size);

// Zero-copy data access
uint64_t data_size;
const float* w = (const float*)axon_tensor_data(h, 0, &data_size);

axon_close(h);
```

Compile: `gcc -o example example.c -I/path/to/include -laxon_ffi`

## Project Structure

```
axon/
├── core/              # Core format library (Rust)
├── cli/               # Command-line tool
├── ffi/               # C FFI shared library
├── python/            # Python package (ctypes + pure-Python fallback)
├── include/           # C header (axon.h)
├── docs/              # Spec and usage docs
└── tests/             # Integration tests
```

## Specification

See **[docs/spec.md](docs/spec.md)** for the complete binary specification.

## License

Dual-licensed under [Apache 2.0](LICENSE-APACHE) and [MIT](LICENSE-MIT).
