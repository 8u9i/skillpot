# .axon Usage Guide

## Installation

### Prerequisites
- Rust 1.75+ (for building from source)
- Python 3.10+ (for Python bindings)

### Build from Source

```bash
git clone <repo>/axon.git
cd axon

# Build all crates
cargo build --release

# The CLI binary is at: target/release/axon
# The shared library is at: target/release/libaxon_ffi.so
```

### Python Package

```bash
# Install from local source
pip install ./python

# Or use directly with the built shared library
cd python
python3 -c "import axon; print(axon.__version__)"
```

## CLI Usage

### Create a test .axon file

```bash
# Create a synthetic 1.1GB model with 17 tensors
axon create --model "MyModel-7B" --architecture llama model.axon
```

### Inspect an .axon file

```bash
# Show structure and tensor list
axon inspect model.axon

# Show raw hex dump of the first 256 bytes
axon inspect --hex model.axon
```

### List tensors

```bash
# Compact list
axon list model.axon

# Verbose with shapes and sizes
axon list --verbose model.axon
```

### Validate checksums

```bash
# Verify all tensor checksums
axon validate model.axon

# Skip checksum verification
axon validate --no-checksums model.axon
```

### Extract tensors

```bash
# Extract a single tensor as raw binary
axon extract model.axon --name layer_0_q --output layer_0_q.bin

# Extract all tensors
axon unpack model.axon --output ./weights/

# Extract all tensors as raw binaries (not .npy)
axon unpack model.axon --output ./weights/ --raw
```

### Pack tensors into .axon

```bash
# Pack from JSON manifest + data directory
axon pack \
  --manifest manifest.json \
  --data-dir ./weights/ \
  --output model.axon \
  --model "MyModel-7B" \
  --architecture llama
```

### Convert between formats

```bash
# Export .axon manifest as JSON
axon convert model.axon manifest.json
```

### Benchmark loading performance

```bash
# Run 10 load iterations
axon bench model.axon --iterations 10
```

## Python Usage

```python
import axon

# Load an .axon file
model = axon.load("model.axon")

# Print summary
print(model.summary())

# Access tensors
weights = model["layer_0_q"]  # raw bytes
print(f"Tensor size: {len(weights)} bytes")

# List all tensor names
for name in model.names:
    info = model.info(name)
    print(f"  {info.name}  {info.dtype_name}  {info.shape}")

# Use with numpy
import numpy as np
tensor = model["emb_weight"]
arr = np.frombuffer(tensor, dtype=np.float16).reshape(32000, 4096)
print(arr.shape)
```

## C FFI Usage

```c
#include "axon.h"
#include <stdio.h>

int main() {
    AxonHandle* handle = axon_open("model.axon");
    if (!handle) {
        printf("Failed to open\n");
        return 1;
    }

    uint64_t count = axon_tensor_count(handle);
    printf("Tensors: %lu\n", count);

    // Get info for first tensor
    char name[64];
    uint32_t dtype;
    uint32_t rank;
    uint64_t shape[8];
    uint64_t offset, size;

    axon_tensor_info(handle, 0, name, 64,
        &dtype, &rank, shape, &offset, &size);

    printf("Tensor 0: %s  dtype=%u  shape=[%lu, %lu]\n",
        name, dtype, shape[0], shape[1]);

    // Access data directly (zero-copy)
    uint64_t data_size;
    const float* data = (const float*)axon_tensor_data(handle, 0, &data_size);
    printf("Data pointer: %p, size: %lu bytes\n", (void*)data, data_size);

    axon_close(handle);
    return 0;
}
```

Compile with:
```bash
gcc -o example example.c -L/path/to/axon/target/release -laxon_ffi
LD_LIBRARY_PATH=/path/to/axon/target/release ./example
```

## Project Structure

```
axon/
├── Cargo.toml              # Workspace manifest
├── core/                   # Core library (axon-core)
│   └── src/
│       ├── lib.rs          # Public API
│       ├── header.rs       # AxonHeader (64 bytes)
│       ├── tensor.rs       # TensorDescriptor, DType, Affinity
│       ├── manifest.rs     # Manifest, AxonFile, AxonBuilder
│       ├── mmap_loader.rs  # MappedAxonFile, AxonBuilder
│       ├── checksum.rs     # XXH3 checksums
│       └── error.rs        # Error types
├── cli/                    # CLI application (axon)
│   └── src/main.rs
├── ffi/                    # C FFI library (libaxon_ffi)
│   └── src/lib.rs
├── python/                 # Python bindings
│   └── axon/
│       ├── __init__.py
│       ├── _bindings.py    # ctypes-based FFI + pure-Python fallback
│       └── setup.py
├── include/                # C header files
│   └── axon.h
├── docs/                   # Documentation
│   ├── spec.md             # Full binary specification
│   └── usage.md            # This file
├── examples/               # Example code
└── tests/                  # Test files
```
