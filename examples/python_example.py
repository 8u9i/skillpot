#!/usr/bin/env python3
"""
Example: Using the .axon Python bindings to load and inspect a model.

Usage:
    python3 examples/python_example.py output/test.axon
"""

import sys
import os

# Add the python package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import axon


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.axon>")
        sys.exit(1)

    path = sys.argv[1]

    # Load the axon file
    print(f"Loading: {path}")
    print(f"Axon version: {axon.__version__}")
    print()

    model = axon.load(path)

    # Print full summary
    print(model.summary())
    print()

    # Access individual tensors
    for name in model.names[:5]:  # First 5 tensors
        info = model.info(name)
        print(f"  Tensor: {info}")
        data = model[name]
        print(f"    Raw:   {len(data)} bytes")
        print(f"    Hex:   {data[:8].hex()}")

        # Try numpy
        try:
            arr = info.numpy()
            print(f"    NumPy: {arr.dtype} {arr.shape}")
        except Exception as e:
            print(f"    NumPy: not available ({e})")
        print()

    # Verify checksums
    print("Verifying checksums...")
    results = model.verify()
    all_ok = all(results.values())
    print(f"  All checksums valid: {all_ok}")
    print()

    print(f"Model: {model.model_name}")
    print(f"Total tensors: {model.tensor_count}")


if __name__ == "__main__":
    main()
