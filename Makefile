.PHONY: all build clean test install python-test cffi-test benchmark

# Default target: build everything
all: build

# Build Rust workspace (release mode)
build:
	cargo build --release

# Install Python package
python-install:
	pip install -e ./python

# Run Python example
python-test: build
	./target/release/axon create --model "TestModel" --architecture test output/test.axon 2>/dev/null
	python3 examples/python_example.py output/test.axon

# Compile and run C FFI example
cffi-test: build
	gcc -o examples/cffi_example examples/cffi_example.c \
		-I./include -L./target/release -laxon_ffi \
		-Wl,-rpath,./target/release
	./examples/cffi_example output/test.axon

# Run benchmark
benchmark: build
	./target/release/axon bench output/test.axon --iterations 10

# Run all validation
test: build
	./target/release/axon validate output/test.axon
	./target/release/axon list output/test.axon --verbose

# Clean everything
clean:
	rm -rf output/
	rm -f examples/cffi_example
	cargo clean

# Full pipeline: create → validate → inspect → unpack → list
pipeline: build
	@echo "=== CREATE ==="
	./target/release/axon create -m "Pipeline-Test" -a test output/model.axon
	@echo ""
	@echo "=== VALIDATE ==="
	./target/release/axon validate output/model.axon
	@echo ""
	@echo "=== INSPECT ==="
	./target/release/axon inspect output/model.axon
	@echo ""
	@echo "=== LIST ==="
	./target/release/axon list output/model.axon --verbose
	@echo ""
	@echo "=== UNPACK ==="
	./target/release/axon unpack output/model.axon --output output/weights/
	@echo ""
	@echo "=== EXTRACT ==="
	./target/release/axon extract output/model.axon -n emb_weight -o output/emb_weight.bin
	ls -lh output/emb_weight.bin
	@echo ""
	@echo "=== DONE ==="
