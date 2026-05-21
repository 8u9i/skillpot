/**
 * Example: Using the .axon C FFI to load and inspect a model.
 *
 * Compile:
 *   gcc -o cffi_example cffi_example.c -I../include -L../target/release -laxon_ffi
 *   LD_LIBRARY_PATH=../target/release ./cffi_example ../output/test.axon
 */

#include <stdio.h>
#include <stdlib.h>
#include "axon.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.axon>\n", argv[0]);
        return 1;
    }

    // Open the file
    AxonHandle* handle = axon_open(argv[1]);
    if (!handle) {
        fprintf(stderr, "Failed to open %s\n", argv[1]);
        return 1;
    }

    // Get version
    char ver[64];
    axon_version(ver, 64);
    printf("Library: %s\n", ver);

    // Model info
    char model_name[256];
    axon_model_name(handle, model_name, 256);
    printf("Model:   %s\n", model_name);
    printf("Tensors: %lu\n", axon_tensor_count(handle));
    printf("Payload: %.2f MB\n\n", axon_payload_size(handle) / (1024.0 * 1024.0));

    // Iterate all tensors
    uint64_t count = axon_tensor_count(handle);
    for (uint64_t i = 0; i < count && i < 10; i++) {
        char name[64];
        uint32_t dtype, rank;
        uint64_t shape[8], offset, size;

        if (axon_tensor_info(handle, i, name, 64, &dtype, &rank, shape, &offset, &size)) {
            // DType names
            const char* dtype_names[] = {
                "FP32", "FP16", "BF16", "I32", "I64", "U8",
                "Q4", "Q8", "FP8_E4M3", "FP8_E5M2", "I8", "I16"
            };
            const char* dn = (dtype <= 11) ? dtype_names[dtype] : "UNKNOWN";

            printf("[%2lu] %-30s %-8s [", i, name, dn);
            for (uint32_t d = 0; d < rank; d++) {
                if (d > 0) printf(", ");
                printf("%lu", shape[d]);
            }
            printf("]  %lu bytes", size);

            // Get data pointer
            uint64_t data_size;
            const void* data = axon_tensor_data(handle, i, &data_size);
            if (data) {
                printf("  ptr=%p", data);
            }
            printf("\n");
        }
    }

    // Verify checksums
    uint64_t failed_buf[256];
    uint64_t failed_count;
    uint64_t valid = axon_verify_checksums(handle, failed_buf, &failed_count);
    printf("\nChecksums: %lu valid, %lu failed\n", valid, failed_count);

    axon_close(handle);
    return 0;
}
