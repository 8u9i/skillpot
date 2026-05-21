#ifndef AXON_H
#define AXON_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AxonHandle AxonHandle;

AxonHandle* axon_open(const char* path);
void axon_close(AxonHandle* handle);
uint64_t axon_tensor_count(const AxonHandle* handle);
uint64_t axon_payload_size(const AxonHandle* handle);
uint64_t axon_model_name(const AxonHandle* handle, char* buf, uint64_t buf_size);

int axon_tensor_info(
    const AxonHandle* handle, uint64_t index,
    char* name_buf, uint64_t name_buf_size,
    uint32_t* dtype_out, uint32_t* rank_out,
    uint64_t* shape_out, uint64_t* data_offset_out,
    uint64_t* data_size_out);

const void* axon_tensor_data(const AxonHandle* handle, uint64_t index, uint64_t* data_size);
uint64_t axon_verify_checksums(const AxonHandle* handle, uint64_t* failed_indices, uint64_t* failed_count);
uint64_t axon_version(char* buf, uint64_t buf_size);
uint64_t axon_last_error(char* buf, uint64_t buf_size);

#define AXON_DTYPE_F32      0
#define AXON_DTYPE_F16      1
#define AXON_DTYPE_BF16     2
#define AXON_DTYPE_I32      3
#define AXON_DTYPE_I64      4
#define AXON_DTYPE_U8       5
#define AXON_DTYPE_Q4       6
#define AXON_DTYPE_Q8       7
#define AXON_DTYPE_F8E4M3   8
#define AXON_DTYPE_F8E5M2   9
#define AXON_DTYPE_I8       10
#define AXON_DTYPE_I16      11

#ifdef __cplusplus
}
#endif

#endif
