#ifndef METAL_IMPL_H
#define METAL_IMPL_H

#include <stdbool.h>  // For bool
#include <stddef.h>
#include <stdint.h>

// Forward declare opaque pointers if needed by Objective-C implementation
typedef void* MTLDevice_t;
typedef void* MTLCommandQueue_t;
typedef void* MTLBuffer_t;
typedef void* MTLLibrary_t;
typedef void* MTLFunction_t;
typedef void* MTLComputePipelineState_t;
typedef void* MTLCommandBuffer_t;
typedef void* MTLComputeCommandEncoder_t;
typedef void* MTLBlitCommandEncoder_t;
typedef void* MTLCompileOptions_t;
typedef void* NSString_t;  // For returning strings

// --- General ---
void metal_release_object(void* obj);  // Generic release for ARC/manual release
const char* metal_string_to_utf8(NSString_t ns_str);  // Helper for NSString

// --- Device Management ---
MTLDevice_t metal_create_system_default_device(void);
NSString_t metal_device_get_name(
    MTLDevice_t device);  // Returns retained NSString

// --- Command Queue ---
MTLCommandQueue_t metal_device_new_command_queue(MTLDevice_t device);

// --- Buffer Management ---
MTLBuffer_t metal_device_new_buffer(
    MTLDevice_t device, size_t length,
    unsigned int options);  // options is MTLResourceOptions
MTLBuffer_t metal_device_new_buffer_with_bytes_no_copy(
    MTLDevice_t device, void* bytes, size_t length,
    unsigned int options);  // options is MTLResourceOptions

void* metal_buffer_contents(
    MTLBuffer_t buffer);  // Returns raw pointer, NULL if not CPU accessible
size_t metal_buffer_length(MTLBuffer_t buffer);
void metal_buffer_did_modify_range(MTLBuffer_t buffer, size_t offset,
                                   size_t length);  // For managed buffers

// --- Compile Options ---
MTLCompileOptions_t metal_create_compile_options(void);
void metal_compile_options_set_fast_math_enabled(MTLCompileOptions_t options,
                                                 bool enabled);
// Add other option setters here...

// --- Library Management ---
MTLLibrary_t metal_device_new_library_with_source(
    MTLDevice_t device, const char* source,
    MTLCompileOptions_t options);  // options can be NULL
MTLLibrary_t metal_device_new_library_with_data(MTLDevice_t device,
                                                const void* data,
                                                size_t length);

// --- Function Management ---
MTLFunction_t metal_library_new_function_with_name(MTLLibrary_t library,
                                                   const char* name);

// --- Pipeline State Management ---
MTLComputePipelineState_t metal_device_new_compute_pipeline_state_with_function(
    MTLDevice_t device, MTLFunction_t function);
size_t metal_pipeline_state_get_max_total_threads_per_threadgroup(
    MTLComputePipelineState_t pipeline_state);

// --- Command Buffer Management ---
MTLCommandBuffer_t metal_queue_new_command_buffer(MTLCommandQueue_t queue);
void metal_command_buffer_commit(MTLCommandBuffer_t buffer);
void metal_command_buffer_wait_until_completed(MTLCommandBuffer_t buffer);

// --- Compute Command Encoder ---
MTLComputeCommandEncoder_t metal_command_buffer_compute_command_encoder(
    MTLCommandBuffer_t buffer);
void metal_compute_command_encoder_set_pipeline_state(
    MTLComputeCommandEncoder_t encoder,
    MTLComputePipelineState_t pipeline_state);
void metal_compute_command_encoder_set_buffer(
    MTLComputeCommandEncoder_t encoder, MTLBuffer_t buffer, size_t offset,
    uint32_t index);  // offset is size_t now
void metal_compute_command_encoder_set_bytes(MTLComputeCommandEncoder_t encoder,
                                             const void* bytes, size_t length,
                                             uint32_t index);
void metal_compute_command_encoder_dispatch_thread_groups(
    MTLComputeCommandEncoder_t encoder, size_t grid_x, size_t grid_y,
    size_t grid_z, size_t thread_x, size_t thread_y, size_t thread_z);
void metal_compute_command_encoder_end_encoding(
    MTLComputeCommandEncoder_t encoder);

// --- Blit Command Encoder ---
MTLBlitCommandEncoder_t metal_command_buffer_blit_command_encoder(
    MTLCommandBuffer_t buffer);
void metal_blit_command_encoder_copy_from_buffer(
    MTLBlitCommandEncoder_t encoder, MTLBuffer_t src_buffer, size_t src_offset,
    MTLBuffer_t dst_buffer, size_t dst_offset, size_t size);
void metal_blit_command_encoder_fill_buffer(MTLBlitCommandEncoder_t encoder,
                                            MTLBuffer_t buffer, size_t offset,
                                            size_t length, uint8_t value);
void metal_blit_command_encoder_end_encoding(
    MTLBlitCommandEncoder_t encoder);  // New distinct function

#endif  // METAL_IMPL_H