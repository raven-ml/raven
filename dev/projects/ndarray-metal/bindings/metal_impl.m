#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>

#include "metal_impl.h"

// --- General ---

void metal_release_object(void* obj) {
  @autoreleasepool {
    if (obj != NULL) {
      id objc_obj = ((__bridge_transfer id)obj);
      // ARC automatically handles the release when objc_obj goes out of scope.
      (void)objc_obj; // Avoid unused variable warning
    }
  }
}

const char* metal_string_to_utf8(NSString_t ns_str) {
  @autoreleasepool {
    if (ns_str == NULL) {
      return "";
    }
    // The returned pointer is valid only until the pool is drained or the string released.
    return [(__bridge NSString*)ns_str UTF8String];
  }
}

// --- Device Management ---

MTLDevice_t metal_create_system_default_device(void) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      printf("Debug: MTLCreateSystemDefaultDevice returned NULL\n");
      return NULL;
    }
    return ((__bridge_retained void*)device);
  }
}

NSString_t metal_device_get_name(MTLDevice_t device_ptr) {
  @autoreleasepool {
    if (device_ptr == NULL) {
      return NULL;
    }
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    NSString* name = [device name];
    if (!name) {
      return NULL;
    }
    return ((__bridge_retained void*)[name copy]);
  }
}

// --- Command Queue ---

MTLCommandQueue_t metal_device_new_command_queue(MTLDevice_t device_ptr) {
  @autoreleasepool {
    if (device_ptr == NULL) {
      return NULL;
    }
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    return ((__bridge_retained void*)queue);
  }
}

// --- Buffer Management ---

MTLBuffer_t metal_device_new_buffer(MTLDevice_t device_ptr, size_t length,
                                    unsigned int options) {
  @autoreleasepool {
    if (device_ptr == NULL) {
      return NULL;
    }
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    id<MTLBuffer> buffer = [device newBufferWithLength:length
                                               options:(MTLResourceOptions)options];
    return ((__bridge_retained void*)buffer);
  }
}

MTLBuffer_t metal_device_new_buffer_with_bytes_no_copy(MTLDevice_t device_ptr, void* bytes,
                                                       size_t length, unsigned int options) {
  @autoreleasepool {
    if (device_ptr == NULL || bytes == NULL) {
      return NULL;
    }
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:bytes
                                                     length:length
                                                    options:(MTLResourceOptions)options
                                                deallocator:nil];
    return ((__bridge_retained void*)buffer);
  }
}

void* metal_buffer_contents(MTLBuffer_t buffer_ptr) {
  @autoreleasepool {
    if (buffer_ptr == NULL) {
      return NULL;
    }
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffer_ptr;
    return [buffer contents];
  }
}

size_t metal_buffer_length(MTLBuffer_t buffer_ptr) {
  @autoreleasepool {
    if (buffer_ptr == NULL) {
      return 0;
    }
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffer_ptr;
    return [buffer length];
  }
}

void metal_buffer_did_modify_range(MTLBuffer_t buffer_ptr, size_t offset, size_t length) {
    @autoreleasepool {
        if (buffer_ptr == NULL || length == 0) {
            return;
        }
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffer_ptr;
        NSRange range = NSMakeRange(offset, length);
        [buffer didModifyRange:range];
    }
}

// --- Compile Options ---

MTLCompileOptions_t metal_create_compile_options(void) {
    @autoreleasepool {
        // Class
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        return ((__bridge_retained void*)options);
    }
}

void metal_compile_options_set_fast_math_enabled(MTLCompileOptions_t options_ptr, bool enabled) {
    @autoreleasepool {
        if (options_ptr == NULL) {
            return;
        }
        MTLCompileOptions *options = (__bridge MTLCompileOptions*)options_ptr;
        options.mathMode = enabled ? MTLMathModeFast : MTLMathModeSafe;
    }
}

// --- Library Management ---

MTLLibrary_t metal_device_new_library_with_source(MTLDevice_t device_ptr, const char* source,
                                                  MTLCompileOptions_t compile_options_ptr) {
  @autoreleasepool {
    if (device_ptr == NULL || source == NULL) {
      return NULL;
    }
    NSError* err = nil;
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    MTLCompileOptions* options = (__bridge MTLCompileOptions*)compile_options_ptr;

    NSString *sourceString = [NSString stringWithUTF8String:source ?: ""];

    id<MTLLibrary> library = [device newLibraryWithSource:sourceString
                                                  options:options
                                                    error:&err];
    if (err) {
      printf("Debug: Failed to create library from source: %s\n",
             [[err localizedDescription] UTF8String]);
      return NULL;
    }
    return ((__bridge_retained void*)library);
  }
}

MTLLibrary_t metal_device_new_library_with_data(MTLDevice_t device_ptr, const void* data,
                                                size_t length) {
  @autoreleasepool {
    if (device_ptr == NULL || data == NULL || length == 0) {
      return NULL;
    }
    NSError* error = nil;
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    dispatch_data_t library_data = dispatch_data_create(data, length, dispatch_get_main_queue(),
                                                        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    id<MTLLibrary> library = [device newLibraryWithData:library_data
                                                   error:&error];

    if (error) {
      printf("Debug: Failed to create library from data: %s\n",
             [[error localizedDescription] UTF8String]);
      // library_data is managed by ARC/GCD, no manual release needed typically
      return NULL;
    }
    return ((__bridge_retained void*)library);
  }
}

// --- Function Management ---

MTLFunction_t metal_library_new_function_with_name(MTLLibrary_t library_ptr, const char* name) {
  @autoreleasepool {
    if (library_ptr == NULL || name == NULL) {
      return NULL;
    }
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)library_ptr;

    NSString *functionName = [NSString stringWithUTF8String:name ?: ""];

    id<MTLFunction> function = [library newFunctionWithName:functionName];
    if (!function) {
        printf("Debug: Failed to find function named: %s\n", name);
        return NULL;
    }
    return ((__bridge_retained void*)function);
  }
}

// --- Pipeline State Management ---

MTLComputePipelineState_t metal_device_new_compute_pipeline_state_with_function(
    MTLDevice_t device_ptr, MTLFunction_t function_ptr) {
  @autoreleasepool {
    if (device_ptr == NULL || function_ptr == NULL) {
      return NULL;
    }
    NSError* error = nil;
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    id<MTLFunction> function = (__bridge id<MTLFunction>)function_ptr;
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function
                                                                                     error:&error];
    if (error) {
      printf("Debug: Failed to create compute pipeline state: %s\n",
             [[error localizedDescription] UTF8String]);
      return NULL;
    }
    return ((__bridge_retained void*)pipelineState);
  }
}

size_t metal_pipeline_state_get_max_total_threads_per_threadgroup(
    MTLComputePipelineState_t pipeline_state_ptr) {
  @autoreleasepool {
    if (pipeline_state_ptr == NULL) {
      return 0;
    }
    id<MTLComputePipelineState> pipeline_state = (__bridge id<MTLComputePipelineState>)pipeline_state_ptr;
    return [pipeline_state maxTotalThreadsPerThreadgroup];
  }
}

// --- Command Buffer Management ---

MTLCommandBuffer_t metal_queue_new_command_buffer(MTLCommandQueue_t queue_ptr) {
  @autoreleasepool {
    if (queue_ptr == NULL) {
      return NULL;
    }
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queue_ptr;
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    return ((__bridge_retained void*)buffer);
  }
}

void metal_command_buffer_commit(MTLCommandBuffer_t buffer_ptr) {
  @autoreleasepool {
    if (buffer_ptr == NULL) {
      return;
    }
    id<MTLCommandBuffer> buffer = (__bridge id<MTLCommandBuffer>)buffer_ptr;
    [buffer commit];
  }
}

void metal_command_buffer_wait_until_completed(MTLCommandBuffer_t buffer_ptr) {
  @autoreleasepool {
    if (buffer_ptr == NULL) {
      return;
    }
    id<MTLCommandBuffer> buffer = (__bridge id<MTLCommandBuffer>)buffer_ptr;
    [buffer waitUntilCompleted];
  }
}

// --- Compute Command Encoder ---

MTLComputeCommandEncoder_t metal_command_buffer_compute_command_encoder(
    MTLCommandBuffer_t buffer_ptr) {
  @autoreleasepool {
    if (buffer_ptr == NULL) {
      return NULL;
    }
    id<MTLCommandBuffer> buffer = (__bridge id<MTLCommandBuffer>)buffer_ptr;
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    return ((__bridge_retained void*)encoder);
  }
}

void metal_compute_command_encoder_set_pipeline_state(
    MTLComputeCommandEncoder_t encoder_ptr, MTLComputePipelineState_t pipeline_state_ptr) {
  @autoreleasepool {
    if (encoder_ptr == NULL || pipeline_state_ptr == NULL) {
      return;
    }
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)encoder_ptr;
    id<MTLComputePipelineState> pipeline_state = (__bridge id<MTLComputePipelineState>)pipeline_state_ptr;
    [encoder setComputePipelineState:pipeline_state];
  }
}

void metal_compute_command_encoder_set_buffer(MTLComputeCommandEncoder_t encoder_ptr,
                                              MTLBuffer_t buffer_ptr, size_t offset, uint32_t index) {
  @autoreleasepool {
    if (encoder_ptr == NULL) {
      return;
    }
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)encoder_ptr;
    id<MTLBuffer> buffer = (buffer_ptr != NULL) ? (__bridge id<MTLBuffer>)buffer_ptr : nil;
    [encoder setBuffer:buffer offset:offset atIndex:index];
  }
}

void metal_compute_command_encoder_set_bytes(MTLComputeCommandEncoder_t encoder_ptr,
                                             const void* bytes, size_t length, uint32_t index) {
    @autoreleasepool {
        if (encoder_ptr == NULL || bytes == NULL) {
            return;
        }
        id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)encoder_ptr;
        [encoder setBytes:bytes length:length atIndex:index];
    }
}


void metal_compute_command_encoder_dispatch_thread_groups(
    MTLComputeCommandEncoder_t encoder_ptr, size_t grid_x, size_t grid_y, size_t grid_z,
    size_t thread_x, size_t thread_y, size_t thread_z) {
  @autoreleasepool {
    if (encoder_ptr == NULL) {
      return;
    }
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)encoder_ptr;
    MTLSize gridSize = MTLSizeMake(grid_x > 0 ? grid_x : 1, grid_y > 0 ? grid_y : 1, grid_z > 0 ? grid_z : 1);
    MTLSize threadGroupSize = MTLSizeMake(thread_x > 0 ? thread_x : 1, thread_y > 0 ? thread_y : 1, thread_z > 0 ? thread_z : 1);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
  }
}

void metal_compute_command_encoder_end_encoding(MTLComputeCommandEncoder_t encoder_ptr) {
  @autoreleasepool {
    if (encoder_ptr == NULL) {
      return;
    }
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)encoder_ptr;
    [encoder endEncoding];
  }
}

// --- Blit Command Encoder ---

MTLBlitCommandEncoder_t metal_command_buffer_blit_command_encoder(MTLCommandBuffer_t buffer_ptr) {
  @autoreleasepool {
    if (buffer_ptr == NULL) {
      return NULL;
    }
    id<MTLCommandBuffer> buffer = (__bridge id<MTLCommandBuffer>)buffer_ptr;
    id<MTLBlitCommandEncoder> encoder = [buffer blitCommandEncoder];
    return ((__bridge_retained void*)encoder);
  }
}

void metal_blit_command_encoder_copy_from_buffer(MTLBlitCommandEncoder_t encoder_ptr,
                                                 MTLBuffer_t src_buffer_ptr, size_t src_offset,
                                                 MTLBuffer_t dst_buffer_ptr, size_t dst_offset,
                                                 size_t size) {
  @autoreleasepool {
    if (encoder_ptr == NULL || src_buffer_ptr == NULL || dst_buffer_ptr == NULL) {
      return;
    }
    id<MTLBlitCommandEncoder> encoder = (__bridge id<MTLBlitCommandEncoder>)encoder_ptr;
    id<MTLBuffer> src_buffer = (__bridge id<MTLBuffer>)src_buffer_ptr;
    id<MTLBuffer> dst_buffer = (__bridge id<MTLBuffer>)dst_buffer_ptr;
    [encoder copyFromBuffer:src_buffer sourceOffset:src_offset
                   toBuffer:dst_buffer destinationOffset:dst_offset
                       size:size];
  }
}

void metal_blit_command_encoder_fill_buffer(MTLBlitCommandEncoder_t encoder_ptr, MTLBuffer_t buffer_ptr,
                                            size_t offset, size_t length, uint8_t value) {
    @autoreleasepool {
        if (encoder_ptr == NULL || buffer_ptr == NULL || length == 0) {
            return;
        }
        id<MTLBlitCommandEncoder> encoder = (__bridge id<MTLBlitCommandEncoder>)encoder_ptr;
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffer_ptr;
        NSRange range = NSMakeRange(offset, length);
        [encoder fillBuffer:buffer range:range value:value];
    }
}


void metal_blit_command_encoder_end_encoding(MTLBlitCommandEncoder_t encoder_ptr) {
  @autoreleasepool {
    if (encoder_ptr == NULL) {
      return;
    }
    id<MTLBlitCommandEncoder> encoder = (__bridge id<MTLBlitCommandEncoder>)encoder_ptr;
    [encoder endEncoding];
  }
}