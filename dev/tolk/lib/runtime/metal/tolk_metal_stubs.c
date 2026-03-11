#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#import <dispatch/dispatch.h>
#include <dlfcn.h>
#import <objc/message.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// 13 is the undocumented request type Metal uses to compile source into MTLB.
// This mirrors tinygrad's Metal compiler path.
#define REQUEST_TYPE_COMPILE 13

typedef struct {
  id<MTLLibrary> library;
  id<MTLFunction> function;
  id<MTLComputePipelineState> pipeline;
  // Cached to avoid repeated ObjC message sends (tinygrad: "cache these msg
  // calls"). Used to validate local threadgroup size before dispatch.
  uint64_t max_total_threads;
  char* name;
  NSString* label; // cached NSString for command buffer labeling
} tolk_metal_program;

static void fail_with_nserror(NSError* error, const char* fallback) {
  if (error != nil) {
    NSString* desc = [error localizedDescription];
    const char* msg = desc != nil ? [desc UTF8String] : fallback;
    caml_failwith(msg);
  }
  caml_failwith(fallback);
}

// METAL_FAST_MATH mirrors tinygrad's fast-math toggle for source compilation.
static bool metal_fast_math_enabled(void) {
  const char* raw = getenv("METAL_FAST_MATH");
  if (raw == NULL) return false;
  while (*raw == ' ' || *raw == '\t' || *raw == '\n') raw++;
  if (*raw == '\0') return false;
  return atoi(raw) != 0;
}

static NSString* metal_cache_dir(void) {
  const char* xdg = getenv("XDG_CACHE_HOME");
  NSString* base = nil;
  if (xdg != NULL && xdg[0] != '\0') {
    base = [NSString stringWithUTF8String:xdg];
  } else {
    base = [[NSHomeDirectory() stringByAppendingPathComponent:@"Library"]
        stringByAppendingPathComponent:@"Caches"];
  }
  NSString* dir = [base stringByAppendingPathComponent:@"tolk"];
  [[NSFileManager defaultManager] createDirectoryAtPath:dir
                            withIntermediateDirectories:YES
                                             attributes:nil
                                                  error:nil];
  return dir;
}

CAMLprim value caml_tolk_metal_create_device(value unit) {
  CAMLparam1(unit);
  @autoreleasepool {
    // MTLCreateSystemDefaultDevice can return nil on unsupported/virtualized
    // setups. The OCaml side will surface the failure if that happens.
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) caml_failwith("Metal device unavailable");
    [device retain];
    CAMLreturn(caml_copy_nativeint((intnat)device));
  }
}

CAMLprim value caml_tolk_metal_release_device(value v_device) {
  CAMLparam1(v_device);
  @autoreleasepool {
    id<MTLDevice> device = (id<MTLDevice>)Nativeint_val(v_device);
    [device release];
    CAMLreturn(Val_unit);
  }
}

CAMLprim value caml_tolk_metal_create_command_queue(value v_device) {
  CAMLparam1(v_device);
  @autoreleasepool {
    id<MTLDevice> device = (id<MTLDevice>)Nativeint_val(v_device);
    id<MTLCommandQueue> queue =
        [device newCommandQueueWithMaxCommandBufferCount:1024];
    if (queue == nil) caml_failwith("Cannot allocate Metal command queue");
    CAMLreturn(caml_copy_nativeint((intnat)queue));
  }
}

CAMLprim value caml_tolk_metal_release_command_queue(value v_queue) {
  CAMLparam1(v_queue);
  @autoreleasepool {
    id<MTLCommandQueue> queue = (id<MTLCommandQueue>)Nativeint_val(v_queue);
    [queue release];
    CAMLreturn(Val_unit);
  }
}

CAMLprim value caml_tolk_metal_buffer_alloc(value v_device, value v_size) {
  CAMLparam2(v_device, v_size);
  @autoreleasepool {
    id<MTLDevice> device = (id<MTLDevice>)Nativeint_val(v_device);
    NSUInteger size = (NSUInteger)Long_val(v_size);
    id<MTLBuffer> buf =
        [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (buf == nil) caml_failwith("Metal OOM while allocating buffer");
    CAMLreturn(caml_copy_nativeint((intnat)buf));
  }
}

CAMLprim value caml_tolk_metal_buffer_free(value v_buf) {
  CAMLparam1(v_buf);
  @autoreleasepool {
    id<MTLBuffer> buf = (id<MTLBuffer>)Nativeint_val(v_buf);
    [buf release];
    CAMLreturn(Val_unit);
  }
}

CAMLprim value caml_tolk_metal_buffer_copyin(value v_buf, value v_bytes) {
  CAMLparam2(v_buf, v_bytes);
  id<MTLBuffer> buf = (id<MTLBuffer>)Nativeint_val(v_buf);
  void* dst = [buf contents];
  size_t len = (size_t)caml_string_length(v_bytes);
  memcpy(dst, Bytes_val(v_bytes), len);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_metal_buffer_copyout(value v_bytes, value v_buf) {
  CAMLparam2(v_bytes, v_buf);
  id<MTLBuffer> buf = (id<MTLBuffer>)Nativeint_val(v_buf);
  void* src = [buf contents];
  size_t len = (size_t)caml_string_length(v_bytes);
  memcpy(Bytes_val(v_bytes), src, len);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_metal_program_create(value v_device, value v_name,
                                         value v_lib) {
  CAMLparam3(v_device, v_name, v_lib);
  @autoreleasepool {
    id<MTLDevice> device = (id<MTLDevice>)Nativeint_val(v_device);
    const char* name = String_val(v_name);
    size_t lib_len = (size_t)caml_string_length(v_lib);
    const uint8_t* lib = (const uint8_t*)String_val(v_lib);
    id<MTLLibrary> library = nil;
    if (lib_len >= 4 && memcmp(lib, "MTLB", 4) == 0) {
      void* copy = malloc(lib_len);
      if (copy == NULL) caml_failwith("Metal library allocation failed");
      memcpy(copy, lib, lib_len);
      dispatch_data_t data = dispatch_data_create(
          copy, lib_len, NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
      NSError* error = nil;
      library = [device newLibraryWithData:data error:&error];
      dispatch_release(data);
      if (library == nil)
        fail_with_nserror(error, "Failed to load Metal library");
    } else {
      NSString* src = [[NSString alloc] initWithBytes:lib
                                               length:lib_len
                                             encoding:NSUTF8StringEncoding];
      if (src == nil) caml_failwith("Metal source is not valid UTF-8");
      MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
      BOOL fast_math = metal_fast_math_enabled();
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
      if (@available(macOS 15.0, *)) {
        options.mathMode = fast_math ? MTLMathModeFast : MTLMathModeSafe;
      } else {
        // Use ObjC runtime to avoid deprecation warnings on older SDKs.
        if ([options respondsToSelector:@selector(setFastMathEnabled:)]) {
          ((void (*)(id, SEL, BOOL))objc_msgSend)(
              options, @selector(setFastMathEnabled:), fast_math);
        }
      }
#else
      options.fastMathEnabled = fast_math;
#endif
      NSError* error = nil;
      library = [device newLibraryWithSource:src options:options error:&error];
      [options release];
      [src release];
      if (library == nil)
        fail_with_nserror(error, "Metal source compile failed");
    }
    NSString* ns_name = [NSString stringWithUTF8String:name];
    id<MTLFunction> function = [library newFunctionWithName:ns_name];
    if (function == nil) {
      [library release];
      caml_failwith("Metal function not found");
    }
    MTLComputePipelineDescriptor* desc =
        [[MTLComputePipelineDescriptor alloc] init];
    desc.computeFunction = function;
    desc.supportIndirectCommandBuffers = YES;
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithDescriptor:desc
                                              options:MTLPipelineOptionNone
                                           reflection:nil
                                                error:&error];
    [desc release];
    if (pipeline == nil) {
      [function release];
      [library release];
      fail_with_nserror(error, "Metal pipeline creation failed");
    }
    tolk_metal_program* prog =
        (tolk_metal_program*)calloc(1, sizeof(tolk_metal_program));
    if (prog == NULL) {
      [pipeline release];
      [function release];
      [library release];
      caml_failwith("Metal program allocation failed");
    }
    prog->library = library;
    prog->function = function;
    prog->pipeline = pipeline;
    prog->max_total_threads =
        (uint64_t)[pipeline maxTotalThreadsPerThreadgroup];
    prog->name = strdup(name);
    prog->label = [[NSString stringWithUTF8String:name] retain];
    CAMLreturn(caml_copy_nativeint((intnat)prog));
  }
}

CAMLprim value caml_tolk_metal_program_free(value v_prog) {
  CAMLparam1(v_prog);
  @autoreleasepool {
    tolk_metal_program* prog = (tolk_metal_program*)Nativeint_val(v_prog);
    if (prog != NULL) {
      [prog->pipeline release];
      [prog->function release];
      [prog->library release];
      if (prog->label != nil) [prog->label release];
      free(prog->name);
      free(prog);
    }
    CAMLreturn(Val_unit);
  }
}

CAMLprim value caml_tolk_metal_program_dispatch(value v_queue, value v_prog,
                                           value v_buffers, value v_offsets,
                                           value v_args, value v_global,
                                           value v_local) {
  CAMLparam5(v_queue, v_prog, v_buffers, v_offsets, v_args);
  CAMLxparam2(v_global, v_local);
  @autoreleasepool {
    id<MTLCommandQueue> queue = (id<MTLCommandQueue>)Nativeint_val(v_queue);
    tolk_metal_program* prog = (tolk_metal_program*)Nativeint_val(v_prog);
    mlsize_t buf_count = Wosize_val(v_buffers);
    mlsize_t arg_count = Wosize_val(v_args);
    if (Wosize_val(v_offsets) != buf_count) {
      caml_failwith("Metal dispatch: buffer and offset array length mismatch");
    }
    if (Wosize_val(v_global) != 3 || Wosize_val(v_local) != 3) {
      caml_failwith("Metal dispatch expects 3D sizes");
    }
    int gx = Int_val(Field(v_global, 0));
    int gy = Int_val(Field(v_global, 1));
    int gz = Int_val(Field(v_global, 2));
    int lx = Int_val(Field(v_local, 0));
    int ly = Int_val(Field(v_local, 1));
    int lz = Int_val(Field(v_local, 2));
    uint64_t local_threads = (uint64_t)lx * (uint64_t)ly * (uint64_t)lz;
    if (local_threads > prog->max_total_threads) {
      caml_failwith("Metal local size exceeds max threads per threadgroup");
    }

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) caml_failwith("Metal command buffer creation failed");
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    if (encoder == nil) caml_failwith("Metal compute encoder creation failed");
    [encoder setComputePipelineState:prog->pipeline];

    for (mlsize_t i = 0; i < buf_count; ++i) {
      id<MTLBuffer> buf = (id<MTLBuffer>)Nativeint_val(Field(v_buffers, i));
      NSUInteger offset = (NSUInteger)Int_val(Field(v_offsets, i));
      [encoder setBuffer:buf offset:offset atIndex:i];
    }
    for (mlsize_t i = 0; i < arg_count; ++i) {
      int32_t arg_value = (int32_t)Int_val(Field(v_args, i));
      [encoder setBytes:&arg_value
                 length:sizeof(arg_value)
                atIndex:(buf_count + i)];
    }

    MTLSize global =
        MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, (NSUInteger)gz);
    MTLSize local = MTLSizeMake((NSUInteger)lx, (NSUInteger)ly, (NSUInteger)lz);
    [encoder dispatchThreadgroups:global threadsPerThreadgroup:local];
    [encoder endEncoding];

    if (prog->label != nil) [cmd setLabel:prog->label];
    [cmd commit];
    [cmd retain];
    CAMLreturn(caml_copy_nativeint((intnat)cmd));
  }
}

CAMLprim value caml_tolk_metal_program_dispatch_bc(value* argv, int argc) {
  (void)argc;
  // Bytecode stub for the 7-arg native entrypoint.
  return caml_tolk_metal_program_dispatch(argv[0], argv[1], argv[2], argv[3],
                                     argv[4], argv[5], argv[6]);
}

CAMLprim value caml_tolk_metal_icb_create(value v_device, value v_count) {
  CAMLparam2(v_device, v_count);
  @autoreleasepool {
    id<MTLDevice> device = (id<MTLDevice>)Nativeint_val(v_device);
    NSUInteger count = (NSUInteger)Long_val(v_count);
    MTLIndirectCommandBufferDescriptor* desc =
        [[MTLIndirectCommandBufferDescriptor alloc] init];
    desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    desc.inheritBuffers = NO;
    desc.inheritPipelineState = NO;
    // 31 is Metal's hardware limit on kernel buffer bindings per command.
    desc.maxKernelBufferBindCount = 31;
    id<MTLIndirectCommandBuffer> icb =
        [device newIndirectCommandBufferWithDescriptor:desc
                                       maxCommandCount:count
                                               options:MTLResourceCPUCacheModeDefaultCache];
    [desc release];
    if (icb == nil) caml_failwith("Metal ICB creation failed");
    CAMLreturn(caml_copy_nativeint((intnat)icb));
  }
}

CAMLprim value caml_tolk_metal_icb_encode(value v_icb, value v_index, value v_prog,
                                     value v_buffers, value v_arg_buf,
                                     value v_arg_offsets, value v_global,
                                     value v_local) {
  CAMLparam5(v_icb, v_index, v_prog, v_buffers, v_arg_buf);
  CAMLxparam3(v_arg_offsets, v_global, v_local);
  @autoreleasepool {
    id<MTLIndirectCommandBuffer> icb =
        (id<MTLIndirectCommandBuffer>)Nativeint_val(v_icb);
    NSUInteger index = (NSUInteger)Int_val(v_index);
    tolk_metal_program* prog = (tolk_metal_program*)Nativeint_val(v_prog);
    mlsize_t buf_count = Wosize_val(v_buffers);
    mlsize_t arg_count = Wosize_val(v_arg_offsets);
    if (Wosize_val(v_global) != 3 || Wosize_val(v_local) != 3) {
      caml_failwith("Metal ICB expects 3D sizes");
    }
    int gx = Int_val(Field(v_global, 0));
    int gy = Int_val(Field(v_global, 1));
    int gz = Int_val(Field(v_global, 2));
    int lx = Int_val(Field(v_local, 0));
    int ly = Int_val(Field(v_local, 1));
    int lz = Int_val(Field(v_local, 2));
    uint64_t local_threads = (uint64_t)lx * (uint64_t)ly * (uint64_t)lz;
    if (local_threads > prog->max_total_threads) {
      caml_failwith("Metal local size exceeds max threads per threadgroup");
    }

    id<MTLIndirectComputeCommand> cmd =
        [icb indirectComputeCommandAtIndex:index];
    [cmd setComputePipelineState:prog->pipeline];

    for (mlsize_t i = 0; i < buf_count; ++i) {
      id<MTLBuffer> buf = (id<MTLBuffer>)Nativeint_val(Field(v_buffers, i));
      [cmd setKernelBuffer:buf offset:0 atIndex:i];
    }
    if (Nativeint_val(v_arg_buf) != 0 && arg_count > 0) {
      id<MTLBuffer> arg_buf = (id<MTLBuffer>)Nativeint_val(v_arg_buf);
      for (mlsize_t i = 0; i < arg_count; ++i) {
        NSUInteger offset = (NSUInteger)Int_val(Field(v_arg_offsets, i));
        [cmd setKernelBuffer:arg_buf offset:offset atIndex:(buf_count + i)];
      }
    }

    MTLSize global =
        MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, (NSUInteger)gz);
    MTLSize local = MTLSizeMake((NSUInteger)lx, (NSUInteger)ly, (NSUInteger)lz);
    [cmd concurrentDispatchThreadgroups:global threadsPerThreadgroup:local];
    // Barrier ensures sequential execution: each command completes before the
    // next begins. Without this, commands in the ICB execute concurrently.
    [cmd setBarrier];
    CAMLreturn(Val_unit);
  }
}

CAMLprim value caml_tolk_metal_icb_encode_bc(value* argv, int argc) {
  (void)argc;
  return caml_tolk_metal_icb_encode(argv[0], argv[1], argv[2], argv[3], argv[4],
                              argv[5], argv[6], argv[7]);
}

CAMLprim value caml_tolk_metal_icb_update_buffer(value v_icb, value v_index,
                                           value v_buf_index, value v_buf) {
  CAMLparam4(v_icb, v_index, v_buf_index, v_buf);
  @autoreleasepool {
    id<MTLIndirectCommandBuffer> icb =
        (id<MTLIndirectCommandBuffer>)Nativeint_val(v_icb);
    NSUInteger index = (NSUInteger)Int_val(v_index);
    NSUInteger buf_index = (NSUInteger)Int_val(v_buf_index);
    id<MTLBuffer> buf = (id<MTLBuffer>)Nativeint_val(v_buf);
    id<MTLIndirectComputeCommand> cmd =
        [icb indirectComputeCommandAtIndex:index];
    [cmd setKernelBuffer:buf offset:0 atIndex:buf_index];
    CAMLreturn(Val_unit);
  }
}

CAMLprim value caml_tolk_metal_icb_update_dispatch(value v_icb, value v_index,
                                             value v_global, value v_local) {
  CAMLparam3(v_icb, v_index, v_global);
  CAMLxparam1(v_local);
  @autoreleasepool {
    id<MTLIndirectCommandBuffer> icb =
        (id<MTLIndirectCommandBuffer>)Nativeint_val(v_icb);
    NSUInteger index = (NSUInteger)Int_val(v_index);
    if (Wosize_val(v_global) != 3 || Wosize_val(v_local) != 3) {
      caml_failwith("Metal ICB expects 3D sizes");
    }
    int gx = Int_val(Field(v_global, 0));
    int gy = Int_val(Field(v_global, 1));
    int gz = Int_val(Field(v_global, 2));
    int lx = Int_val(Field(v_local, 0));
    int ly = Int_val(Field(v_local, 1));
    int lz = Int_val(Field(v_local, 2));

    id<MTLIndirectComputeCommand> cmd =
        [icb indirectComputeCommandAtIndex:index];
    MTLSize global =
        MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, (NSUInteger)gz);
    MTLSize local = MTLSizeMake((NSUInteger)lx, (NSUInteger)ly, (NSUInteger)lz);
    [cmd concurrentDispatchThreadgroups:global threadsPerThreadgroup:local];
    CAMLreturn(Val_unit);
  }
}

CAMLprim value caml_tolk_metal_icb_update_dispatch_bc(value* argv, int argc) {
  (void)argc;
  return caml_tolk_metal_icb_update_dispatch(argv[0], argv[1], argv[2], argv[3]);
}

CAMLprim value caml_tolk_metal_icb_execute(value v_queue, value v_icb,
                                     value v_count, value v_resources,
                                     value v_pipelines) {
  CAMLparam5(v_queue, v_icb, v_count, v_resources, v_pipelines);
  @autoreleasepool {
    id<MTLCommandQueue> queue = (id<MTLCommandQueue>)Nativeint_val(v_queue);
    id<MTLIndirectCommandBuffer> icb =
        (id<MTLIndirectCommandBuffer>)Nativeint_val(v_icb);
    NSUInteger count = (NSUInteger)Long_val(v_count);
    mlsize_t res_count = Wosize_val(v_resources);
    mlsize_t pipeline_count = Wosize_val(v_pipelines);

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) caml_failwith("Metal command buffer creation failed");
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    if (encoder == nil) caml_failwith("Metal compute encoder creation failed");

    if (res_count > 0) {
      id<MTLResource>* resources =
          (id<MTLResource>*)malloc(sizeof(id<MTLResource>) * res_count);
      if (resources == NULL) caml_failwith("Metal resource allocation failed");
      for (mlsize_t i = 0; i < res_count; ++i) {
        id<MTLBuffer> buf =
            (id<MTLBuffer>)Nativeint_val(Field(v_resources, i));
        resources[i] = buf;
      }
      [encoder useResources:resources
                      count:res_count
                      usage:MTLResourceUsageRead | MTLResourceUsageWrite];
      free(resources);
    }

    // M1/M2 workaround: dummy dispatch with each pipeline to mark them as used.
    // Without this, ICB execution can crash on AGXG<15 (pre-M3) GPUs.
    for (mlsize_t i = 0; i < pipeline_count; ++i) {
      tolk_metal_program* prog =
          (tolk_metal_program*)Nativeint_val(Field(v_pipelines, i));
      [encoder setComputePipelineState:prog->pipeline];
      [encoder dispatchThreadgroups:MTLSizeMake(0, 0, 0)
           threadsPerThreadgroup:MTLSizeMake(0, 0, 0)];
    }

    NSRange range = NSMakeRange(0, count);
    [encoder executeCommandsInBuffer:icb withRange:range];
    [encoder endEncoding];
    [cmd commit];
    [cmd retain];
    CAMLreturn(caml_copy_nativeint((intnat)cmd));
  }
}

CAMLprim value caml_tolk_metal_icb_release(value v_icb) {
  CAMLparam1(v_icb);
  @autoreleasepool {
    id<MTLIndirectCommandBuffer> icb =
        (id<MTLIndirectCommandBuffer>)Nativeint_val(v_icb);
    [icb release];
    CAMLreturn(Val_unit);
  }
}

// Detect whether this GPU needs the M1/M2 ICB workaround.
// Returns true for AGXG<15 (pre-M3) families.
CAMLprim value caml_tolk_metal_needs_icb_fix(value v_device) {
  CAMLparam1(v_device);
  @autoreleasepool {
    id<MTLDevice> device = (id<MTLDevice>)Nativeint_val(v_device);
    NSString* desc = [device description];
    if (desc == nil) CAMLreturn(Val_true);
    NSRange range = [desc rangeOfString:@"AGXG"];
    if (range.location == NSNotFound) CAMLreturn(Val_true);
    NSString* rest = [desc substringFromIndex:range.location + 4];
    int family = atoi([rest UTF8String]);
    CAMLreturn(Val_bool(family < 15));
  }
}

CAMLprim value caml_tolk_metal_blit_copy(value v_queue, value v_src_buf,
                                    value v_src_offset, value v_dst_buf,
                                    value v_dst_offset, value v_size) {
  CAMLparam5(v_queue, v_src_buf, v_src_offset, v_dst_buf, v_dst_offset);
  CAMLxparam1(v_size);
  @autoreleasepool {
    id<MTLCommandQueue> queue = (id<MTLCommandQueue>)Nativeint_val(v_queue);
    id<MTLBuffer> src = (id<MTLBuffer>)Nativeint_val(v_src_buf);
    NSUInteger src_offset = (NSUInteger)Long_val(v_src_offset);
    id<MTLBuffer> dst = (id<MTLBuffer>)Nativeint_val(v_dst_buf);
    NSUInteger dst_offset = (NSUInteger)Long_val(v_dst_offset);
    NSUInteger size = (NSUInteger)Long_val(v_size);

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) caml_failwith("Metal command buffer creation failed");
    id<MTLBlitCommandEncoder> encoder = [cmd blitCommandEncoder];
    if (encoder == nil) caml_failwith("Metal blit encoder creation failed");
    [encoder copyFromBuffer:src
               sourceOffset:src_offset
                   toBuffer:dst
          destinationOffset:dst_offset
                       size:size];
    [encoder endEncoding];
    [cmd commit];
    [cmd retain];
    CAMLreturn(caml_copy_nativeint((intnat)cmd));
  }
}

CAMLprim value caml_tolk_metal_blit_copy_bc(value* argv, int argc) {
  (void)argc;
  return caml_tolk_metal_blit_copy(argv[0], argv[1], argv[2], argv[3], argv[4],
                              argv[5]);
}

CAMLprim value caml_tolk_metal_create_shared_event(value v_device) {
  CAMLparam1(v_device);
  @autoreleasepool {
    id<MTLDevice> device = (id<MTLDevice>)Nativeint_val(v_device);
    id<MTLSharedEvent> event = [device newSharedEvent];
    if (event == nil) caml_failwith("Metal shared event creation failed");
    CAMLreturn(caml_copy_nativeint((intnat)event));
  }
}

CAMLprim value caml_tolk_metal_release_shared_event(value v_event) {
  CAMLparam1(v_event);
  @autoreleasepool {
    id<MTLSharedEvent> event = (id<MTLSharedEvent>)Nativeint_val(v_event);
    [event release];
    CAMLreturn(Val_unit);
  }
}

CAMLprim value caml_tolk_metal_encode_signal_event(value v_cmd, value v_event,
                                              value v_timeline_value) {
  CAMLparam3(v_cmd, v_event, v_timeline_value);
  @autoreleasepool {
    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)Nativeint_val(v_cmd);
    id<MTLEvent> event = (id<MTLEvent>)Nativeint_val(v_event);
    uint64_t val = (uint64_t)Long_val(v_timeline_value);
    [cmd encodeSignalEvent:event value:val];
    CAMLreturn(Val_unit);
  }
}

CAMLprim value caml_tolk_metal_encode_wait_event(value v_cmd, value v_event,
                                            value v_timeline_value) {
  CAMLparam3(v_cmd, v_event, v_timeline_value);
  @autoreleasepool {
    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)Nativeint_val(v_cmd);
    id<MTLEvent> event = (id<MTLEvent>)Nativeint_val(v_event);
    uint64_t val = (uint64_t)Long_val(v_timeline_value);
    [cmd encodeWaitForEvent:event value:val];
    CAMLreturn(Val_unit);
  }
}

CAMLprim value caml_tolk_metal_command_buffer_gpu_time(value v_cmd) {
  CAMLparam1(v_cmd);
  CAMLlocal1(v_pair);
  id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)Nativeint_val(v_cmd);
  double start = [cmd GPUStartTime];
  double end = [cmd GPUEndTime];
  v_pair = caml_alloc(2 * Double_wosize, Double_array_tag);
  Store_double_field(v_pair, 0, start);
  Store_double_field(v_pair, 1, end);
  CAMLreturn(v_pair);
}

CAMLprim value caml_tolk_metal_device_name(value v_device) {
  CAMLparam1(v_device);
  @autoreleasepool {
    id<MTLDevice> device = (id<MTLDevice>)Nativeint_val(v_device);
    NSString* name = [device name];
    const char* str = name != nil ? [name UTF8String] : "unknown";
    CAMLreturn(caml_copy_string(str));
  }
}

CAMLprim value caml_tolk_metal_command_buffer_wait(value v_cmd) {
  CAMLparam1(v_cmd);
  id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)Nativeint_val(v_cmd);

  caml_release_runtime_system();
  [cmd waitUntilCompleted];
  caml_acquire_runtime_system();

  @autoreleasepool {
    NSError* error = [cmd error];
    if (error != nil) {
      NSString* desc = [error localizedDescription];
      const char* msg =
          desc != nil ? [desc UTF8String] : "Metal command buffer failed";
      char buf[512];
      snprintf(buf, sizeof(buf), "%s", msg);
      [cmd release];
      caml_failwith(buf);
    }
    [cmd release];
  }
  CAMLreturn(Val_unit);
}

typedef void* (*MTLCodeGenServiceCreate_t)(const char* label);
typedef void (*MTLCodeGenServiceBuildRequest_t)(void* cgs, void* queue,
                                                int request_type,
                                                const void* request,
                                                size_t request_len,
                                                void* callback);

static void* mtlcompiler_handle = NULL;
static MTLCodeGenServiceCreate_t mtl_create = NULL;
static MTLCodeGenServiceBuildRequest_t mtl_build = NULL;
static void* mtl_service = NULL;

// MTLCompiler is a private framework used for fast source->MTLB compilation.
// If it can't be loaded, we fall back to runtime source compilation.
static int ensure_mtlcompiler(void) {
  if (mtl_create != NULL && mtl_build != NULL && mtl_service != NULL) return 1;
  if (mtlcompiler_handle == NULL) {
    mtlcompiler_handle = dlopen(
        "/System/Library/PrivateFrameworks/MTLCompiler.framework/MTLCompiler",
        RTLD_LAZY);
    if (mtlcompiler_handle == NULL) {
      mtlcompiler_handle = dlopen("MTLCompiler", RTLD_LAZY);
    }
  }
  if (mtlcompiler_handle == NULL) return 0;
  if (mtl_create == NULL) {
    mtl_create = (MTLCodeGenServiceCreate_t)dlsym(mtlcompiler_handle,
                                                  "MTLCodeGenServiceCreate");
  }
  if (mtl_build == NULL) {
    mtl_build = (MTLCodeGenServiceBuildRequest_t)dlsym(
        mtlcompiler_handle, "MTLCodeGenServiceBuildRequest");
  }
  if (mtl_create == NULL || mtl_build == NULL) return 0;
  if (mtl_service == NULL) {
    mtl_service = mtl_create("tolk");
  }
  return mtl_service != NULL;
}

typedef struct {
  int error;
  char* error_msg;
  uint8_t* data;
  size_t len;
} compile_result;

static size_t round_up(size_t value, size_t align) {
  size_t rem = value % align;
  if (rem == 0) return value;
  return value + (align - rem);
}

// Compile Metal source to MTLB binary via Apple's private MTLCompiler.
// Returns Some(bytes) on success, None if MTLCompiler is unavailable.
// The request format is: [src_len:8][params_len:8][src_padded][params].
// The reply format is: [?:8][header_size:4][warning_size:4][header][warnings][MTLB].
CAMLprim value caml_tolk_metal_compile(value v_src) {
  CAMLparam1(v_src);
  CAMLlocal2(v_bytes, v_some);
  @autoreleasepool {
    if (!ensure_mtlcompiler()) {
      CAMLreturn(Val_int(0));
    }
    const char* src = String_val(v_src);
    size_t src_len = (size_t)caml_string_length(v_src);

    NSOperatingSystemVersion ver =
        [[NSProcessInfo processInfo] operatingSystemVersion];
    int major = (int)ver.majorVersion;
    const char* metal_version = "macos-metal2.0";
    if (major >= 14)
      metal_version = "metal3.1";
    else if (major >= 13)
      metal_version = "metal3.0";

    NSString* cache_dir = metal_cache_dir();
    const char* cache_path = [cache_dir UTF8String];

    char params[1024];
    snprintf(params, sizeof(params),
             "-fno-fast-math -std=%s --driver-mode=metal -x metal "
             "-fmodules-cache-path=\"%s\" -fno-caret-diagnostics",
             metal_version, cache_path);

    size_t src_padded_len = round_up(src_len + 1, 4);
    size_t params_len = strlen(params) + 1;
    size_t request_len = 16 + src_padded_len + params_len;
    uint8_t* request = (uint8_t*)malloc(request_len);
    if (request == NULL)
      caml_failwith("Metal compiler request allocation failed");

    uint64_t src_len64 = (uint64_t)src_padded_len;
    uint64_t params_len64 = (uint64_t)params_len;
    memcpy(request, &src_len64, 8);
    memcpy(request + 8, &params_len64, 8);
    memcpy(request + 16, src, src_len);
    request[16 + src_len] = '\0';
    if (src_padded_len > src_len + 1) {
      memset(request + 16 + src_len + 1, 0, src_padded_len - (src_len + 1));
    }
    memcpy(request + 16 + src_padded_len, params, params_len);

    __block compile_result res = {0, NULL, NULL, 0};
    void* service = mtl_service;
    // MTLCodeGenServiceBuildRequest expects a block (Apple's C extension).
    // We use a stack block here to mirror tinygrad's callback behavior.
    mtl_build(service, NULL, REQUEST_TYPE_COMPILE, request, request_len,
              ^(void* blockptr, int32_t error, void* dataPtr, size_t dataLen,
                const char* errorMessage) {
                (void)blockptr;
                if (error == 0 && dataPtr != NULL && dataLen > 0) {
                  res.data = (uint8_t*)malloc(dataLen);
                  if (res.data != NULL) {
                    memcpy(res.data, dataPtr, dataLen);
                    res.len = dataLen;
                  }
                } else {
                  res.error = error != 0 ? (int)error : -1;
                  if (errorMessage != NULL)
                    res.error_msg = strdup(errorMessage);
                }
              });
    free(request);

    if (res.error != 0 || res.data == NULL) {
      char buf[256];
      const char* msg =
          res.error_msg != NULL ? res.error_msg : "Metal compiler failed";
      snprintf(buf, sizeof(buf), "%s", msg);
      free(res.error_msg);
      free(res.data);
      caml_failwith(buf);
    }

    if (res.len < 16) {
      free(res.data);
      caml_failwith("Invalid Metal compiler output");
    }
    // The compiler reply includes a header + warnings before the MTLB blob.
    uint32_t header_size = 0;
    uint32_t warning_size = 0;
    memcpy(&header_size, res.data + 8, 4);
    memcpy(&warning_size, res.data + 12, 4);
    size_t offset = (size_t)header_size + (size_t)warning_size;
    if (offset > res.len) {
      free(res.data);
      caml_failwith("Invalid Metal compiler output");
    }
    uint8_t* mtlb = res.data + offset;
    size_t mtlb_len = res.len - offset;
    if (mtlb_len < 8 || memcmp(mtlb, "MTLB", 4) != 0 ||
        memcmp(mtlb + mtlb_len - 4, "ENDT", 4) != 0) {
      free(res.data);
      caml_failwith("Invalid Metal library output");
    }

    v_bytes = caml_alloc_string(mtlb_len);
    memcpy((char*)String_val(v_bytes), mtlb, mtlb_len);
    free(res.data);

    v_some = caml_alloc(1, 0);
    Store_field(v_some, 0, v_bytes);
    CAMLreturn(v_some);
  }
}
