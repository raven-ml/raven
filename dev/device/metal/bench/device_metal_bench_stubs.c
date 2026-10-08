/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Floors: the raw Metal calls each driver row makes, on a queue of their
   own. Objective-C on macOS; elsewhere nothing here is called. A stub that
   waits for the GPU releases the runtime; the others hold it. */

#define _GNU_SOURCE

#include <stdint.h>
#include <stdlib.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#ifdef __APPLE__

#import <Metal/Metal.h>

struct floor {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> step;
  id<MTLBuffer> args, out;
  dispatch_data_t metallib;
};

#define Floor_val(v) ((struct floor *)Nativeint_val(v))

/* A floor over the metallib [v_b], whose [step] it dispatches; never
   freed. */
value device_metal_bench_floor(value v_b) {
  struct floor *f = calloc(1, sizeof *f);
  @autoreleasepool {
    f->device = MTLCreateSystemDefaultDevice();
    f->queue = [f->device newCommandQueueWithMaxCommandBufferCount:1024];
    f->metallib = dispatch_data_create(String_val(v_b), caml_string_length(v_b),
                                       NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    id<MTLLibrary> l = [f->device newLibraryWithData:f->metallib error:NULL];
    id<MTLFunction> step = [l newFunctionWithName:@"step"];
    f->step = [f->device newComputePipelineStateWithFunction:step error:NULL];
    f->out =
        [f->device newBufferWithLength:16 options:MTLResourceStorageModeShared];
    f->args =
        [f->device newBufferWithLength:16 options:MTLResourceStorageModeShared];
    *(uint64_t *)f->args.contents = f->out.gpuAddress;
    [step release];
    [l release];
  }
  if (f->step == nil) caml_failwith("device_metal_bench_floor: no pipeline");
  return caml_copy_nativeint((intnat)f);
}

/* [v_n] command buffers committed, then a wait for the last. */
value device_metal_bench_release(value v_f, value v_n) {
  struct floor *f = Floor_val(v_f);
  @autoreleasepool {
    id<MTLCommandBuffer> b = nil;
    for (long i = 0; i < Long_val(v_n); i++) {
      b = [f->queue commandBuffer];
      [b commit];
    }
    caml_release_runtime_system();
    [b waitUntilCompleted];
    caml_acquire_runtime_system();
  }
  return Val_unit;
}

/* [v_n] one-thread dispatches of [step], each after the one before,
   encoded directly into one command buffer, waited. */
value device_metal_bench_launch(value v_f, value v_n) {
  struct floor *f = Floor_val(v_f);
  @autoreleasepool {
    id<MTLCommandBuffer> b = [f->queue commandBuffer];
    id<MTLComputeCommandEncoder> e =
        [b computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    [e useResource:f->out usage:MTLResourceUsageWrite];
    [e setComputePipelineState:f->step];
    [e setBuffer:f->args offset:0 atIndex:0];
    for (long i = 0; i < Long_val(v_n); i++)
      [e dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [e endEncoding];
    [b commit];
    caml_release_runtime_system();
    [b waitUntilCompleted];
    caml_acquire_runtime_system();
  }
  return Val_unit;
}

/* [v_n] command buffers of one one-thread dispatch of [step] each, all
   committed, then a wait for the last. */
value device_metal_bench_buffers(value v_f, value v_n) {
  struct floor *f = Floor_val(v_f);
  @autoreleasepool {
    id<MTLCommandBuffer> b = nil;
    for (long i = 0; i < Long_val(v_n); i++) {
      b = [f->queue commandBuffer];
      id<MTLComputeCommandEncoder> e =
          [b computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
      [e useResource:f->out usage:MTLResourceUsageWrite];
      [e setComputePipelineState:f->step];
      [e setBuffer:f->args offset:0 atIndex:0];
      [e dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [e endEncoding];
      [b commit];
    }
    caml_release_runtime_system();
    [b waitUntilCompleted];
    caml_acquire_runtime_system();
  }
  return Val_unit;
}

/* A shared buffer of [v_n] bytes, released. */
value device_metal_bench_alloc(value v_f, value v_n) {
  struct floor *f = Floor_val(v_f);
  [[f->device newBufferWithLength:(NSUInteger)Long_val(v_n)
                          options:MTLResourceStorageModeShared] release];
  return Val_unit;
}

/* The [v_n] bytes at [v_p], page-aligned, wrapped and released. */
value device_metal_bench_map_host(value v_f, value v_p, value v_n) {
  struct floor *f = Floor_val(v_f);
  [[f->device newBufferWithBytesNoCopy:(void *)Nativeint_val(v_p)
                                length:(NSUInteger)Long_val(v_n)
                               options:MTLResourceStorageModeShared
                           deallocator:nil] release];
  return Val_unit;
}

/* The floor's metallib loaded, with a pipeline of each function usable
   from an indirect command buffer, released. */
value device_metal_bench_image(value v_f) {
  struct floor *f = Floor_val(v_f);
  @autoreleasepool {
    id<MTLLibrary> l = [f->device newLibraryWithData:f->metallib error:NULL];
    for (NSString *name in l.functionNames) {
      MTLComputePipelineDescriptor *d =
          [[MTLComputePipelineDescriptor alloc] init];
      d.computeFunction = [[l newFunctionWithName:name] autorelease];
      d.supportIndirectCommandBuffers = YES;
      [[f->device newComputePipelineStateWithDescriptor:d
                                                options:0
                                             reflection:nil
                                                  error:NULL] release];
      [d release];
    }
    [l release];
  }
  return Val_unit;
}

#else

CAMLnoret static void no_metal(void) {
  caml_invalid_argument("device_metal_bench: Metal exists on macOS only");
}

value device_metal_bench_floor(value a) { (void)a, no_metal(); }
value device_metal_bench_release(value a, value b) {
  (void)a, (void)b, no_metal();
}
value device_metal_bench_launch(value a, value b) {
  (void)a, (void)b, no_metal();
}
value device_metal_bench_alloc(value a, value b) {
  (void)a, (void)b, no_metal();
}
value device_metal_bench_buffers(value a, value b) {
  (void)a, (void)b, no_metal();
}
value device_metal_bench_map_host(value a, value b, value c) {
  (void)a, (void)b, (void)c, no_metal();
}
value device_metal_bench_image(value a) { (void)a, no_metal(); }

#endif
