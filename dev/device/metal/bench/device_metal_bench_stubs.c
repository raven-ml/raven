/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Floors: the raw Metal calls each driver row makes, on a queue of their
   own; and the load that rows of waits run under. Objective-C on macOS;
   elsewhere nothing here is called. A stub that waits for the GPU releases
   the runtime; the others hold it. */

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
#include <pthread.h>
#include <pthread/qos.h>

/* [step] for direct dispatches; [indirect], the same function for an
   indirect command buffer's. */
struct floor {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> step, indirect;
  id<MTLBuffer> args, out;
  dispatch_data_t metallib;
};

#define Floor_val(v) ((struct floor *)Nativeint_val(v))

/* A floor over the metallib [v_b], whose [step] it dispatches; never
   freed. */
value device_metal_bench_floor(value v_b) {
  struct floor *f = calloc(1, sizeof *f);
  if (f == NULL) caml_raise_out_of_memory();
  @autoreleasepool {
    f->device = MTLCreateSystemDefaultDevice();
    f->queue = [f->device newCommandQueueWithMaxCommandBufferCount:1024];
    f->metallib = dispatch_data_create(String_val(v_b), caml_string_length(v_b),
                                       NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    id<MTLLibrary> l = [f->device newLibraryWithData:f->metallib error:NULL];
    id<MTLFunction> step = [l newFunctionWithName:@"step"];
    f->step = [f->device newComputePipelineStateWithFunction:step error:NULL];
    MTLComputePipelineDescriptor *d =
        [[MTLComputePipelineDescriptor alloc] init];
    d.computeFunction = step;
    d.supportIndirectCommandBuffers = YES;
    f->indirect = [f->device newComputePipelineStateWithDescriptor:d
                                                           options:0
                                                        reflection:nil
                                                             error:NULL];
    [d release];
    f->out =
        [f->device newBufferWithLength:16 options:MTLResourceStorageModeShared];
    f->args =
        [f->device newBufferWithLength:16 options:MTLResourceStorageModeShared];
    *(uint64_t *)f->args.contents = f->out.gpuAddress;
    [step release];
    [l release];
  }
  if (f->step == nil || f->indirect == nil)
    caml_failwith("device_metal_bench_floor: no pipeline");
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

/* An indirect command buffer of [v_n] one-thread dispatches of [step],
   each after the one before, as the driver records them; never freed. */
value device_metal_bench_icb(value v_f, value v_n) {
  struct floor *f = Floor_val(v_f);
  MTLIndirectCommandBufferDescriptor *d =
      [[MTLIndirectCommandBufferDescriptor alloc] init];
  d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
  d.inheritBuffers = NO;
  d.inheritPipelineState = NO;
  d.maxKernelBufferBindCount = 1;
  id<MTLIndirectCommandBuffer> b =
      [f->device newIndirectCommandBufferWithDescriptor:d
                                        maxCommandCount:(NSUInteger)Long_val(v_n)
                                                options:0];
  [d release];
  if (b == nil) caml_failwith("device_metal_bench_icb: no buffer");
  for (long i = 0; i < Long_val(v_n); i++) {
    id<MTLIndirectComputeCommand> c = [b indirectComputeCommandAtIndex:i];
    [c setComputePipelineState:f->indirect];
    [c setKernelBuffer:f->args offset:0 atIndex:0];
    [c setBarrier];
    [c concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  }
  return caml_copy_nativeint((intnat)b);
}

/* The [v_n] commands of the indirect command buffer [v_b] in one command
   buffer, after the empty dispatch that sets their pipeline, waited. */
value device_metal_bench_execute(value v_f, value v_b, value v_n) {
  struct floor *f = Floor_val(v_f);
  @autoreleasepool {
    id<MTLCommandBuffer> b = [f->queue commandBuffer];
    id<MTLComputeCommandEncoder> e =
        [b computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    [e useResource:f->out usage:MTLResourceUsageWrite];
    [e setComputePipelineState:f->indirect];
    [e dispatchThreadgroups:MTLSizeMake(0, 0, 0)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [e executeCommandsInBuffer:(id)Nativeint_val(v_b)
                     withRange:NSMakeRange(0, (NSUInteger)Long_val(v_n))];
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

/* Load */

/* Moves the calling thread to the default class, the one of a thread made
   without a class, such as a domain's. */
value device_metal_bench_default_class(value unit) {
  (void)unit;
  pthread_set_qos_class_self_np(QOS_CLASS_DEFAULT, 0);
  return Val_unit;
}

/* Threads at the default class that spin until [load_stop]: the load a
   wait shares the cores with. */
static int spinning;
static pthread_t spinners[64];
static long nspinners;

static void *spin(void *unused) {
  (void)unused;
  while (__atomic_load_n(&spinning, __ATOMIC_RELAXED)) {
  }
  return NULL;
}

value device_metal_bench_load_stop(value unit) {
  (void)unit;
  __atomic_store_n(&spinning, 0, __ATOMIC_RELAXED);
  for (long i = 0; i < nspinners; i++) pthread_join(spinners[i], NULL);
  nspinners = 0;
  return Val_unit;
}

/* [v_n] spinning threads, at most 64. */
value device_metal_bench_load_start(value v_n) {
  long n = Long_val(v_n) < 64 ? Long_val(v_n) : 64;
  __atomic_store_n(&spinning, 1, __ATOMIC_RELAXED);
  for (; nspinners < n; nspinners++)
    if (pthread_create(&spinners[nspinners], NULL, spin, NULL) != 0) {
      device_metal_bench_load_stop(Val_unit);
      caml_failwith("device_metal_bench_load_start: no thread");
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
value device_metal_bench_icb(value a, value b) { (void)a, (void)b, no_metal(); }
value device_metal_bench_execute(value a, value b, value c) {
  (void)a, (void)b, (void)c, no_metal();
}
value device_metal_bench_default_class(value a) { (void)a, no_metal(); }
value device_metal_bench_load_start(value a) { (void)a, no_metal(); }
value device_metal_bench_load_stop(value a) { (void)a, no_metal(); }

#endif
