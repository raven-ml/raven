/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A submitter for the tests: compiles Metal source, encodes work on the
   device's queue, and signals its event, as the libraries that submit work
   do. */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <dlfcn.h>
#import <objc/message.h>
#include <stdlib.h>
#include <string.h>

#define Object_val(v) ((id)(intptr_t)Nativeint_val(v))

typedef void *(*create_service)(const char *);
typedef void (*build_request)(void *, void *, int, const void *, size_t,
                              void *);

/* Compiles [v_src] to a metallib with the system's Metal compiler service. */
value test_metal_compile(value v_src) {
  CAMLparam1(v_src);
  CAMLlocal1(v_lib);
  void *framework = dlopen(
      "/System/Library/PrivateFrameworks/MTLCompiler.framework/MTLCompiler",
      RTLD_LAZY);
  if (framework == NULL) caml_failwith("MTLCompiler is unavailable");
  create_service create = (create_service)dlsym(framework, "MTLCodeGenServiceCreate");
  build_request build = (build_request)dlsym(framework, "MTLCodeGenServiceBuildRequest");
  if (create == NULL || build == NULL) caml_failwith("MTLCompiler is unavailable");
  void *service = create("nx");
  const char *params =
      "-fno-fast-math -std=metal3.0 --driver-mode=metal -x metal";
  size_t src_len = caml_string_length(v_src);
  size_t src_padded = (src_len + 1 + 3) & ~(size_t)3;
  size_t params_len = strlen(params) + 1;
  size_t request_len = 16 + src_padded + params_len;
  uint8_t *request = calloc(1, request_len);
  uint64_t lens[2] = {src_padded, params_len};
  memcpy(request, lens, 16);
  memcpy(request + 16, String_val(v_src), src_len);
  memcpy(request + 16 + src_padded, params, params_len);
  __block uint8_t *reply = NULL;
  __block size_t reply_len = 0;
  build(service, NULL, 13, request, request_len,
        ^(int32_t error, void *data, size_t len, const char *msg) {
          (void)msg;
          if (error == 0 && data != NULL) {
            reply = malloc(len);
            memcpy(reply, data, len);
            reply_len = len;
          }
        });
  free(request);
  if (reply == NULL || reply_len < 16) caml_failwith("Metal compilation failed");
  uint32_t sizes[2];
  memcpy(sizes, reply + 8, 8);
  size_t offset = (size_t)sizes[0] + sizes[1];
  v_lib = caml_alloc_initialized_string(reply_len - offset, (char *)reply + offset);
  free(reply);
  CAMLreturn(v_lib);
}

/* Whether the residency set [v_set] holds the allocation [v_buffer]. */
value test_metal_contains(value v_set, value v_buffer) {
  BOOL held = ((BOOL(*)(id, SEL, id))objc_msgSend)(
      Object_val(v_set), sel_registerName("containsAllocation:"),
      Object_val(v_buffer));
  return Val_bool(held);
}

value test_metal_allocation_count(value v_set) {
  NSUInteger count = ((NSUInteger(*)(id, SEL))objc_msgSend)(
      Object_val(v_set), sel_registerName("allocationCount"));
  return Val_long((intnat)count);
}

value test_metal_set_signaled(value v_event, value v_value) {
  id<MTLSharedEvent> event = Object_val(v_event);
  event.signaledValue = (uint64_t)Long_val(v_value);
  return Val_unit;
}

/* Runs [v_pipeline] over [v_threads] threads, with the GPU address [v_address]
   as its one argument, and signals [v_value] on [v_event] when done. Unless
   [v_stamps] is 0, it leaves the command buffer, retained, in the stamp word of
   the first of the two slots there and 0 in the second's, to be timed. */
value test_metal_dispatch(value v_queue, value v_event, value v_fence,
                          value v_resources, value v_pipeline,
                          value v_address, value v_threads, value v_value,
                          value v_stamps) {
  CAMLparam5(v_queue, v_event, v_fence, v_resources, v_pipeline);
  CAMLxparam4(v_address, v_threads, v_value, v_stamps);
  id<MTLCommandQueue> queue = Object_val(v_queue);
  @autoreleasepool {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder waitForFence:Object_val(v_fence)];
    mlsize_t count = Wosize_val(v_resources);
    if (count > 0) {
      id<MTLResource> *resources = malloc(count * sizeof(id));
      for (mlsize_t i = 0; i < count; i++)
        resources[i] = Object_val(Field(v_resources, i));
      [encoder useResources:resources
                      count:count
                      usage:MTLResourceUsageRead | MTLResourceUsageWrite];
      free(resources);
    }
    [encoder setComputePipelineState:Object_val(v_pipeline)];
    uint64_t address = (uint64_t)Nativeint_val(v_address);
    [encoder setBytes:&address length:sizeof(address) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)Long_val(v_threads), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder updateFence:Object_val(v_fence)];
    [encoder endEncoding];
    [command encodeSignalEvent:Object_val(v_event)
                         value:(uint64_t)Long_val(v_value)];
    uint64_t *stamps = (uint64_t *)Nativeint_val(v_stamps);
    if (stamps != NULL) {
      stamps[1] = (uint64_t)(uintptr_t)[command retain];
      stamps[3] = 0;
    }
    [command commit];
  }
  CAMLreturn(Val_unit);
}

value test_metal_dispatch_byte(value *argv, int argc) {
  (void)argc;
  return test_metal_dispatch(argv[0], argv[1], argv[2], argv[3], argv[4],
                             argv[5], argv[6], argv[7], argv[8]);
}

/* Runs the [v_count] commands of the indirect command buffer [v_icb] as timeline
   work signalling [v_value]. */
value test_metal_execute(value v_queue, value v_event, value v_fence,
                         value v_resources, value v_icb, value v_count,
                         value v_value) {
  CAMLparam5(v_queue, v_event, v_fence, v_resources, v_icb);
  CAMLxparam2(v_count, v_value);
  id<MTLCommandQueue> queue = Object_val(v_queue);
  @autoreleasepool {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder waitForFence:Object_val(v_fence)];
    mlsize_t count = Wosize_val(v_resources);
    if (count > 0) {
      id<MTLResource> *resources = malloc(count * sizeof(id));
      for (mlsize_t i = 0; i < count; i++)
        resources[i] = Object_val(Field(v_resources, i));
      [encoder useResources:resources
                      count:count
                      usage:MTLResourceUsageRead | MTLResourceUsageWrite];
      free(resources);
    }
    [encoder executeCommandsInBuffer:Object_val(v_icb)
                           withRange:NSMakeRange(0, (NSUInteger)Long_val(v_count))];
    [encoder updateFence:Object_val(v_fence)];
    [encoder endEncoding];
    [command encodeSignalEvent:Object_val(v_event)
                         value:(uint64_t)Long_val(v_value)];
    [command commit];
  }
  CAMLreturn(Val_unit);
}

value test_metal_execute_byte(value *argv, int argc) {
  (void)argc;
  return test_metal_execute(argv[0], argv[1], argv[2], argv[3], argv[4],
                            argv[5], argv[6]);
}
