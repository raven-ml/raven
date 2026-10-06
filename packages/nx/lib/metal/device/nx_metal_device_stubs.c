/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <math.h>
#include <os/lock.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Objective-C objects cross to OCaml as retained pointers in nativeints. */
#define Object_val(v) ((id)(intptr_t)Nativeint_val(v))

extern void *objc_autoreleasePoolPush(void);
extern void objc_autoreleasePoolPop(void *pool);

/* An error message copied out of an autorelease pool, so that raising does
   not leave one. */
typedef struct {
  char text[512];
} message;

static void set_message(message *m, NSError *error, const char *fallback) {
  const char *text = error != nil && error.localizedDescription != nil
                         ? error.localizedDescription.UTF8String
                         : fallback;
  snprintf(m->text, sizeof(m->text), "%s", text);
}

value caml_nx_metal_create_device(value unit) {
  (void)unit;
  @autoreleasepool {
    return caml_copy_nativeint((intnat)MTLCreateSystemDefaultDevice());
  }
}

/* The GPU family, as its enumeration names it: the highest Apple family the
   device supports, else its Mac family. */
value caml_nx_metal_arch(value v_device) {
  CAMLparam1(v_device);
  id<MTLDevice> device = Object_val(v_device);
  char arch[16] = "";
  for (int family = 10; family >= 1 && arch[0] == '\0'; family--)
    if ([device supportsFamily:(MTLGPUFamily)(1000 + family)])
      snprintf(arch, sizeof(arch), "Apple%d", family);
  for (int family = 2; family >= 1 && arch[0] == '\0'; family--)
    if ([device supportsFamily:(MTLGPUFamily)(2000 + family)])
      snprintf(arch, sizeof(arch), "Mac%d", family);
  CAMLreturn(caml_copy_string(arch));
}

value caml_nx_metal_working_set(value v_device) {
  id<MTLDevice> device = Object_val(v_device);
  return Val_long((intnat)device.recommendedMaxWorkingSetSize);
}

value caml_nx_metal_unified(value v_device) {
  id<MTLDevice> device = Object_val(v_device);
  return Val_bool(device.hasUnifiedMemory);
}

value caml_nx_metal_new_queue(value v_device) {
  id<MTLDevice> device = Object_val(v_device);
  id<MTLCommandQueue> queue =
      [device newCommandQueueWithMaxCommandBufferCount:1024];
  if (queue == nil) caml_failwith("cannot allocate a command queue");
  return caml_copy_nativeint((intnat)queue);
}

value caml_nx_metal_new_event(value v_device) {
  id<MTLDevice> device = Object_val(v_device);
  id<MTLSharedEvent> event = [device newSharedEvent];
  if (event == nil) caml_failwith("cannot create a shared event");
  return caml_copy_nativeint((intnat)event);
}

/* Signals values on a device's event from the completed handlers of its
   command buffers, so that a value is signaled only once its command buffer,
   and every one before it, completed without error: after a failure, nothing
   is signaled again. A failed command buffer
   still runs the signals it encodes, so a signal encoded on the GPU would
   hide the failure. The first failure's reason is kept, and the device's
   waits raise it. */
@interface NxMetalSignaler : NSObject {
 @public
  id<MTLSharedEvent> event;
  os_unfair_lock lock;
  char *failure;
}
- (void)signal:(id<MTLCommandBuffer>)command value:(uint64_t)v;
@end

@implementation NxMetalSignaler
- (void)signal:(id<MTLCommandBuffer>)command value:(uint64_t)v {
  [command addCompletedHandler:^(id<MTLCommandBuffer> done) {
    os_unfair_lock_lock(&lock);
    if (done.status == MTLCommandBufferStatusError) {
      if (failure == NULL) {
        NSString *why = done.error.localizedDescription;
        failure = strdup(why != nil ? why.UTF8String
                                    : "a command buffer failed");
      }
    } else if (failure == NULL && v > event.signaledValue) {
      event.signaledValue = v;
    }
    os_unfair_lock_unlock(&lock);
  }];
}
@end

value caml_nx_metal_new_signaler(value v_event) {
  NxMetalSignaler *signaler = [[NxMetalSignaler alloc] init];
  signaler->event = [Object_val(v_event) retain];
  signaler->lock = OS_UNFAIR_LOCK_INIT;
  signaler->failure = NULL;
  return caml_copy_nativeint((intnat)signaler);
}

/* Raises the first failure of [signaler]'s command buffers, if any. */
static void check_failure(NxMetalSignaler *signaler) {
  char text[512];
  text[0] = '\0';
  os_unfair_lock_lock(&signaler->lock);
  if (signaler->failure != NULL)
    snprintf(text, sizeof(text), "%s", signaler->failure);
  os_unfair_lock_unlock(&signaler->lock);
  if (text[0] != '\0') caml_failwith(text);
}

value caml_nx_metal_new_fence(value v_device) {
  id<MTLDevice> device = Object_val(v_device);
  id<MTLFence> fence = [device newFence];
  if (fence == nil) caml_failwith("cannot create a fence");
  return caml_copy_nativeint((intnat)fence);
}

/* A residency set added to [queue], or 0 where Metal has none. Called by name
   so that the library builds against SDKs that predate residency sets. */
value caml_nx_metal_new_residency_set(value v_device, value v_queue) {
  id device = Object_val(v_device);
  id queue = Object_val(v_queue);
  id set = nil;
  @autoreleasepool {
    Class descriptor_class = NSClassFromString(@"MTLResidencySetDescriptor");
    SEL make = sel_registerName("newResidencySetWithDescriptor:error:");
    if (descriptor_class != nil && [device respondsToSelector:make]) {
      id descriptor = [[descriptor_class alloc] init];
      NSError *error = nil;
      set = ((id(*)(id, SEL, id, NSError **))objc_msgSend)(device, make,
                                                           descriptor, &error);
      [descriptor release];
      if (set != nil)
        ((void (*)(id, SEL, id))objc_msgSend)(
            queue, sel_registerName("addResidencySet:"), set);
    }
  }
  return caml_copy_nativeint((intnat)set);
}

value caml_nx_metal_residency(value v_set, value v_buffer, value v_add) {
  id set = Object_val(v_set);
  id buffer = Object_val(v_buffer);
  SEL change = sel_registerName(Bool_val(v_add) ? "addAllocation:"
                                                : "removeAllocation:");
  ((void (*)(id, SEL, id))objc_msgSend)(set, change, buffer);
  ((void (*)(id, SEL))objc_msgSend)(set, sel_registerName("commit"));
  return Val_unit;
}

/* [Some (host, device address, buffer, bytes)] for a buffer, [None] for nil.
 */
static value region_of_buffer(id<MTLBuffer> buffer, intnat host) {
  CAMLparam0();
  CAMLlocal2(region, v);
  if (buffer == nil) CAMLreturn(Val_none);
  region = caml_alloc_tuple(4);
  v = caml_copy_nativeint(host);
  Store_field(region, 0, v);
  v = caml_copy_nativeint((intnat)buffer.gpuAddress);
  Store_field(region, 1, v);
  v = caml_copy_nativeint((intnat)buffer);
  Store_field(region, 2, v);
  Store_field(region, 3, Val_long(buffer.length));
  CAMLreturn(caml_alloc_some(region));
}

value caml_nx_metal_alloc(value v_device, value v_size) {
  CAMLparam2(v_device, v_size);
  id<MTLDevice> device = Object_val(v_device);
  id<MTLBuffer> buffer =
      [device newBufferWithLength:(NSUInteger)Long_val(v_size)
                          options:MTLResourceStorageModeShared];
  CAMLreturn(region_of_buffer(buffer, (intnat)buffer.contents));
}

/* A buffer over the pages that hold the [v_size] bytes at [v_ptr], without a
   copy: Metal wraps whole pages only. Its host address is the first page. */
value caml_nx_metal_wrap(value v_device, value v_ptr, value v_size) {
  CAMLparam3(v_device, v_ptr, v_size);
  id<MTLDevice> device = Object_val(v_device);
  uintptr_t page = (uintptr_t)getpagesize();
  uintptr_t ptr = (uintptr_t)Nativeint_val(v_ptr);
  uintptr_t first = ptr & ~(page - 1);
  uintptr_t last = (ptr + (uintptr_t)Long_val(v_size) + page - 1) & ~(page - 1);
  id<MTLBuffer> buffer =
      [device newBufferWithBytesNoCopy:(void *)first
                                length:last - first
                               options:MTLResourceStorageModeShared
                           deallocator:nil];
  CAMLreturn(region_of_buffer(buffer, (intnat)first));
}

value caml_nx_metal_release(value v_object) {
  [Object_val(v_object) release];
  return Val_unit;
}

/* The pipeline of function [v_name] of the metallib [v_binary]. */
value caml_nx_metal_pipeline(value v_device, value v_binary, value v_name) {
  CAMLparam3(v_device, v_binary, v_name);
  id<MTLDevice> device = Object_val(v_device);
  id<MTLComputePipelineState> pipeline = nil;
  message error_message;
  error_message.text[0] = '\0';
  @autoreleasepool {
    dispatch_data_t data = dispatch_data_create(
        String_val(v_binary), caml_string_length(v_binary), NULL,
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSString *name = [NSString stringWithUTF8String:String_val(v_name)];
    caml_release_runtime_system();
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
    dispatch_release(data);
    if (library == nil) {
      set_message(&error_message, error, "invalid library");
    } else {
      id<MTLFunction> function = [library newFunctionWithName:name];
      if (function == nil) {
        snprintf(error_message.text, sizeof(error_message.text),
                 "the library has no function %s", name.UTF8String);
      } else {
        MTLComputePipelineDescriptor *descriptor =
            [[MTLComputePipelineDescriptor alloc] init];
        descriptor.computeFunction = function;
        descriptor.supportIndirectCommandBuffers = YES;
        pipeline = [device
            newComputePipelineStateWithDescriptor:descriptor
                                          options:MTLPipelineOptionNone
                                       reflection:nil
                                            error:&error];
        if (pipeline == nil)
          set_message(&error_message, error, "cannot build a pipeline");
        [descriptor release];
        [function release];
      }
      [library release];
    }
    caml_acquire_runtime_system();
  }
  if (pipeline == nil) caml_failwith(error_message.text);
  CAMLreturn(caml_copy_nativeint((intnat)pipeline));
}

/* The last value [v_signaler] signaled, or the failure of a command buffer
   of its device. */
value caml_nx_metal_signaled(value v_signaler) {
  NxMetalSignaler *signaler = (NxMetalSignaler *)Object_val(v_signaler);
  check_failure(signaler);
  return Val_long((intnat)signaler->event.signaledValue);
}

/* Waits until [v_signaler]'s event reaches [v_value], for at most [v_ms]
   milliseconds, and raises the failure of a command buffer of its device. */
value caml_nx_metal_wait(value v_signaler, value v_value, value v_ms) {
  NxMetalSignaler *signaler = (NxMetalSignaler *)Object_val(v_signaler);
  uint64_t target = (uint64_t)Long_val(v_value);
  uint64_t ms = (uint64_t)Long_val(v_ms);
  check_failure(signaler);
  caml_release_runtime_system();
  BOOL signaled = [signaler->event waitUntilSignaledValue:target timeoutMS:ms];
  caml_acquire_runtime_system();
  check_failure(signaler);
  return Val_bool(signaled);
}

/* Writes the GPU start and end times of the command buffer that the stamp word
   of the first of the two slots at [v_words] holds, if the second's is 0, over
   them as host nanoseconds, and releases the command buffer. GPUStartTime and
   GPUEndTime are mach time in seconds, the host clock. */
value caml_nx_metal_resolve(value v_words) {
  uint64_t *words = (uint64_t *)Nativeint_val(v_words);
  if (words[1] != 0 && words[3] == 0) {
    id<MTLCommandBuffer> command = (id)(uintptr_t)words[1];
    caml_release_runtime_system();
    [command waitUntilCompleted];
    caml_acquire_runtime_system();
    words[1] = (uint64_t)llround(command.GPUStartTime * 1e9);
    words[3] = (uint64_t)llround(command.GPUEndTime * 1e9);
    [command release];
  }
  return Val_unit;
}

value caml_nx_metal_msg_send(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&objc_msgSend);
}

/* A selector is an opaque word: its bits cross as they are. */
value caml_nx_metal_selector(value v_name) {
  SEL selector = sel_registerName(String_val(v_name));
  intnat word;
  memcpy(&word, &selector, sizeof word);
  return caml_copy_nativeint(word);
}

value caml_nx_metal_max_threads(value v_pipeline) {
  id<MTLComputePipelineState> pipeline = Object_val(v_pipeline);
  return Val_long((intnat)pipeline.maxTotalThreadsPerThreadgroup);
}

/* An indirect command buffer of one concurrent dispatch per pipeline of
   [v_pipelines], each after the ones before it, with [v_buffer] as its kernel
   buffer 0: the words [v_launches] give each its offset in the buffer, then
   its threadgroups per grid and threads per threadgroup on three axes. The
   result is the indirect command buffer, then each command, all retained. */
value caml_nx_metal_new_icb(value v_device, value v_buffer, value v_pipelines,
                            value v_launches) {
  CAMLparam4(v_device, v_buffer, v_pipelines, v_launches);
  CAMLlocal2(result, v);
  id<MTLDevice> device = Object_val(v_device);
  id<MTLBuffer> buffer = Object_val(v_buffer);
  mlsize_t n = Wosize_val(v_pipelines);
  id<MTLIndirectCommandBuffer> icb = nil;
  @autoreleasepool {
    MTLIndirectCommandBufferDescriptor *descriptor =
        [[MTLIndirectCommandBufferDescriptor alloc] init];
    descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    descriptor.maxKernelBufferBindCount = 1;
    icb = [device newIndirectCommandBufferWithDescriptor:descriptor
                                         maxCommandCount:(n > 0 ? n : 1)
                                                 options:0];
    [descriptor release];
  }
  if (icb == nil)
    caml_failwith("cannot create an indirect command buffer: does the GPU "
                  "support them?");
  result = caml_alloc(n + 1, 0);
  v = caml_copy_nativeint((intnat)icb);
  Store_field(result, 0, v);
  for (mlsize_t i = 0; i < n; i++) {
    id<MTLIndirectComputeCommand> command =
        [[icb indirectComputeCommandAtIndex:i] retain];
    intnat w[7];
    for (int k = 0; k < 7; k++) w[k] = Long_val(Field(v_launches, 7 * i + k));
    [command setComputePipelineState:Object_val(Field(v_pipelines, i))];
    [command setKernelBuffer:buffer offset:(NSUInteger)w[0] atIndex:0];
    [command concurrentDispatchThreadgroups:MTLSizeMake(w[1], w[2], w[3])
                      threadsPerThreadgroup:MTLSizeMake(w[4], w[5], w[6])];
    [command setBarrier];
    v = caml_copy_nativeint((intnat)command);
    Store_field(result, i + 1, v);
  }
  CAMLreturn(result);
}

/* The autorelease pool of the thread that synchronizes, cycled at the end of
   each synchronization so that objects autoreleased by submissions outside
   any pool are drained. */
static _Thread_local void *pool = NULL;

value caml_nx_metal_cycle_pool(value unit) {
  (void)unit;
  if (pool != NULL) objc_autoreleasePoolPop(pool);
  pool = objc_autoreleasePoolPush();
  return Val_unit;
}
