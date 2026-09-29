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
#include <stdio.h>
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
  if (queue == nil) caml_failwith("Metal: cannot allocate a command queue");
  return caml_copy_nativeint((intnat)queue);
}

value caml_nx_metal_new_event(value v_device) {
  id<MTLDevice> device = Object_val(v_device);
  id<MTLSharedEvent> event = [device newSharedEvent];
  if (event == nil) caml_failwith("Metal: cannot create a shared event");
  return caml_copy_nativeint((intnat)event);
}

value caml_nx_metal_new_fence(value v_device) {
  id<MTLDevice> device = Object_val(v_device);
  id<MTLFence> fence = [device newFence];
  if (fence == nil) caml_failwith("Metal: cannot create a fence");
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

/* [Some { host = Some host; device; handle }] for a buffer, [None] for nil. */
static value memory_of_buffer(id<MTLBuffer> buffer, intnat host) {
  CAMLparam0();
  CAMLlocal2(memory, v);
  if (buffer == nil) CAMLreturn(Val_none);
  memory = caml_alloc_tuple(3);
  v = caml_copy_nativeint(host);
  v = caml_alloc_some(v);
  Store_field(memory, 0, v);
  v = caml_copy_nativeint((intnat)buffer.gpuAddress);
  Store_field(memory, 1, v);
  v = caml_copy_nativeint((intnat)buffer);
  Store_field(memory, 2, v);
  CAMLreturn(caml_alloc_some(memory));
}

value caml_nx_metal_alloc(value v_device, value v_size) {
  CAMLparam2(v_device, v_size);
  id<MTLDevice> device = Object_val(v_device);
  id<MTLBuffer> buffer =
      [device newBufferWithLength:(NSUInteger)Long_val(v_size)
                          options:MTLResourceStorageModeShared];
  CAMLreturn(memory_of_buffer(buffer, (intnat)buffer.contents));
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
  CAMLreturn(memory_of_buffer(buffer, (intnat)first));
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
      set_message(&error_message, error, "Metal: invalid library");
    } else {
      id<MTLFunction> function = [library newFunctionWithName:name];
      if (function == nil) {
        snprintf(error_message.text, sizeof(error_message.text),
                 "Metal: the library has no function %s", name.UTF8String);
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
          set_message(&error_message, error, "Metal: cannot build a pipeline");
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

value caml_nx_metal_signaled(value v_event) {
  id<MTLSharedEvent> event = Object_val(v_event);
  return Val_long((intnat)event.signaledValue);
}

value caml_nx_metal_wait(value v_event, value v_value, value v_timeout_ms) {
  id<MTLSharedEvent> event = Object_val(v_event);
  uint64_t target = (uint64_t)Long_val(v_value);
  uint64_t timeout_ms = (uint64_t)Long_val(v_timeout_ms);
  caml_release_runtime_system();
  BOOL signaled = [event waitUntilSignaledValue:target timeoutMS:timeout_ms];
  caml_acquire_runtime_system();
  return Val_bool(signaled);
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
