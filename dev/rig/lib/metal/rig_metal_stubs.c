/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Opening, memory, images, indirect command buffers and the OCaml side of
   the timeline. Objective-C objects cross to OCaml as retained pointers in
   nativeints, pipelines in ints. A stub that waits or compiles releases the
   runtime without running pending signals, which run at the caller's next
   poll point; the others hold it, since none blocks. Off macOS no device
   opens, and the stubs that serve an open device are never called. */

#define _GNU_SOURCE

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

#include "rig_metal.h"

/* Why open fails, as rig_metal.ml reads it from a negative address. */
enum {
  not_macos = 1,
  no_device,
  before_15,
  no_family,
  no_queue,
  no_fence,
  no_set,
  no_word,
  no_memory
};

#ifdef __APPLE__

#include "rig_metal_stubs.h"

#define Device_val(v) ((struct rig_metal *)Long_val(v))
#define Object_val(v) ((id)(intptr_t)Nativeint_val(v))

static value object(id o) { return caml_copy_nativeint((intnat)o); }

static value triple(value a, value b, value c) {
  CAMLparam3(a, b, c);
  CAMLlocal1(v);
  v = caml_alloc_tuple(3);
  Store_field(v, 0, a);
  Store_field(v, 1, b);
  Store_field(v, 2, c);
  CAMLreturn(v);
}

value caml_rig_metal_count(value unit) {
  (void)unit;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  [device release];
  return Val_int(device != nil);
}

/* The highest Apple GPU family [device] supports (1 to 10), else 0 for
   the Mac2 family, else -1. */
static int family(id<MTLDevice> device) {
  for (int f = 10; f >= 1; f--)
    if ([device supportsFamily:(MTLGPUFamily)(MTLGPUFamilyApple1 + f - 1)])
      return f;
  return [device supportsFamily:MTLGPUFamilyMac2] ? 0 : -1;
}

static const struct rig_driver driver = {rig_metal_room, rig_metal_submit,
                                         rig_metal_commit};

static intnat open_device(void) API_AVAILABLE(macos(15.0)) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device == nil) return -no_device;
  int f = family(device);
  struct rig_metal *d = f >= 0 ? calloc(1, sizeof *d) : NULL;
  if (d == NULL) {
    [device release];
    return f >= 0 ? -no_memory : -no_family;
  }
  MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
  d->driver = &driver;
  d->device = device;
  d->queue = [device newCommandQueueWithMaxCommandBufferCount:ring_slots];
  d->fence = [device newFence];
  d->set = [device newResidencySetWithDescriptor:desc error:NULL];
  d->word = [device newBufferWithLength:sizeof(uint64_t)
                                options:MTLResourceStorageModeShared];
  [desc release];
  int missing = !d->queue   ? no_queue
                : !d->fence ? no_fence
                : !d->set   ? no_set
                : !d->word  ? no_word
                            : 0;
  if (missing) {
    [d->queue release], [d->fence release], [d->set release];
    [d->word release], [device release], free(d);
    return -missing;
  }
  [d->queue addResidencySet:d->set];
  pthread_mutex_init(&d->set_mutex, NULL);
  pthread_mutex_init(&d->open_mutex, NULL);
  d->open = (struct rig_metal_queue){nil, nil, -1, d};
  rig_metal_ring_init(&d->ring, d->slots, ring_slots, d->word.contents);
  return (intnat)d;
}

value caml_rig_metal_open(value unit) {
  (void)unit;
  intnat d = -before_15;
  @autoreleasepool {
    if (@available(macOS 15, *)) d = open_device();
  }
  return Val_long(d);
}

/* [(b, b's GPU address, b's host address)]. */
static value buffer(id<MTLBuffer> b) {
  CAMLparam0();
  CAMLlocal1(handle);
  handle = object(b);
  CAMLreturn(triple(handle, Val_long((intnat)b.gpuAddress),
                    Val_long((intnat)b.contents)));
}

/* The device's family, budget and word. */
value caml_rig_metal_facts(value v_d) {
  CAMLparam1(v_d);
  CAMLlocal1(word);
  struct rig_metal *d = Device_val(v_d);
  word = buffer(d->word);
  intnat budget = (intnat)d->device.recommendedMaxWorkingSetSize;
  CAMLreturn(triple(Val_int(family(d->device)), Val_long(budget), word));
}

/* Memory */

/* [Some (buffer b)], resident from the device's next submission, or [None]
   for nil. */
static value resident(struct rig_metal *d, id<MTLBuffer> b) {
  if (b == nil) return Val_none;
  pthread_mutex_lock(&d->set_mutex);
  [d->set addAllocation:b];
  d->changed = 1;
  pthread_mutex_unlock(&d->set_mutex);
  return caml_alloc_some(buffer(b));
}

value caml_rig_metal_alloc(value v_d, value v_n) {
  struct rig_metal *d = Device_val(v_d);
  return resident(d,
                  [d->device newBufferWithLength:(NSUInteger)Long_val(v_n)
                                         options:MTLResourceStorageModeShared]);
}

/* The pages holding the [v_n] bytes at [v_p], wrapped without a copy. */
value caml_rig_metal_map_host(value v_d, value v_p, value v_n) {
  struct rig_metal *d = Device_val(v_d);
  uintptr_t page = (uintptr_t)getpagesize(), p = (uintptr_t)Long_val(v_p);
  uintptr_t first = p & ~(page - 1);
  uintptr_t last = (p + (uintptr_t)Long_val(v_n) + page - 1) & ~(page - 1);
  return resident(
      d, [d->device newBufferWithBytesNoCopy:(void *)first
                                      length:last - first
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil]);
}

/* Takes [v_buffer] out of the residency set at once, then releases it. */
value caml_rig_metal_free(value v_d, value v_buffer) {
  struct rig_metal *d = Device_val(v_d);
  pthread_mutex_lock(&d->set_mutex);
  [d->set removeAllocation:Object_val(v_buffer)];
  [d->set commit];
  d->changed = 0;
  pthread_mutex_unlock(&d->set_mutex);
  [Object_val(v_buffer) release];
  return Val_unit;
}

/* Releases the stopped device's word, which no handler writes any more:
   the word is in no residency set. */
value caml_rig_metal_free_word(value v_d) {
  struct rig_metal *d = Device_val(v_d);
  [d->word release];
  d->word = nil;
  return Val_unit;
}

/* Images */

/* The metallib [v_b], which makes no pipeline: [("", library, names)],
   or [(why, 0, [||])] if Metal loads no library of [v_b] or one of its
   functions is no compute kernel. Loading compiles nothing for the GPU. */
value caml_rig_metal_image(value v_d, value v_b) {
  CAMLparam2(v_d, v_b);
  CAMLlocal3(names, why, lib);
  id<MTLDevice> device = Device_val(v_d)->device;
  dispatch_data_t data =
      dispatch_data_create(String_val(v_b), caml_string_length(v_b), NULL,
                           DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  char text[512] = "";
  @autoreleasepool {
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
    NSArray<NSString *> *fs = library ? library.functionNames : @[];
    if (library == nil)
      snprintf(text, sizeof text, "loading the image: %s",
               error.localizedDescription.UTF8String ?: "no reason given");
    for (NSUInteger i = 0; i < fs.count && text[0] == '\0'; i++) {
      id<MTLFunction> f = [[library newFunctionWithName:fs[i]] autorelease];
      if (f.functionType != MTLFunctionTypeKernel)
        snprintf(text, sizeof text, "the function \"%s\" is no compute kernel",
                 fs[i].UTF8String);
    }
    if (text[0]) {
      [library release];
      library = nil;
      fs = @[];
    }
    names = caml_alloc_tuple(fs.count);
    for (NSUInteger i = 0; i < fs.count; i++)
      Store_field(names, i, caml_copy_string(fs[i].UTF8String));
    lib = object(library);
  }
  dispatch_release(data);
  why = caml_copy_string(text);
  CAMLreturn(triple(why, lib, names));
}

/* The pipeline, usable from an indirect command buffer, of the function
   [v_f] of the library [v_lib]: [("", pipeline, most threads per
   threadgroup)], or [(why, 0, 0)] with Metal's reason if Metal makes none.
   It releases the runtime while Metal compiles. */
value caml_rig_metal_pipeline(value v_lib, value v_f) {
  CAMLparam2(v_lib, v_f);
  CAMLlocal1(why);
  id<MTLLibrary> library = Object_val(v_lib);
  char *f = caml_stat_strdup(String_val(v_f));
  char text[512] = "";
  id<MTLComputePipelineState> p = nil;
  caml_enter_blocking_section_no_pending();
  @autoreleasepool {
    MTLComputePipelineDescriptor *desc =
        [[MTLComputePipelineDescriptor alloc] init];
    desc.computeFunction = [[library
        newFunctionWithName:[NSString stringWithUTF8String:f]] autorelease];
    desc.supportIndirectCommandBuffers = YES;
    NSError *error = nil;
    p = [library.device newComputePipelineStateWithDescriptor:desc
                                                       options:0
                                                    reflection:nil
                                                         error:&error];
    if (p == nil)
      snprintf(text, sizeof text, "%s",
               error.localizedDescription.UTF8String ?: "no reason given");
    [desc release];
  }
  caml_leave_blocking_section();
  caml_stat_free(f);
  why = caml_copy_string(text);
  intnat max = p == nil ? 0 : (intnat)p.maxTotalThreadsPerThreadgroup;
  CAMLreturn(triple(why, Val_long((intnat)p), Val_long(max)));
}

value caml_rig_metal_release(value v_pipeline) {
  [(id)Long_val(v_pipeline) release];
  return Val_unit;
}

value caml_rig_metal_release_library(value v_lib) {
  [Object_val(v_lib) release];
  return Val_unit;
}

/* Indirect command buffers */

/* What an indirect command buffer's release frees: the buffer, its
   commands and the pipelines they hold, one per command. */
struct icb {
  id<MTLIndirectCommandBuffer> icb;
  int n;
  id commands[];
};

static void free_icb(struct icb *b) {
  for (int i = 0; i < 2 * b->n; i++) [b->commands[i] release];
  [b->icb release];
  free(b);
}

/* An indirect command buffer of one dispatch per pipeline of [v_pipelines],
   each after the one before, with [v_buffer] as kernel buffer 0; [v_sizes]
   gives each its offset, threadgroups per grid and threads per threadgroup,
   seven words, all checked by the caller. The result is the release, the
   buffer, then each command, or [[||]] if Metal made no buffer. */
value caml_rig_metal_icb(value v_d, value v_buffer, value v_pipelines,
                            value v_sizes) {
  CAMLparam4(v_d, v_buffer, v_pipelines, v_sizes);
  CAMLlocal2(objects, v);
  struct rig_metal *d = Device_val(v_d);
  id<MTLBuffer> args = Object_val(v_buffer);
  int n = (int)Wosize_val(v_pipelines);
  struct icb *b = calloc(1, sizeof *b + 2 * n * sizeof(id));
  if (b == NULL) caml_raise_out_of_memory();
  MTLIndirectCommandBufferDescriptor *desc =
      [[MTLIndirectCommandBufferDescriptor alloc] init];
  desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
  desc.inheritBuffers = NO;
  desc.inheritPipelineState = NO;
  desc.maxKernelBufferBindCount = 1;
  b->icb = [d->device newIndirectCommandBufferWithDescriptor:desc
                                             maxCommandCount:n > 0 ? n : 1
                                                     options:0];
  [desc release];
  if (b->icb == nil) {
    free(b);
    CAMLreturn(Atom(0));
  }
  b->n = n;
  /* Metal returns each command autoreleased: without a pool of its own,
     the calling thread's would keep it until the thread ends. */
  @autoreleasepool {
    for (int i = 0; i < n; i++) {
      id<MTLIndirectComputeCommand> c =
          [[b->icb indirectComputeCommandAtIndex:(NSUInteger)i] retain];
      id<MTLComputePipelineState> p =
          [(id)Long_val(Field(v_pipelines, i)) retain];
      intnat w[7];
      for (int k = 0; k < 7; k++) w[k] = Long_val(Field(v_sizes, 7 * i + k));
      [c setComputePipelineState:p];
      [c setKernelBuffer:args offset:(NSUInteger)w[0] atIndex:0];
      [c setBarrier];
      [c concurrentDispatchThreadgroups:MTLSizeMake(w[1], w[2], w[3])
                  threadsPerThreadgroup:MTLSizeMake(w[4], w[5], w[6])];
      b->commands[2 * i] = c;
      b->commands[2 * i + 1] = p;
    }
  }
  objects = caml_alloc_tuple(2 + (mlsize_t)n);
  v = caml_copy_nativeint((intnat)b);
  Store_field(objects, 0, v);
  v = object(b->icb);
  Store_field(objects, 1, v);
  for (int i = 0; i < n; i++) {
    v = object(b->commands[2 * i]);
    Store_field(objects, 2 + i, v);
  }
  CAMLreturn(objects);
}

value caml_rig_metal_icb_release(value v_icb) {
  free_icb((struct icb *)Nativeint_val(v_icb));
  return Val_unit;
}

/* The timeline */

intnat caml_rig_metal_signaled(intnat d) {
  return (intnat)__atomic_load_n(((struct rig_metal *)d)->ring.word,
                                 __ATOMIC_ACQUIRE);
}

/* 0 once the word differs from [v_seen] or [v_ms] passed, 1 if a
   submission failed. */
value caml_rig_metal_sleep(value v_d, value v_seen, value v_ms) {
  struct rig_metal *d = Device_val(v_d);
  uint64_t seen = (uint64_t)Long_val(v_seen);
  int ms = Int_val(v_ms);
  caml_enter_blocking_section_no_pending();
  const char *why = rig_metal_ring_sleep(&d->ring, seen, ms);
  caml_leave_blocking_section();
  return Val_int(why != NULL);
}

/* The device's first failure. */
value caml_rig_metal_failure(value v_d) {
  const char *why = rig_metal_ring_failure(&Device_val(v_d)->ring);
  return caml_copy_string(why ? why : "");
}

/* Stops the device. Once its ring is empty no handler runs, so the queue
   and fence go; the residency set stays for the frees that may follow. */
value caml_rig_metal_stop(value v_d) {
  struct rig_metal *d = Device_val(v_d);
  rig_metal_drop(d);
  int idle = rig_metal_ring_stop(&d->ring, d->last);
  if (idle) {
    [d->queue release];
    [d->fence release];
    d->queue = nil;
    d->fence = nil;
  }
  return Val_unit;
}

static int (*const split)(void *, uint64_t *, uint64_t *) = rig_metal_split;

#else

value caml_rig_metal_count(value unit) {
  (void)unit;
  return Val_int(0);
}

value caml_rig_metal_open(value unit) {
  (void)unit;
  return Val_long(-not_macos);
}

/* Unreachable: off macOS no device opens, so nothing reaches a stub that
   serves one. */
CAMLnoret static void no_metal(void) { abort(); }

#define NO_METAL1(f) \
  value f(value a) { (void)a, no_metal(); }
#define NO_METAL2(f) \
  value f(value a, value b) { (void)a, (void)b, no_metal(); }
#define NO_METAL3(f) \
  value f(value a, value b, value c) { (void)a, (void)b, (void)c, no_metal(); }

NO_METAL1(caml_rig_metal_facts)
NO_METAL2(caml_rig_metal_alloc)
NO_METAL3(caml_rig_metal_map_host)
NO_METAL2(caml_rig_metal_free)
NO_METAL1(caml_rig_metal_free_word)
NO_METAL1(caml_rig_metal_release)
NO_METAL2(caml_rig_metal_image)
NO_METAL2(caml_rig_metal_pipeline)
NO_METAL1(caml_rig_metal_release_library)
NO_METAL1(caml_rig_metal_icb_release)
NO_METAL3(caml_rig_metal_sleep)
NO_METAL1(caml_rig_metal_failure)
NO_METAL1(caml_rig_metal_stop)

value caml_rig_metal_icb(value a, value b, value c, value d) {
  (void)a, (void)b, (void)c, (void)d, no_metal();
}

intnat caml_rig_metal_signaled(intnat a) { (void)a, no_metal(); }

static int (*const split)(void *, uint64_t *, uint64_t *) = NULL;

#endif

value caml_rig_metal_signaled_byte(value v_d) {
  return Val_long(caml_rig_metal_signaled(Long_val(v_d)));
}

/* The C entry rig_metal_split. */
value caml_rig_metal_split(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)split);
}
