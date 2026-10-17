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

#include <sched.h>
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
  no_memory
};

#ifdef __APPLE__

#include "rig_metal_stubs.h"

#define Device_val(v) ((struct rig_metal *)Long_val(v))
#define Run_val(v) (Device_val(v)->run)
#define Object_val(v) ((id)(intptr_t)Nativeint_val(v))

static value object(id o) { return caml_copy_nativeint((intnat)o); }

_Atomic intptr_t rig_metal_pipelines, rig_metal_buffers;

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
                                         rig_metal_commit, NULL};

/* A device whose word is the 8 bytes at [word], its values starting after
   theirs. */
static intnat open_device(_Atomic uint64_t *word) API_AVAILABLE(macos(15.0)) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device == nil) return -no_device;
  int f = family(device);
  struct rig_metal *d = f >= 0 ? calloc(1, sizeof *d) : NULL;
  struct rig_metal_run *r = f >= 0 ? calloc(1, sizeof *r) : NULL;
  if (d == NULL || r == NULL) {
    free(d), free(r);
    [device release];
    return f >= 0 ? -no_memory : -no_family;
  }
  MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
  d->driver = &driver;
  r->device = device;
  r->longest = (uint64_t)device.maxBufferLength;
  r->queue = [device newCommandQueueWithMaxCommandBufferCount:ring_slots];
  r->fence = [device newFence];
  r->set = [device newResidencySetWithDescriptor:desc error:NULL];
  [desc release];
  int missing = !r->queue   ? no_queue
                : !r->fence ? no_fence
                : !r->set   ? no_set
                            : 0;
  if (missing) {
    [r->queue release], [r->fence release], [r->set release];
    [device release], free(r), free(d);
    return -missing;
  }
  [r->queue addResidencySet:r->set];
  pthread_mutex_init(&r->set_mutex, NULL);
  pthread_mutex_init(&r->open_mutex, NULL);
  pthread_mutex_init(&r->grown_mutex, NULL);
  r->open = (struct rig_metal_queue){nil, nil, -1, r};
  d->at = (uint64_t *)word;
  r->last = atomic_load_explicit(word, memory_order_acquire);
  rig_metal_ring_init(&r->ring, r->slots, ring_slots, d->at);
  r->holds = 1;
  d->run = r;
  return (intnat)d;
}

void rig_metal_hold(struct rig_metal_run *r) {
  atomic_fetch_add_explicit(&r->holds, 1, memory_order_relaxed);
}

/* The last release frees [r]: no handler and no call of rig's uses it
   then. */
void rig_metal_release(struct rig_metal_run *r) {
  if (atomic_fetch_sub_explicit(&r->holds, 1, memory_order_acq_rel) != 1)
    return;
  [r->queue release], [r->fence release], [r->set release];
  [r->arena release], [r->grown release];
  pthread_mutex_destroy(&r->grown_mutex);
  [r->device release];
  pthread_mutex_destroy(&r->set_mutex);
  pthread_mutex_destroy(&r->open_mutex);
  rig_metal_ring_end(&r->ring);
  free(r);
}

/* Makes the arena the submission whose room answered RIG_GROW on this
   thread needs, for the next room to install: the least power of two from
   ARENA_MIN bytes that holds twice its parameters, at most the device's
   longest buffer. Growths of other submissions run beside it, so the
   arena handed to room is the largest one made, as NV's rings are
   (caml_rig_nv_hand): a growth that finds one as large made makes none,
   and one that makes a smaller one gives it back. The handed arena is read
   under [grown_mutex], since room installs and may release it meanwhile.
   0, or the bytes Metal made no buffer of. */
value caml_rig_metal_grow(value v_d) {
  struct rig_metal_run *r = Run_val(v_d);
  uint64_t n = ARENA_MIN;
  while (n < 2 * rig_metal_asked) n *= 2;
  if (n > r->longest) n = r->longest;
  pthread_mutex_lock(&r->grown_mutex);
  int held = r->grown != nil && r->grown.length >= n;
  pthread_mutex_unlock(&r->grown_mutex);
  if (held) return Val_long(0);
  id<MTLBuffer> b =
      [r->device newBufferWithLength:n
                             options:MTLResourceStorageModeShared |
                                     MTLResourceHazardTrackingModeUntracked];
  if (b == nil) return Val_long((intnat)n);
  pthread_mutex_lock(&r->grown_mutex);
  id<MTLBuffer> lost = b;
  if (r->grown == nil || r->grown.length < n) {
    lost = r->grown;
    r->grown = b;
  }
  pthread_mutex_unlock(&r->grown_mutex);
  [lost release];
  return Val_long(0);
}

value caml_rig_metal_open(value v_word) {
  _Atomic uint64_t *word = (_Atomic uint64_t *)Nativeint_val(v_word);
  intnat d = -before_15;
  @autoreleasepool {
    if (@available(macOS 15, *)) d = open_device(word);
  }
  return Val_long(d);
}

/* Where [b] lies, a Rig_edge.location: [{ address = Some (b's GPU
   address); host = Some (b's host address); handle = b }]. */
static value location(id<MTLBuffer> b) {
  CAMLparam0();
  CAMLlocal3(address, host, handle);
  address = caml_alloc_some(Val_long((intnat)b.gpuAddress));
  host = caml_alloc_some(Val_long((intnat)b.contents));
  handle = object(b);
  CAMLreturn(triple(address, host, handle));
}

/* The device's family and budget. */
value caml_rig_metal_facts(value v_d) {
  CAMLparam1(v_d);
  CAMLlocal1(v);
  id<MTLDevice> device = Device_val(v_d)->run->device;
  intnat budget = (intnat)device.recommendedMaxWorkingSetSize;
  v = caml_alloc_tuple(2);
  Store_field(v, 0, Val_int(family(device)));
  Store_field(v, 1, Val_long(budget));
  CAMLreturn(v);
}

/* Memory */

/* [Some (location b)], resident from the device's next submission, or
   [None] for nil. */
static value resident(struct rig_metal_run *r, id<MTLBuffer> b) {
  if (b == nil) return Val_none;
  atomic_fetch_add(&rig_metal_buffers, 1);
  pthread_mutex_lock(&r->set_mutex);
  [r->set addAllocation:b];
  r->changed = 1;
  pthread_mutex_unlock(&r->set_mutex);
  return caml_alloc_some(location(b));
}

value caml_rig_metal_alloc(value v_d, value v_n) {
  struct rig_metal_run *r = Run_val(v_d);
  return resident(r,
                  [r->device newBufferWithLength:(NSUInteger)Long_val(v_n)
                                         options:MTLResourceStorageModeShared]);
}

/* The pages holding the [v_n] bytes at [v_p], wrapped without a copy. */
value caml_rig_metal_map_host(value v_d, value v_p, value v_n) {
  struct rig_metal_run *r = Run_val(v_d);
  uintptr_t page = (uintptr_t)getpagesize(), p = (uintptr_t)Long_val(v_p);
  uintptr_t first = p & ~(page - 1);
  uintptr_t last = (p + (uintptr_t)Long_val(v_n) + page - 1) & ~(page - 1);
  return resident(
      r, [r->device newBufferWithBytesNoCopy:(void *)first
                                      length:last - first
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil]);
}

/* Takes [v_buffer] out of the residency set at once, then releases it. A
   stopped device's set goes with its run state. The caller excludes the
   stop. */
value caml_rig_metal_free(value v_d, value v_buffer) {
  struct rig_metal_run *r = Run_val(v_d);
  if (r != NULL) {
    pthread_mutex_lock(&r->set_mutex);
    [r->set removeAllocation:Object_val(v_buffer)];
    [r->set commit];
    r->changed = 0;
    pthread_mutex_unlock(&r->set_mutex);
  }
  [Object_val(v_buffer) release];
  atomic_fetch_sub(&rig_metal_buffers, 1);
  return Val_unit;
}

/* Images */

/* The metallib [v_b], which makes no pipeline: [("", library, names)],
   or [(why, 0, [||])] if Metal loads no library of [v_b] or one of its
   functions is no compute kernel. Loading compiles nothing for the GPU. */
value caml_rig_metal_image(value v_d, value v_b) {
  CAMLparam2(v_d, v_b);
  CAMLlocal3(names, why, lib);
  id<MTLDevice> device = Run_val(v_d)->device;
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

/* The bytes of parameters a function reads, as [reflection] binds them:
   the size of the struct at buffer 0, which a launch sets with
   [setBytes:]; 0 with no buffer 0; -1 where buffer 0 is no struct, such as
   a pointer to words, whose length the function does not state. */
static int64_t params_of(MTLComputePipelineReflection *reflection) {
  for (id<MTLBinding> b in reflection.bindings) {
    if (b.index != 0 || b.type != MTLBindingTypeBuffer) continue;
    id<MTLBufferBinding> buffer = (id<MTLBufferBinding>)b;
    MTLPointerType *p = buffer.bufferPointerType;
    if (p == nil || p.elementType != MTLDataTypeStruct) return -1;
    return (int64_t)buffer.bufferDataSize;
  }
  return 0;
}

/* The entry of the function [v_f] of the library [v_lib], whose pipeline
   an indirect command buffer may use too, as the address of its C state;
   raises [Failure] with Metal's reason if Metal makes no pipeline. A
   threadgroup holds at most the pipeline's [maxTotalThreadsPerThreadgroup]
   threads, and at most the device's along each axis: their least bounds
   both. It releases the runtime while Metal compiles. */
value caml_rig_metal_pipeline(value v_lib, value v_f) {
  CAMLparam2(v_lib, v_f);
  id<MTLLibrary> library = Object_val(v_lib);
  char *f = caml_stat_strdup(String_val(v_f));
  struct rig_metal_entry *e = calloc(1, sizeof *e);
  if (e == NULL) {
    caml_stat_free(f);
    caml_raise_out_of_memory();
  }
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
    MTLComputePipelineReflection *reflection = nil;
    p = [library.device
        newComputePipelineStateWithDescriptor:desc
                                      options:MTLPipelineOptionBindingInfo
                                   reflection:&reflection
                                        error:&error];
    if (p == nil)
      snprintf(text, sizeof text, "%s",
               error.localizedDescription.UTF8String ?: "no reason given");
    else
      e->params = params_of(reflection);
    [desc release];
  }
  caml_leave_blocking_section();
  caml_stat_free(f);
  if (p == nil) {
    free(e);
    caml_failwith(text);
  }
  MTLSize axes = library.device.maxThreadsPerThreadgroup;
  NSUInteger n = p.maxTotalThreadsPerThreadgroup;
  n = MIN(n, MIN(axes.width, MIN(axes.height, axes.depth)));
  e->pipeline = p;
  atomic_fetch_add(&rig_metal_pipelines, 1);
  e->threads = (uint32_t)n;
  e->shared = (uint32_t)(library.device.maxThreadgroupMemoryLength -
                         p.staticThreadgroupMemoryLength);
  CAMLreturn(caml_copy_nativeint((intnat)e));
}

/* An entry's pipeline, as an int, and its most threads per threadgroup. */
value caml_rig_metal_entry_code(value v_e) {
  struct rig_metal_entry *e = (struct rig_metal_entry *)Nativeint_val(v_e);
  return Val_long((intnat)e->pipeline);
}

value caml_rig_metal_entry_threads(value v_e) {
  struct rig_metal_entry *e = (struct rig_metal_entry *)Nativeint_val(v_e);
  return Val_long(e->threads);
}

/* An entry's bytes of parameters, or -1. */
value caml_rig_metal_entry_params(value v_e) {
  struct rig_metal_entry *e = (struct rig_metal_entry *)Nativeint_val(v_e);
  return Val_long((intnat)e->params);
}

/* Releases the entry [e] and its pipeline. */
static void free_entry(struct rig_metal_entry *e) {
  [e->pipeline release];
  free(e);
  atomic_fetch_sub(&rig_metal_pipelines, 1);
}

value caml_rig_metal_release(value v_e) {
  free_entry((struct rig_metal_entry *)Nativeint_val(v_e));
  return Val_unit;
}

/* Releases each entry of the array [v_entries] whose code is not 0, those an
   [entry] made, then the library [v_library]: an image's unload, in one
   call that allocates nothing. */
value caml_rig_metal_unload(value v_entries, value v_library) {
  for (mlsize_t k = 0; k < Wosize_val(v_entries); k++) {
    value v_e = Field(v_entries, k);
    if (Long_val(Field(v_e, 0)) == 0) continue;
    free_entry((struct rig_metal_entry *)Nativeint_val(Field(v_e, 1)));
  }
  [Object_val(v_library) release];
  return Val_unit;
}

/* Indirect command buffers */

/* What an indirect command buffer's release frees: the buffer, its
   commands and the pipelines they hold, one per command. */
struct icb {
  id<MTLIndirectCommandBuffer> icb;
  pid_t process; /* the process that made it: a forked child calls no Metal */
  int n;
  id commands[];
};

static void free_icb(struct icb *b) {
  if (getpid() != b->process) return;
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
  struct rig_metal_run *r = Run_val(v_d);
  id<MTLBuffer> args = Object_val(v_buffer);
  int n = (int)Wosize_val(v_pipelines);
  /* Allocate every OCaml result before acquiring native objects. */
  objects = caml_alloc_tuple(2 + (mlsize_t)n);
  for (mlsize_t i = 0; i < 2 + (mlsize_t)n; i++) {
    v = caml_copy_nativeint(0);
    Store_field(objects, i, v);
  }
  struct icb *b = calloc(1, sizeof *b + 2 * n * sizeof(id));
  if (b == NULL) caml_raise_out_of_memory();
  b->process = getpid();
  MTLIndirectCommandBufferDescriptor *desc =
      [[MTLIndirectCommandBufferDescriptor alloc] init];
  desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
  desc.inheritBuffers = NO;
  desc.inheritPipelineState = NO;
  desc.maxKernelBufferBindCount = 1;
  b->icb = [r->device newIndirectCommandBufferWithDescriptor:desc
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
  Nativeint_val(Field(objects, 0)) = (intnat)b;
  Nativeint_val(Field(objects, 1)) = (intnat)b->icb;
  for (int i = 0; i < n; i++)
    Nativeint_val(Field(objects, 2 + i)) = (intnat)b->commands[2 * i];
  CAMLreturn(objects);
}

value caml_rig_metal_icb_release(value v_icb) {
  free_icb((struct icb *)Nativeint_val(v_icb));
  return Val_unit;
}

/* The timeline */

intnat caml_rig_metal_signaled(intnat d) {
  return (intnat)__atomic_load_n(((struct rig_metal *)d)->at,
                                 __ATOMIC_ACQUIRE);
}

/* 0 once the word differs from [v_seen] or [v_ms] passed, 1 if a
   submission failed. Raises [Invalid_argument] once the device stopped. */
value caml_rig_metal_sleep(value v_d, value v_seen, value v_ms) {
  struct rig_metal_run *r = Run_val(v_d);
  if (r == NULL) caml_invalid_argument("Rig_metal.sleep: the device stopped");
  uint64_t seen = (uint64_t)Long_val(v_seen);
  int ms = Int_val(v_ms);
  caml_enter_blocking_section_no_pending();
  const char *why = rig_metal_ring_sleep(&r->ring, seen, ms);
  caml_leave_blocking_section();
  return Val_int(why != NULL);
}

/* The device's first failure. Raises [Invalid_argument] once the device
   stopped. */
value caml_rig_metal_failure(value v_d) {
  struct rig_metal_run *r = Run_val(v_d);
  if (r == NULL)
    caml_invalid_argument("Rig_metal.failure: the device stopped");
  const char *why = rig_metal_ring_failure(&r->ring);
  return caml_copy_string(why ? why : "");
}

/* Stops the device: the word shows its last value once its ring is
   empty. The run state goes with the device's hold: each committed command
   buffer holds it, and the command queue, until it completes. */
value caml_rig_metal_stop(value v_d) {
  struct rig_metal *d = Device_val(v_d);
  struct rig_metal_run *r = d->run;
  d->run = NULL;
  rig_metal_drop(r);
  rig_metal_ring_stop(&r->ring, r->last);
  rig_metal_release(r);
  return Val_unit;
}

static int (*const split)(void *, uint64_t *, uint64_t *) = rig_metal_split;

#else

value caml_rig_metal_count(value unit) {
  (void)unit;
  return Val_int(0);
}

value caml_rig_metal_open(value v_word) {
  (void)v_word;
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
NO_METAL1(caml_rig_metal_release)
NO_METAL2(caml_rig_metal_image)
NO_METAL2(caml_rig_metal_pipeline)
NO_METAL1(caml_rig_metal_entry_code)
NO_METAL1(caml_rig_metal_entry_threads)
NO_METAL1(caml_rig_metal_entry_params)
NO_METAL2(caml_rig_metal_unload)
NO_METAL1(caml_rig_metal_icb_release)
NO_METAL3(caml_rig_metal_sleep)
NO_METAL1(caml_rig_metal_failure)
NO_METAL1(caml_rig_metal_stop)
NO_METAL1(caml_rig_metal_grow)

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
