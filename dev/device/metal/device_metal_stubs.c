/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Opening, memory, images, indirect command buffers and the OCaml side of
   the timeline. Objective-C objects cross to OCaml as retained pointers in
   nativeints. A stub that waits or compiles releases the runtime without
   running pending signals, which run at the caller's next poll point; the
   others hold it, since none blocks. Off macOS no device opens, and the
   stubs that serve an open device are never called. */

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

#include "device_metal.h"

/* Why open fails, as device_metal.ml reads it from a negative address. */
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

static value tuple(int n, value a, value b, value c) {
  CAMLparam3(a, b, c);
  CAMLlocal1(v);
  v = caml_alloc_tuple(n);
  Store_field(v, 0, a);
  Store_field(v, 1, b);
  if (n > 2) Store_field(v, 2, c);
  CAMLreturn(v);
}

#ifdef __APPLE__

#include "device_metal_stubs.h"

#define Device_val(v) ((struct device_metal *)Long_val(v))
#define Object_val(v) ((id)(intptr_t)Nativeint_val(v))

static value object(id o) { return caml_copy_nativeint((intnat)o); }

value caml_device_metal_count(value unit) {
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

static intnat open_device(void) API_AVAILABLE(macos(15.0)) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device == nil) return -no_device;
  int f = family(device);
  struct device_metal *d = f >= 0 ? calloc(1, sizeof *d) : NULL;
  if (d == NULL) {
    [device release];
    return f >= 0 ? -no_memory : -no_family;
  }
  MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
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
  device_metal_ring_init(&d->ring, d->slots, ring_slots, d->word.contents);
  return (intnat)d;
}

value caml_device_metal_open(value unit) {
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
  CAMLlocal2(handle, host);
  handle = object(b);
  host = caml_copy_nativeint((intnat)b.contents);
  CAMLreturn(tuple(3, handle, Val_long((intnat)b.gpuAddress), host));
}

/* The device's family, budget and word. */
value caml_device_metal_facts(value v_d) {
  CAMLparam1(v_d);
  CAMLlocal1(word);
  struct device_metal *d = Device_val(v_d);
  word = buffer(d->word);
  intnat budget = (intnat)d->device.recommendedMaxWorkingSetSize;
  CAMLreturn(tuple(3, Val_int(family(d->device)), Val_long(budget), word));
}

/* Memory */

/* [Some (buffer b)], resident from the device's next submission, or [None]
   for nil. */
static value resident(struct device_metal *d, id<MTLBuffer> b) {
  if (b == nil) return Val_none;
  pthread_mutex_lock(&d->set_mutex);
  [d->set addAllocation:b];
  d->changed = 1;
  pthread_mutex_unlock(&d->set_mutex);
  return caml_alloc_some(buffer(b));
}

value caml_device_metal_alloc(value v_d, value v_n) {
  struct device_metal *d = Device_val(v_d);
  return resident(d,
                  [d->device newBufferWithLength:(NSUInteger)Long_val(v_n)
                                         options:MTLResourceStorageModeShared]);
}

/* The pages holding the [v_n] bytes at [v_p], wrapped without a copy. */
value caml_device_metal_map_host(value v_d, value v_p, value v_n) {
  struct device_metal *d = Device_val(v_d);
  uintptr_t page = (uintptr_t)getpagesize(), p = (uintptr_t)Nativeint_val(v_p);
  uintptr_t first = p & ~(page - 1);
  uintptr_t last = (p + (uintptr_t)Long_val(v_n) + page - 1) & ~(page - 1);
  return resident(
      d, [d->device newBufferWithBytesNoCopy:(void *)first
                                      length:last - first
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil]);
}

/* Takes [v_buffer] out of the residency set at once, then releases it. */
value caml_device_metal_free(value v_d, value v_buffer) {
  struct device_metal *d = Device_val(v_d);
  pthread_mutex_lock(&d->set_mutex);
  [d->set removeAllocation:Object_val(v_buffer)];
  [d->set commit];
  d->changed = 0;
  pthread_mutex_unlock(&d->set_mutex);
  [Object_val(v_buffer) release];
  return Val_unit;
}

value caml_device_metal_release(value v_object) {
  [Object_val(v_object) release];
  return Val_unit;
}

/* Images */

/* The metallib [v_b] with a pipeline for each of its functions:
   [("", names, pipelines)], or [(why, [||], [||])]. */
value caml_device_metal_image(value v_d, value v_b) {
  CAMLparam2(v_d, v_b);
  CAMLlocal4(names, pipelines, why, v);
  id<MTLDevice> device = Device_val(v_d)->device;
  dispatch_data_t data =
      dispatch_data_create(String_val(v_b), caml_string_length(v_b), NULL,
                           DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  char text[512] = "";
  int oom;
  @autoreleasepool {
    caml_enter_blocking_section_no_pending();
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
    NSArray<NSString *> *fs = library ? library.functionNames : @[];
    id *ps = calloc(fs.count + 1, sizeof(id));
    if (ps == NULL) fs = @[];
    oom = ps == NULL;
    if (library == nil)
      snprintf(text, sizeof text, "loading the image: %s",
               error.localizedDescription.UTF8String);
    for (NSUInteger i = 0; i < fs.count && text[0] == '\0'; i++) {
      MTLComputePipelineDescriptor *desc =
          [[MTLComputePipelineDescriptor alloc] init];
      desc.computeFunction = [[library newFunctionWithName:fs[i]] autorelease];
      desc.supportIndirectCommandBuffers = YES;
      ps[i] = [device newComputePipelineStateWithDescriptor:desc
                                                    options:0
                                                 reflection:nil
                                                      error:&error];
      if (ps[i] == nil)
        snprintf(text, sizeof text, "building the pipeline of \"%s\": %s",
                 fs[i].UTF8String, error.localizedDescription.UTF8String);
      [desc release];
    }
    [library release];
    caml_leave_blocking_section();
    mlsize_t n = text[0] ? 0 : fs.count;
    names = caml_alloc_tuple(n);
    pipelines = caml_alloc_tuple(n);
    for (mlsize_t i = 0; i < fs.count; i++) {
      if (i >= n) {
        [ps[i] release];
        continue;
      }
      v = caml_copy_string(fs[i].UTF8String);
      Store_field(names, i, v);
      v = object(ps[i]);
      Store_field(pipelines, i, v);
    }
    free(ps);
  }
  dispatch_release(data);
  if (oom) caml_raise_out_of_memory();
  why = caml_copy_string(text);
  CAMLreturn(tuple(3, why, names, pipelines));
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

/* Raises [Invalid_argument] for what [icb] cannot record, and is why Metal
   cannot record it, or "". */
static void check(struct device_metal *d, id<MTLBuffer> args, value v_pipelines,
                  value v_sizes, char *why, size_t n) {
  char m[160];
  if (args.device != d->device)
    caml_invalid_argument(
        "Device_metal_abi.icb: the argument buffer is another GPU's");
  for (mlsize_t i = 0; i < Wosize_val(v_pipelines); i++) {
    id<MTLComputePipelineState> p = (id)Long_val(Field(v_pipelines, i));
    intnat offset = Long_val(Field(v_sizes, 7 * i)), threads = 1;
    for (int k = 4; k < 7; k++) threads *= Long_val(Field(v_sizes, 7 * i + k));
    if (p.device != d->device)
      snprintf(m, sizeof m, "dispatch %d's pipeline is another GPU's", (int)i);
    else if (offset >= (intnat)args.length)
      snprintf(m, sizeof m,
               "dispatch %d's offset %ld lies outside the buffer's %lu bytes",
               (int)i, (long)offset, (unsigned long)args.length);
    else {
      if (why[0] == '\0' && threads > (intnat)p.maxTotalThreadsPerThreadgroup)
        snprintf(
            why, n,
            "dispatch %d asks for %ld threads per threadgroup, expected at "
            "most %lu",
            (int)i, (long)threads,
            (unsigned long)p.maxTotalThreadsPerThreadgroup);
      continue;
    }
    char full[200];
    snprintf(full, sizeof full, "Device_metal_abi.icb: %s", m);
    caml_invalid_argument(full);
  }
}

/* An indirect command buffer of one dispatch per pipeline of [v_pipelines],
   each after the one before, with [v_buffer] as kernel buffer 0; [v_sizes]
   gives each its offset, threadgroups per grid and threads per threadgroup,
   seven words. The result is [("", objects)], the objects being the
   release, the buffer, then each command, or [(why, [||])]. */
value caml_device_metal_icb(value v_d, value v_buffer, value v_pipelines,
                            value v_sizes) {
  CAMLparam4(v_d, v_buffer, v_pipelines, v_sizes);
  CAMLlocal3(objects, why, v);
  struct device_metal *d = Device_val(v_d);
  id<MTLBuffer> args = Object_val(v_buffer);
  int n = (int)Wosize_val(v_pipelines);
  char text[160] = "";
  check(d, args, v_pipelines, v_sizes, text, sizeof text);
  struct icb *b = NULL;
  if (text[0] == '\0') {
    b = calloc(1, sizeof *b + 2 * n * sizeof(id));
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
  }
  if (b == NULL || b->icb == nil) {
    free(b);
    if (text[0] == '\0')
      snprintf(text, sizeof text, "Metal made no indirect command buffer");
    why = caml_copy_string(text);
    CAMLreturn(tuple(2, why, Atom(0), Val_unit));
  }
  b->n = n;
  objects = caml_alloc_tuple(2 + (mlsize_t)n);
  v = caml_copy_nativeint((intnat)b);
  Store_field(objects, 0, v);
  v = object(b->icb);
  Store_field(objects, 1, v);
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
    v = object(c);
    Store_field(objects, 2 + i, v);
  }
  why = caml_copy_string("");
  CAMLreturn(tuple(2, why, objects, Val_unit));
}

value caml_device_metal_icb_release(value v_icb) {
  free_icb((struct icb *)Nativeint_val(v_icb));
  return Val_unit;
}

/* The timeline */

intnat caml_device_metal_signaled(intnat d) {
  return (intnat)__atomic_load_n(((struct device_metal *)d)->ring.word,
                                 __ATOMIC_ACQUIRE);
}

intnat caml_device_metal_last(intnat d) {
  return (intnat)((struct device_metal *)d)->last;
}

/* 0 once the word differs from [v_seen] or [v_ms] passed, 1 if a
   submission failed. */
value caml_device_metal_sleep(value v_d, value v_seen, value v_ms) {
  struct device_metal *d = Device_val(v_d);
  uint64_t seen = (uint64_t)Long_val(v_seen);
  int ms = Int_val(v_ms);
  caml_enter_blocking_section_no_pending();
  const char *why = device_metal_ring_sleep(&d->ring, seen, ms);
  caml_leave_blocking_section();
  return Val_int(why != NULL);
}

/* The device's first failure. */
value caml_device_metal_failure(value v_d) {
  const char *why = device_metal_ring_failure(&Device_val(v_d)->ring);
  return caml_copy_string(why ? why : "");
}

/* Stops the device. Once its ring is empty no handler runs, so the queue
   and fence go; the residency set stays for the frees that may follow. */
value caml_device_metal_stop(value v_d) {
  struct device_metal *d = Device_val(v_d);
  int idle = device_metal_ring_stop(&d->ring, d->last);
  if (idle) {
    [d->queue release];
    [d->fence release];
    d->queue = nil;
    d->fence = nil;
  }
  return Val_bool(idle);
}

static int (*const split)(void *, uint64_t *, uint64_t *) = device_metal_split;

#else

value caml_device_metal_count(value unit) {
  (void)unit;
  return Val_int(0);
}

value caml_device_metal_open(value unit) {
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

NO_METAL1(caml_device_metal_facts)
NO_METAL2(caml_device_metal_alloc)
NO_METAL3(caml_device_metal_map_host)
NO_METAL2(caml_device_metal_free)
NO_METAL1(caml_device_metal_release)
NO_METAL2(caml_device_metal_image)
NO_METAL1(caml_device_metal_icb_release)
NO_METAL3(caml_device_metal_sleep)
NO_METAL1(caml_device_metal_failure)
NO_METAL1(caml_device_metal_stop)

value caml_device_metal_icb(value a, value b, value c, value d) {
  (void)a, (void)b, (void)c, (void)d, no_metal();
}

intnat caml_device_metal_signaled(intnat a) { (void)a, no_metal(); }
intnat caml_device_metal_last(intnat a) { (void)a, no_metal(); }

static int (*const split)(void *, uint64_t *, uint64_t *) = NULL;

#endif

value caml_device_metal_signaled_byte(value v_d) {
  return Val_long(caml_device_metal_signaled(Long_val(v_d)));
}

value caml_device_metal_last_byte(value v_d) {
  return Val_long(caml_device_metal_last(Long_val(v_d)));
}

/* The C entries room, submit and split, the first two typed as the edge
   states. */
value caml_device_metal_entries(value unit) {
  CAMLparam1(unit);
  CAMLlocal3(room, submit, v_split);
  nx_room_fn *r = device_metal_room;
  nx_submit_fn *s = device_metal_submit;
  room = caml_copy_nativeint((intnat)r);
  submit = caml_copy_nativeint((intnat)s);
  v_split = caml_copy_nativeint((intnat)split);
  CAMLreturn(tuple(3, room, submit, v_split));
}

/* Device_metal.submit's C side. A part is a record whose second field
   holds ints: nx_part's int fields in the order below, then its [after]
   indices. The parts are copied out of the OCaml heap, then submitted
   without the runtime: NX_OK, or NX_FAILED. */
enum {
  part_queue,
  part_fill,
  part_arg,
  part_ring_units,
  part_segment_bytes,
  part_copy_dst,
  part_copy_dst_offset,
  part_copy_src,
  part_copy_src_offset,
  part_copy_bytes,
  part_after
};

static intnat at(value ints, int f) { return Long_val(Field(ints, f)); }

value caml_device_metal_submit(value v_d, value v_v, value v_parts) {
  CAMLparam3(v_d, v_v, v_parts);
  int n = (int)Wosize_val(v_parts);
  size_t nafter = 0;
  for (int i = 0; i < n; i++)
    nafter += Wosize_val(Field(Field(v_parts, i), 1)) - part_after;
  size_t size = n * sizeof(struct nx_part) + nafter * sizeof(int);
  struct nx_part *p = size == 0 ? NULL : malloc(size);
  if (size != 0 && p == NULL) caml_raise_out_of_memory();
  int *after = p ? (int *)(p + n) : NULL;
  for (int i = 0; i < n; i++) {
    value k = Field(Field(v_parts, i), 1);
    p[i] = (struct nx_part){
        .queue = (int)at(k, part_queue),
        .fill = (int (*)(void *, void *, uint64_t))at(k, part_fill),
        .arg = (void *)at(k, part_arg),
        .ring_units = (size_t)at(k, part_ring_units),
        .segment_bytes = (size_t)at(k, part_segment_bytes),
        .copy_dst = (uint64_t)at(k, part_copy_dst),
        .copy_dst_offset = (uint64_t)at(k, part_copy_dst_offset),
        .copy_src = (uint64_t)at(k, part_copy_src),
        .copy_src_offset = (uint64_t)at(k, part_copy_src_offset),
        .copy_bytes = (uint64_t)at(k, part_copy_bytes),
        .after = after,
        .nafter = (int)(Wosize_val(k) - part_after)};
    for (int j = 0; j < p[i].nafter; j++) *after++ = (int)at(k, part_after + j);
  }
  void *self = (void *)Long_val(v_d);
  uint64_t v = (uint64_t)Long_val(v_v);
  const char *why = NULL;
  caml_enter_blocking_section_no_pending();
  int rc = device_metal_submit(self, v, NULL, 0, p, n, NULL, 0, &why);
  free(p);
  caml_leave_blocking_section();
  CAMLreturn(Val_int(rc));
}
