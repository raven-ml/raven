/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Fills, as compiled code would write them, and probes of Metal objects
   and host memory. A fill's argument is C memory that its custom block
   frees. Objective-C on macOS; elsewhere no device opens and nothing here
   is called. Every stub holds the runtime: none blocks. */

#define _GNU_SOURCE

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#define Ptr_val(v) ((void *)Nativeint_val(v))
#define Addr_val(v) ((void *)Long_val(v))

/* Host memory */

value device_metal_test_get64(value v_p, value v_i) {
  return caml_copy_int64(((int64_t *)Addr_val(v_p))[Long_val(v_i)]);
}

value device_metal_test_set64(value v_p, value v_i, value v_x) {
  ((int64_t *)Addr_val(v_p))[Long_val(v_i)] = Int64_val(v_x);
  return Val_unit;
}

value device_metal_test_get8(value v_p, value v_i) {
  return Val_int(((uint8_t *)Addr_val(v_p))[Long_val(v_i)]);
}

value device_metal_test_set8(value v_p, value v_i, value v_x) {
  ((uint8_t *)Addr_val(v_p))[Long_val(v_i)] = (uint8_t)Int_val(v_x);
  return Val_unit;
}

value device_metal_test_get32(value v_p, value v_i) {
  return Val_long(((uint32_t *)Addr_val(v_p))[Long_val(v_i)]);
}

value device_metal_test_set32(value v_p, value v_i, value v_x) {
  ((uint32_t *)Addr_val(v_p))[Long_val(v_i)] = (uint32_t)Long_val(v_x);
  return Val_unit;
}

value device_metal_test_macos(value unit) {
  (void)unit;
#ifdef __APPLE__
  return Val_true;
#else
  return Val_false;
#endif
}

/* Fills */

#define Arg_val(v) (*(void **)Data_custom_val(v))

static void finalize_arg(value v) { free(Arg_val(v)); }

static struct custom_operations arg_ops = {
    "device_metal_test.arg",    finalize_arg,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default,
};

/* A block owning [n] zeroed bytes of C memory, the argument of a fill. */
static value arg(size_t n) {
  value v = caml_alloc_custom(&arg_ops, sizeof(void *), 0, 1);
  void *p = calloc(1, n);
  if (p == NULL) caml_raise_out_of_memory();
  Arg_val(v) = p;
  return v;
}

value device_metal_test_arg_address(value v_arg) {
  return caml_copy_nativeint((intnat)Arg_val(v_arg));
}

/* A fill that returns [code]. */
struct failing {
  int code;
};

static int failing(void *queue, void *arg, uint64_t v) {
  (void)queue, (void)v;
  return ((struct failing *)arg)->code;
}

value device_metal_test_failing(value v_code) {
  CAMLparam1(v_code);
  CAMLlocal1(v);
  v = arg(sizeof(struct failing));
  ((struct failing *)Arg_val(v))->code = Int_val(v_code);
  CAMLreturn(v);
}

value device_metal_test_failing_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)failing);
}

#ifdef __APPLE__

#import <Metal/Metal.h>
#import <objc/runtime.h>

#define Object_val(v) ((id)(intptr_t)Nativeint_val(v))

/* [v_n] bytes of host memory at a multiple of the page; never freed. */
value device_metal_test_pages(value v_n) {
  size_t page = (size_t)getpagesize();
  size_t n = ((size_t)Long_val(v_n) + page - 1) / page * page;
  void *p = aligned_alloc(page, n);
  if (p == NULL) caml_raise_out_of_memory();
  memset(p, 0, n);
  return Val_long((intnat)p);
}

/* A dispatch of [pipeline] over [groups] threadgroups of [threads] threads,
   with [buffer] at [offset] as kernel buffer 0, then [splits] splits, each
   writing its times at [times] and dispatching again. */
struct dispatch {
  id<MTLComputePipelineState> pipeline;
  id<MTLBuffer> buffer;
  uint64_t offset, groups, threads, splits;
  int (*split)(void *queue, uint64_t *start, uint64_t *end);
  uint64_t *times;
};

static void encode(id<MTLComputeCommandEncoder> e, struct dispatch *a) {
  [e setComputePipelineState:a->pipeline];
  [e setBuffer:a->buffer offset:a->offset atIndex:0];
  [e dispatchThreadgroups:MTLSizeMake(a->groups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(a->threads, 1, 1)];
}

static int dispatch(void *queue, void *arg, uint64_t v) {
  (void)v;
  struct dispatch *a = arg;
  encode(*(id *)queue, a);
  for (uint64_t k = 0; k < a->splits; k++) {
    int rc = a->split(queue, a->times ? &a->times[2 * k] : NULL,
                      a->times ? &a->times[2 * k + 1] : NULL);
    if (rc != 0) return rc;
    encode(*(id *)queue, a);
  }
  return 0;
}

value device_metal_test_dispatch(value v_pipeline, value v_buffer,
                                 value v_offset, value v_groups,
                                 value v_threads) {
  CAMLparam5(v_pipeline, v_buffer, v_offset, v_groups, v_threads);
  CAMLlocal1(v);
  v = arg(sizeof(struct dispatch));
  struct dispatch *a = Arg_val(v);
  a->pipeline = Object_val(v_pipeline);
  a->buffer = Object_val(v_buffer);
  a->offset = (uint64_t)Long_val(v_offset);
  a->groups = (uint64_t)Long_val(v_groups);
  a->threads = (uint64_t)Long_val(v_threads);
  CAMLreturn(v);
}

/* Makes the dispatch [v_arg] split [v_k] times through [v_split], writing
   the times at [v_times] unless it is 0. */
value device_metal_test_split(value v_arg, value v_split, value v_k,
                              value v_times) {
  struct dispatch *a = Arg_val(v_arg);
  a->split = Ptr_val(v_split);
  a->splits = (uint64_t)Long_val(v_k);
  a->times = Addr_val(v_times);
  return Val_unit;
}

value device_metal_test_dispatch_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)dispatch);
}

/* The [n] commands of an indirect command buffer, after setting each of
   [pipelines] on the encoder with an empty dispatch, which GPUs before the
   Apple9 family need before they run an indirect command buffer's
   pipelines (tinygrad's runtime/graph/metal.py does the same). */
struct execute {
  id<MTLIndirectCommandBuffer> icb;
  uint64_t n, npipelines;
  id<MTLComputePipelineState> pipelines[];
};

static int execute(void *queue, void *arg, uint64_t v) {
  (void)v;
  struct execute *a = arg;
  id<MTLComputeCommandEncoder> e = *(id *)queue;
  for (uint64_t i = 0; i < a->npipelines; i++) {
    [e setComputePipelineState:a->pipelines[i]];
    [e dispatchThreadgroups:MTLSizeMake(0, 0, 0)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  }
  [e executeCommandsInBuffer:a->icb withRange:NSMakeRange(0, a->n)];
  return 0;
}

value device_metal_test_execute(value v_icb, value v_n, value v_pipelines) {
  CAMLparam3(v_icb, v_n, v_pipelines);
  CAMLlocal1(v);
  size_t np = Wosize_val(v_pipelines);
  v = arg(sizeof(struct execute) + np * sizeof(id));
  struct execute *a = Arg_val(v);
  a->icb = Object_val(v_icb);
  a->n = (uint64_t)Long_val(v_n);
  a->npipelines = np;
  for (size_t i = 0; i < np; i++)
    a->pipelines[i] = Object_val(Field(v_pipelines, i));
  CAMLreturn(v);
}

value device_metal_test_execute_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)execute);
}

/* A fill that encodes nothing and stores, at [*slot], a weak reference to
   its encoder, which lives as long as the encoder's command buffer. The
   slot is C memory never freed, since the runtime clears it when the
   encoder goes. */
struct watching {
  id *slot;
};

static int watching(void *queue, void *arg, uint64_t v) {
  (void)v;
  objc_storeWeak(((struct watching *)arg)->slot, *(id *)queue);
  return 0;
}

/* [(arg, slot)]. */
value device_metal_test_watching(value unit) {
  CAMLparam1(unit);
  CAMLlocal3(v, slot, r);
  id *p = calloc(1, sizeof(id));
  if (p == NULL) caml_raise_out_of_memory();
  v = arg(sizeof(struct watching));
  ((struct watching *)Arg_val(v))->slot = p;
  slot = caml_copy_nativeint((intnat)p);
  r = caml_alloc_tuple(2);
  Store_field(r, 0, v);
  Store_field(r, 1, slot);
  CAMLreturn(r);
}

value device_metal_test_watching_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)watching);
}

/* Sets the sizes of the indirect compute command [v_command]. */
value device_metal_test_resize(value v_command, value v_groups,
                               value v_threads) {
  id<MTLIndirectComputeCommand> c = Object_val(v_command);
  [c concurrentDispatchThreadgroups:MTLSizeMake(Long_val(v_groups), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(Long_val(v_threads), 1, 1)];
  return Val_unit;
}

/* Objects */

/* A weak reference to [v_object], in C memory never freed. */
value device_metal_test_weak(value v_object) {
  id *slot = calloc(1, sizeof(id));
  if (slot == NULL) caml_raise_out_of_memory();
  objc_storeWeak(slot, Object_val(v_object));
  return caml_copy_nativeint((intnat)slot);
}

value device_metal_test_alive(value v_weak) {
  BOOL alive;
  @autoreleasepool {
    alive = objc_loadWeak(Ptr_val(v_weak)) != nil;
  }
  return Val_bool(alive);
}

/* The host clock of Metal's times, in nanoseconds. */
value device_metal_test_uptime(value unit) {
  (void)unit;
  return Val_long((intnat)clock_gettime_nsec_np(CLOCK_UPTIME_RAW));
}

#else

CAMLnoret static void no_metal(void) {
  caml_invalid_argument("Device_metal_support: Metal exists on macOS only");
}

value device_metal_test_pages(value a) { (void)a, no_metal(); }
value device_metal_test_dispatch(value a, value b, value c, value d, value e) {
  (void)a, (void)b, (void)c, (void)d, (void)e, no_metal();
}
value device_metal_test_split(value a, value b, value c, value d) {
  (void)a, (void)b, (void)c, (void)d, no_metal();
}
value device_metal_test_dispatch_fill(value a) { (void)a, no_metal(); }
value device_metal_test_execute(value a, value b, value c) {
  (void)a, (void)b, (void)c, no_metal();
}
value device_metal_test_execute_fill(value a) { (void)a, no_metal(); }
value device_metal_test_watching(value a) { (void)a, no_metal(); }
value device_metal_test_watching_fill(value a) { (void)a, no_metal(); }
value device_metal_test_resize(value a, value b, value c) {
  (void)a, (void)b, (void)c, no_metal();
}
value device_metal_test_weak(value a) { (void)a, no_metal(); }
value device_metal_test_alive(value a) { (void)a, no_metal(); }
value device_metal_test_uptime(value a) { (void)a, no_metal(); }

#endif
