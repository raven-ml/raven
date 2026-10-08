/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A fill for a Metal device: it runs the dispatches an indirect command
   buffer recorded, in the compute encoder the device hands it. Its argument
   is the indirect command buffer and its number of dispatches. Objective-C
   on macOS; elsewhere no Metal device opens, and the fill answers a
   failure. */

#define _GNU_SOURCE

#include <stdint.h>

#include <caml/alloc.h>
#include <caml/mlvalues.h>

struct run {
  uint64_t icb, n;
};

#if defined(__APPLE__)

#import <Metal/Metal.h>

static int run(void *queue, void *arg, uint64_t v) {
  (void)v;
  const struct run *a = arg;
  id<MTLComputeCommandEncoder> e = *(id *)queue;
  id<MTLIndirectCommandBuffer> icb = (id)(uintptr_t)a->icb;
  [e executeCommandsInBuffer:icb withRange:NSMakeRange(0, a->n)];
  return 0;
}

#else

static int run(void *queue, void *arg, uint64_t v) {
  (void)queue;
  (void)arg;
  (void)v;
  return 1;
}

#endif

/* Does not release the runtime: it returns an address. */
value caml_rig_example_run(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&run);
}
