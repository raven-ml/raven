/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A fill for a CUDA device: it launches one kernel on the stream the device
   hands it. Its argument holds the address of cuLaunchKernel, which the
   device's capability found in the CUDA library, the kernel's CUfunction,
   the launch's sizes and the kernel's parameters, each a 64-bit word. The
   last parameter is 32 bits: CUDA reads its low half, which comes first in
   the little-endian hosts CUDA runs on. No CUDA header is needed: the
   function's type is written here. */

#define _GNU_SOURCE

#include <stddef.h>
#include <stdint.h>

#include <caml/alloc.h>
#include <caml/mlvalues.h>

typedef int (*launch_fn)(void *f, unsigned grid_x, unsigned grid_y,
                         unsigned grid_z, unsigned block_x, unsigned block_y,
                         unsigned block_z, unsigned shared, void *stream,
                         void **params, void **extra);

struct launch {
  uint64_t launch, function, grid, block;
  uint64_t a, b, out, n;
};

/* Answers cuLaunchKernel's CUresult: 0 on success. */
static int run(void *queue, void *arg, uint64_t v) {
  (void)v;
  struct launch *l = arg;
  void *params[] = {&l->a, &l->b, &l->out, &l->n};
  launch_fn launch = (launch_fn)(uintptr_t)l->launch;
  return launch((void *)(uintptr_t)l->function, (unsigned)l->grid, 1, 1,
                (unsigned)l->block, 1, 1, 0, queue, params, NULL);
}

/* Does not release the runtime: it returns an address. */
value caml_rig_example_run(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&run);
}
