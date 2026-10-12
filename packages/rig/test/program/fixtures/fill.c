/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <stdint.h>

/* rig calls a fill as a C function of the process, by the platform's
   convention: on x86_64 Windows, ms_abi (gen.sh defines RIG_WINDOWS for that
   object). A host entry the fill calls stays linked code, by System V. */
#ifdef RIG_WINDOWS
#define PLATFORM __attribute__((ms_abi))
#else
#define PLATFORM
#endif

PLATFORM int fill_empty(void *queue, const void *params, uint64_t point) {
  (void)queue, (void)params, (void)point;
  return 0;
}

/* The run supplies a destination through its ref and a word beside it. */
PLATFORM int fill(void *queue, const uint64_t *params, uint64_t point) {
  (void)queue, (void)point;
  *(uint64_t *)(uintptr_t)params[0] = params[1];
  return 0;
}

/* The first parameter names a container initialized with its destination. */
PLATFORM int fill_indirect(void *queue, const uint64_t *params,
                           uint64_t point) {
  (void)queue, (void)point;
  const uint64_t *container = (const uint64_t *)(uintptr_t)params[0];
  *(uint64_t *)(uintptr_t)container[0] = params[1];
  return 0;
}

/* The first parameter names a container initialized with the address of a
   host function of test/fixtures' affine. Its call adds one to the word at
   the second parameter. */
struct affine {
  int64_t *out;
  const int64_t *in;
  int64_t n, a, c;
};

PLATFORM int fill_call(void *queue, const uint64_t *params, uint64_t point) {
  (void)queue, (void)point;
  const uint64_t *container = (const uint64_t *)(uintptr_t)params[0];
  void (*f)(const struct affine *, int64_t, int64_t) =
      (void (*)(const struct affine *, int64_t, int64_t))(uintptr_t)
          container[0];
  int64_t *w = (int64_t *)(uintptr_t)params[1];
  const struct affine p = {w, w, 1, 1, 1};
  f(&p, 0, 1);
  return 0;
}
