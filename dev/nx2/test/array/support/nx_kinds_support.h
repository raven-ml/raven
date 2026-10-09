/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The kinds' loops per target, which the support stubs call. */

#ifndef NX_KINDS_SUPPORT_H
#define NX_KINDS_SUPPORT_H

#include <stdint.h>

typedef struct {
  const char *name;
  int arity;
  void (*f32)(const float *, const float *, const float *, float *, long);
  void (*f64)(const double *, const double *, const double *, double *, long);
} nx_real_loop;

typedef struct {
  const char *name;
  int arity;
  void (*run)(const uint64_t *, const uint64_t *, const uint64_t *, uint64_t *,
              long);
} nx_int_loop;

#define NX_REAL_LOOPS 40
#define NX_INT_LOOPS 22

typedef struct {
  nx_real_loop real[NX_REAL_LOOPS];
  nx_int_loop ints[NX_INT_LOOPS];
} nx_kinds_loops;

/* The base target's loops, and on x86-64 the v3 target's (AVX2 and FMA),
   NULL elsewhere. */
extern const nx_kinds_loops *const nx_kinds_loops_base;
extern const nx_kinds_loops *const nx_kinds_loops_v3;

#endif /* NX_KINDS_SUPPORT_H */
