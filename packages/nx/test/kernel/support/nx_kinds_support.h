/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The kinds' loops, which the support stubs call. */

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

/* Every kind's loop, as nx.cpu's rows compute it on this CPU. */
extern const nx_kinds_loops nx_kinds_support_loops;

#endif /* NX_KINDS_SUPPORT_H */
