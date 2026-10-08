/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The kinds' loops per target, shared by the support stubs and the bench. */

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
extern const nx_kinds_loops nx_kinds_loops_base;
extern const nx_kinds_loops *const nx_kinds_loops_v3;

/* The loops of the target nx.cpu would pick on this CPU. */
const nx_kinds_loops *nx_kinds_loops_best(void);

const nx_real_loop *nx_kinds_real_loop(const nx_kinds_loops *l,
                                       const char *name);
const nx_int_loop *nx_kinds_int_loop(const nx_kinds_loops *l,
                                     const char *name);

#endif /* NX_KINDS_SUPPORT_H */
