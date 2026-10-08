/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The base target's loops, and how a target is picked. */

#include <string.h>

#include "nx_kinds.h"
#include "nx_kinds_support.h"

#define NX_LOOPS const nx_kinds_loops nx_kinds_loops_base
#include "nx_kinds_support_loops.inc"

#if !defined(__x86_64__)
const nx_kinds_loops *const nx_kinds_loops_v3 = NULL;
#endif

const nx_kinds_loops *nx_kinds_loops_best(void) {
#if defined(__x86_64__)
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
    return nx_kinds_loops_v3;
#endif
  return &nx_kinds_loops_base;
}

const nx_real_loop *nx_kinds_real_loop(const nx_kinds_loops *l,
                                       const char *name) {
  for (int i = 0; i < NX_REAL_LOOPS; i++)
    if (strcmp(l->real[i].name, name) == 0) return &l->real[i];
  return NULL;
}

const nx_int_loop *nx_kinds_int_loop(const nx_kinds_loops *l,
                                     const char *name) {
  for (int i = 0; i < NX_INT_LOOPS; i++)
    if (strcmp(l->ints[i].name, name) == 0) return &l->ints[i];
  return NULL;
}
