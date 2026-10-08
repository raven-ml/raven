/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The v3 target's loops: x86-64 with AVX2, FMA, F16C and BMI2, the headers
   compiled inside the region as nx.cpu's are. Empty elsewhere. */

#if defined(__x86_64__)

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma,f16c,bmi2"))), \
                             apply_to = function)
#else
#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c,bmi2")
#endif

#include "nx_kinds.h"
#include "nx_kinds_support.h"

#define NX_LOOPS static const nx_kinds_loops nx_kinds_loops_v3_table
#include "nx_kinds_support_loops.inc"

#if defined(__clang__)
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif

const nx_kinds_loops *const nx_kinds_loops_v3 = &nx_kinds_loops_v3_table;

#else

/* ISO C forbids an empty unit. */
typedef int nx_kinds_support_v3_empty;

#endif
