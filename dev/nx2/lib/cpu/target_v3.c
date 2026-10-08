/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The v3 target: x86-64 with AVX2, FMA, F16C and BMI2. Its runs are the
   base target's compiled for these instructions, the headers' inline codecs
   included: the region starts before them. gcc then defines __F16C__ and
   __AVX2__, and nx_array.h's runs use F16C and AVX2; clang defines neither,
   and its v3 runs are the base target's loops compiled for these
   instructions. */

#if defined(__x86_64__)

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma,f16c,bmi2"))), \
                             apply_to = function)
#else
#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c,bmi2")
#endif

#include "cpu.h"

#include "convert.inc"

void nx_cpu_fill_v3(nx_cpu_target *t) { fill(t); }

#if defined(__clang__)
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif

#endif
