/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The base target: the instructions every host of the architecture has,
   SSE2 on x86-64 and NEON on arm64. */

#include "cpu.h"

#include "convert.inc"

void nx_cpu_fill_base(nx_cpu_target *t) { fill(t); }
