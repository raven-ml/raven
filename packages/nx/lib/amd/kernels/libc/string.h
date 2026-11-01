/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The C library the kernels see: size_t, and memcpy as the compiler's
   builtin. */

#ifndef NX_AMD_STRING_H
#define NX_AMD_STRING_H

typedef __SIZE_TYPE__ size_t;

#define memcpy(d, s, n) __builtin_memcpy(d, s, n)

#endif
