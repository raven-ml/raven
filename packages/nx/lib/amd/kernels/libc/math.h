/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The C library the kernels see, which compile with no system header: what
   nx_dtype.h and nx_kinds.h use of math.h, as the compiler's builtins,
   which compile to instructions or to inline code, never to a call. */

#ifndef NX_AMD_MATH_H
#define NX_AMD_MATH_H

#define NAN __builtin_nanf("")
#define INFINITY __builtin_inff()
#define fabs(x) __builtin_fabs(x)
#define fma(a, b, c) __builtin_fma(a, b, c)
#define fmaf(a, b, c) __builtin_fmaf(a, b, c)
#define fmod(a, b) __builtin_fmod(a, b)
#define sqrt(x) __builtin_sqrt(x)
#define sqrtf(x) __builtin_sqrtf(x)

#endif
