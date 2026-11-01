/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The C library the kernels see, which compile with no system header: what
   they and nx_dtype.h use of math.h, as the compiler's builtins. */

#ifndef NX_AMD_MATH_H
#define NX_AMD_MATH_H

#define NAN __builtin_nanf("")
#define INFINITY __builtin_inff()
#define fabs(x) __builtin_fabs(x)
#define ldexp(x, e) __builtin_ldexp(x, e)
#define sqrt(x) __builtin_sqrt(x)
#define sqrtf(x) __builtin_sqrtf(x)

#endif
