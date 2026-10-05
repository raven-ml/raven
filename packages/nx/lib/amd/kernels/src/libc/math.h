/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The C library the kernels see, which compile with no system header: what
   nx_dtype.h uses of math.h, as the compiler's builtins. */

#ifndef NX_AMD_MATH_H
#define NX_AMD_MATH_H

#define NAN __builtin_nanf("")
#define INFINITY __builtin_inff()
#define isnan(x) __builtin_isnan(x)
#define isinf(x) __builtin_isinf(x)
#define isfinite(x) __builtin_isfinite(x)
#define signbit(x) __builtin_signbit(x)
#define fabs(x) __builtin_fabs(x)
#define ldexpf(x, e) __builtin_ldexpf(x, e)
#define copysignf(x, y) __builtin_copysignf(x, y)

#endif
