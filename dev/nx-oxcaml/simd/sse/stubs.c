/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*/

#include <assert.h>

#define BUILTIN(name) void name() { assert(!"Didn't use [@@builtin] intrinsic."); }

/* Prefetch (no-op in bytecode; native compiler inlines the instruction) */
void caml_prefetch_ignore() { }
BUILTIN(caml_prefetch_read_high_val_offset_untagged)

/* Shared */
BUILTIN(caml_sse2_unreachable)
BUILTIN(caml_vec128_cast)

/* Int64x2 */
BUILTIN(caml_sse2_int64x2_add)
BUILTIN(caml_sse2_int64x2_sub)
BUILTIN(caml_sse2_int64x2_neg)
BUILTIN(caml_sse2_vec128_and)
BUILTIN(caml_sse2_vec128_or)
BUILTIN(caml_sse2_vec128_xor)
BUILTIN(caml_sse2_vec128_not)
BUILTIN(caml_sse42_int64x2_cmpgt)
BUILTIN(caml_int64x2_const1)
BUILTIN(caml_int64x2_low_of_int64)
BUILTIN(caml_int64x2_low_to_int64)
BUILTIN(caml_sse2_int64x2_dup)
BUILTIN(caml_simd_vec128_low_64_to_high_64)
BUILTIN(caml_simd_vec128_high_64_to_low_64)

/* Int32x4 */
BUILTIN(caml_sse2_int32x4_add)
BUILTIN(caml_sse2_int32x4_sub)
BUILTIN(caml_sse2_int32x4_neg)
BUILTIN(caml_ssse3_int32x4_abs)
BUILTIN(caml_sse41_int32x4_min)
BUILTIN(caml_sse41_int32x4_max)
BUILTIN(caml_int32x4_const1)
BUILTIN(caml_int32x4_low_of_int32)
BUILTIN(caml_int32x4_low_to_int32)
BUILTIN(caml_sse2_int32x4_shuffle_0000)
BUILTIN(caml_simd_vec128_interleave_low_32)
BUILTIN(caml_simd_vec128_interleave_low_64)
BUILTIN(caml_sse2_int32x4_dup_lane)

/* Float64x2 */
BUILTIN(caml_sse2_float64x2_add)
BUILTIN(caml_sse2_float64x2_sub)
BUILTIN(caml_sse2_float64x2_mul)
BUILTIN(caml_sse2_float64x2_div)
BUILTIN(caml_sse2_float64x2_sqrt)
BUILTIN(caml_fma_float64x2_fmadd)
BUILTIN(caml_sse3_float64x2_hadd)
BUILTIN(caml_sse2_float64x2_min)
BUILTIN(caml_sse2_float64x2_max)
BUILTIN(caml_float64x2_const1)
BUILTIN(caml_float64x2_low_of_float)
BUILTIN(caml_float64x2_low_to_float)

/* Float32x4 */
BUILTIN(caml_sse_float32x4_add)
BUILTIN(caml_sse_float32x4_sub)
BUILTIN(caml_sse_float32x4_mul)
BUILTIN(caml_sse_float32x4_div)
BUILTIN(caml_sse_float32x4_sqrt)
BUILTIN(caml_fma_float32x4_fmadd)
BUILTIN(caml_sse3_float32x4_hadd)
BUILTIN(caml_sse_float32x4_min)
BUILTIN(caml_sse_float32x4_max)
BUILTIN(caml_float32x4_const1)
BUILTIN(caml_float32x4_low_of_float32)
BUILTIN(caml_float32x4_low_to_float32)
