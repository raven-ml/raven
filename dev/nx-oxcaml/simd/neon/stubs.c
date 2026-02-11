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

/* Shared */
BUILTIN(caml_vec128_unreachable)
BUILTIN(caml_vec128_cast)

/* Int64x2 */
BUILTIN(caml_neon_int64x2_add)
BUILTIN(caml_neon_int64x2_sub)
BUILTIN(caml_neon_int64x2_neg)
BUILTIN(caml_neon_int64x2_bitwise_and)
BUILTIN(caml_neon_int64x2_bitwise_or)
BUILTIN(caml_neon_int64x2_bitwise_xor)
BUILTIN(caml_neon_int64x2_bitwise_not)
BUILTIN(caml_neon_int64x2_cmpgt)
BUILTIN(caml_int64x2_const1)
BUILTIN(caml_int64x2_low_of_int64)
BUILTIN(caml_int64x2_low_to_int64)
BUILTIN(caml_neon_int64x2_dup)
BUILTIN(caml_simd_vec128_low_64_to_high_64)
BUILTIN(caml_simd_vec128_high_64_to_low_64)

/* Int32x4 */
BUILTIN(caml_neon_int32x4_add)
BUILTIN(caml_neon_int32x4_sub)
BUILTIN(caml_neon_int32x4_neg)
BUILTIN(caml_neon_int32x4_abs)
BUILTIN(caml_neon_int32x4_min)
BUILTIN(caml_neon_int32x4_max)
BUILTIN(caml_neon_int32x4_bitwise_and)
BUILTIN(caml_neon_int32x4_bitwise_or)
BUILTIN(caml_neon_int32x4_bitwise_xor)
BUILTIN(caml_neon_int32x4_bitwise_not)
BUILTIN(caml_int32x4_const1)
BUILTIN(caml_int32x4_low_of_int32)
BUILTIN(caml_int32x4_low_to_int32)
BUILTIN(caml_neon_int32x4_dup)
BUILTIN(caml_simd_vec128_interleave_low_32)
BUILTIN(caml_simd_vec128_interleave_low_64)
BUILTIN(caml_neon_int32x4_dup_lane)

/* Float64x2 */
BUILTIN(caml_neon_float64x2_add)
BUILTIN(caml_neon_float64x2_sub)
BUILTIN(caml_neon_float64x2_mul)
BUILTIN(caml_neon_float64x2_div)
BUILTIN(caml_neon_float64x2_sqrt)
BUILTIN(caml_neon_float64x2_hadd)
BUILTIN(caml_neon_float64x2_min)
BUILTIN(caml_neon_float64x2_max)
BUILTIN(caml_float64x2_const1)
BUILTIN(caml_float64x2_low_of_float)
BUILTIN(caml_float64x2_low_to_float)

/* Float32x4 */
BUILTIN(caml_neon_float32x4_add)
BUILTIN(caml_neon_float32x4_sub)
BUILTIN(caml_neon_float32x4_mul)
BUILTIN(caml_neon_float32x4_div)
BUILTIN(caml_neon_float32x4_sqrt)
BUILTIN(caml_neon_float32x4_hadd)
BUILTIN(caml_neon_float32x4_min)
BUILTIN(caml_neon_float32x4_max)
BUILTIN(caml_float32x4_const1)
BUILTIN(caml_float32x4_low_of_float32)
BUILTIN(caml_float32x4_low_to_float32)

/* The ARM64 OxCaml backend does not support Cprefetch (operation_supported
   returns false), so [@@builtin] is not inlined. Provide a real C stub that
   emits the PRFM instruction via __builtin_prefetch. */

#include <caml/mlvalues.h>

void caml_prefetch_read_high_val_offset_untagged(value v, intnat offset) {
    __builtin_prefetch((char *)v + offset, 0, 3);
}

/* FMA — NEON has hardware FMA (vfmaq_f32/f64), but the OxCaml compiler
   doesn't expose them as [@@builtin] intrinsics. The OCaml externals use
   [@unboxed] parameters, so the native compiler passes raw NEON registers.
   These stubs accept the unboxed types directly. */

#ifdef __ARM_NEON
#include <arm_neon.h>

float32x4_t caml_neon_float32x4_fma(float32x4_t a, float32x4_t b, float32x4_t c) {
    return vfmaq_f32(a, b, c);
}

float64x2_t caml_neon_float64x2_fma(float64x2_t a, float64x2_t b, float64x2_t c) {
    return vfmaq_f64(a, b, c);
}

#endif
