/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Elements as the families store them: a sub-byte element into the
   32-bit word that holds it, and a narrow float's code. */

#ifndef NX_CUDA_WORDS_CUH
#define NX_CUDA_WORDS_CUH

#include <stdint.h>

#include "nx_dtype.h"

/* The aligned 32-bit word holding element [p] of [bits] bits at [base],
   and the element's shift in it. The word may hold bytes outside the
   array: rig orders memory in 4-byte words (Rig.word), so work may write
   each whole word holding a byte of its buffer, and rig orders the word's
   other bytes, its allocation's, with the buffer. */
__device__ __forceinline__ uint32_t *nx_word_at(const void *base, int64_t p,
                                                int bits, unsigned &shift) {
  uint64_t bit = (uint64_t)p * bits;
  uintptr_t a = (uintptr_t)base + (bit >> 3);
  shift = (unsigned)((a & 3) * 8 + (bit & 7));
  return (uint32_t *)(a & ~(uintptr_t)3);
}

/* Stores the low [bits] bits of [v] as element [p] of [bits] bits, 1 or 4,
   at [base]. The element shares its byte with others, which other threads
   store: it changes its own bits of the word, by and/or, which commute
   with other threads' on other bits. */
__device__ __forceinline__ void nx_store_bits(void *base, int64_t p, int bits,
                                              uint32_t v) {
  unsigned shift;
  uint32_t *w = nx_word_at(base, p, bits, shift);
  uint32_t mask = ((1u << bits) - 1) << shift;
  atomicAnd(w, ~mask);
  atomicOr(w, (v << shift) & mask);
}

/* The code of the narrow float [dt] nearest the float32 [f]. */
__device__ uint32_t nx_narrow_code(int dt, float f) {
  switch (dt) {
  case NX_FLOAT16: return nx_float_to_f16(f);
  case NX_BFLOAT16: return nx_float_to_bf16(f);
  case NX_FLOAT8_E4M3FN: return nx_float_to_e4m3fn(f);
  case NX_FLOAT8_E5M2: return nx_float_to_e5m2(f);
  default: return nx_float_to_e2m1fn(f);
  }
}

#endif
