/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Runs: the codecs of nx_dtype.h over [n] contiguous elements, for
   convert.c's table. Every scalar codec selects rather than branches, or
   branches only where gcc and clang select, as bfloat16's encoder does, so
   a loop of it vectorises at the width of the target the including code
   compiles for, under -fno-trapping-math, and is a run. The runs below are
   named because they are faster than such a loop: the hardware converts a
   vector at a time with one instruction where it can, bit operations on
   vectors encode and decode bfloat16 and e4m3fn in fewer instructions
   than a compiler finds, and a 64-bit integer converts with the scalar
   instruction, which no vector form matches. A run's tail is the scalar
   codec. */

#ifndef NX_CPU_RUNS_H
#define NX_CPU_RUNS_H

#include <stddef.h>
#include <stdint.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__AVX__)
#include <immintrin.h>
#endif

#include "nx_dtype.h"

/* float16 from and to binary32. On arm64 a loop of FCVT vectorises. On
   x86-64 code compiled for F16C, VCVTPS2PH and VCVTPH2PS convert eight
   elements at a time with the bits the scalar codecs give, NaN payloads
   included. */

static inline void nx_float_to_f16_run(const float *src, uint16_t *dst,
                                       size_t n) {
  size_t i = 0;
#if defined(__F16C__) && defined(__AVX__)
  for (; i + 8 <= n; i += 8)
    _mm_storeu_si128((__m128i *)(dst + i),
                     _mm256_cvtps_ph(_mm256_loadu_ps(src + i),
                                     _MM_FROUND_TO_NEAREST_INT));
#endif
  for (; i < n; i++) dst[i] = nx_float_to_f16(src[i]);
}

static inline void nx_f16_to_float_run(const uint16_t *src, float *dst,
                                       size_t n) {
  size_t i = 0;
#if defined(__F16C__) && defined(__AVX__)
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(_mm_loadu_si128(
                                  (const __m128i *)(src + i))));
#endif
  for (; i < n; i++) dst[i] = nx_f16_to_float(src[i]);
}

/* binary32 to int32, int8 and uint8, as nx_float_to_int stores them. On
   arm64 FCVTZS and FCVTZU truncate, saturate and send NaN to 0, which is
   that rule, and SQXTN and UQXTN narrow with saturation. On x86-64 code
   compiled for AVX2, CVTTPS2DQ gives 0x80000000 past int32's range and for
   NaN: the run selects the bound and 0, or converts values already cleared
   of NaN and clamped. */

static inline void nx_float_to_i32_run(const float *src, int32_t *dst,
                                       size_t n) {
  size_t i = 0;
#if defined(__aarch64__)
  for (; i + 16 <= n; i += 16) {
    float32x4x4_t x = vld1q_f32_x4(src + i);
    int32x4x4_t y = {{vcvtq_s32_f32(x.val[0]), vcvtq_s32_f32(x.val[1]),
                      vcvtq_s32_f32(x.val[2]), vcvtq_s32_f32(x.val[3])}};
    vst1q_s32_x4(dst + i, y);
  }
#elif defined(__AVX2__)
  const __m256 top = _mm256_set1_ps(0x1p31f);
  for (; i + 8 <= n; i += 8) {
    __m256 x = _mm256_loadu_ps(src + i);
    __m256i t = _mm256_cvttps_epi32(x);
    /* 0x80000000 flipped is 0x7FFFFFFF. */
    __m256i over = _mm256_castps_si256(_mm256_cmp_ps(x, top, _CMP_GE_OQ));
    __m256i nan = _mm256_castps_si256(_mm256_cmp_ps(x, x, _CMP_UNORD_Q));
    t = _mm256_andnot_si256(nan, _mm256_xor_si256(t, over));
    _mm256_storeu_si256((__m256i *)(dst + i), t);
  }
#endif
  for (; i < n; i++)
    dst[i] = (int32_t)nx_float_to_int(src[i], INT32_MIN, INT32_MAX);
}

#if defined(__AVX2__) && !defined(__aarch64__)
/* The eight binary32 values from [src] truncated to int32 lanes, NaN as 0
   and those past [hi] as [hi], the largest of an integer dtype of at most
   16 bits; AVX2. Packs that saturate then give nx_float_to_int's values
   for the dtype's range: a value below it truncates below it, or to
   INT32_MIN past int32's. */
static inline __m256i nx_float_to_int_x8(const float *src, float hi) {
  __m256 x = _mm256_loadu_ps(src);
  x = _mm256_and_ps(x, _mm256_cmp_ps(x, x, _CMP_ORD_Q));
  return _mm256_cvttps_epi32(_mm256_min_ps(x, _mm256_set1_ps(hi)));
}

/* Four vectors of int32 lanes a, b, c and d packed to bytes, first in
   pairs, hold their groups of four lanes as a0 b0 c0 d0 a1 b1 c1 d1, since
   a pack works within 128-bit halves: a permute of 32-bit lanes by
   NX_PACKED_ORDER puts them back in sequence. */
#define NX_PACKED_ORDER _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)
#endif

static inline void nx_float_to_i8_run(const float *src, int8_t *dst,
                                      size_t n) {
  size_t i = 0;
#if defined(__aarch64__)
  for (; i + 16 <= n; i += 16) {
    float32x4x4_t x = vld1q_f32_x4(src + i);
    int16x8_t lo = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(x.val[0])),
                                vqmovn_s32(vcvtq_s32_f32(x.val[1])));
    int16x8_t hi = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(x.val[2])),
                                vqmovn_s32(vcvtq_s32_f32(x.val[3])));
    vst1q_s8(dst + i, vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi)));
  }
#elif defined(__AVX2__)
  for (; i + 32 <= n; i += 32) {
    __m256i a = nx_float_to_int_x8(src + i, 127.0f);
    __m256i b = nx_float_to_int_x8(src + i + 8, 127.0f);
    __m256i c = nx_float_to_int_x8(src + i + 16, 127.0f);
    __m256i d = nx_float_to_int_x8(src + i + 24, 127.0f);
    __m256i y = _mm256_packs_epi16(_mm256_packs_epi32(a, b),
                                   _mm256_packs_epi32(c, d));
    _mm256_storeu_si256((__m256i *)(dst + i),
                        _mm256_permutevar8x32_epi32(y, NX_PACKED_ORDER));
  }
  for (; i + 8 <= n; i += 8) {
    __m256i t = nx_float_to_int_x8(src + i, 127.0f);
    __m128i h = _mm_packs_epi32(_mm256_castsi256_si128(t),
                                _mm256_extracti128_si256(t, 1));
    _mm_storel_epi64((__m128i *)(dst + i), _mm_packs_epi16(h, h));
  }
#endif
  for (; i < n; i++)
    dst[i] = (int8_t)nx_float_to_int(src[i], INT8_MIN, INT8_MAX);
}

static inline void nx_float_to_u8_run(const float *src, uint8_t *dst,
                                      size_t n) {
  size_t i = 0;
#if defined(__aarch64__)
  for (; i + 16 <= n; i += 16) {
    float32x4x4_t x = vld1q_f32_x4(src + i);
    uint16x8_t lo = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(x.val[0])),
                                 vqmovn_u32(vcvtq_u32_f32(x.val[1])));
    uint16x8_t hi = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(x.val[2])),
                                 vqmovn_u32(vcvtq_u32_f32(x.val[3])));
    vst1q_u8(dst + i, vcombine_u8(vqmovn_u16(lo), vqmovn_u16(hi)));
  }
#elif defined(__AVX2__)
  for (; i + 32 <= n; i += 32) {
    __m256i a = nx_float_to_int_x8(src + i, 255.0f);
    __m256i b = nx_float_to_int_x8(src + i + 8, 255.0f);
    __m256i c = nx_float_to_int_x8(src + i + 16, 255.0f);
    __m256i d = nx_float_to_int_x8(src + i + 24, 255.0f);
    __m256i y = _mm256_packus_epi16(_mm256_packs_epi32(a, b),
                                    _mm256_packs_epi32(c, d));
    _mm256_storeu_si256((__m256i *)(dst + i),
                        _mm256_permutevar8x32_epi32(y, NX_PACKED_ORDER));
  }
  for (; i + 8 <= n; i += 8) {
    __m256i t = nx_float_to_int_x8(src + i, 255.0f);
    __m128i h = _mm_packs_epi32(_mm256_castsi256_si128(t),
                                _mm256_extracti128_si256(t, 1));
    _mm_storel_epi64((__m128i *)(dst + i), _mm_packus_epi16(h, h));
  }
#endif
  for (; i < n; i++) dst[i] = (uint8_t)nx_float_to_int(src[i], 0, UINT8_MAX);
}

/* bfloat16 to binary32, sixteen at a time with AVX2: the quiet bit set on a
   NaN as the scalar decoder does, in 16-bit lanes (a magnitude past the
   infinity's makes 0x7F80 minus it wrap, and its top bit, shifted to bit
   6, is the quiet bit), then each code moved to the top half of a 32-bit
   lane by interleaving with zeros, the 64-bit groups first permuted so
   that the interleave keeps their order. On fudge it takes 0.033 ns an
   element on one core against the loop's 0.049 to 0.061. */
static inline void nx_bf16_to_float_run(const uint16_t *src, float *dst,
                                        size_t n) {
  size_t i = 0;
#if defined(__AVX2__)
  const __m256i zero = _mm256_setzero_si256();
  const __m256i inf = _mm256_set1_epi16(0x7F80);
  const __m256i mag = _mm256_set1_epi16(0x7FFF);
  const __m256i quiet = _mm256_set1_epi16(0x40);
  for (; i + 16 <= n; i += 16) {
    __m256i c = _mm256_loadu_si256((const __m256i *)(src + i));
    __m256i past = _mm256_sub_epi16(inf, _mm256_and_si256(c, mag));
    __m256i q = _mm256_or_si256(
        c, _mm256_and_si256(_mm256_srli_epi16(past, 9), quiet));
    q = _mm256_permute4x64_epi64(q, 0xD8);
    _mm256_storeu_si256((__m256i *)(dst + i), _mm256_unpacklo_epi16(zero, q));
    _mm256_storeu_si256((__m256i *)(dst + i + 8),
                        _mm256_unpackhi_epi16(zero, q));
  }
#endif
  for (; i < n; i++) dst[i] = nx_bf16_to_float(src[i]);
}

/* binary32 to bfloat16, rounded to nearest even, eight at a time: a
   float's high half, plus one where its low half passes 0x8000, or reaches
   it with the high half odd; a NaN's high half with the quiet bit. arm64
   splits the halves of eight floats into two vectors of 16-bit lanes; AVX2
   rounds in 32-bit lanes and packs. */
static inline void nx_float_to_bf16_run(const float *src, uint16_t *dst,
                                        size_t n) {
  size_t i = 0;
#if defined(__aarch64__)
  const uint16x8_t one = vdupq_n_u16(1), tie = vdupq_n_u16(0x8000);
  for (; i + 8 <= n; i += 8) {
    float32x4_t a = vld1q_f32(src + i), b = vld1q_f32(src + i + 4);
    uint16x8_t wa = vreinterpretq_u16_f32(a), wb = vreinterpretq_u16_f32(b);
    uint16x8_t hi = vuzp2q_u16(wa, wb), lo = vuzp1q_u16(wa, wb);
    uint16x8_t up = vcgtq_u16(lo, vsubq_u16(tie, vandq_u16(hi, one)));
    uint16x8_t ok = vuzp1q_u16(vreinterpretq_u16_u32(vceqq_f32(a, a)),
                               vreinterpretq_u16_u32(vceqq_f32(b, b)));
    vst1q_u16(dst + i, vbslq_u16(ok, vsubq_u16(hi, up),
                                 vorrq_u16(hi, vdupq_n_u16(0x40))));
  }
#elif defined(__AVX2__)
  for (; i + 16 <= n; i += 16) {
    __m256i r[2];
    for (int k = 0; k < 2; k++) {
      __m256 x = _mm256_loadu_ps(src + i + 8 * k);
      __m256i w = _mm256_castps_si256(x), hi = _mm256_srli_epi32(w, 16);
      __m256i lsb = _mm256_and_si256(hi, _mm256_set1_epi32(1));
      __m256i up = _mm256_srli_epi32(
          _mm256_add_epi32(_mm256_add_epi32(w, _mm256_set1_epi32(0x7FFF)), lsb),
          16);
      __m256i nan = _mm256_castps_si256(_mm256_cmp_ps(x, x, _CMP_UNORD_Q));
      r[k] = _mm256_blendv_epi8(
          up, _mm256_or_si256(hi, _mm256_set1_epi32(0x40)), nan);
    }
    /* The pack interleaves 128-bit lanes: the permute restores order. */
    _mm256_storeu_si256(
        (__m256i *)(dst + i),
        _mm256_permute4x64_epi64(_mm256_packus_epi32(r[0], r[1]), 0xD8));
  }
#endif
  for (; i < n; i++) dst[i] = nx_float_to_bf16(src[i]);
}

/* binary32 to e4m3fn, sixteen at a time on arm64: nx_mini_round's two
   cases in 32-bit lanes, as the scalar encoder computes them, then the
   lanes narrowed with saturation to bytes, where the largest finite code,
   NaN and the sign apply once for sixteen elements. 60 vector operations
   for sixteen against the loop's 71. With AVX2, 32 at a time: the code,
   NaN's and the sign are set in 32-bit lanes, which one chain of packs
   then narrows, as nx_float_to_i8_run's does; 0.13 ns an element on a
   fudge core against the loop's 0.59. */

/* nx_mini_round(f, 3, 7)'s constants: the binary32 whose addition rounds a
   subnormal code's bits into place, the bias that moves a normal's
   exponent and adds half a unit less one, and the bits of the least
   normal. */
#define NX_E4M3FN_MAGIC ((uint32_t)(127 + 24 - 7 - 3) << 23)
#define NX_E4M3FN_BIAS (0u - ((uint32_t)(127 - 7) << 23) + (1u << 19) - 1)
#define NX_E4M3FN_NORMAL ((uint32_t)(128 - 7) << 23)

#if defined(__aarch64__)
/* nx_mini_round(f, 3, 7) of four lanes. */
static inline uint32x4_t nx_e4m3fn_round_x4(float32x4_t f) {
  uint32x4_t u = vreinterpretq_u32_f32(vabsq_f32(f));
  const float32x4_t magic = vreinterpretq_f32_u32(vdupq_n_u32(NX_E4M3FN_MAGIC));
  uint32x4_t sub = vsubq_u32(
      vreinterpretq_u32_f32(vaddq_f32(vreinterpretq_f32_u32(u), magic)),
      vreinterpretq_u32_f32(magic));
  uint32x4_t odd = vandq_u32(vshrq_n_u32(u, 20), vdupq_n_u32(1));
  uint32x4_t bias = vdupq_n_u32(NX_E4M3FN_BIAS);
  uint32x4_t normal = vshrq_n_u32(vaddq_u32(vaddq_u32(u, bias), odd), 20);
  uint32x4_t small = vcltq_u32(u, vdupq_n_u32(NX_E4M3FN_NORMAL));
  return vbslq_u32(small, sub, normal);
}
#elif defined(__AVX2__)
/* The e4m3fn code of eight lanes, its sign included, NaN's 0x7F. */
static inline __m256i nx_e4m3fn_x8(const float *src) {
  __m256i b = _mm256_castps_si256(_mm256_loadu_ps(src));
  __m256i u = _mm256_and_si256(b, _mm256_set1_epi32(0x7FFFFFFF));
  const __m256 magic =
      _mm256_castsi256_ps(_mm256_set1_epi32((int32_t)NX_E4M3FN_MAGIC));
  __m256i sub = _mm256_sub_epi32(
      _mm256_castps_si256(_mm256_add_ps(_mm256_castsi256_ps(u), magic)),
      _mm256_castps_si256(magic));
  __m256i odd =
      _mm256_and_si256(_mm256_srli_epi32(u, 20), _mm256_set1_epi32(1));
  __m256i bias = _mm256_set1_epi32((int32_t)NX_E4M3FN_BIAS);
  __m256i normal =
      _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(u, bias), odd), 20);
  __m256i small =
      _mm256_cmpgt_epi32(_mm256_set1_epi32((int32_t)NX_E4M3FN_NORMAL), u);
  __m256i q = _mm256_min_epu32(_mm256_blendv_epi8(normal, sub, small),
                               _mm256_set1_epi32(0x7E));
  __m256i nan = _mm256_cmpgt_epi32(u, _mm256_set1_epi32(0x7F800000));
  __m256i sign = _mm256_and_si256(_mm256_srli_epi32(b, 24),
                                  _mm256_set1_epi32(0x80));
  return _mm256_or_si256(
      _mm256_or_si256(q, _mm256_and_si256(nan, _mm256_set1_epi32(0x7F))),
      sign);
}
#endif

static inline void nx_float_to_e4m3fn_run(const float *src, uint8_t *dst,
                                          size_t n) {
  size_t i = 0;
#if defined(__aarch64__)
  for (; i + 16 <= n; i += 16) {
    float32x4x4_t f = vld1q_f32_x4(src + i);
    uint16x8_t q01 = vcombine_u16(vqmovn_u32(nx_e4m3fn_round_x4(f.val[0])),
                                  vqmovn_u32(nx_e4m3fn_round_x4(f.val[1])));
    uint16x8_t q23 = vcombine_u16(vqmovn_u32(nx_e4m3fn_round_x4(f.val[2])),
                                  vqmovn_u32(nx_e4m3fn_round_x4(f.val[3])));
    uint8x16_t q = vminq_u8(vcombine_u8(vqmovn_u16(q01), vqmovn_u16(q23)),
                            vdupq_n_u8(0x7E));
    /* Lanes equal to themselves, narrowed: not NaN. */
    uint16x8_t n01 =
        vuzp1q_u16(vreinterpretq_u16_u32(vceqq_f32(f.val[0], f.val[0])),
                   vreinterpretq_u16_u32(vceqq_f32(f.val[1], f.val[1])));
    uint16x8_t n23 =
        vuzp1q_u16(vreinterpretq_u16_u32(vceqq_f32(f.val[2], f.val[2])),
                   vreinterpretq_u16_u32(vceqq_f32(f.val[3], f.val[3])));
    uint8x16_t ok = vuzp1q_u8(vreinterpretq_u8_u16(n01),
                              vreinterpretq_u8_u16(n23));
    /* Each float's top byte, whose top bit is its sign. */
    uint16x8_t h01 = vuzp2q_u16(vreinterpretq_u16_f32(f.val[0]),
                                vreinterpretq_u16_f32(f.val[1]));
    uint16x8_t h23 = vuzp2q_u16(vreinterpretq_u16_f32(f.val[2]),
                                vreinterpretq_u16_f32(f.val[3]));
    uint8x16_t sign = vandq_u8(
        vuzp2q_u8(vreinterpretq_u8_u16(h01), vreinterpretq_u8_u16(h23)),
        vdupq_n_u8(0x80));
    vst1q_u8(dst + i, vorrq_u8(sign, vbslq_u8(ok, q, vdupq_n_u8(0x7F))));
  }
#elif defined(__AVX2__)
  for (; i + 32 <= n; i += 32) {
    __m256i a = _mm256_packs_epi32(nx_e4m3fn_x8(src + i),
                                   nx_e4m3fn_x8(src + i + 8));
    __m256i b = _mm256_packs_epi32(nx_e4m3fn_x8(src + i + 16),
                                   nx_e4m3fn_x8(src + i + 24));
    _mm256_storeu_si256((__m256i *)(dst + i),
                        _mm256_permutevar8x32_epi32(_mm256_packus_epi16(a, b),
                                                    NX_PACKED_ORDER));
  }
#endif
  for (; i < n; i++) dst[i] = nx_float_to_e4m3fn(src[i]);
}

/* e4m3fn to binary32: the magnitude bits placed in binary16's fields read
   as the value times 2^-8, as nx_mini_value does on arm64, widened by the
   hardware eight at a time and scaled; a NaN code becomes binary16's quiet
   NaN of its sign, which widens to the NaN the scalar decoder gives. */
static inline void nx_e4m3fn_to_float_run(const uint8_t *src, float *dst,
                                          size_t n) {
  size_t i = 0;
#if defined(__aarch64__)
  const float32x4_t scale = vdupq_n_f32(256.0f);
  for (; i + 8 <= n; i += 8) {
    uint16x8_t c = vmovl_u8(vld1_u8(src + i));
    /* c << 7 puts the magnitude in place and the sign at bit 14; adding
       bit 14 moves it to bit 15. */
    uint16x8_t s7 = vshlq_n_u16(c, 7);
    uint16x8_t h = vaddq_u16(s7, vandq_u16(s7, vdupq_n_u16(0x4000)));
    uint16x8_t nan = vceqq_u16(vandq_u16(c, vdupq_n_u16(0x7F)),
                               vdupq_n_u16(0x7F));
    h = vbslq_u16(nan, vorrq_u16(vandq_u16(h, vdupq_n_u16(0x8000)),
                                 vdupq_n_u16(0x7E00)),
                  h);
    float16x8_t f = vreinterpretq_f16_u16(h);
    vst1q_f32(dst + i, vmulq_f32(vcvt_f32_f16(vget_low_f16(f)), scale));
    vst1q_f32(dst + i + 4, vmulq_f32(vcvt_high_f32_f16(f), scale));
  }
#elif defined(__F16C__) && defined(__AVX2__)
  const __m256 scale = _mm256_set1_ps(256.0f);
  for (; i + 16 <= n; i += 16) {
    __m256i c =
        _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(src + i)));
    __m256i s7 = _mm256_slli_epi16(c, 7);
    __m256i h =
        _mm256_add_epi16(s7, _mm256_and_si256(s7, _mm256_set1_epi16(0x4000)));
    __m256i nan = _mm256_cmpeq_epi16(
        _mm256_and_si256(c, _mm256_set1_epi16(0x7F)), _mm256_set1_epi16(0x7F));
    h = _mm256_blendv_epi8(
        h,
        _mm256_or_si256(_mm256_and_si256(h, _mm256_set1_epi16((short)0x8000)),
                        _mm256_set1_epi16(0x7E00)),
        nan);
    __m256 lo = _mm256_cvtph_ps(_mm256_castsi256_si128(h));
    __m256 hi = _mm256_cvtph_ps(_mm256_extracti128_si256(h, 1));
    _mm256_storeu_ps(dst + i, _mm256_mul_ps(lo, scale));
    _mm256_storeu_ps(dst + i + 8, _mm256_mul_ps(hi, scale));
  }
#endif
  for (; i < n; i++) dst[i] = nx_e4m3fn_to_float(src[i]);
}

/* 64-bit integers to binary32, rounded once. No vector instruction of
   arm64 or AVX2 converts them, and a vector of rounding to odd costs three
   to four times the scalar conversion, which rounds once. On x86-64 gcc and
   clang compile C's cast to CVTSI2SS (VCVTQQ2PS under AVX-512), which
   rounds once; on arm64 clang vectorises it through float64, rounding
   twice, so the run converts with SCVTF and UCVTF from general registers. */
#if defined(__aarch64__)
/* Four conversions at a time, independent, so that they issue together. */
#define NX_CVT4(op, src, dst, i)                                         \
  do {                                                                   \
    float f0, f1, f2, f3;                                                \
    __asm__(op " %s0, %x4\n\t" op " %s1, %x5\n\t" op " %s2, %x6\n\t" op    \
            " %s3, %x7"                                                  \
            : "=&w"(f0), "=&w"(f1), "=&w"(f2), "=&w"(f3)                 \
            : "r"(src[i]), "r"(src[i + 1]), "r"(src[i + 2]),             \
              "r"(src[i + 3]));                                          \
    dst[i] = f0, dst[i + 1] = f1, dst[i + 2] = f2, dst[i + 3] = f3;      \
  } while (0)
#endif

static inline void nx_i64_to_float_run(const int64_t *src, float *dst,
                                       size_t n) {
  size_t i = 0;
#if defined(__aarch64__)
  for (; i + 4 <= n; i += 4) NX_CVT4("scvtf", src, dst, i);
  for (; i < n; i++) dst[i] = (float)nx_i64_odd(src[i]);
#else
  for (; i < n; i++) dst[i] = (float)src[i];
#endif
}

/* x86-64 has no unsigned conversion before AVX-512, and C's cast branches
   on the top bit. Both cases convert and one is selected: below 2^63 the
   signed conversion, from it the half rounded to odd, converted and
   doubled, which rounds once (0.32 ns an element on kimchi, against 0.37
   for C's cast and 0.50 for the codec). */
static inline void nx_u64_to_float_run(const uint64_t *src, float *dst,
                                       size_t n) {
  size_t i = 0;
#if defined(__aarch64__)
  for (; i + 4 <= n; i += 4) NX_CVT4("ucvtf", src, dst, i);
  for (; i < n; i++) dst[i] = (float)nx_u64_odd(src[i]);
#else
  for (; i < n; i++) {
    uint64_t x = src[i];
    float low = (float)(int64_t)x;
    float high = (float)(int64_t)((x >> 1) | (x & 1));
    dst[i] = (int64_t)x < 0 ? high + high : low;
  }
#endif
}

#if defined(__aarch64__)
#undef NX_CVT4
#endif

/* Doubles to float16. arm64 narrows a double to __fp16 in one rounding:
   clang narrows a vector through FCVTXN (round to odd) then FCVTN, gcc
   converts each element with FCVT from d to h. With F16C and AVX2, a double rounds to odd at binary32's
   precision on its bits, as highway's DemoteTo from float64 to float16
   does (Apache-2.0): the 29 bits binary32 drops are cleared and the
   lowest one kept is set where one of them was. The conversion to
   binary32 is then exact, save below binary32's normal range, where
   float16 rounds to zero either way, and past its largest, where it
   rounds to infinity. 1 Mi doubles take 21 us on fudge's six P-cores
   against 46 through nx_float_odd's form, whose compares and shuffles
   queue on one port. */

#if defined(__aarch64__)
/* The run writes uint16_t memory as __fp16. */
typedef __fp16 __attribute__((may_alias)) nx_fp16;
#endif

static inline void nx_double_to_f16_run(const double *src, uint16_t *dst,
                                        size_t n) {
#if defined(__aarch64__)
  nx_fp16 *h = (nx_fp16 *)dst;
  for (size_t i = 0; i < n; i++) h[i] = (__fp16)src[i];
#else
  size_t i = 0;
#if defined(__F16C__) && defined(__AVX2__)
  const __m256i kept = _mm256_set1_epi64x((int64_t)0xFFFFFFFFE0000000u);
  const __m256i dropped = _mm256_set1_epi64x(0x1FFFFFFF);
  const __m256i last = _mm256_set1_epi64x(0x20000000);
  for (; i + 8 <= n; i += 8) {
    __m128 f[2];
    for (int k = 0; k < 2; k++) {
      __m256i x = _mm256_loadu_si256((const __m256i *)(src + i + 4 * k));
      /* Adding the dropped bits' mask carries into the last kept bit
         where one of them is set; the carry flips that bit, so the kept
         bits are or-ed back after: the bit ends set where it was or where
         a carry came. */
      __m256i carry = _mm256_and_si256(_mm256_add_epi64(x, dropped), last);
      __m256i odd = _mm256_or_si256(_mm256_and_si256(x, kept), carry);
      f[k] = _mm256_cvtpd_ps(_mm256_castsi256_pd(odd));
    }
    _mm_storeu_si128((__m128i *)(dst + i),
                     _mm256_cvtps_ph(_mm256_set_m128(f[1], f[0]),
                                     _MM_FROUND_TO_NEAREST_INT));
  }
#endif
  for (; i < n; i++) dst[i] = nx_double_to_f16(src[i]);
#endif
}

#endif /* NX_CPU_RUNS_H */
