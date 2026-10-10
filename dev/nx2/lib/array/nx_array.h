/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Arrays from C.

   A host kernel reads arrays only through the door: nx_read checks every
   operand of a call and claims every operand's memory, or refuses and claims
   none, then waits under the claims for the device work on them; the kernel
   runs over the descriptors nx_read filled, and nx_done ends the claims:

     nx_operand in[3] = { { vz, dt, 1 }, { vx, dt, 0 }, { vy, dt, 0 } };
     nx_array a[3];
     nx_loop l;
     int e = nx_read(3, in, a);
     if (e) return Val_int(e);
     if (!(e = nx_coalesce(3, a, &l))) run(&l);
     nx_done(3, a);
     return Val_int(e);

   The kernel answers Val_int of a code, an Nx_array.answer: NX_OK is Done,
   NX_DECLINED a case the kernel does not compute, and every other code a
   refusal, which Nx_array.refused raises.

   A descriptor's element at index (i0, …, ik-1), 0 <= ij < dim[j], lies at
   position offset + Σ ij·dim[rank + j], counted in elements from base, as
   Layout places it (layout.mli). */

#ifndef NX_ARRAY_H
#define NX_ARRAY_H

#include <stdint.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#if (defined(__F16C__) && defined(__AVX__)) || defined(__AVX2__)
#include <immintrin.h>
#endif

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_dtype.h"

#define NX_MAX_RANK 32
#define NX_MAX_OPERANDS 4

/* A layout's flags, computed when it is made. */
enum {
  NX_CONTIGUOUS = 1, /* Layout.is_contiguous */
  NX_DISTINCT = 2,   /* Layout.is_distinct */
  NX_EMPTY = 4       /* no element */
};

/* Codes: the constructors of Nx_array.answer, in its order. */

enum {
  NX_OK,           /* Done */
  NX_DECLINED,     /* the kernel does not compute the case */
  NX_DTYPE,        /* an operand's dtype is not the one named */
  NX_DEAD,         /* an operand's buffer is dead */
  NX_NOT_HOST,     /* the host does not address an operand's memory */
  NX_EXCLUSIVE,    /* an operand's memory is held exclusive */
  NX_READ_ONLY,    /* a written operand's memory is Read */
  NX_NOT_DISTINCT, /* a written operand reaches a position twice */
  NX_OVERLAP,      /* a written operand shares a byte with another written
                      operand, or a read one not identical to it */
  NX_SHAPE,        /* the operands' shapes do not fit the operation */
  NX_ARITY         /* no operand, or more than NX_MAX_OPERANDS, to a loop */
};

/* The door

   From nx_read to nx_done, each operand's memory is claimed and its buffer
   is a local root of the domain: a kernel needs no CAMLparam to keep its
   operands reachable. A descriptor holds no pointer into the OCaml heap, so
   it stays valid when the kernel allocates or releases the domain lock. In
   return the kernel keeps these rules:

   - nx_read may wait for device work, which runs OCaml code: the collector,
     signal handlers, other threads, the device's driver. A kernel that calls
     it is an external that is not [@@noalloc], and keeps as a root
     registered with CAMLparam every OCaml value other than its operands that
     it uses after nx_read. nx_read raises what Rig.Buffer.wait raises,
     holding no claim: the kernel acquires nothing before it that must be
     undone.
   - nx_read and nx_done run on one thread, with the domain lock held.
   - Descriptors stay where nx_read filled them, in the frame of the C
     function that called it, and are passed by pointer, never copied or
     moved, until nx_done.
   - nx_done runs once per successful nx_read, never after a refusal or a
     raise, before that function returns or raises.
   - nx_read and nx_done nest with root frames as CAMLparam and CAMLreturn
     do: a frame registered before nx_read is popped after nx_done, and one
     registered after nx_read is popped before nx_done. A kernel registers
     its roots with CAMLparam before it calls nx_read. nx_done ends the
     process if it finds the descriptors' roots popped.
   - While the domain lock is released, the kernel reads no OCaml value. */

/* An operand of a call: an OCaml array, the dtype the kernel's loads assume,
   and whether the kernel writes it. nx_read reads [array] while it runs;
   the descriptor it fills is what the kernel uses after. */
typedef struct {
  value array;
  int dtype;
  int written;
} nx_operand;

/* Layout.max_numel: a descriptor's extents, the product of its extents
   other than 0, and the end of its span are at most NX_MAX_NUMEL. A product
   of any of its extents, a position times a width of at most 128 bits, and
   a sum of a few of these fit in an int64_t, unchecked. */
#define NX_MAX_NUMEL ((int64_t)1 << 53)

/* An operand, read: its descriptor. [base] is NULL for an operand with no
   element. */
typedef struct {
  uint8_t *base; /* host address of the buffer's first byte */
  int dtype, bits, rank, flags;
  int64_t offset;               /* elements from base */
  int64_t dim[2 * NX_MAX_RANK]; /* rank extents, then rank strides */
  /* Read-only: a read operand identical to a written one, whose claim
     covers it. The kernel writes its memory, so it may read it at an index
     only before it writes that index. */
  int alias;
  /* Private: the claimed buffer, a local root until nx_done, and whether
     nx_read waits for device work on it. */
  value buffer;
  int wait;
  struct caml__roots_block roots;
  struct caml__roots_block **local; /* the domain's local roots */
} nx_array;

/* The dtype of the OCaml array [v], a code. */
int nx_array_dtype(value v);

/* The layout of the array [v], an Nx_array.t, read without a claim: its
   rank, then at [dim] its [rank] extents and its [rank] strides, as
   nx_array's dim holds them, and at [offset] its first element, in
   elements. [dim] holds 2 · NX_MAX_RANK. It reads [v] alone and allocates
   nothing; what it writes is C memory, so it stays valid after [v]
   moves. */
int nx_array_layout(value v, int64_t *dim, int64_t *offset);

/* Reads the [n] operands [in] into [out] and answers NX_OK, or answers why
   it refuses one, claims nothing and leaves [out] unspecified. [n] is any
   number; with no operand it answers NX_OK. Past NX_MAX_OPERANDS it
   allocates, before any claim, and raises Out_of_memory if host memory
   runs out. It checks in this order and answers the first refusal: each
   operand's dtype; then each operand's claim of its memory, for writing if
   written: NX_DEAD if its buffer is dead, NX_EXCLUSIVE if the memory is
   held exclusive, NX_READ_ONLY if a written operand's memory is Read; then,
   per operand, that a written one is NX_DISTINCT and that the host
   addresses one with an element; then that a written operand shares no
   byte with another written operand, nor with a read operand unless the
   two are identical: one width, and every index at one byte (equal extents
   and strides, one first bit). A read operand identical to a written one
   is claimed through it, once. Under the claims it waits for the device
   work each operand's access must follow, as Rig.Buffer.wait does; this
   alone runs OCaml code, and only while such work is unfinished. If the
   wait raises, as Rig.Lost does for a lost device, nx_read releases every
   claim and raises it. */
int nx_read(int n, const nx_operand *in, nx_array *out);

/* Releases the claims of the [n] operands a successful nx_read filled and
   unlinks their roots. */
void nx_done(int n, nx_array *a);

/* Coalescing */

/* A loop over operands of one shape: [rank] axes, at least one, of
   [extent]s; operand k's element at a loop index is at first[k] + Σ
   index·step[k], in elements from its base. A loop with no element has
   extent[0] = 0; one of one element has rank 1 and extent 1. */
typedef struct {
  int rank;
  int64_t extent[NX_MAX_RANK];
  int64_t step[NX_MAX_OPERANDS][NX_MAX_RANK]; /* elements */
  int64_t first[NX_MAX_OPERANDS];             /* elements from base */
} nx_loop;

/* Fills [l] for the [n] operands [a] and answers NX_OK, or answers
   NX_ARITY if [n] is not in [1, NX_MAX_OPERANDS], and NX_SHAPE if their
   shapes differ. The loop keeps the C order of indices: axes are dropped
   and merged, never reordered. */
int nx_coalesce(int n, const nx_array *a, nx_loop *l);

/* Coalesces in place the loop of [n] operands over [rank] axes, [rank] 0
   or more: their [extent]s and operand k's steps step[k], by axis. Drops
   the axes of extent 1 and merges an axis into the one before it where
   every operand's step there is this axis's step times its extent,
   keeping the C order. Answers the merged rank, at least 1: a loop with no
   element has rank 1, extent[0] = 0 and every step 0; one of one element,
   a rank-0 loop included, rank 1, extent 1 and every step 0. nx_coalesce's
   loop is this over its operands' shapes and strides. */
int nx_coalesce_dims(int n, int rank, int64_t *extent,
                     int64_t (*step)[NX_MAX_RANK]);

/* Block copies */

/* A box: [extent[0]] planes of [extent[1]] rows of [extent[2]] elements, in
   C order, as nx_loop's axes. Element (p, i, j) of operand k, 0 the
   destination and 1 the source, lies at position
   first[k] + p·step[k][0] + i·step[k][1] + j·step[k][2], counted in
   elements from its base. Positions are at least 0; steps have either
   sign. */
typedef struct {
  int64_t extent[3];
  int64_t first[2];
  int64_t step[2][3]; /* elements, by operand then axis */
} nx_box;

/* Copies the box [b] of elements of [bits] bits from [src] to [dst], bits
   for bits. The destination's elements are distinct and share no byte with
   the source's. Bytes that hold only copied elements take plain stores; a
   byte shared with other elements takes a compare-and-swap, as
   nx_sub_store does. Each plane copies as one block: rows of adjacent
   elements are memcpy; where one operand steps one element across rows and
   the other one along them, whichever operand is transposed, it moves
   square blocks through registers, 8x8 of 4-byte elements, 32 bytes from
   each of eight columns into each of eight rows, and 4x4 of the others.
   Two adjacent runs of the source interleaved into adjacent elements, as a
   window two elements wide reads its rows, zip in registers. The case is
   chosen once per box. */
void nx_copy_box(uint8_t *dst, const uint8_t *src, const nx_box *b,
                 int bits);

/* Sub-byte elements

   Element p of a dtype of [bits] bits (1 or 4) is bits p·bits to p·bits +
   bits - 1 of the bytes from [base], LSB first. A store is one
   compare-and-swap of its byte and a load one relaxed atomic load of it, so
   threads may load and store elements of one byte at once: no store loses
   another element's bits, and no load sees an element no store wrote. */

static inline uint32_t nx_sub_load(const uint8_t *base, int bits, int64_t p) {
  uint64_t bit = (uint64_t)p * (uint64_t)bits;
  uint8_t byte = __atomic_load_n(base + (bit >> 3), __ATOMIC_RELAXED);
  return (byte >> (bit & 7)) & ((1u << bits) - 1);
}

/* Stores the bits of [set] under [mask] into [*byte], keeping its other
   bits, with one compare-and-swap. */
static inline void nx_sub_put(uint8_t *byte, uint8_t mask, uint8_t set) {
  uint8_t old = __atomic_load_n(byte, __ATOMIC_RELAXED);
  set &= mask;
  while (!__atomic_compare_exchange_n(byte, &old,
                                      (uint8_t)((old & ~mask) | set), 1,
                                      __ATOMIC_RELAXED, __ATOMIC_RELAXED))
    ;
}

/* Stores the low [bits] of [v] at element [p]. */
static inline void nx_sub_store(uint8_t *base, int bits, int64_t p,
                                uint32_t v) {
  uint64_t bit = (uint64_t)p * (uint64_t)bits;
  uint8_t mask = (uint8_t)(((1u << bits) - 1) << (bit & 7));
  nx_sub_put(base + (bit >> 3), mask, (uint8_t)((v << (bit & 7)) & mask));
}

/* Runs of sub-byte elements

   The [n] elements from element [p], one per byte; [p] is at least 0, and
   a run of no element touches no byte. A run's whole bytes hold only its
   elements and are plain loads and stores; a byte it shares with other
   elements, at either end, is one atomic load or one compare-and-swap. */

/* Reads the elements of [dt], a sub-byte dtype, into [dst]: int4's
   sign-extended, the others' zero-extended. Both runs are always inlined,
   so that a caller that passes a constant [dt] or [bits] gets a loop over
   whole bytes the compiler vectorises. */
static inline __attribute__((always_inline)) void nx_sub_unpack_run(
    const uint8_t *base, int dt, int64_t p, uint8_t *dst, int64_t n) {
  int bits = dt == NX_BIT ? 1 : 4, per = 8 / bits;
  uint32_t mask = (1u << bits) - 1, half = dt == NX_INT4 ? 8 : 0;
  int64_t i = 0;
#define NX_EXTEND(v) ((uint8_t)((((v) & mask) ^ half) - half))
  for (; i < n && (p + i) % per != 0; i++)
    dst[i] = NX_EXTEND(nx_sub_load(base, bits, p + i));
  const uint8_t *b = base + (p + i) / per;
  int64_t whole = (n - i) / per;
  if (bits == 4)
    for (int64_t k = 0; k < whole; k++, i += 2) {
      dst[i] = NX_EXTEND(b[k]);
      dst[i + 1] = NX_EXTEND(b[k] >> 4);
    }
  else
    for (int64_t k = 0; k < whole; k++, i += 8)
      for (int j = 0; j < 8; j++) dst[i + j] = NX_EXTEND(b[k] >> j);
  for (; i < n; i++) dst[i] = NX_EXTEND(nx_sub_load(base, bits, p + i));
#undef NX_EXTEND
}

/* Writes the low [bits] of each byte of [src] to the elements. */
static inline __attribute__((always_inline)) void nx_sub_pack_run(
    uint8_t *base, int bits, int64_t p, const uint8_t *src, int64_t n) {
  if (n <= 0) return;
  int per = 8 / bits;
  uint32_t mask = (1u << bits) - 1;
  int64_t i = 0;
  /* The elements of the first byte, if the run starts inside it. */
  if (p % per != 0) {
    int at = (int)(p % per) * bits;
    uint8_t m = 0, set = 0;
    for (; i < n && (p + i) % per != 0; i++, at += bits) {
      m |= (uint8_t)(mask << at);
      set |= (uint8_t)((src[i] & mask) << at);
    }
    nx_sub_put(base + p / per, m, set);
  }
  uint8_t *b = base + (p + i) / per;
  int64_t whole = (n - i) / per;
  if (bits == 4)
    for (int64_t k = 0; k < whole; k++, i += 2)
      b[k] = (uint8_t)((src[i] & 15) | (src[i + 1] << 4));
  else
    for (int64_t k = 0; k < whole; k++, i += 8) {
      uint8_t x = 0;
      for (int j = 0; j < 8; j++) x |= (uint8_t)((src[i + j] & 1) << j);
      b[k] = x;
    }
  /* The elements of the last byte, if the run ends inside it. */
  if (i < n) {
    uint8_t m = 0, set = 0;
    for (int at = 0; i < n; i++, at += bits) {
      m |= (uint8_t)(mask << at);
      set |= (uint8_t)((src[i] & mask) << at);
    }
    nx_sub_put(b + whole, m, set);
  }
}

/* Runs

   The codecs of nx_dtype.h over [n] contiguous elements, for host
   kernels. Every scalar codec selects rather than branches, or branches
   only where gcc and clang select, as bfloat16's encoder does, so a loop
   of it vectorises at the width of the target the including code compiles
   for, under -fno-trapping-math, and is a run. The runs below are named
   because they are faster than such a loop: the hardware converts a
   vector at a time with one instruction where it can, bit operations on
   vectors encode bfloat16 and decode e4m3fn in fewer instructions than a
   compiler finds, and a 64-bit integer converts with the scalar
   instruction, which no vector form matches. A run's tail is the scalar
   codec. Device kernels convert per element with nx_dtype.h. */

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
/* The eight binary32 values from [src] as nx_float_to_int gives them for
   [[lo, hi]], in int32 lanes; AVX2. [[lo, hi]] is an integer dtype's range
   of at most 16 bits, whose bounds binary32 holds. */
static inline __m256i nx_float_to_int_x8(const float *src, float lo,
                                         float hi) {
  __m256 x = _mm256_loadu_ps(src);
  x = _mm256_and_ps(x, _mm256_cmp_ps(x, x, _CMP_ORD_Q));
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(lo)), _mm256_set1_ps(hi));
  return _mm256_cvttps_epi32(x);
}
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
  for (; i + 8 <= n; i += 8) {
    __m256i t = nx_float_to_int_x8(src + i, -128.0f, 127.0f);
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
  for (; i + 8 <= n; i += 8) {
    __m256i t = nx_float_to_int_x8(src + i, 0.0f, 255.0f);
    __m128i h = _mm_packs_epi32(_mm256_castsi256_si128(t),
                                _mm256_extracti128_si256(t, 1));
    _mm_storel_epi64((__m128i *)(dst + i), _mm_packus_epi16(h, h));
  }
#endif
  for (; i < n; i++) dst[i] = (uint8_t)nx_float_to_int(src[i], 0, UINT8_MAX);
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
   for sixteen against the loop's 71. */
#if defined(__aarch64__)
/* nx_mini_round(f, 3, 7) of four lanes. */
static inline uint32x4_t nx_e4m3fn_round_x4(float32x4_t f) {
  uint32x4_t u = vreinterpretq_u32_f32(vabsq_f32(f));
  const float32x4_t magic =
      vreinterpretq_f32_u32(vdupq_n_u32((uint32_t)(127 + 24 - 7 - 3) << 23));
  uint32x4_t sub = vsubq_u32(
      vreinterpretq_u32_f32(vaddq_f32(vreinterpretq_f32_u32(u), magic)),
      vreinterpretq_u32_f32(magic));
  uint32x4_t odd = vandq_u32(vshrq_n_u32(u, 20), vdupq_n_u32(1));
  uint32x4_t bias =
      vdupq_n_u32(0u - ((uint32_t)(127 - 7) << 23) + (1u << 19) - 1);
  uint32x4_t normal = vshrq_n_u32(vaddq_u32(vaddq_u32(u, bias), odd), 20);
  uint32x4_t small = vcltq_u32(u, vdupq_n_u32((uint32_t)(128 - 7) << 23));
  return vbslq_u32(small, sub, normal);
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

/* From and to doubles. arm64 widens float16 with FCVT and FCVTL, exactly.
   A double narrows to __fp16 in one rounding: clang narrows a vector through
   FCVTXN (round to odd) then FCVTN, gcc converts each element with FCVT
   from d to h. With F16C and AVX2, doubles round to odd as binary32 four
   at a time, as nx_float_odd does, and convert eight at a time. */

#if defined(__aarch64__)
/* These runs read and write uint16_t memory as __fp16. */
typedef __fp16 __attribute__((may_alias)) nx_fp16;
#endif

static inline void nx_f16_to_double_run(const uint16_t *src, double *dst,
                                        size_t n) {
#if defined(__aarch64__)
  const nx_fp16 *h = (const nx_fp16 *)src;
  for (size_t i = 0; i < n; i++) dst[i] = (double)h[i];
#else
  for (size_t i = 0; i < n; i++) dst[i] = nx_f16_to_float(src[i]);
#endif
}

static inline void nx_double_to_f16_run(const double *src, uint16_t *dst,
                                        size_t n) {
#if defined(__aarch64__)
  nx_fp16 *h = (nx_fp16 *)dst;
  for (size_t i = 0; i < n; i++) h[i] = (__fp16)src[i];
#else
  size_t i = 0;
#if defined(__F16C__) && defined(__AVX2__)
  const __m256d mag = _mm256_castsi256_pd(_mm256_set1_epi64x(INT64_MAX));
  /* A 64-bit mask's even halves, in its first four lanes. */
  const __m256i even = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
  for (; i + 8 <= n; i += 8) {
    __m128 h[2];
    for (int k = 0; k < 2; k++) {
      __m256d x = _mm256_loadu_pd(src + i + 4 * k);
      __m128 f = _mm256_cvtpd_ps(x);
      __m256d b = _mm256_cvtps_pd(f);
      __m256d away = _mm256_cmp_pd(_mm256_and_pd(b, mag),
                                   _mm256_and_pd(x, mag), _CMP_GT_OQ);
      __m256d inexact = _mm256_cmp_pd(b, x, _CMP_NEQ_OQ);
      __m128i a = _mm256_castsi256_si128(
          _mm256_permutevar8x32_epi32(_mm256_castpd_si256(away), even));
      __m128i e = _mm256_castsi256_si128(
          _mm256_permutevar8x32_epi32(_mm256_castpd_si256(inexact), even));
      /* a is -1 where f rounded away from zero. */
      __m128i odd = _mm_or_si128(_mm_add_epi32(_mm_castps_si128(f), a),
                                 _mm_srli_epi32(e, 31));
      h[k] = _mm_castsi128_ps(odd);
    }
    _mm_storeu_si128((__m128i *)(dst + i),
                     _mm256_cvtps_ph(_mm256_set_m128(h[1], h[0]),
                                     _MM_FROUND_TO_NEAREST_INT));
  }
#endif
  for (; i < n; i++) dst[i] = nx_double_to_f16(src[i]);
#endif
}

static inline void nx_bf16_to_double_run(const uint16_t *src, double *dst,
                                         size_t n) {
  for (size_t i = 0; i < n; i++) dst[i] = nx_bf16_to_float(src[i]);
}

static inline void nx_double_to_bf16_run(const double *src, uint16_t *dst,
                                         size_t n) {
  for (size_t i = 0; i < n; i++) dst[i] = nx_double_to_bf16(src[i]);
}

#endif /* NX_ARRAY_H */
