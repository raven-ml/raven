/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Elements as kernels hold them: each dtype's width, a value held in up
   to four 32-bit words, the casts between dtypes as Prog.Cast stores
   them, and an element's read and write at any width, sub-byte ones
   included. The loops and the contractions share them, so that every
   cast of nx.metal has nx_kinds.h's bits. */

#ifndef NX_METAL_VALUES_H
#define NX_METAL_VALUES_H

#include "device.h"
#include "nx_dtype.h"

typedef ulong u64;

/* A value of up to four words, as held: [lo] its words 0 and 1, [hi] its
   words 2 and 3. */
struct held {
  u64 lo, hi;
};

static inline held held_of(u64 lo, u64 hi) {
  held h;
  h.lo = lo;
  h.hi = hi;
  return h;
}

/* Dtypes

   [bits] of an element; the words its held value takes; its bits [r], the
   low bits of a word, as held: a signed integer sign-extended to 32 bits,
   the others as they are; and a held value's bits. */

constexpr int log2_of(int b) { return b <= 1 ? 0 : 1 + log2_of(b / 2); }
#define LOG_BITS(NAME, name, bits, kind) | u64(log2_of(bits)) << (3 * NX_##NAME)
constant constexpr u64 log_bits = 0 NX_DTYPES(LOG_BITS);
#undef LOG_BITS

static inline uint bits_of(int dt) { return 1u << ((log_bits >> (3 * dt)) & 7); }

static inline uint words_of(int dt) {
  const uint b = bits_of(dt);
  return b <= 32 ? 1 : b / 32;
}

static inline u64 of_raw(int dt, u64 r) {
  switch (dt) {
  case NX_INT16: return uint(int(short(ushort(r))));
  case NX_INT8: return uint(int(char(uchar(r))));
  case NX_INT4: return uint(int(uint(r) << 28) >> 28);
  default: {
    const uint b = bits_of(dt);
    return b >= 64 ? r : r & ((u64(1) << b) - 1);
  }
  }
}

static inline u64 raw_of(int dt, u64 h) {
  const uint b = bits_of(dt);
  return b >= 64 ? h : h & ((u64(1) << b) - 1);
}

/* The float32 of the narrow float [dt]'s code [h]. */
static inline float narrow_value(int dt, uint h) {
  if (dt == NX_FLOAT16) return nx_metal_f16_to_float(ushort(h));
  return nx_bits_to_float(dt, h);
}

/* The code of the narrow float [dt] nearest the float32 [f]. */
static inline uint narrow_code(int dt, float f) {
  switch (dt) {
  case NX_FLOAT16: return nx_metal_float_to_f16(f);
  case NX_BFLOAT16: return nx_float_to_bf16(f);
  case NX_FLOAT8_E4M3FN: return nx_float_to_e4m3fn(f);
  case NX_FLOAT8_E5M2: return nx_float_to_e5m2(f);
  default: return nx_float_to_e2m1fn(f);
  }
}

/* Casts

   Every cast here reads and writes bits, so a subnormal float32 casts as
   on every library: Metal flushes subnormals in arithmetic alone. The host
   declines a cast from or into float64 or complex128 but into their own
   dtype. */

static inline bool float_dt(int dt) { return dt <= NX_FLOAT4_E2M1FN; }
static inline bool signed_dt(int dt) {
  return dt == NX_INT64 || dt == NX_INT32 || dt == NX_INT16 || dt == NX_INT8 ||
         dt == NX_INT4;
}
static inline bool bool_dt(int dt) { return dt == NX_BOOL || dt == NX_BIT; }

/* The magnitude [m] as a float32 rounded to odd: exact below 2^24, else
   its top 24 bits, the last set where a bit below them is. Binary32's
   grid is at least four times finer than a narrow float's at every
   magnitude, so the encoders then round [m] once. */
static inline float odd_f32(u64 m) {
  if (m < (u64(1) << 24)) return float(m);
  const uint shift = 40 - uint(metal::clz(m));
  const u64 top = (m >> shift) | ((m & ((u64(1) << shift) - 1)) != 0 ? 1 : 0);
  return as_type<float>(as_type<uint>(float(top)) + (shift << 23));
}

/* A float32's value [f] into the integer dtype [dt], toward zero,
   saturated, NaN to 0: the bits a store writes, sign-extended. */
static inline u64 float_to_int(int dt, float f) {
  switch (dt) {
  case NX_INT64: {
    if (f != f) return 0;
    if (f >= 0x1p63f) return u64(0x7fffffffffffffff);
    if (f < -0x1p63f) return u64(1) << 63;
    return u64(long(f));
  }
  case NX_UINT64: {
    if (f != f || f <= 0.0f) return 0;
    if (f >= 0x1p64f) return ~u64(0);
    return u64(f);
  }
  case NX_INT32: return u64(long(int(nx_float_to_int(f, INT_MIN, INT_MAX))));
  case NX_UINT32: return nx_float_to_int(f, 0, UINT_MAX);
  case NX_INT16: return of_raw(dt, nx_float_to_int(f, -32768, 32767));
  case NX_UINT16: return nx_float_to_int(f, 0, 65535);
  case NX_INT8: return of_raw(dt, nx_float_to_int(f, -128, 127));
  case NX_UINT8: return nx_float_to_int(f, 0, 255);
  case NX_INT4: return of_raw(dt, nx_float_to_int(f, -8, 7));
  default: return nx_float_to_int(f, 0, 15); /* uint4 */
  }
}

/* [x] held of the real dtype [from] cast into the real dtype [to], held,
   as Prog.Cast stores it: each value rounded once. A float that keeps its
   format keeps its bits. */
static inline held cast_real(int from, int to, u64 x) {
  if (from == to) return held_of(x, 0);
  /* A float's value as float32, exactly: float32 itself or a narrow
     float. */
  if (float_dt(from)) {
    const float f = from == NX_FLOAT32 ? as_type<float>(uint(x))
                                       : narrow_value(from, uint(x));
    if (bool_dt(to)) return held_of((as_type<uint>(f) & 0x7fffffffu) != 0, 0);
    if (to == NX_FLOAT32) return held_of(as_type<uint>(f), 0);
    if (float_dt(to)) return held_of(narrow_code(to, f), 0);
    return held_of(of_raw(to, float_to_int(to, f)), 0);
  }
  /* An integer's value: a uint64's bits, else the held value read
     signed or not. */
  long s;
  if (bool_dt(from)) s = x != 0;
  else if (from == NX_INT64 || from == NX_UINT64) s = long(x);
  else if (signed_dt(from)) s = int(uint(x));
  else s = long(uint(x));
  const bool unsig = !signed_dt(from);
  if (bool_dt(to)) return held_of(s != 0, 0);
  if (to == NX_FLOAT32)
    return held_of(as_type<uint>(unsig ? float(u64(s)) : float(s)), 0);
  if (float_dt(to)) {
    const bool neg = !unsig && s < 0;
    const u64 m = neg ? u64(0) - u64(s) : u64(s);
    float f = odd_f32(m);
    if (neg) f = -f;
    return held_of(narrow_code(to, f), 0);
  }
  /* Into an integer: modulo its width. */
  return held_of(of_raw(to, u64(s)), 0);
}

static inline held cast(int from, int to, held x) {
  if (from == to) return x;
  if (from != NX_COMPLEX64) {
    if (to != NX_COMPLEX64) return cast_real(from, to, x.lo);
    return held_of(cast_real(from, NX_FLOAT32, x.lo).lo, 0);
  }
  /* From complex64: its parts' bits. */
  const uint re = uint(x.lo), im = uint(x.lo >> 32);
  if (bool_dt(to))
    return held_of(((re | im) & 0x7fffffffu) != 0, 0);
  return cast_real(NX_FLOAT32, to, re);
}

/* [x] of [from] read in [to], of one width. */
static inline held bitcast(int from, int to, held x) {
  if (bits_of(from) == 128) return x;
  return held_of(of_raw(to, raw_of(from, x.lo)), 0);
}

/* Memory */

/* The held value of [dt] at element [p] of [b], a bit [phase] past it for
   a dtype of fewer than 8 bits. Elements of 32 bits or more read as their
   32-bit words: a complex64's lie on 4 bytes alone. */
static inline held read(device const uchar *b, u64 p, uint phase, int dt) {
  held h = held_of(0, 0);
  switch (bits_of(dt)) {
  case 1: {
    const u64 bit = phase + p;
    h.lo = (b[bit >> 3] >> (bit & 7)) & 1;
    break;
  }
  case 4: {
    const u64 bit = phase + 4 * p;
    h.lo = (b[bit >> 3] >> (bit & 7)) & 15;
    break;
  }
  case 8: h.lo = b[p]; break;
  case 16: h.lo = ((device const ushort *)b)[p]; break;
  case 32: h.lo = ((device const uint *)b)[p]; break;
  case 64: {
    device const uint *q = (device const uint *)b + 2 * p;
    h.lo = u64(q[0]) | u64(q[1]) << 32;
    break;
  }
  default: {
    device const uint *q = (device const uint *)b + 4 * p;
    h.lo = u64(q[0]) | u64(q[1]) << 32;
    h.hi = u64(q[2]) | u64(q[3]) << 32;
    return h;
  }
  }
  h.lo = of_raw(dt, h.lo);
  return h;
}

/* Stores the low [bits] bits of [v] as the element of [bits] bits, 1 or 4,
   at bit [bit] of [base], whose address lies on 4 bytes. The element
   shares its 32-bit word with others, which other threads store: it
   changes its own bits of the word, by and then or, which commute with
   other threads' on other bits. rig orders memory in 4-byte words, so a
   launch may write each word holding a byte of its buffer. */
static inline void store_bits(device uchar *base, u64 bit, uint bits, uint v) {
  device metal::atomic_uint *w =
      (device metal::atomic_uint *)base + (bit >> 5);
  const uint shift = uint(bit & 31);
  const uint mask = ((1u << bits) - 1) << shift;
  metal::atomic_fetch_and_explicit(w, ~mask, metal::memory_order_relaxed);
  metal::atomic_fetch_or_explicit(w, (v << shift) & mask,
                                 metal::memory_order_relaxed);
}

/* Stores [h] of [dt] at element [p] of [b], a bit [phase] past it. */
static inline void write(device uchar *b, u64 p, uint phase, int dt, held h) {
  const uint n = bits_of(dt);
  switch (n) {
  case 1:
  case 4: store_bits(b, phase + n * p, n, uint(h.lo)); return;
  case 8: b[p] = uchar(h.lo); return;
  case 16: ((device ushort *)b)[p] = ushort(h.lo); return;
  case 32: ((device uint *)b)[p] = uint(h.lo); return;
  case 64: {
    device uint *q = (device uint *)b + 2 * p;
    q[0] = uint(h.lo);
    q[1] = uint(h.lo >> 32);
    return;
  }
  default: {
    device uint *q = (device uint *)b + 4 * p;
    q[0] = uint(h.lo);
    q[1] = uint(h.lo >> 32);
    q[2] = uint(h.hi);
    q[3] = uint(h.hi >> 32);
  }
  }
}

#endif
