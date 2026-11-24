/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What the spectral kernels share: the two formats they compute in (float
   and its complex numbers), scalar arithmetic that reads the same on
   each, the formats' constants, a group's place and meetings, and the
   split of a flat index.

   Matrices are row-major, element (i, j) of an r x c matrix at x[i c + j],
   one after another along the batch, but where a header says otherwise.
   A step specialises by format alone: T is the element, float or cf, and
   its real part is always float.

   The GPU flushes every arithmetic result below 2^-126 to a zero of its
   sign, and reads a subnormal operand of an instruction as one. A step
   reads each element of a routine's operand that it does not use as an
   arithmetic operand (a copy, a comparison, a magnitude taken by bits)
   through flush, so it computes on the operand's flushed values. Scaling
   by a power of two goes through nx_ldexp_f32, whose multiplies flush as
   the arithmetic does, never through Metal's ldexp, whose subnormal
   results the language does not state. So no subnormal arises past the
   reads, and the least positive number a step meets is 2^-126.

   This follows nx.amd's sp_common.h. */

#ifndef NX_METAL_SP_COMMON_H
#define NX_METAL_SP_COMMON_H

#include "device.h"
#include "nx_kinds.h"

namespace sp {

/* A group's place: a thread's index in its group, the group's threads,
   and a word of the group's memory from which a meeting that answers
   takes its answer. */
struct grp {
  int tid, nthr;
  threadgroup int *any;
};

/* Every barrier orders threadgroup and device memory: a step's working
   arrays lie in either. */
static inline void barrier() {
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup |
                             metal::mem_flags::mem_device);
}

/* Whether [x] holds for some thread of the group, which each receive. */
static inline bool barrier_or(grp g, bool x) {
  barrier();
  if (g.tid == 0) *g.any = 0;
  barrier();
  if (x)
    metal::atomic_store_explicit((threadgroup metal::atomic_int *)g.any, 1,
                                 metal::memory_order_relaxed);
  barrier();
  return *g.any != 0;
}

/* Arithmetic of float, each an instruction or inline code. */
static inline float fabs_(float x) { return metal::fabs(x); }
static inline float sqrt_(float x) { return nx_sqrtf(x); }
static inline float fmax_(float a, float b) { return metal::fmax(a, b); }
static inline float fmin_(float a, float b) { return metal::fmin(a, b); }
static inline float fma_(float a, float b, float c) { return nx_fmaf(a, b, c); }
static inline float copysign_(float a, float b) {
  return metal::copysign(a, b);
}
static inline float ldexp_(float x, int e) { return nx_ldexp_f32(x, e); }

/* The exponent of a finite nonzero normal [x]: x = m 2^e with
   1 <= |m| < 2. */
static inline int ilogb_(float x) { return int(nx_ilogb_f32(x)); }

static inline float inf_() { return as_type<float>(0x7F800000u); }
static inline float nan_() { return as_type<float>(0x7FC00000u); }

/* sqrt(a² + b²) without undue overflow or underflow: the larger magnitude
   times sqrt(1 + r²), r the ratio of the smaller to it, within a few
   ulps; infinite where either is, NaN where either is NaN and neither is
   infinite. */
static inline float hypot_(float a, float b) {
  a = fabs_(a), b = fabs_(b);
  const float big = fmax_(a, b), small = fmin_(a, b);
  if (big == inf_() || small == inf_()) return inf_();
  if (a != a || b != b) return a + b;
  if (big == 0) return 0;
  const float r = small / big;
  return big * sqrt_(1 + r * r);
}

/* A complex number of float parts. */
struct cf {
  float re, im;
};

static inline cf C(float r, float i) {
  cf x;
  x.re = r;
  x.im = i;
  return x;
}

template <class T> struct real_of;
template <> struct real_of<float> {
  static constexpr bool cplx() { return false; }
};
template <> struct real_of<cf> {
  static constexpr bool cplx() { return true; }
};

/* Scalars. A real T reads as a complex one of zero imaginary part. */
static inline float re(float x) { return x; }
static inline float re(cf x) { return x.re; }
static inline float im(float) { return 0; }
static inline float im(cf x) { return x.im; }
static inline float conj(float x) { return x; }
static inline cf conj(cf x) { return C(x.re, -x.im); }
static inline float abs2(float x) { return x * x; }
static inline float abs2(cf x) { return x.re * x.re + x.im * x.im; }
static inline float mag(float x) { return fabs_(x); }
static inline float mag(cf x) { return hypot_(x.re, x.im); }

/* |re| + |im|, LAPACK's cheap magnitude. */
static inline float mag1(float x) { return fabs_(x); }
static inline float mag1(cf x) { return fabs_(x.re) + fabs_(x.im); }

static inline cf operator+(cf a, cf b) { return C(a.re + b.re, a.im + b.im); }
static inline cf operator-(cf a, cf b) { return C(a.re - b.re, a.im - b.im); }
static inline cf operator-(cf a) { return C(-a.re, -a.im); }
static inline cf operator*(cf a, cf b) {
  return C(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
static inline cf operator*(float s, cf a) { return C(s * a.re, s * a.im); }
static inline cf operator*(cf a, float s) { return C(s * a.re, s * a.im); }
static inline cf operator/(cf a, float s) { return C(a.re / s, a.im / s); }
static inline thread cf &operator+=(thread cf &a, cf b) { return a = a + b; }
static inline thread cf &operator-=(thread cf &a, cf b) { return a = a - b; }
static inline bool operator==(cf a, cf b) {
  return a.re == b.re && a.im == b.im;
}
static inline bool operator!=(cf a, cf b) { return !(a == b); }

/* An element of device or threadgroup memory updated in place: a cf
   lvalue there takes += and -= as a thread one does. */
static inline device cf &operator+=(device cf &a, cf b) { return a = a + b; }
static inline device cf &operator-=(device cf &a, cf b) { return a = a - b; }
static inline threadgroup cf &operator+=(threadgroup cf &a, cf b) {
  return a = a + b;
}
static inline threadgroup cf &operator-=(threadgroup cf &a, cf b) {
  return a = a - b;
}

/* a / b by Smith's algorithm, which overflows only where the quotient
   does. */
static inline float cdiv(float a, float b) { return a / b; }
static inline cf cdiv(cf a, cf b) {
  if (fabs_(b.re) >= fabs_(b.im)) {
    const float r = b.im / b.re, d = b.re + b.im * r;
    return C((a.re + a.im * r) / d, (a.im - a.re * r) / d);
  }
  const float r = b.re / b.im, d = b.re * r + b.im;
  return C((a.re * r + a.im) / d, (a.im * r - a.re) / d);
}
static inline cf cdiv(cf a, float b) { return a / b; }
static inline cf operator/(cf a, cf b) { return cdiv(a, b); }

/* The element of real part [r] and imaginary part [i], [i] dropped on a
   real T. */
template <class T> struct mk;
template <> struct mk<float> {
  static float of(float r, float) { return r; }
};
template <> struct mk<cf> {
  static cf of(float r, float i) { return C(r, i); }
};
template <class T> static inline T elt(float r, float i = 0) {
  return mk<T>::of(r, i);
}

/* The unit roundoff, the least normal number, and the least positive
   number a step meets: the least normal number, since the arithmetic
   flushes below it. */
struct lim {
  static float eps() { return 0x1p-24f; }
  static float tiny() { return 0x1p-126f; }
  static float eta() { return 0x1p-126f; }
};

/* [x] as its operand's flushed value: each subnormal part a zero of its
   sign. */
static inline float flush(float x) { return nx_flush_f32(x); }
static inline cf flush(cf x) {
  return C(nx_flush_f32(x.re), nx_flush_f32(x.im));
}

/* A sum of squares that carries its rounding errors along: Neumaier's
   compensated sum, each square's own error taken exactly by a fused
   multiply-add, up to the flushing of an error below 2^-126. Within a
   few u of the exact sum whatever the count. A plain running sum loses
   every term below half its last bit, so it comes out low, and a vector
   divided by its root comes out long. */
struct sumsq {
  float s = 0, c = 0;
  void add(float t) thread {
    const float y = s + t;
    c += fabs_(s) >= fabs_(t) ? (s - y) + t : (t - y) + s;
    s = y;
  }
  void sq(float x) thread {
    const float p = x * x;
    add(p);
    c += fma_(x, x, -p);
  }
  void sq(cf x) thread { sq(x.re), sq(x.im); }
  float get() const thread { return s + c; }
};

/* The exponent e whose 2^e brings [big], a matrix's largest magnitude,
   into [1, 2) where it lies outside [lo, hi], 0 where it lies inside or
   is 0. Scaling by nx_ldexp_f32 is exact both ways for every element that
   stays normal. */
static inline int scale_exp(float big, float lo, float hi) {
  if (big == 0 || (big >= lo && big <= hi)) return 0;
  return -ilogb_(big);
}
static inline float scaled(float x, int e) { return ldexp_(x, e); }
static inline cf scaled(cf x, int e) {
  return C(ldexp_(x.re, e), ldexp_(x.im, e));
}

/* Flat indices: [i] % [n], [i] left as [i] / [n], in 32-bit arithmetic
   where [small], every index of the loop below 2^32, since a 64-bit
   division is a long sequence on the GPU. */
static inline bool is_small(long count) { return ulong(count) >> 32 == 0; }
static inline long split(thread long &i, long n, bool small) {
  if (small) {
    const uint q = uint(i) / uint(n), r = uint(i) - q * uint(n);
    i = long(q);
    return long(r);
  }
  const long q = i / n, r = i - q * n;
  i = q;
  return r;
}

/* The same by [dn], [n]'s divisor word from the host (Plan.divisor):
   where [small], device.h's divide, a product's high half, an add and a
   shift; otherwise the 64-bit division, which the host charges. */
static inline long split(thread long &i, long n, ulong dn, bool small) {
  if (small) {
    const uint q = divide(uint(i), dn), r = uint(i) - q * uint(n);
    i = long(q);
    return long(r);
  }
  const long q = i / n, r = i - q * n;
  i = q;
  return r;
}

} // namespace sp

#endif
