/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* binary: a function of the elements of two operands at each index, as nx.cpu
   computes it (cpu/nx_c_map.c). BINARY(add, float32) defines the kernels of add
   at float32. Signed integers add, subtract and multiply in the unsigned width,
   so they wrap. A float result that is NaN is the first NaN operand, its bits
   unchanged, as on the host, or else the operation's own. */

#include "common.h"
#include "ocml.h"

/* [r], unless an operand is NaN: then the first NaN operand. */
template <class T> DEVICE T nan2(T a, T b, T r) {
  r = b != b ? b : r;
  return a != a ? a : r;
}

/* base^e of integers by squaring, wrapping. A negative exponent has a value
   for a base of 1 or -1 alone, and is 0 otherwise. */
DEVICE int64_t ipow(int64_t base, int64_t e) {
  if (e < 0) return base == 1 ? 1 : base == -1 ? ((e & 1) ? -1 : 1) : 0;
  uint64_t b = (uint64_t)base, r = 1;
  while (e > 0) {
    if (e & 1) r *= b;
    e >>= 1;
    if (e) b *= b;
  }
  return (int64_t)r;
}

DEVICE uint64_t ipow(uint64_t b, uint64_t e) {
  uint64_t r = 1;
  while (e > 0) {
    if (e & 1) r *= b;
    e >>= 1;
    if (e) b *= b;
  }
  return r;
}

namespace kind {

#define ARITH(k, op)                                                           \
  struct k {                                                                   \
    static DEVICE float f(float a, float b) { return nan2(a, b, a op b); }     \
    static DEVICE double f(double a, double b) { return nan2(a, b, a op b); }  \
    static DEVICE int64_t f(int64_t a, int64_t b) {                            \
      return (int64_t)((uint64_t)a op(uint64_t) b);                            \
    }                                                                          \
    static DEVICE uint64_t f(uint64_t a, uint64_t b) { return a op b; }        \
  };

ARITH(add, +)
ARITH(sub, -)
ARITH(mul, *)

struct fdiv {
  static DEVICE float f(float a, float b) { return nan2(a, b, a / b); }
  static DEVICE double f(double a, double b) { return nan2(a, b, a / b); }
};

/* The quotient truncated: by 0 it is 0, and a signed least value by -1 is
   itself. */
struct idiv {
  static DEVICE float f(float a, float b) { return __builtin_truncf(a / b); }
  static DEVICE double f(double a, double b) { return __builtin_trunc(a / b); }
  static DEVICE int64_t f(int64_t a, int64_t b) {
    return b == 0 ? 0 : b == -1 ? (int64_t)(-(uint64_t)a) : a / b;
  }
  static DEVICE uint64_t f(uint64_t a, uint64_t b) { return b == 0 ? 0 : a / b; }
};

/* The remainder of the dividend's sign: by 0 it is the dividend. */
struct mod {
  static DEVICE float f(float a, float b) { return __ocml_fmod_f32(a, b); }
  static DEVICE double f(double a, double b) { return __ocml_fmod_f64(a, b); }
  static DEVICE int64_t f(int64_t a, int64_t b) {
    return b == 0 ? a : b == -1 ? 0 : a % b;
  }
  static DEVICE uint64_t f(uint64_t a, uint64_t b) { return b == 0 ? a : a % b; }
};

struct pow {
  static DEVICE float f(float a, float b) { return __ocml_pow_f32(a, b); }
  static DEVICE double f(double a, double b) { return __ocml_pow_f64(a, b); }
  static DEVICE int64_t f(int64_t a, int64_t b) { return ipow(a, b); }
  static DEVICE uint64_t f(uint64_t a, uint64_t b) { return ipow(a, b); }
};

struct atan2 {
  static DEVICE float f(float a, float b) { return __ocml_atan2_f32(a, b); }
  static DEVICE double f(double a, double b) { return __ocml_atan2_f64(a, b); }
};

#define EXTREME(k, op, fop)                                                    \
  struct k {                                                                   \
    static DEVICE float f(float a, float b) {                                  \
      return fop<float, uint32_t>(a, b);                                       \
    }                                                                          \
    static DEVICE double f(double a, double b) {                               \
      return fop<double, uint64_t>(a, b);                                      \
    }                                                                          \
    static DEVICE int64_t f(int64_t a, int64_t b) { return a op b ? a : b; }   \
    static DEVICE uint64_t f(uint64_t a, uint64_t b) {                         \
      return a op b ? a : b;                                                   \
    }                                                                          \
    static DEVICE uint8_t f(uint8_t a, uint8_t b) { return a op b ? a : b; }   \
  };

EXTREME(maximum, >, fmax)
EXTREME(minimum, <, fmin)

#define BITWISE(k, op)                                                         \
  struct k {                                                                   \
    static DEVICE int64_t f(int64_t a, int64_t b) { return a op b; }           \
    static DEVICE uint64_t f(uint64_t a, uint64_t b) { return a op b; }        \
    static DEVICE uint8_t f(uint8_t a, uint8_t b) { return a op b; }           \
  };

BITWISE(and_, &)
BITWISE(or_, |)
BITWISE(xor_, ^)

} // namespace kind

template <class K, class D, class I>
DEVICE void map2(typename D::storage *dst, const typename D::storage *a,
                 const typename D::storage *b, i64 n, i64 groups, I index) {
  for (i64 i = first(); i < n; i += groups * THREADS)
    dst[i] = store<D>(
        K::f(D::load(a[index.at(0, i)]), D::load(b[index.at(1, i)])));
}

#define BINARY(K, D)                                                           \
  KERNEL void NAME(c)(D::storage *dst, const D::storage *a,                    \
                      const D::storage *b, i64 n, i64 groups) {                \
    map2<kind::K, D>(dst, a, b, n, groups, contiguous());                      \
  }                                                                            \
  KERNEL void NAME(s)(D::storage *dst, const D::storage *a,                    \
                      const D::storage *b, meta m) {                           \
    map2<kind::K, D>(dst, a, b, m.n, m.groups, strided{&m});                   \
  }
