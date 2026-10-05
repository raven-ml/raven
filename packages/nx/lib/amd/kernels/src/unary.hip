/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* unary: a function of each element, as nx.cpu computes it (cpu/nx_c_map.c),
   in the dtype's compute type. UNARY(sin, float32) is the module of sin at
   float32. Integers negate and take absolute values in the unsigned width, so
   they wrap; the transcendental functions are OCML's, held to nx's accuracy
   contract where OCML misses it. */

#include "common.h"
#include "ocml.h"

namespace kind {

struct neg {
  static DEVICE float f(float v) { return -v; }
  static DEVICE double f(double v) { return -v; }
  static DEVICE int64_t f(int64_t v) { return (int64_t)(-(uint64_t)v); }
  static DEVICE uint64_t f(uint64_t v) { return -v; }
};

struct recip {
  static DEVICE float f(float v) { return 1.0f / v; }
  static DEVICE double f(double v) { return 1.0 / v; }
  static DEVICE int64_t f(int64_t v) { return v == 0 ? 0 : 1 / v; }
  static DEVICE uint64_t f(uint64_t v) { return v == 0 ? 0 : 1 / v; }
};

struct abs {
  static DEVICE float f(float v) { return __builtin_fabsf(v); }
  static DEVICE double f(double v) { return __builtin_fabs(v); }
  static DEVICE int64_t f(int64_t v) {
    return v < 0 ? (int64_t)(-(uint64_t)v) : v;
  }
  static DEVICE uint64_t f(uint64_t v) { return v; }
};

/* -1, 0 or 1, and a NaN as it is. */
struct sign {
  static DEVICE float f(float v) {
    return v != v ? v : (float)((v > 0) - (v < 0));
  }
  static DEVICE double f(double v) {
    return v != v ? v : (double)((v > 0) - (v < 0));
  }
  static DEVICE int64_t f(int64_t v) { return (v > 0) - (v < 0); }
  static DEVICE uint64_t f(uint64_t v) { return v != 0; }
};

#define BUILTIN(k, b)                                                          \
  struct k {                                                                   \
    static DEVICE float f(float v) { return __builtin_##b##f(v); }             \
    static DEVICE double f(double v) { return __builtin_##b(v); }              \
  };

BUILTIN(sqrt, sqrt)
BUILTIN(trunc, trunc)
BUILTIN(ceil, ceil)
BUILTIN(floor, floor)
BUILTIN(round, round)

#define TRANSCENDENTAL(k)                                                      \
  struct k {                                                                   \
    static DEVICE float f(float v) { return __ocml_##k##_f32(v); }             \
    static DEVICE double f(double v) { return __ocml_##k##_f64(v); }           \
  };

/* An odd function keeps the sign of a zero, as C99's Annex F has it: OCML's
   expm1 and log1p give +0 for -0. */
#define ODD(k)                                                                 \
  struct k {                                                                   \
    static DEVICE float f(float v) { return v == 0 ? v : __ocml_##k##_f32(v); } \
    static DEVICE double f(double v) {                                         \
      return v == 0 ? v : __ocml_##k##_f64(v);                                 \
    }                                                                          \
  };

TRANSCENDENTAL(exp)
TRANSCENDENTAL(cos)
TRANSCENDENTAL(acos)
TRANSCENDENTAL(cosh)
ODD(log1p)
ODD(expm1)
ODD(sin)
ODD(tan)
ODD(asin)
ODD(atan)
ODD(sinh)
ODD(tanh)
ODD(erf)

/* OCML's binary32 log is 2 ulps off at points, past nx's bound of 1: its
   binary64 log, rounded once, is within it. */
struct log {
  static DEVICE float f(float v) { return (float)__ocml_log_f64((double)v); }
  static DEVICE double f(double v) { return __ocml_log_f64(v); }
};

} // namespace kind

template <class K, class D, class I>
DEVICE void map1(typename D::storage *dst, const typename D::storage *x, i64 n,
                 i64 groups, I index) {
  for (i64 i = first(); i < n; i += groups * THREADS)
    dst[i] = store<D>(K::f(D::load(x[index.at(0, i)])));
}

#define UNARY(K, D)                                                            \
  KERNEL void c(D::storage *dst, const D::storage *x, i64 n, i64 groups) {     \
    map1<kind::K, D>(dst, x, n, groups, contiguous());                         \
  }                                                                            \
  KERNEL void s(D::storage *dst, const D::storage *x, meta m) {                \
    map1<kind::K, D>(dst, x, m.n, m.groups, strided{&m});                      \
  }
