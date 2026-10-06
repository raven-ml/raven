/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What every kernel of nx.amd shares: the kernel ABI, the dtypes, and the
   conversions between them.

   A module holds a family's kernels at one dtype, or at one element width
   for a family that moves bytes, such as binary.float32 or contiguous.4. A
   macro such as BINARY(add, float32) defines the kernels of one kind, each
   named NAME(form), which the module defines before it: c_add and s_add. The
   kernels of a family of one kind are named by their form alone. The forms
   are [c], the contiguous form, and [s], the strided form; reduce.hip and
   the others state their own. Their parameters are pointers, then 64-bit
   integers or structs of them; the workgroup size is fixed at THREADS and
   the workgroup count comes as a parameter, so that no kernel reads implicit
   arguments.
   - [c (dst, op1, ..., n, groups)]: every operand C-contiguous from its
     pointer, [n] elements.
   - [s (dst, op1, ..., meta)]: [dst] C-contiguous, operand [k] at element
     [off[k] + sum_d i_d str[k][d]] for the index [(i_0, ..., i_rank-1)] of
     extents [ext].
   Each thread walks the elements from its global index by the grid's size.
   Index arithmetic is 64-bit. */

#include <stdint.h>

/* The storage formats come from nx's own codecs, so that a conversion here is
   nx.cpu's bit for bit. */
#pragma clang force_cuda_host_device begin
#include "nx_dtype.h"
#pragma clang force_cuda_host_device end

typedef int64_t i64;

#define KERNEL                                                                 \
  extern "C" __attribute__((global))                                           \
  __attribute__((amdgpu_flat_work_group_size(256, 256)))
#define DEVICE __attribute__((device)) inline

enum { THREADS = 256, MAX_RANK = 32, MAX_OPERANDS = 4 };

struct meta {
  i64 n, rank, groups;
  i64 ext[MAX_RANK];
  i64 off[MAX_OPERANDS];
  i64 str[MAX_OPERANDS][MAX_RANK];
};

/* The thread's first element, and the elements between two of its own. */
DEVICE i64 first() {
  return (i64)__builtin_amdgcn_workgroup_id_x() * THREADS +
         __builtin_amdgcn_workitem_id_x();
}

/* Waits for the workgroup's threads, each seeing the others' writes to the
   workgroup's memory before it. */
DEVICE void barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

/* The index templates a family's body is written over: where operand [k]'s
   element [i] in C order lies. */
struct contiguous {
  DEVICE i64 at(int, i64 i) const { return i; }
};

struct strided {
  const meta *m;
  DEVICE i64 at(int k, i64 i) const {
    i64 o = m->off[k];
    for (i64 d = m->rank - 1; d >= 0; d--) {
      i64 e = m->ext[d];
      o += (i % e) * m->str[k][d];
      i /= e;
    }
    return o;
  }
};

/* Dtypes

   As nx.cpu's table (cpu/nx_c.h): each dtype's storage, the type it computes
   in, how a stored element loads into it, and its category. */

enum category { FLOAT, SINT, UINT, BOOL };

#define DTYPE(name, storage_, compute_, cat_, load_)                           \
  struct name {                                                                \
    typedef storage_ storage;                                                  \
    typedef compute_ compute;                                                  \
    static const category cat = cat_;                                          \
    static DEVICE compute load(storage x) { return (compute)(load_(x)); }      \
  };

#define ID(x) (x)
#define BOOL_LD(x) ((x) != 0)

DTYPE(float16, uint16_t, float, FLOAT, half_to_float)
DTYPE(float32, float, float, FLOAT, ID)
DTYPE(float64, double, double, FLOAT, ID)
DTYPE(bfloat16, uint16_t, float, FLOAT, bfloat16_to_float)
DTYPE(float8_e4m3, uint8_t, float, FLOAT, fp8_e4m3_to_float)
DTYPE(float8_e5m2, uint8_t, float, FLOAT, fp8_e5m2_to_float)
DTYPE(int8, int8_t, int64_t, SINT, ID)
DTYPE(uint8, uint8_t, int64_t, UINT, ID)
DTYPE(int16, int16_t, int64_t, SINT, ID)
DTYPE(uint16, uint16_t, int64_t, UINT, ID)
DTYPE(int32, int32_t, int64_t, SINT, ID)
DTYPE(uint32, uint32_t, uint64_t, UINT, ID)
DTYPE(int64, int64_t, int64_t, SINT, ID)
DTYPE(uint64, uint64_t, uint64_t, UINT, ID)
DTYPE(bool_, uint8_t, uint8_t, BOOL, BOOL_LD)

/* The storage of a computed value of a dtype: a narrow float encoded from its
   float, an integer wrapped, a boolean normalized. */
template <class D> DEVICE typename D::storage store(typename D::compute v) {
  return (typename D::storage)v;
}
template <> DEVICE uint16_t store<float16>(float v) { return float_to_half(v); }
template <> DEVICE uint16_t store<bfloat16>(float v) {
  return float_to_bfloat16(v);
}
template <> DEVICE uint8_t store<float8_e4m3>(float v) {
  return float_to_fp8_e4m3(v);
}
template <> DEVICE uint8_t store<float8_e5m2>(float v) {
  return float_to_fp8_e5m2(v);
}
template <> DEVICE uint8_t store<bool_>(uint8_t v) { return v != 0; }

/* Conversions, as nx.cpu's cast (cpu/nx_c_map.c) */

/* A value on its way to a float narrower than binary32, as binary32: a wider
   value rounds to odd, so that the encoder rounds it once. */
DEVICE float narrow_odd(float v) { return v; }
DEVICE float narrow_odd(double v) { return double_to_float_odd(v); }
DEVICE float narrow_odd(int64_t v) {
  return double_to_float_odd(i64_to_double_odd(v));
}
DEVICE float narrow_odd(uint64_t v) {
  return double_to_float_odd(u64_to_double_odd(v));
}
DEVICE float narrow_odd(uint8_t v) { return double_to_float_odd((double)v); }

/* A float to an integer of [w] bits truncated toward zero, held at the ends of
   its range, and NaN to 0. The bounds 2^(w-1) and 2^w are exact doubles. */
template <class S> DEVICE S to_sint(double v) {
  const int w = sizeof(S) * 8;
  double lim = __builtin_ldexp(1.0, w - 1);
  if (v != v) return 0;
  if (v <= -lim) return (S)((uint64_t)1 << (w - 1));
  if (v >= lim) return (S)(((uint64_t)1 << (w - 1)) - 1);
  return (S)v;
}

template <class S> DEVICE S to_uint(double v) {
  const int w = sizeof(S) * 8;
  double lim = __builtin_ldexp(1.0, w);
  if (v != v) return 0;
  if (v <= 0.0) return 0;
  if (v >= lim) return (S)(~(uint64_t)0);
  return (S)v;
}

template <class D, class S, category Dcat, category Scat> struct convert;

/* To a float: the float dtypes of binary32 and wider by C's conversion, the
   narrower through [narrow_odd]. */
template <class D, class S, category Scat> struct convert<D, S, FLOAT, Scat> {
  static DEVICE typename D::compute of(typename S::compute v) {
    if (sizeof(typename D::storage) >= 4) return (typename D::compute)v;
    return narrow_odd(v);
  }
};

template <class D, class S> struct convert<D, S, SINT, FLOAT> {
  static DEVICE typename D::compute of(typename S::compute v) {
    return to_sint<typename D::storage>((double)v);
  }
};

template <class D, class S> struct convert<D, S, UINT, FLOAT> {
  static DEVICE typename D::compute of(typename S::compute v) {
    return to_uint<typename D::storage>((double)v);
  }
};

template <class D, class S, category Scat> struct convert<D, S, SINT, Scat> {
  static DEVICE typename D::compute of(typename S::compute v) {
    return (typename D::compute)v;
  }
};

template <class D, class S, category Scat> struct convert<D, S, UINT, Scat> {
  static DEVICE typename D::compute of(typename S::compute v) {
    return (typename D::compute)v;
  }
};

template <class D, class S, category Scat> struct convert<D, S, BOOL, Scat> {
  static DEVICE typename D::compute of(typename S::compute v) { return v != 0; }
};

/* An element of [S] converted to [D]'s storage. */
template <class S, class D>
DEVICE typename D::storage cast(typename S::storage x) {
  return store<D>(convert<D, S, D::cat, S::cat>::of(S::load(x)));
}

/* Arithmetic shared by families */

/* The IEEE 754 maximum and minimum: NaN propagates, [a]'s when it is one, and
   -0 orders below +0; between equal operands, and-ing (maximum) or or-ing
   (minimum) their bits picks the zero of the right sign. */
template <class T, class U> DEVICE T fmax(T a, T b) {
  T g = (a > b || a != a) ? a : b;
  U gb = __builtin_bit_cast(U, g), ab = __builtin_bit_cast(U, a);
  U tie = (U)0 - (U)(a == b);
  return __builtin_bit_cast(T, (U)(gb & (ab | ~tie)));
}

template <class T, class U> DEVICE T fmin(T a, T b) {
  T g = (a < b || a != a) ? a : b;
  U gb = __builtin_bit_cast(U, g), ab = __builtin_bit_cast(U, a);
  U tie = (U)0 - (U)(a == b);
  return __builtin_bit_cast(T, (U)(gb | (ab & tie)));
}

/* The greater and the lesser of [a] and [b] at a compute type, as IEEE 754
   maximum and minimum for floats; for booleans, their or and and. */
DEVICE float maximum(float a, float b) { return fmax<float, uint32_t>(a, b); }
DEVICE double maximum(double a, double b) {
  return fmax<double, uint64_t>(a, b);
}
template <class T> DEVICE T maximum(T a, T b) { return a > b ? a : b; }
DEVICE float minimum(float a, float b) { return fmin<float, uint32_t>(a, b); }
DEVICE double minimum(double a, double b) {
  return fmin<double, uint64_t>(a, b);
}
template <class T> DEVICE T minimum(T a, T b) { return a < b ? a : b; }

/* [a] and [b] added and multiplied at a compute type, integers in the
   unsigned width, so that they wrap. */
DEVICE float add(float a, float b) { return a + b; }
DEVICE double add(double a, double b) { return a + b; }
DEVICE int64_t add(int64_t a, int64_t b) {
  return (int64_t)((uint64_t)a + (uint64_t)b);
}
DEVICE uint64_t add(uint64_t a, uint64_t b) { return a + b; }
DEVICE float mul(float a, float b) { return a * b; }
DEVICE double mul(double a, double b) { return a * b; }
DEVICE int64_t mul(int64_t a, int64_t b) {
  return (int64_t)((uint64_t)a * (uint64_t)b);
}
DEVICE uint64_t mul(uint64_t a, uint64_t b) { return a * b; }

/* The extremes of a compute type, which seed max and min. */
template <class A> DEVICE A lowest();
template <class A> DEVICE A highest();
template <> DEVICE float lowest<float>() { return -__builtin_inff(); }
template <> DEVICE float highest<float>() { return __builtin_inff(); }
template <> DEVICE double lowest<double>() { return -__builtin_inf(); }
template <> DEVICE double highest<double>() { return __builtin_inf(); }
template <> DEVICE int64_t lowest<int64_t>() { return INT64_MIN; }
template <> DEVICE int64_t highest<int64_t>() { return INT64_MAX; }
template <> DEVICE uint64_t lowest<uint64_t>() { return 0; }
template <> DEVICE uint64_t highest<uint64_t>() { return UINT64_MAX; }
template <> DEVICE uint8_t lowest<uint8_t>() { return 0; }
template <> DEVICE uint8_t highest<uint8_t>() { return UINT8_MAX; }
