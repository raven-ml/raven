/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c.h — the dtype table and ABIs for the backend_c CPU backend.

   This is the single source of truth below the kernel line. From one X-macro
   table it generates the dtype enum, the element-size and class tables, and
   the per-dtype load/store and saturating float->int converters. It also
   defines the metadata struct crossing the FFI, the inner-loop kernel ABIs and
   their dispatch-table types, the status protocol, and the parallel-policy
   declarations. Everything else in the backend is generated from or built
   against this file; see README.md for the maintained architecture.

   Layering: this header pulls in the caml value header, nx.device's buffer
   accessor and nx.dtype's f16/bf16/fp8 converters, but NOT
   caml/fail.h or caml/threads.h. A translation unit that includes only this
   header therefore *cannot* raise an OCaml exception or touch the runtime
   lock — the "kernels never call the runtime" rule is enforced by what is
   reachable, not by convention. Only the engine's funnel (which includes the
   raise headers itself) may raise. */

#ifndef NX_C_C_H
#define NX_C_C_H

#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <malloc.h>
#endif

#include <caml/mlvalues.h>

#include "nx_device.h" /* buffer host addresses */
#include "nx_dtype.h"  /* f16/bf16/fp8 converters */

/* Prefixed to avoid colliding with backend_c's unprefixed complex typedefs
   should a translation unit ever pull in both backends' headers. */
typedef float _Complex nx_c_complex32;
typedef double _Complex nx_c_complex64;

/* mingw-w64's <complex.h> lacks C11's CMPLX constructors. The builtin is exact
   where `re + I * im` is not: an infinite or NaN imaginary part stays put. */
#ifndef CMPLX
#define CMPLX(re, im) __builtin_complex((double)(re), (double)(im))
#endif
#ifndef CMPLXF
#define CMPLXF(re, im) __builtin_complex((float)(re), (float)(im))
#endif

/* 64-byte (cache-line) aligned heap blocks for packing panels and per-worker
   scratch. Windows has no aligned allocation that free() accepts, so both ends
   go through this pair; nx_c_parallel_for's free_on_exit is always one of
   these. Zero rounds up to one line; the size is rounded to a multiple of the
   alignment as C11 aligned_alloc requires. Either function accepts NULL. */
static inline void *nx_c_aligned_alloc(size_t bytes) {
  if (bytes == 0) bytes = 64;
  bytes = (bytes + 63u) & ~(size_t)63u;
#if defined(_WIN32)
  return _aligned_malloc(bytes, 64);
#else
  return aligned_alloc(64, bytes);
#endif
}

static inline void nx_c_aligned_free(void *p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  free(p);
#endif
}

/* Highest tensor rank the backend accepts (test_nx_basics.ml exercises a
   rank-32 tensor). Enforced once, at extraction, into the caller-stack arrays
   in nx_c_ndarray — no dynamic allocation on the FFI path. */
#define NX_C_MAX_NDIM 32

/* An operand is an Nx_array.t, read at fixed slots, and those slots ARE ABI:
       Nx_array.t  0 dtype    Nx_dtype.t, a constant constructor whose index
                              is the tag
                   1 view     View.t
                   2 buffer   a host Nx_device.Buffer.t
       View.t      0 shape    int array
                   1 strides  int array, ELEMENT units
                   2 offset   int, ELEMENT units
   Both OCaml declarations say so: reordering their fields misreads every
   operand. */
#define NX_C_FFI_DTYPE 0
#define NX_C_FFI_VIEW 1
#define NX_C_FFI_DATA 2
#define NX_C_FFI_VIEW_SHAPE 0
#define NX_C_FFI_VIEW_STRIDES 1
#define NX_C_FFI_VIEW_OFFSET 2

#if defined(__GNUC__) || defined(__clang__)
#define NX_C_NORETURN __attribute__((noreturn))
#else
#define NX_C_NORETURN
#endif

/* ── The dtype table ──────────────────────────────────────────────────────

   The one place a dtype is described. Adding a dtype is exactly one new row.

   ONE table in Nx_dtype.t declaration order, so the generated enum values
   equal its constructor indices (0=Float16 … 18=Bool) and NX_C_DTYPE_COUNT
   falls out as the trailing enumerator — the correspondence is pinned by a
   _Static_assert below and each row's facts by the binding's dtype test, never
   by hand. Two iterators project the single table: NX_C_FOR_EACH_DTYPE walks
   all 19 rows (enum, class, size); NX_C_FOR_EACH_COMPUTE_DTYPE walks only the
   compute rows (load/store, float->int, kernel dispatch tables) — packed rows
   expand to nothing via the `sel` selector, so no compute code is ever
   emitted for int4/uint4 and no second list exists to drift.

   Row: X(A, suffix, storage, compute, load, store, cat, sel)
     A        threaded generator (supplied by the iterator, not by callers).
     suffix   identifier tail; yields NX_C_DTYPE_<suffix>, nx_c_ld_<suffix>, ...
     storage  C type of one stored element. Element size is sizeof(storage)
              (packed rows report 0), so there is no separate, driftable size.
     compute  C type kernels compute in. Small ints widen so wrap-on-store
              gives modular semantics and reductions gain headroom; f16/bf16/
              fp8 compute in float. `void` on packed rows (never instantiated).
     load     storage-value -> compute-value converter (NX_C_ID, an
              nx_dtype.h converter, or the bool normalizer). Never
              redefine the buffer converters.
     store    compute-value -> storage-value converter.
     cat      category as a BARE token (not a macro, so it survives argument
              prescan for pasting): NX_C_CAT_SINT / NX_C_CAT_UINT / NX_C_CAT_FLOAT
              / NX_C_CAT_COMPLEX / NX_C_CAT_BOOL. Drives the class bitmask and the
              signed/unsigned float->int converter — category lives once.
     sel      NX_C_COMPUTE or NX_C_PACKED — the single source of packed-ness. It
              drives the compute/full iterator split, the size zeroing, and the
              NX_C_CLASS_PACKED bit.

   X-macro hygiene: every expansion site does `#define <GEN>` before the
   iterator and `#undef <GEN>` right after; columns used in expressions are
   parenthesized at the use site.

   Compute widths: signed ints and u8/u16 widen to int64_t (single products
   fit; wrap-on-store gives modular semantics). u32/u64 widen to uint64_t so
   modular multiply has no signed-overflow UB. Caveat: SUMS of i32 products
   (and i64 arithmetic generally) can overflow int64_t — signed overflow is
   UB in principle; two's-complement wrap in practice, matching the wrap the
   references expect. Kernels accumulating many products (matmul, reduce)
   inherit this; near-2^31-magnitude i32 inputs at k>=2 are the exposed
   regime, matching numpy's int64-accumulate behavior only below it. */

#define NX_C_CLASS_INT 0x01
#define NX_C_CLASS_FLOAT 0x02
#define NX_C_CLASS_COMPLEX 0x04
#define NX_C_CLASS_BOOL 0x08
#define NX_C_CLASS_PACKED 0x10

/* Identity converter for native-storage dtypes; bool normalizers guarantee the
   stored-0/1 invariant on both directions (nonzero -> 1). */
#define NX_C_ID(x) (x)
#define NX_C_BOOL_LD(x) ((x) != 0)
#define NX_C_BOOL_ST(x) ((x) != 0)

#define NX_C_DTYPE_TABLE(X, A)                                                  \
  X(A, f16, uint16_t, float, half_to_float, float_to_half,                     \
    NX_C_CAT_FLOAT, NX_C_COMPUTE)                                                \
  X(A, f32, float, float, NX_C_ID, NX_C_ID, NX_C_CAT_FLOAT,                       \
    NX_C_COMPUTE)                                                               \
  X(A, f64, double, double, NX_C_ID, NX_C_ID, NX_C_CAT_FLOAT,                     \
    NX_C_COMPUTE)                                                               \
  X(A, bf16, uint16_t, float, bfloat16_to_float,                               \
    float_to_bfloat16, NX_C_CAT_FLOAT, NX_C_COMPUTE)                             \
  X(A, f8e4m3, uint8_t, float, fp8_e4m3_to_float,                              \
    float_to_fp8_e4m3, NX_C_CAT_FLOAT, NX_C_COMPUTE)                             \
  X(A, f8e5m2, uint8_t, float, fp8_e5m2_to_float,                              \
    float_to_fp8_e5m2, NX_C_CAT_FLOAT, NX_C_COMPUTE)                             \
  X(A, i4, uint8_t, void, NX_C_ID, NX_C_ID, NX_C_CAT_SINT,                        \
    NX_C_PACKED)                                                                \
  X(A, u4, uint8_t, void, NX_C_ID, NX_C_ID, NX_C_CAT_UINT,                        \
    NX_C_PACKED)                                                                \
  X(A, i8, int8_t, int64_t, NX_C_ID, NX_C_ID, NX_C_CAT_SINT,                      \
    NX_C_COMPUTE)                                                               \
  X(A, u8, uint8_t, int64_t, NX_C_ID, NX_C_ID, NX_C_CAT_UINT,                     \
    NX_C_COMPUTE)                                                               \
  X(A, i16, int16_t, int64_t, NX_C_ID, NX_C_ID, NX_C_CAT_SINT,                    \
    NX_C_COMPUTE)                                                               \
  X(A, u16, uint16_t, int64_t, NX_C_ID, NX_C_ID, NX_C_CAT_UINT,                   \
    NX_C_COMPUTE)                                                               \
  X(A, i32, int32_t, int64_t, NX_C_ID, NX_C_ID, NX_C_CAT_SINT,                    \
    NX_C_COMPUTE)                                                               \
  X(A, u32, uint32_t, uint64_t, NX_C_ID, NX_C_ID,                                \
    NX_C_CAT_UINT, NX_C_COMPUTE)                                                 \
  X(A, i64, int64_t, int64_t, NX_C_ID, NX_C_ID, NX_C_CAT_SINT,                    \
    NX_C_COMPUTE)                                                               \
  X(A, u64, uint64_t, uint64_t, NX_C_ID, NX_C_ID,                                \
    NX_C_CAT_UINT, NX_C_COMPUTE)                                                 \
  X(A, c32, nx_c_complex32, nx_c_complex32, NX_C_ID, NX_C_ID,                      \
    NX_C_CAT_COMPLEX, NX_C_COMPUTE)                                              \
  X(A, c64, nx_c_complex64, nx_c_complex64, NX_C_ID, NX_C_ID,                      \
    NX_C_CAT_COMPLEX, NX_C_COMPUTE)                                              \
  X(A, bool_, uint8_t, uint8_t, NX_C_BOOL_LD, NX_C_BOOL_ST,                      \
    NX_C_CAT_BOOL, NX_C_COMPUTE)

/* Full iteration: G receives all seven columns (including cat and sel). */
#define NX_C_FULL(G, sfx, storage, compute, ld, st, cat, sel)                  \
  G(sfx, storage, compute, ld, st, cat, sel)
#define NX_C_FOR_EACH_DTYPE(G) NX_C_DTYPE_TABLE(NX_C_FULL, G)

/* Compute-only iteration: packed rows expand to nothing; G receives the first
   six columns (sel is consumed by the filter). */
#define NX_C_FILTER(G, sfx, storage, compute, ld, st, cat, sel)                \
  NX_C_FILTER_##sel(G, sfx, storage, compute, ld, st, cat)
#define NX_C_FILTER_NX_C_COMPUTE(G, sfx, storage, compute, ld, st, cat)          \
  G(sfx, storage, compute, ld, st, cat)
#define NX_C_FILTER_NX_C_PACKED(G, sfx, storage, compute, ld, st, cat)
#define NX_C_FOR_EACH_COMPUTE_DTYPE(G) NX_C_DTYPE_TABLE(NX_C_FILTER, G)

/* Dense dtype enum in tag order. NX_C_DTYPE_COUNT is the slot count. */
typedef enum {
#define NX_C_ENUM_ROW(sfx, storage, compute, ld, st, cat, sel)                 \
  NX_C_DTYPE_##sfx,
  NX_C_FOR_EACH_DTYPE(NX_C_ENUM_ROW)
#undef NX_C_ENUM_ROW
      NX_C_DTYPE_COUNT
} nx_c_dtype;

/* nx_c_dtype must equal Nx_dtype.t's constructor index: every row is pinned at
   compile time. test/test_backend_c.ml checks each row's size, class and
   signedness against Nx_dtype. */
_Static_assert(NX_C_DTYPE_f16 == 0 && NX_C_DTYPE_f32 == 1 &&
                   NX_C_DTYPE_f64 == 2 && NX_C_DTYPE_bf16 == 3 &&
                   NX_C_DTYPE_f8e4m3 == 4 && NX_C_DTYPE_f8e5m2 == 5 &&
                   NX_C_DTYPE_i4 == 6 && NX_C_DTYPE_u4 == 7 &&
                   NX_C_DTYPE_i8 == 8 && NX_C_DTYPE_u8 == 9 &&
                   NX_C_DTYPE_i16 == 10 && NX_C_DTYPE_u16 == 11 &&
                   NX_C_DTYPE_i32 == 12 && NX_C_DTYPE_u32 == 13 &&
                   NX_C_DTYPE_i64 == 14 && NX_C_DTYPE_u64 == 15 &&
                   NX_C_DTYPE_c32 == 16 && NX_C_DTYPE_c64 == 17 &&
                   NX_C_DTYPE_bool_ == 18 && NX_C_DTYPE_COUNT == 19,
               "nx_c_dtype must equal Nx_dtype.t's constructor index");

/* ── Per-dtype load/store and saturating float->int (compute dtypes only) ──

   nx_c_ld_<suffix>(p)    reads one stored element at byte pointer p and returns
                         it in the dtype's compute type.
   nx_c_st_<suffix>(p, v) converts compute value v to storage and writes it at
                         byte pointer p (wrapping for integers — the modular
                         store; NOT for a float source, see nx_c_f2i below).
   nx_c_f2i_<suffix>(v)   converts double v to the target int storage with
                         saturation: NaN -> 0, +/-inf and out-of-range clamp to
                         the dtype's [min, max]. This is the ONLY correct
                         float->int narrowing (a plain (int)double is UB on
                         NaN/inf/overflow); nx_c_cast MUST use it and is the sole
                         owner, so wrap-vs-saturate cannot diverge across ops.
   Pointers are plain byte addresses so kernels walk arbitrary strides;
   elements are naturally aligned (offsets are element multiples of an aligned
   base), so the aliased access is well-defined. Unused instances are
   `static inline`, so they draw no warnings. */
#define NX_C_LDST_ROW(sfx, storage, compute, ld, st, cat)                      \
  static inline compute nx_c_ld_##sfx(const void *p) {                          \
    return (compute)(ld(*(const storage *)(p)));                              \
  }                                                                            \
  static inline void nx_c_st_##sfx(void *p, compute v) {                        \
    *(storage *)(p) = (storage)(st(v));                                        \
  }
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_LDST_ROW)
#undef NX_C_LDST_ROW

/* Saturating float->int, emitted per category (int rows only), signed and
   unsigned handled separately so no runtime branch and no per-width special
   case: the thresholds 2^(w-1) and 2^w are exact doubles for every width, so
   the 2^63 / 2^64 boundary that makes (int64_t)double UB is caught before the
   cast. */
#define NX_C_F2I_NX_C_CAT_SINT(sfx, storage)                                    \
  static inline storage nx_c_f2i_##sfx(double v) {                             \
    double lim = ldexp(1.0, (int)(sizeof(storage) * 8 - 1)); /* 2^(w-1) */    \
    if (isnan(v)) return 0;                                                   \
    if (v <= -lim) return (storage)((uintmax_t)1 << (sizeof(storage) * 8 - 1)); \
    if (v >= lim)                                                             \
      return (storage)(((uintmax_t)1 << (sizeof(storage) * 8 - 1)) - 1);      \
    return (storage)v;                                                        \
  }
#define NX_C_F2I_NX_C_CAT_UINT(sfx, storage)                                    \
  static inline storage nx_c_f2i_##sfx(double v) {                             \
    double lim = ldexp(1.0, (int)(sizeof(storage) * 8)); /* 2^w */            \
    if (isnan(v)) return 0;                                                   \
    if (v <= 0.0) return 0;                                                   \
    if (v >= lim) return (storage)(~(uintmax_t)0);                            \
    return (storage)v;                                                        \
  }
#define NX_C_F2I_NX_C_CAT_FLOAT(sfx, storage)
#define NX_C_F2I_NX_C_CAT_COMPLEX(sfx, storage)
#define NX_C_F2I_NX_C_CAT_BOOL(sfx, storage)
#define NX_C_F2I_ROW(sfx, storage, compute, ld, st, cat)                       \
  NX_C_F2I_##cat(sfx, storage)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_F2I_ROW)
#undef NX_C_F2I_ROW

/* ── Derived dtype accessors ──────────────────────────────────────────────
   All generated from the one table, keeping size handling in one place. dt is
   bounds-checked as `(unsigned)dt < COUNT`, which rejects negatives and
   out-of-range in one comparison with no -Wtype-limits risk. */

static inline int nx_c_dtype_class(nx_c_dtype dt) {
  static const int classes[NX_C_DTYPE_COUNT] = {
#define NX_C_CATBITS_NX_C_CAT_SINT NX_C_CLASS_INT
#define NX_C_CATBITS_NX_C_CAT_UINT NX_C_CLASS_INT
#define NX_C_CATBITS_NX_C_CAT_FLOAT NX_C_CLASS_FLOAT
#define NX_C_CATBITS_NX_C_CAT_COMPLEX NX_C_CLASS_COMPLEX
#define NX_C_CATBITS_NX_C_CAT_BOOL NX_C_CLASS_BOOL
#define NX_C_PACKEDBIT_NX_C_COMPUTE 0
#define NX_C_PACKEDBIT_NX_C_PACKED NX_C_CLASS_PACKED
#define NX_C_CLASS_ROW(sfx, storage, compute, ld, st, cat, sel)                \
  [NX_C_DTYPE_##sfx] = NX_C_CATBITS_##cat | NX_C_PACKEDBIT_##sel,
      NX_C_FOR_EACH_DTYPE(NX_C_CLASS_ROW)
#undef NX_C_CLASS_ROW
#undef NX_C_CATBITS_NX_C_CAT_SINT
#undef NX_C_CATBITS_NX_C_CAT_UINT
#undef NX_C_CATBITS_NX_C_CAT_FLOAT
#undef NX_C_CATBITS_NX_C_CAT_COMPLEX
#undef NX_C_CATBITS_NX_C_CAT_BOOL
#undef NX_C_PACKEDBIT_NX_C_COMPUTE
#undef NX_C_PACKEDBIT_NX_C_PACKED
  };
  return ((unsigned)dt < NX_C_DTYPE_COUNT) ? classes[dt] : 0;
}

static inline bool nx_c_dtype_is_packed(nx_c_dtype dt) {
  return (nx_c_dtype_class(dt) & NX_C_CLASS_PACKED) != 0;
}
static inline bool nx_c_dtype_is_int(nx_c_dtype dt) {
  return (nx_c_dtype_class(dt) & NX_C_CLASS_INT) != 0;
}
static inline bool nx_c_dtype_is_float(nx_c_dtype dt) {
  return (nx_c_dtype_class(dt) & NX_C_CLASS_FLOAT) != 0;
}
static inline bool nx_c_dtype_is_complex(nx_c_dtype dt) {
  return (nx_c_dtype_class(dt) & NX_C_CLASS_COMPLEX) != 0;
}
static inline bool nx_c_dtype_is_bool(nx_c_dtype dt) {
  return (nx_c_dtype_class(dt) & NX_C_CLASS_BOOL) != 0;
}

/* Bytes of one element. Packed dtypes report 0 (a sub-byte element has no byte
   size); use nx_c_dtype_is_packed and nx_c_dtype_bytes for their extent. 0 is a
   poison value: an engine that ever fed a packed dtype through here would
   produce a zero-length run and fail immediately rather than subtly. */
static inline int64_t nx_c_elem_size(nx_c_dtype dt) {
  static const int64_t sizes[NX_C_DTYPE_COUNT] = {
#define NX_C_SIZE_NX_C_COMPUTE(storage) (int64_t)sizeof(storage)
#define NX_C_SIZE_NX_C_PACKED(storage) 0
#define NX_C_SIZE_ROW(sfx, storage, compute, ld, st, cat, sel)                 \
  [NX_C_DTYPE_##sfx] = NX_C_SIZE_##sel(storage),
      NX_C_FOR_EACH_DTYPE(NX_C_SIZE_ROW)
#undef NX_C_SIZE_ROW
#undef NX_C_SIZE_NX_C_COMPUTE
#undef NX_C_SIZE_NX_C_PACKED
  };
  return ((unsigned)dt < NX_C_DTYPE_COUNT) ? sizes[dt] : 0;
}

/* Byte extent of `count` contiguous elements — the one place packed nibble
   arithmetic lives (two elements per byte, rounded up). */
static inline int64_t nx_c_dtype_bytes(nx_c_dtype dt, int64_t count) {
  if (nx_c_dtype_is_packed(dt)) return (count + 1) / 2;
  return count * nx_c_elem_size(dt);
}

/* ── Dtype semantics the kernel families must honor ───────────────────────

   These are policy, stated once here; the conformance suite encodes them.

   - Integer store is modular (wrap): nx_c_st_<int> truncates to the storage
     width. Correct for int->int cast and integer arithmetic.
   - Float->int cast is the ONLY float-source narrowing and MUST go through
     nx_c_f2i_<dst> (NaN -> 0, +/-inf and out-of-range clamp to range). It lives
     entirely in nx_c_cast — the single owner — so wrap-vs-saturate cannot
     diverge across operations.
   - bool storage is 0/1: nx_c_ld_/nx_c_st_bool_ normalize (nonzero -> 1), and
     bool participates only in logical/comparison/min/max/where/select — all
     0/1-preserving. Arithmetic that could break the invariant is promoted away
     by the frontend and never reaches a bool kernel.
   - Integer div/mod/recip by zero return 0 (total, never trap).
   - Small-int reductions accumulate in 64-bit (the compute widths above);
     bool folds as 0/1 bytes; f16/bf16/fp8 accumulate in float.
   - Complex has no mod and no ordered comparison; rounding/abs/sign on complex
     are the kernel's concern (rejected loudly, never identity), not the ABI's.
   - Float max/min, elementwise, reduced or scanned, are IEEE 754-2019 maximum
     and minimum: NaN propagates and -0 orders below +0. A NaN result is the
     first NaN met, the left operand's elementwise and the earliest in a
     reduction's or a scan's order, so every grouping of a reduction gives the
     same bits. argmax/argmin and sort order the zeros the same way, and
     argmax/argmin find the first NaN.
   - A float sum (a reduction, a scan, scatter's additions, fold's overlaps, a
     matmul's contraction) is +0 plus its terms, so one that is exactly zero is
     +0 whatever the association and the layout. */

/* ── Float extremes ───────────────────────────────────────────────────────

   nx_c_fmax and nx_c_fmin are IEEE 754-2019 maximum and minimum of two floats
   or two doubles: a NaN operand gives a NaN (a's when a is one), and -0 orders
   below +0. A select takes the greater operand, or a when it is NaN; between
   equal operands, and-ing (max) or or-ing (min) their bits picks the zero of
   the right sign and leaves equal nonzero values as they are. Selects and bit
   masks leave no branch, so a loop over independent lanes vectorizes. */
#define NX_C_DEFINE_FEXTREMES(T, U)                                            \
  static inline T nx_c_fmax_##T(T a, T b) {                                    \
    T g = (a > b || a != a) ? a : b;                                           \
    U gb, ab, tie = (U)0 - (U)(a == b);                                        \
    memcpy(&gb, &g, sizeof g);                                                 \
    memcpy(&ab, &a, sizeof a);                                                 \
    gb &= ab | ~tie;                                                           \
    memcpy(&g, &gb, sizeof g);                                                 \
    return g;                                                                  \
  }                                                                            \
  static inline T nx_c_fmin_##T(T a, T b) {                                    \
    T g = (a < b || a != a) ? a : b;                                           \
    U gb, ab, tie = (U)0 - (U)(a == b);                                        \
    memcpy(&gb, &g, sizeof g);                                                 \
    memcpy(&ab, &a, sizeof a);                                                 \
    gb |= ab & tie;                                                            \
    memcpy(&g, &gb, sizeof g);                                                 \
    return g;                                                                  \
  }
NX_C_DEFINE_FEXTREMES(float, uint32_t)
NX_C_DEFINE_FEXTREMES(double, uint64_t)
#undef NX_C_DEFINE_FEXTREMES
#define nx_c_fmax(a, b)                                                        \
  _Generic((a), float: nx_c_fmax_float, double: nx_c_fmax_double)(a, b)
#define nx_c_fmin(a, b)                                                        \
  _Generic((a), float: nx_c_fmin_float, double: nx_c_fmin_double)(a, b)

/* ── Associations ─────────────────────────────────────────────────────────

   Lanes. A float sum over a stretch of terms (a piece of a reduction's run, a
   dot's chunk) keeps NX_C_LANES partial sums: term p of the stretch goes to
   lane p mod NX_C_LANES whatever the stride, and NX_C_LANE_TREE combines the
   lanes by one fixed balanced tree. A contiguous stretch vectorizes over
   independent accumulators. LANE(i) names lane i and ADD is the compute type's
   addition.

   The matmul's dot-shaped paths (the split path, a 1x1 output among them; the
   row path; the direct loop) share one order for an output: its contraction
   in fixed chunks of MM_DOT_CHUNK (65536) elements, each chunk in these lanes
   and this tree, the chunks added in order. Each such output has the bits of
   the dot of its row and column. The blocked kernel, which sums along k in
   order, and Accelerate on macOS do not.

   Every other kernel that combines elements fixes its order from its
   operands' shapes and layouts, never from the number of threads that run it;
   only Accelerate makes no such promise. Integer sums and products are
   modular and float max/min keep the first NaN, so those give the same bits
   under every grouping, and the orders below decide the bits of float and
   complex sums and products.

   Reductions (nx_c_fold_run). The fold driver orders the reduced axes by the
   layout and chooses one of two paths by the layout (nx_c_engine.h). On both,
   an output's terms, in the order of the ordered reduced axes, are cut into
   blocks of NX_C_FOLD_BLOCK consecutive terms, the last shorter, and the
   blocks' results combine in order by the left-complete binary tree: a power
   of two of them pairs neighbours level by level, and any other count splits
   after the largest power of two below it, the left part complete. A binary
   counter computes it in one pass, combining the two newest results once for
   each trailing zero of the number of blocks done.
   - On the per-output path, a block is folded from the identity by one step
     per piece of a run it holds. A float sum folds a piece in lanes; complex
     sums and the other ops fold it in order.
   - On the streaming path, a block is NX_C_FOLD_BLOCK consecutive rows, and it
     folds each output's terms in row order.
   A block's sum starts from +0 and is never -0, so a float sum that is exactly
   zero is +0 on both paths.

   Scans (nx_c_scan_run). A slice is cut into chunks of NX_C_SCAN_CHUNK
   elements counted from its start. Its first chunk is scanned in order from
   the identity. Every later chunk is rescanned in order from its carry, the
   in-order combination of the totals of the chunks before it, each total
   folded in order from the identity. So an output's association depends only
   on its position, and the scan of a prefix of a slice is the prefix of the
   slice's scan. An integer or bool scan, exact under every grouping, is
   walked in one chunk.

   Scatter (nx_c_move.c). Updates apply one at a time in the row-major order
   of their index space: under Set the last update to a position wins, and
   under Add a position that an update reaches holds +0 plus its value, then
   each of its updates added in that order; one that none reaches keeps its
   value. float16, bfloat16 and the float8 dtypes round to
   storage after every update.

   NX_C_FOLD_BLOCK and NX_C_SCAN_CHUNK fix bits outside nx's contract, which
   leaves sum's association unspecified: changing either changes bits and
   nothing a caller may rely on. */
#define NX_C_LANES 16
#define NX_C_LANE_TREE(LANE, ADD)                                             \
  ADD(ADD(ADD(ADD(LANE(0), LANE(1)), ADD(LANE(2), LANE(3))),                  \
          ADD(ADD(LANE(4), LANE(5)), ADD(LANE(6), LANE(7)))),                 \
      ADD(ADD(ADD(LANE(8), LANE(9)), ADD(LANE(10), LANE(11))),                \
          ADD(ADD(LANE(12), LANE(13)), ADD(LANE(14), LANE(15)))))
_Static_assert(NX_C_LANES == 16, "NX_C_LANE_TREE combines sixteen lanes");
#define NX_C_FOLD_BLOCK 1024
#define NX_C_SCAN_CHUNK 4096

/* ── Status protocol ──────────────────────────────────────────────────────

   A status is NULL on success, otherwise a static, never-freed string. No
   status string is ever heap-allocated, so no error path can leak. Kernels and
   drivers return status; ONLY the engine's funnel turns a non-NULL status into
   an OCaml exception, and only with the runtime lock held. Nothing aborts: a
   precondition believed unreachable is still a status, since a raised error
   names what failed and leaves the process running, where abort() takes the
   host program down without a trace.

   Testing a status against a specific error compares by CONTENT (strcmp), or
   routes it to nx_c_raise_status (nx_c_engine.h) — never by pointer: identical
   string literals are not pooled across translation units, so `status ==
   NX_C_ERR_X` is unspecified. Success is still the pointer test `status ==
   NX_C_OK` (i.e. == NULL). */
typedef const char *nx_c_status;
#define NX_C_OK ((nx_c_status)NULL)

#define NX_C_ERR_NDIM "ndim exceeds NX_C_MAX_NDIM"
#define NX_C_ERR_RANK_MISMATCH "shape and strides rank disagree"
#define NX_C_ERR_UNSUPPORTED_DTYPE "dtype not supported for this operation"
#define NX_C_ERR_PACKED "packed dtype not supported for this operation"
#define NX_C_ERR_SHAPE "shape mismatch"
#define NX_C_ERR_EMPTY_REDUCE "reduction over empty axis has no identity"
#define NX_C_ERR_ALLOC "out of memory"
/* The operand's buffer was consumed: the raisers raise Invalid_argument with
   the reason nx_c_consumed holds. */
#define NX_C_ERR_CONSUMED "consumed buffer"

/* Why the operand whose status is NX_C_ERR_CONSUMED was consumed, per thread:
   nx_c_ndarray_of_value copies it here, as the reason is an OCaml string that
   the raise's allocation may move. */
extern _Thread_local char nx_c_consumed[256];

/* Funnel raisers, implemented in nx_c_engine.c. Call ONLY with the runtime lock
   held (before caml_enter_blocking_section, or after re-acquiring). The op name
   is prefixed to the message. */
NX_C_NORETURN void nx_c_raise(const char *op, nx_c_status status);
NX_C_NORETURN void nx_c_raise_invalid(const char *op, nx_c_status status);

/* ── ndarray metadata ─────────────────────────────────────────────────────

   Operand metadata after extraction from the FFI record. shape/strides live
   inline in caller-stack storage bounded by NX_C_MAX_NDIM (no malloc, so no leak
   on any error path; ~536 bytes per operand). strides and offset are in ELEMENT
   units, exactly as OCaml provides; the engine multiplies by nx_c_elem_size to
   get the byte steps the kernel ABI wants — that conversion happens in exactly
   one place. Entries [ndim, NX_C_MAX_NDIM) are unspecified; consumers read only
   [0, ndim). `data` is the buffer's first byte; the first live element is at
   data + offset*elem_size. */
typedef struct {
  void *data;
  int ndim;
  int64_t shape[NX_C_MAX_NDIM];
  int64_t strides[NX_C_MAX_NDIM];
  int64_t offset;
} nx_c_ndarray;

/* Extract operand metadata from an FFI record value into `out`.

   Runs with the runtime lock held, BEFORE the blocking section. Reads OCaml
   fields but performs no allocation and enters no GC point, so `v` (already
   rooted by the funnel) needs no local rooting here. Never raises — validates
   rank cheaply and reports via status; the funnel raises on non-NULL. */
static inline nx_c_status nx_c_ndarray_of_value(value v, nx_c_ndarray *out) {
  value v_buffer = Field(v, NX_C_FFI_DATA);
  if (!nx_device_buffer_live(v_buffer)) {
    snprintf(nx_c_consumed, sizeof nx_c_consumed, "%s",
             nx_device_buffer_why(v_buffer));
    return NX_C_ERR_CONSUMED;
  }
  value v_view = Field(v, NX_C_FFI_VIEW);
  value v_shape = Field(v_view, NX_C_FFI_VIEW_SHAPE);
  value v_strides = Field(v_view, NX_C_FFI_VIEW_STRIDES);
  int ndim = (int)Wosize_val(v_shape);
  if (ndim > NX_C_MAX_NDIM) return NX_C_ERR_NDIM;
  if ((int)Wosize_val(v_strides) != ndim) return NX_C_ERR_RANK_MISMATCH;
  out->data = nx_device_buffer_host(v_buffer);
  out->ndim = ndim;
  out->offset = Long_val(Field(v_view, NX_C_FFI_VIEW_OFFSET));
  for (int i = 0; i < ndim; i++) {
    out->shape[i] = Long_val(Field(v_shape, i));
    out->strides[i] = Long_val(Field(v_strides, i));
  }
  return NX_C_OK;
}

/* Dtype of an FFI operand, its constructor index. Lock held, no allocation. */
static inline nx_c_dtype nx_c_dtype_of_value(value v) {
  return (nx_c_dtype)Long_val(Field(v, NX_C_FFI_DTYPE));
}

/* ── Kernel ABIs ──────────────────────────────────────────────────────────

   Kernels are inner loops over one 1-D run, not whole operations. The engine
   owns coalescing, strategy, threading, and the funnel; a kernel states only
   scalar semantics over a run. Every pointer is a byte address; every step is a
   byte stride and MAY be 0 (a broadcast input repeats one element — output
   steps are never 0). `n` MAY be 0 (empty tensors), and every kernel must be a
   no-op then. Kernels never touch the OCaml runtime and never fail; op
   preconditions with an error (e.g. empty-axis min/max) are checked by the
   funnel before the blocking section. `ctx` is an opaque, op-defined parameter
   block (a fill value, a comparison mode, …) or NULL; its layout belongs to the
   kernel's own file, not this header.

   These ABIs cover the *generated* families (map, fold, argreduce, scan).
   The custom families — sort/argsort, gather/scatter, pad/cat, unfold, matmul,
   linalg, fft — own their own driver signatures in their own files: their
   access is data- or structure-dependent and does not ride a 1-D
   run. They still reuse the engine's extraction, dispatch, parallel policy, and
   status protocol. */

/* map — elementwise, no cross-element state (map1/map2/map3, cast, where).
   ptrs[0]/steps[0] are the output; ptrs[1..k]/steps[1..k] the k inputs, in
   argument order. One shape serves every arity: the kernel reads as many inputs
   as its op has. */
typedef void nx_c_map_loop(char *const *ptrs, const int64_t *steps, int64_t n,
                          void *ctx);

/* An accumulator wide enough for any fold or scan of any compute dtype: a
   single reduced value in the op's accumulate type. Small-int sums use `i`
   (int64), unsigned wide sums `u`, bool `b`, floats `f`/`d`, complex
   `c32`/`c64`: every accumulator holds its compute type at offset 0, which
   combine relies on. Within-run multi-accumulator unrolling is a kernel-local
   concern; only this one reduced value is carried between step() calls. */
typedef union {
  int64_t i;
  uint64_t u;
  uint8_t b;
  float f;
  double d;
  nx_c_complex32 c32;
  nx_c_complex64 c64;
} nx_c_acc;

/* fold — axis reduction (sum, prod, max, min), in blocks combined by a tree
   (Associations). The per-output path drives one output as:
       for each block of the output's terms, in order:
         init(acc, ctx);                    // op+dtype identity
         for each piece of a run in the block:
           step(acc, in, in_step, n, ctx);  // fold the piece's n terms
         push acc; for each trailing zero of the blocks done:
           combine(older, newer, 1, ctx);   // older = older ⊕ newer
       combine what remains, newest first; fini(out, 0, acc, 1, ctx);
   An output with no term stores init's identity through fini. step folds n
   terms of one strided run into whatever acc holds, a float sum in NX_C_LANES
   lanes and every other op in order. sum/prod seed the neutral identity
   (0/1); max/min have no neutral identity for an empty axis (the fold driver
   rejects that case before any kernel runs, gated by its no_identity flag
   which max/min stubs set true), and init seeds a sentinel extreme (±inf /
   INT64_MIN / INT64_MAX) so the first real element always wins.

   combine folds n compute values of `other` into n of `acc`, element by
   element: acc[j] = acc[j] ⊕ other[j], acc's terms preceding other's. It
   combines the block tree's results, one value on the per-output path and a
   row of accumulators on the streaming path, and a scan's carry with a
   chunk's total. acc and other point at n consecutive values of the op's
   compute type, which an nx_c_acc holds one of.

   The streaming path (nx_c_engine.h) serves a reduction across a kept axis
   more contiguous than every reduced axis, such as an axis-0 sum of a
   C-contiguous matrix, where the per-output path would gather each output's
   terms a fresh cache line apart. It walks the input a row at a time: a row
   is one point of the reduced axes, and its elements along the lane, the most
   contiguous kept axis, are one term of each of `n` outputs. Folding a row
   into `n` accumulators vectorizes across the lane, and every row is read
   whole. `accs` is `n` accumulators of the op's compute type, so f16/bf16/fp8
   keep their float accumulation and small ints their 64-bit accumulation, as
   on the per-output path.
     stream folds one row into the `n` accumulators. `first != 0` seeds them
       from this row as init then step would (accs[j] = the identity combined
       with load(in_row + j*lane_step)); the driver passes it at the first row
       of each block. `first == 0` folds the row in (accs[j] <combine>=
       load(in_row + j*lane_step)).

   fini converts `n` accumulators of the compute type to storage and writes
   them (out[j*out_step] = accs[j]): one output on the per-output path, a
   tile's row on the streaming path. It depends only on the dtype, so all four
   reduction tables share one instance per dtype. */
typedef void nx_c_fold_init(nx_c_acc *acc, void *ctx);
typedef void nx_c_fold_step(nx_c_acc *acc, const char *in, int64_t in_step,
                           int64_t n, void *ctx);
typedef void nx_c_fold_combine(void *acc, const void *other, int64_t n,
                              void *ctx);
typedef void nx_c_fold_fini(char *out, int64_t out_step, const void *accs,
                           int64_t n, void *ctx);
typedef void nx_c_fold_stream(void *accs, const char *in_row, int64_t lane_step,
                             int64_t n, int first, void *ctx);

/* argreduce — argmax/argmin over exactly one axis (backend_intf: single axis,
   int64 result). The accumulator carries the running extreme value and its
   index along that axis. init sets index = -1 ("unset"), so step takes the
   first element unconditionally and no per-dtype identity is needed; the empty
   axis is rejected by the funnel. There is exactly one run per output (the
   axis), so the kernel's element counter is the axis index directly. step
   encapsulates NaN-wins, first-index-wins comparison so argmax/argmin agree
   with reduce_max/min on NaN. fini writes the int64 index. */
typedef struct {
  nx_c_acc value;
  int64_t index;
} nx_c_arg_acc;

static inline void nx_c_arg_init(nx_c_arg_acc *acc) { acc->index = -1; }
static inline void nx_c_arg_fini(char *out, const nx_c_arg_acc *acc) {
  *(int64_t *)out = acc->index;
}
typedef void nx_c_arg_step(nx_c_arg_acc *acc, const char *in, int64_t in_step,
                          int64_t n, void *ctx);

/* scan — inclusive cumulative op (cumsum, cumprod, cummax, cummin) along one
   axis, in chunks (Associations). A scan is its op's fold table, for the
   identity and combine, and a step that stores running values:
       step(out, out_step, in, in_step, n, state, total, ctx);
   step reads in[k], folds it into *state and writes the running result to
   out[k]; when `total` is not NULL it also folds in[k] into *total. state and
   total carry across calls. Per slice, with id the op's identity from init,
   the driver runs:
       first chunk:           state = id; step(.., &state, NULL); carry = state
       later chunk, not last: state = carry; total = id;
                              step(.., &state, &total); combine(&carry, &total)
       last chunk, not first: state = carry; step(.., &state, NULL) */
typedef void nx_c_scan_step(char *out, int64_t out_step, const char *in,
                           int64_t in_step, int64_t n, nx_c_acc *state,
                           nx_c_acc *total, void *ctx);

/* ── Dispatch tables ──────────────────────────────────────────────────────

   Per-family kernel tables are indexed by nx_c_dtype and built with
   NX_C_FOR_EACH_COMPUTE_DTYPE (designated initializers), so packed and any op-
   unsupported dtypes are NULL slots. Invariant: kernel files never index these
   directly — the ENGINE's dispatch is the single place that reads a slot, tests
   it for NULL, and turns NULL into an NX_C_ERR_UNSUPPORTED_DTYPE / NX_C_ERR_PACKED
   status before the blocking section. A NULL slot is therefore never called
   (elem_size=0 poison does not help here — dispatch precedes any size math). */
typedef struct {
  nx_c_map_loop *fn[NX_C_DTYPE_COUNT];
} nx_c_map_table;
typedef struct {
  nx_c_fold_init *init[NX_C_DTYPE_COUNT];
  nx_c_fold_step *step[NX_C_DTYPE_COUNT];
  nx_c_fold_combine *combine[NX_C_DTYPE_COUNT];
  nx_c_fold_fini *fini[NX_C_DTYPE_COUNT];
  nx_c_fold_stream *stream[NX_C_DTYPE_COUNT];
} nx_c_fold_table;
typedef struct {
  nx_c_arg_step *step[NX_C_DTYPE_COUNT];
} nx_c_arg_table;
typedef struct {
  const nx_c_fold_table *op; /* the identity and combine of the same op */
  nx_c_scan_step *step[NX_C_DTYPE_COUNT];
} nx_c_scan_table;

/* ── Parallel policy ──────────────────────────────────────────────────────

   One function decides thread counts for the whole backend; its constant table
   encodes the lab/ Apple-Silicon findings (serial SIMD beats parallel-for below
   ~16M elements for bandwidth-bound work). Implemented in nx_c_engine.c. */
typedef enum {
  NX_C_COST_BANDWIDTH, /* memory-bound: copy, add, cast, fill */
  NX_C_COST_COMPUTE,   /* arithmetic-bound: pow, exp, trig, erf */
  NX_C_COST_HEAVY,     /* per-run heavy: sort, per-batch linalg */
} nx_c_cost_class;

/* Threads to use for `runs` independent outer iterations of `run_len` elements
   each, moving `bytes` total, given the op's cost class. Returns a count in
   [1, pool size]; 1 means run serially and skip the blocking-section handshake.
   Reads only its arguments and the engine's constant table. */
int nx_c_threads_for(nx_c_cost_class cls, int64_t runs, int64_t run_len,
                    int64_t bytes);

/* The thread count of a plan of `runs` independent units, for the families
   whose kernels combine elements (fold, argreduce, scan, sort): `threads`
   when it is positive, capped by the pool's size, and otherwise
   nx_c_threads_for's choice; in both cases between 1 and `runs`. A positive
   count skips the policy's serial floors, so a small operand runs on that many
   threads. The CAMLprims of those families take `threads` from OCaml as their
   last argument, and nx.cpu passes 0. No setting changes the pool.
   Implemented in nx_c_engine.c. */
int nx_c_plan_threads(int threads, nx_c_cost_class cls, int64_t runs,
                      int64_t run_len, int64_t bytes);

#endif /* NX_C_C_H */
