/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c.h — the dtype table and ABIs for the backend_c CPU backend.

   This is the single source of truth below the kernel line. From one X-macro
   table it generates the dtype enum, the kind translation, the element-size,
   class, per-dtype load/store, and saturating float->int converters. It also
   defines the metadata struct crossing the FFI, the inner-loop kernel ABIs and
   their dispatch-table types, the status protocol, and the parallel-policy
   declarations. Everything else in the backend is generated from or built
   against this file; see README.md for the maintained architecture.

   Layering: this header pulls in the caml value/bigarray headers and the
   buffer layer's extended kinds and f16/bf16/fp8 converters, but NOT
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

#include <caml/bigarray.h>
#include <caml/mlvalues.h>

#include "nx_buffer_stubs.h" /* extended kinds + f16/bf16/fp8 converters */

/* Prefixed to avoid colliding with backend_c's unprefixed complex typedefs
   should a translation unit ever pull in both backends' headers. */
typedef float _Complex nx_c_complex32;
typedef double _Complex nx_c_complex64;

/* Highest tensor rank the backend accepts (test_nx_basics.ml exercises a
   rank-32 tensor). Enforced once, at extraction, into the caller-stack arrays
   in nx_c_ndarray — no dynamic allocation on the FFI path. */
#define NX_C_MAX_NDIM 32

/* The first four fields of Nx_backend.t cross the FFI in a fixed order, and that
   order IS ABI: an operand value is read at exactly these slots.
       0 data     the bigarray
       1 shape    int array
       2 strides  int array, ELEMENT units
       3 offset   int, ELEMENT units
   Nx_backend.t is {buffer; shape; strides; offset; dtype; context}: C reads
   slots 0-3 and never touches slot 4 (dtype — redundant here,
   since C derives it from the bigarray kind) or later. The layout is pinned by
   the binding layer's echo test, not by convention; reordering these four
   silently misreads every operand. */
#define NX_C_FFI_DATA 0
#define NX_C_FFI_SHAPE 1
#define NX_C_FFI_STRIDES 2
#define NX_C_FFI_OFFSET 3

#if defined(__GNUC__) || defined(__clang__)
#define NX_C_NORETURN __attribute__((noreturn))
#else
#define NX_C_NORETURN
#endif

/* ── The dtype table ──────────────────────────────────────────────────────

   The one place a dtype is described. Adding a dtype is exactly one new row.

   ONE table in Dtype.t declaration order, so the generated enum values equal
   Dtype.Packed.tag (0=Float16 … 18=Bool) and NX_C_DTYPE_COUNT falls out as the
   trailing enumerator — the correspondence is pinned by a _Static_assert below
   and by the binding's kind->tag test, never by hand. Two iterators project the
   single table: NX_C_FOR_EACH_DTYPE walks all 19 rows (enum, kind switch, class,
   size); NX_C_FOR_EACH_COMPUTE_DTYPE walks only the compute rows (load/store,
   float->int, kernel dispatch tables) — packed rows expand to nothing via the
   `sel` selector, so no compute code is ever emitted for int4/uint4 and no
   second list exists to drift.

   Row: X(A, suffix, kind, storage, compute, load, store, cat, sel)
     A        threaded generator (supplied by the iterator, not by callers).
     suffix   identifier tail; yields NX_C_DTYPE_<suffix>, nx_c_ld_<suffix>, ...
     kind     bigarray kind constant (CAML_BA_* / NX_BA_*). The ONLY place
              kinds are named.
     storage  C type of one stored element. Element size is sizeof(storage)
              (packed rows report 0), so there is no separate, driftable size.
     compute  C type kernels compute in. Small ints widen so wrap-on-store
              gives modular semantics and reductions gain headroom; f16/bf16/
              fp8 compute in float. `void` on packed rows (never instantiated).
     load     storage-value -> compute-value converter (NX_C_ID, a
              nx_buffer_stubs.h converter, or the bool normalizer). Never
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
  X(A, f16, CAML_BA_FLOAT16, uint16_t, float, half_to_float, float_to_half,    \
    NX_C_CAT_FLOAT, NX_C_COMPUTE)                                                \
  X(A, f32, CAML_BA_FLOAT32, float, float, NX_C_ID, NX_C_ID, NX_C_CAT_FLOAT,      \
    NX_C_COMPUTE)                                                               \
  X(A, f64, CAML_BA_FLOAT64, double, double, NX_C_ID, NX_C_ID, NX_C_CAT_FLOAT,    \
    NX_C_COMPUTE)                                                               \
  X(A, bf16, NX_BA_BFLOAT16, uint16_t, float, bfloat16_to_float,               \
    float_to_bfloat16, NX_C_CAT_FLOAT, NX_C_COMPUTE)                             \
  X(A, f8e4m3, NX_BA_FP8_E4M3, caml_ba_fp8_e4m3, float, fp8_e4m3_to_float,     \
    float_to_fp8_e4m3, NX_C_CAT_FLOAT, NX_C_COMPUTE)                             \
  X(A, f8e5m2, NX_BA_FP8_E5M2, caml_ba_fp8_e5m2, float, fp8_e5m2_to_float,     \
    float_to_fp8_e5m2, NX_C_CAT_FLOAT, NX_C_COMPUTE)                             \
  X(A, i4, NX_BA_INT4, uint8_t, void, NX_C_ID, NX_C_ID, NX_C_CAT_SINT,            \
    NX_C_PACKED)                                                                \
  X(A, u4, NX_BA_UINT4, uint8_t, void, NX_C_ID, NX_C_ID, NX_C_CAT_UINT,           \
    NX_C_PACKED)                                                                \
  X(A, i8, CAML_BA_SINT8, int8_t, int64_t, NX_C_ID, NX_C_ID, NX_C_CAT_SINT,       \
    NX_C_COMPUTE)                                                               \
  X(A, u8, CAML_BA_UINT8, uint8_t, int64_t, NX_C_ID, NX_C_ID, NX_C_CAT_UINT,      \
    NX_C_COMPUTE)                                                               \
  X(A, i16, CAML_BA_SINT16, int16_t, int64_t, NX_C_ID, NX_C_ID, NX_C_CAT_SINT,    \
    NX_C_COMPUTE)                                                               \
  X(A, u16, CAML_BA_UINT16, uint16_t, int64_t, NX_C_ID, NX_C_ID, NX_C_CAT_UINT,   \
    NX_C_COMPUTE)                                                               \
  X(A, i32, CAML_BA_INT32, int32_t, int64_t, NX_C_ID, NX_C_ID, NX_C_CAT_SINT,     \
    NX_C_COMPUTE)                                                               \
  X(A, u32, NX_BA_UINT32, caml_ba_uint32, uint64_t, NX_C_ID, NX_C_ID,            \
    NX_C_CAT_UINT, NX_C_COMPUTE)                                                 \
  X(A, i64, CAML_BA_INT64, int64_t, int64_t, NX_C_ID, NX_C_ID, NX_C_CAT_SINT,     \
    NX_C_COMPUTE)                                                               \
  X(A, u64, NX_BA_UINT64, caml_ba_uint64, uint64_t, NX_C_ID, NX_C_ID,            \
    NX_C_CAT_UINT, NX_C_COMPUTE)                                                 \
  X(A, c32, CAML_BA_COMPLEX32, nx_c_complex32, nx_c_complex32, NX_C_ID, NX_C_ID,   \
    NX_C_CAT_COMPLEX, NX_C_COMPUTE)                                              \
  X(A, c64, CAML_BA_COMPLEX64, nx_c_complex64, nx_c_complex64, NX_C_ID, NX_C_ID,   \
    NX_C_CAT_COMPLEX, NX_C_COMPUTE)                                              \
  X(A, bool_, NX_BA_BOOL, caml_ba_bool, uint8_t, NX_C_BOOL_LD, NX_C_BOOL_ST,     \
    NX_C_CAT_BOOL, NX_C_COMPUTE)

/* Full iteration: G receives all eight columns (including cat and sel). */
#define NX_C_FULL(G, sfx, kind, storage, compute, ld, st, cat, sel)            \
  G(sfx, kind, storage, compute, ld, st, cat, sel)
#define NX_C_FOR_EACH_DTYPE(G) NX_C_DTYPE_TABLE(NX_C_FULL, G)

/* Compute-only iteration: packed rows expand to nothing; G receives the first
   seven columns (sel is consumed by the filter). */
#define NX_C_FILTER(G, sfx, kind, storage, compute, ld, st, cat, sel)          \
  NX_C_FILTER_##sel(G, sfx, kind, storage, compute, ld, st, cat)
#define NX_C_FILTER_NX_C_COMPUTE(G, sfx, kind, storage, compute, ld, st, cat)    \
  G(sfx, kind, storage, compute, ld, st, cat)
#define NX_C_FILTER_NX_C_PACKED(G, sfx, kind, storage, compute, ld, st, cat)
#define NX_C_FOR_EACH_COMPUTE_DTYPE(G) NX_C_DTYPE_TABLE(NX_C_FILTER, G)

/* Dense dtype enum in tag order. NX_C_DTYPE_COUNT is the slot count and doubles
   as the "not a dtype" sentinel returned by nx_c_dtype_of_kind. */
typedef enum {
#define NX_C_ENUM_ROW(sfx, kind, storage, compute, ld, st, cat, sel)           \
  NX_C_DTYPE_##sfx,
  NX_C_FOR_EACH_DTYPE(NX_C_ENUM_ROW)
#undef NX_C_ENUM_ROW
      NX_C_DTYPE_COUNT
} nx_c_dtype;

/* nx_c_dtype must equal Dtype.Packed.tag; pin the anchors (and the packed
   boundary, the likeliest drift point) at compile time. The binding's per-
   dtype kind->tag test pins the rest. */
_Static_assert(NX_C_DTYPE_f16 == 0 && NX_C_DTYPE_f8e5m2 == 5 && NX_C_DTYPE_i4 == 6 &&
                   NX_C_DTYPE_u4 == 7 && NX_C_DTYPE_i8 == 8 && NX_C_DTYPE_u64 == 15 &&
                   NX_C_DTYPE_bool_ == 18 && NX_C_DTYPE_COUNT == 19,
               "nx_c_dtype must equal Dtype.Packed.tag");

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
#define NX_C_LDST_ROW(sfx, kind, storage, compute, ld, st, cat)                \
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
#define NX_C_F2I_ROW(sfx, kind, storage, compute, ld, st, cat)                 \
  NX_C_F2I_##cat(sfx, storage)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_F2I_ROW)
#undef NX_C_F2I_ROW

/* ── Derived dtype accessors ──────────────────────────────────────────────
   All generated from the one table, keeping size and kind handling in one
   place. dt is bounds-checked as
   `(unsigned)dt < COUNT`, which rejects negatives and out-of-range in one
   comparison with no -Wtype-limits risk. */

static inline int nx_c_dtype_class(nx_c_dtype dt) {
  static const int classes[NX_C_DTYPE_COUNT] = {
#define NX_C_CATBITS_NX_C_CAT_SINT NX_C_CLASS_INT
#define NX_C_CATBITS_NX_C_CAT_UINT NX_C_CLASS_INT
#define NX_C_CATBITS_NX_C_CAT_FLOAT NX_C_CLASS_FLOAT
#define NX_C_CATBITS_NX_C_CAT_COMPLEX NX_C_CLASS_COMPLEX
#define NX_C_CATBITS_NX_C_CAT_BOOL NX_C_CLASS_BOOL
#define NX_C_PACKEDBIT_NX_C_COMPUTE 0
#define NX_C_PACKEDBIT_NX_C_PACKED NX_C_CLASS_PACKED
#define NX_C_CLASS_ROW(sfx, kind, storage, compute, ld, st, cat, sel)          \
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
#define NX_C_SIZE_ROW(sfx, kind, storage, compute, ld, st, cat, sel)           \
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

/* The one kind switch in the entire backend. Returns NX_C_DTYPE_COUNT for an
   unrecognized kind (including the OCaml native-int kinds, which nx's Dtype.t
   cannot construct); the funnel maps that to a raised NX_C_ERR_BAD_KIND. */
static inline nx_c_dtype nx_c_dtype_of_kind(int kind) {
  switch (kind) {
#define NX_C_KIND_ROW(sfx, kind_, storage, compute, ld, st, cat, sel)          \
  case kind_:                                                                  \
    return NX_C_DTYPE_##sfx;
    NX_C_FOR_EACH_DTYPE(NX_C_KIND_ROW)
#undef NX_C_KIND_ROW
    default:
      return NX_C_DTYPE_COUNT;
  }
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
   - Small-int and bool reductions accumulate in 64-bit (the compute widths
     above); f16/bf16/fp8 accumulate in float.
   - Complex has no mod and no ordered comparison; rounding/abs/sign on complex
     are the kernel's concern (rejected loudly, never identity), not the ABI's. */

/* ── Status protocol ──────────────────────────────────────────────────────

   A status is NULL on success, otherwise a static, never-freed string. No
   status string is ever heap-allocated, so no error path can leak. Kernels and
   drivers return status; ONLY the engine's funnel turns a non-NULL status into
   an OCaml exception, and only with the runtime lock held. abort() is reserved
   for provably-unreachable internal invariants, never user data.

   Testing a status against a specific error compares by CONTENT (strcmp), or
   routes it to nx_c_raise_status (nx_c_engine.h) — never by pointer: identical
   string literals are not pooled across translation units, so `status ==
   NX_C_ERR_X` is unspecified. Success is still the pointer test `status ==
   NX_C_OK` (i.e. == NULL). */
typedef const char *nx_c_status;
#define NX_C_OK ((nx_c_status)NULL)

#define NX_C_ERR_NDIM "ndim exceeds NX_C_MAX_NDIM"
#define NX_C_ERR_RANK_MISMATCH "shape and strides rank disagree"
#define NX_C_ERR_BAD_KIND "unsupported bigarray kind"
#define NX_C_ERR_UNSUPPORTED_DTYPE "dtype not supported for this operation"
#define NX_C_ERR_PACKED "packed dtype not supported for this operation"
#define NX_C_ERR_SHAPE "shape mismatch"
#define NX_C_ERR_EMPTY_REDUCE "reduction over empty axis has no identity"
#define NX_C_ERR_ALLOC "out of memory"

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
   [0, ndim). `data` is the bigarray base; the first live element is at
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
  value v_shape = Field(v, NX_C_FFI_SHAPE);
  value v_strides = Field(v, NX_C_FFI_STRIDES);
  int ndim = (int)Wosize_val(v_shape);
  if (ndim > NX_C_MAX_NDIM) return NX_C_ERR_NDIM;
  if ((int)Wosize_val(v_strides) != ndim) return NX_C_ERR_RANK_MISMATCH;
  out->data = Caml_ba_array_val(Field(v, NX_C_FFI_DATA))->data;
  out->ndim = ndim;
  out->offset = Long_val(Field(v, NX_C_FFI_OFFSET));
  for (int i = 0; i < ndim; i++) {
    out->shape[i] = Long_val(Field(v_shape, i));
    out->strides[i] = Long_val(Field(v_strides, i));
  }
  return NX_C_OK;
}

/* Dtype of an FFI operand, from its bigarray kind. Lock held, no allocation. */
static inline nx_c_dtype nx_c_dtype_of_value(value v) {
  return nx_c_dtype_of_kind(
      nx_buffer_get_kind(Caml_ba_array_val(Field(v, NX_C_FFI_DATA))));
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

   These four ABIs cover the *generated* families (map, fold, argreduce, scan).
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
   (int64), unsigned wide sums `u`, floats `f`/`d`, complex `c32`/`c64`.
   Within-run multi-accumulator unrolling is a kernel-local concern; only this
   one reduced value is carried between step() calls. */
typedef union {
  int64_t i;
  uint64_t u;
  float f;
  double d;
  nx_c_complex32 c32;
  nx_c_complex64 c64;
} nx_c_acc;

/* fold — axis reduction (sum, prod, max, min). The engine drives one output
   element as:
       init(acc, ctx);                 // op+dtype seed
       for each strided run feeding this output:
         step(acc, in, in_step, n, ctx);   // fold n elements into acc
       fini(out, acc, ctx);            // convert acc to out dtype and store
   Multiple step() calls cover non-innermost and multi-axis reductions; a single
   call covers the contiguous fast path. sum/prod seed the neutral identity
   (0/1), so an empty reduced extent stores it; max/min have no neutral identity
   for an empty axis (the fold driver rejects that case before any kernel runs,
   gated by its no_identity flag which max/min stubs set true), and
   init seeds a sentinel extreme (±inf / INT64_MIN / INT64_MAX) so the first
   real element always wins. */
typedef void nx_c_fold_init(nx_c_acc *acc, void *ctx);
typedef void nx_c_fold_step(nx_c_acc *acc, const char *in, int64_t in_step,
                           int64_t n, void *ctx);
typedef void nx_c_fold_fini(char *out, const nx_c_acc *acc, void *ctx);

/* argreduce — argmax/argmin over exactly one axis (backend_intf: single axis,
   int32 result). The accumulator carries the running extreme value and its
   index along that axis. init sets index = -1 ("unset"), so step takes the
   first element unconditionally and no per-dtype identity is needed; the empty
   axis is rejected by the funnel. There is exactly one run per output (the
   axis), so the kernel's element counter is the axis index directly. step
   encapsulates NaN-wins, first-index-wins comparison so argmax/argmin agree
   with reduce_max/min on NaN. fini writes the int32 index. */
typedef struct {
  nx_c_acc value;
  int64_t index;
} nx_c_arg_acc;

static inline void nx_c_arg_init(nx_c_arg_acc *acc) { acc->index = -1; }
static inline void nx_c_arg_fini(char *out, const nx_c_arg_acc *acc) {
  *(int32_t *)out = (int32_t)acc->index;
}
typedef void nx_c_arg_step(nx_c_arg_acc *acc, const char *in, int64_t in_step,
                          int64_t n, void *ctx);

/* scan — inclusive cumulative op (cumsum, cumprod, cummax, cummin) along one
   axis. Slices are independent (the engine parallelizes across them); within a
   slice the walk is sequential. Per slice:
       init(state, ctx);              // op+dtype identity
       step(out, out_step, in, in_step, n, state, ctx);
   step reads in[k], folds it into *state, and writes the running result to
   out[k]. state carries across calls so a slice may be handed to step in
   pieces. */
typedef void nx_c_scan_init(nx_c_acc *state, void *ctx);
typedef void nx_c_scan_step(char *out, int64_t out_step, const char *in,
                           int64_t in_step, int64_t n, nx_c_acc *state,
                           void *ctx);

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
  nx_c_fold_fini *fini[NX_C_DTYPE_COUNT];
} nx_c_fold_table;
typedef struct {
  nx_c_arg_step *step[NX_C_DTYPE_COUNT];
} nx_c_arg_table;
typedef struct {
  nx_c_scan_init *init[NX_C_DTYPE_COUNT];
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

#endif /* NX_C_C_H */
