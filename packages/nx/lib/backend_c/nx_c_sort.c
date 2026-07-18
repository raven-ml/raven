/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_sort.c — sort and argsort along one axis for the backend_c backend.

   A custom kernel family (nx_c.h §"Kernel ABIs"): its access is data-dependent
   and does not ride a 1-D map/fold run, so it owns its own driver signature. It
   still reuses the engine's extraction (nx_c_ndarray_of_value), dtype table,
   parallel policy (nx_c_threads_for), and status protocol; only the funnel
   raisers (nx_c_raise / nx_c_raise_status) raise, so this file includes neither
   caml/fail.h nor caml/threads.h.

   The comparator is a per-dtype
   `static inline` inlined into a specialized introsort at compile time — one sort
   kernel per compute dtype, generated over NX_C_FOR_EACH_COMPUTE_DTYPE. NaN is not
   a comparator branch: NaN-class elements are pre-partitioned to the slice tail,
   the finite prefix is sorted with a NaN-free comparator, and NaN stays last in
   both directions by construction.

   Each slice is gathered into a contiguous per-thread scratch, sorted, and
   scattered to the (C-contiguous) output — never sorted through strides, so the
   comparator never chases a stride.

   Threading: independent slices are split across the engine pool via
   nx_c_parallel_for (NX_C_COST_HEAVY), with a private scratch slot per worker; the
   engine owns the runtime-lock handshake, so this file touches neither
   caml/threads.h nor caml/fail.h. A single huge slice stays serial (HEAVY gives
   one thread for one run); parallel single-slice sort is a possible future
   benchmark-gated change. */

#include <stdint.h>
#include <stdlib.h>

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_c_engine.h"

/* argsort indices are int32 (backend_intf), so a sorted axis longer than INT32
   cannot be indexed; reject up front rather than truncate. Maps to Failure. */
#define NX_C_ERR_SORT_CAP "argsort axis length exceeds INT32_MAX"

/* ── Per-dtype total order (finite domain) ─────────────────────────────────

   The comparators the sort inlines, one set per dtype category. NaN never
   reaches them (pre-partitioned away), so the float/complex orders need no
   NaN branch. Complex is lexicographic (real, then imaginary), matching the
   NaN-last-both-directions contract once NaN-class elements are removed.
   __real__/__imag__ are type-generic over both complex compute widths. */

#define NX_C_LT_NX_C_CAT_SINT(a, b) ((a) < (b))
#define NX_C_LT_NX_C_CAT_UINT(a, b) ((a) < (b))
#define NX_C_LT_NX_C_CAT_BOOL(a, b) ((a) < (b))
#define NX_C_LT_NX_C_CAT_FLOAT(a, b) ((a) < (b))
#define NX_C_LT_NX_C_CAT_COMPLEX(a, b)                                           \
  (__real__(a) < __real__(b) ||                                                \
   (__real__(a) == __real__(b) && __imag__(a) < __imag__(b)))

#define NX_C_EQ_NX_C_CAT_SINT(a, b) ((a) == (b))
#define NX_C_EQ_NX_C_CAT_UINT(a, b) ((a) == (b))
#define NX_C_EQ_NX_C_CAT_BOOL(a, b) ((a) == (b))
#define NX_C_EQ_NX_C_CAT_FLOAT(a, b) ((a) == (b))
#define NX_C_EQ_NX_C_CAT_COMPLEX(a, b)                                           \
  (__real__(a) == __real__(b) && __imag__(a) == __imag__(b))

#define NX_C_NAN_NX_C_CAT_SINT(x) (0)
#define NX_C_NAN_NX_C_CAT_UINT(x) (0)
#define NX_C_NAN_NX_C_CAT_BOOL(x) (0)
#define NX_C_NAN_NX_C_CAT_FLOAT(x) (isnan(x))
#define NX_C_NAN_NX_C_CAT_COMPLEX(x) (isnan(__real__(x)) || isnan(__imag__(x)))

/* NaN-class test plus the two "sorts before" predicates, per dtype.

   The value predicate is direction-FREE (ascending "x before y"): descending is
   a reverse of the sorted finite prefix (value sort's equal elements are
   indistinguishable), which keeps the hottest comparison — the 1M-element value
   sort — a single branch-free compare rather than a per-comparison direction
   branch. The argsort predicate cannot reverse (that would flip tie order), so
   it carries `desc` and breaks value ties by original index (first index first);
   the result is a TOTAL order, so the sort is stable by construction and the
   tie-break is direction-independent (ties keep the first index either way). */
#define NX_C_GEN_PRED(sfx, kind, storage, compute, ld, st, cat)                 \
  static inline int nx_c_isnan_##sfx(compute x) {                               \
    (void)x;                                                                   \
    return NX_C_NAN_##cat(x);                                                   \
  }                                                                            \
  static inline int nx_c_vlt_##sfx(compute x, compute y, const void *kctx,      \
                                  int desc) {                                  \
    (void)kctx;                                                                \
    (void)desc;                                                                \
    return NX_C_LT_##cat(x, y);                                                 \
  }                                                                            \
  static inline int nx_c_abefore_##sfx(int32_t i, int32_t j, const void *kctx,  \
                                      int desc) {                              \
    const compute *keys = (const compute *)kctx;                              \
    compute x = keys[i], y = keys[j];                                          \
    if (NX_C_EQ_##cat(x, y)) return i < j;                                      \
    return desc ? NX_C_LT_##cat(y, x) : NX_C_LT_##cat(x, y);                     \
  }
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_GEN_PRED)
#undef NX_C_GEN_PRED

/* ── Introsort skeleton ────────────────────────────────────────────────────

   One generator stamps an introsort per (dtype, mode): median-of-3 quicksort with
   a BRANCHLESS Lomuto partition, insertion sort under a small threshold, and a
   heapsort fallback when the recursion depth exceeds 2·floor(log2 n) — so the
   worst case (including adversarial and all-equal inputs) is O(n log n), never
   quadratic. The branchless partition is the performance point: on random input a
   branchy Hoare scan mispredicts ~50% and runs ~3x slower than a tuned std::sort;
   the unconditional-swap Lomuto loop removes that branch and matches it. Skeleton
   otherwise follows libstdc++'s introsort_loop + final insertion sort.

   CMP(x, y, kctx, desc) is a per-dtype inlined predicate ("element x sorts before
   element y") returning 0/1 — the value drives the branchless boundary. ELT is the
   array element: the value itself for a value sort, or an int32 index (with
   kctx = the key array) for an argsort — so the same skeleton serves both without
   a function-pointer comparator. `kctx`/`desc` thread through the recursion; the
   compiler inlines CMP, leaving a tight NaN-free inner loop. */

#define NX_C_SORT_THRESHOLD 16

#define NX_C_DEFINE_INTROSORT(NAME, ELT, CMP)                                   \
  static void NAME##_sift(ELT *a, int64_t first, int64_t n, int64_t root,      \
                          const void *kctx, int desc) {                        \
    for (;;) {                                                                 \
      int64_t child = 2 * root + 1;                                            \
      if (child >= n) break;                                                   \
      if (child + 1 < n &&                                                     \
          CMP(a[first + child], a[first + child + 1], kctx, desc))             \
        child++;                                                               \
      if (!CMP(a[first + root], a[first + child], kctx, desc)) break;          \
      ELT t = a[first + root];                                                 \
      a[first + root] = a[first + child];                                      \
      a[first + child] = t;                                                    \
      root = child;                                                            \
    }                                                                          \
  }                                                                            \
  static void NAME##_heap(ELT *a, int64_t first, int64_t last,                 \
                          const void *kctx, int desc) {                        \
    int64_t n = last - first;                                                  \
    for (int64_t i = n / 2 - 1; i >= 0; i--)                                   \
      NAME##_sift(a, first, n, i, kctx, desc);                                 \
    for (int64_t i = n - 1; i > 0; i--) {                                      \
      ELT t = a[first];                                                        \
      a[first] = a[first + i];                                                 \
      a[first + i] = t;                                                        \
      NAME##_sift(a, first, i, 0, kctx, desc);                                 \
    }                                                                          \
  }                                                                            \
  static void NAME##_median(ELT *a, int64_t r, int64_t b, int64_t c,           \
                            int64_t d, const void *kctx, int desc) {           \
    if (CMP(a[b], a[c], kctx, desc)) {                                         \
      if (CMP(a[c], a[d], kctx, desc)) {                                       \
        ELT t = a[r];                                                          \
        a[r] = a[c];                                                           \
        a[c] = t;                                                              \
      } else if (CMP(a[b], a[d], kctx, desc)) {                                \
        ELT t = a[r];                                                          \
        a[r] = a[d];                                                           \
        a[d] = t;                                                              \
      } else {                                                                 \
        ELT t = a[r];                                                          \
        a[r] = a[b];                                                           \
        a[b] = t;                                                              \
      }                                                                        \
    } else if (CMP(a[b], a[d], kctx, desc)) {                                  \
      ELT t = a[r];                                                            \
      a[r] = a[b];                                                             \
      a[b] = t;                                                                \
    } else if (CMP(a[c], a[d], kctx, desc)) {                                  \
      ELT t = a[r];                                                            \
      a[r] = a[d];                                                             \
      a[d] = t;                                                                \
    } else {                                                                   \
      ELT t = a[r];                                                            \
      a[r] = a[c];                                                             \
      a[c] = t;                                                                \
    }                                                                          \
  }                                                                            \
  static void NAME##_loop(ELT *a, int64_t first, int64_t last, int depth,      \
                          const void *kctx, int desc) {                        \
    while (last - first > NX_C_SORT_THRESHOLD) {                                \
      if (depth == 0) {                                                        \
        NAME##_heap(a, first, last, kctx, desc);                              \
        return;                                                                \
      }                                                                        \
      depth--;                                                                 \
      int64_t mid = first + ((last - first) >> 1);                             \
      NAME##_median(a, first, first + 1, mid, last - 1, kctx, desc);           \
      /* Branchless Lomuto partition of (first, last) around the median at              \
         a[first]: each element is swapped unconditionally and the boundary k           \
         advances by the 0/1 comparison result, so the hot loop carries no              \
         data-dependent branch — this is the change that closes the gap to a            \
         tuned std::sort on random input (a branchy Hoare scan mispredicts ~50%).       \
         Elements EQUAL to the pivot fall right; a long run of equal keys is            \
         bounded to O(n log n) by the depth-limit heapsort, not partitioned away. */    \
      ELT pivot = a[first];                                                    \
      int64_t k = first + 1;                                                   \
      for (int64_t i = first + 1; i < last; i++) {                             \
        ELT v = a[i];                                                          \
        int sm = CMP(v, pivot, kctx, desc);                                    \
        a[i] = a[k];                                                           \
        a[k] = v;                                                              \
        k += sm;                                                               \
      }                                                                        \
      int64_t p = k - 1;                                                       \
      ELT t = a[first];                                                        \
      a[first] = a[p];                                                         \
      a[p] = t;                                                                \
      NAME##_loop(a, p + 1, last, depth, kctx, desc);                          \
      last = p;                                                                \
    }                                                                          \
  }                                                                            \
  static void NAME##_gins(ELT *a, int64_t lo, int64_t hi, const void *kctx,    \
                          int desc) {                                          \
    for (int64_t i = lo + 1; i < hi; i++) {                                    \
      ELT v = a[i];                                                            \
      int64_t j = i;                                                           \
      while (j > lo && CMP(v, a[j - 1], kctx, desc)) {                         \
        a[j] = a[j - 1];                                                       \
        j--;                                                                   \
      }                                                                        \
      a[j] = v;                                                                \
    }                                                                          \
  }                                                                            \
  static void NAME(ELT *a, int64_t n, const void *kctx, int desc) {            \
    if (n < 2) return;                                                         \
    int lg = 0;                                                                \
    while ((n >> (lg + 1)) > 0) lg++;                                          \
    NAME##_loop(a, 0, n, 2 * lg, kctx, desc);                                  \
    /* introsort_loop leaves the array THRESHOLD-sorted: every element is                \
       within THRESHOLD of its place and the global minimum is in the first                \
       block. Sort that block guarded, then the rest unguarded — the min at                \
       a[0] stops every leftward scan, so the inner loop drops its bound test. */          \
    if (n > NX_C_SORT_THRESHOLD) {                                              \
      NAME##_gins(a, 0, NX_C_SORT_THRESHOLD, kctx, desc);                       \
      for (int64_t i = NX_C_SORT_THRESHOLD; i < n; i++) {                       \
        ELT v = a[i];                                                          \
        int64_t j = i;                                                         \
        while (CMP(v, a[j - 1], kctx, desc)) {                                 \
          a[j] = a[j - 1];                                                     \
          j--;                                                                 \
        }                                                                      \
        a[j] = v;                                                              \
      }                                                                        \
    } else {                                                                   \
      NAME##_gins(a, 0, n, kctx, desc);                                        \
    }                                                                          \
  }

/* Byte offset of the argsort index array within a scratch slot: the compute-typed
   key gather rounded up so the trailing int32 indices are naturally aligned. Used
   by both the kernel (to place idx) and the driver (to size the slot). */
static inline int64_t nx_c_argsort_keys_bytes(int64_t n, int64_t csize) {
  return (n * csize + 7) & ~(int64_t)7;
}

/* ── Per-dtype slice kernels ───────────────────────────────────────────────

   Both gather the strided input slice into contiguous scratch (converting to the
   compute type — so f16/bf16/fp8 sort in float, and u32/u64 in unsigned 64-bit
   with correct high-bit order), pre-partition NaN-class elements to the tail,
   sort the finite prefix, and scatter to the C-contiguous output.

   Value sort scatters the sorted values. Order among values that compare equal is
   unobservable, so an unstable sort — and the descending reversal — is sound; the
   one equal-yet-bit-distinct case is IEEE ±0.0, whose relative order is left
   unspecified (numpy's sort does the same). NaN-class elements are pre-partitioned
   to the tail; their payload bits survive for native float/complex but are
   re-quieted for the converted dtypes (f16/bf16/fp8) by the storage round-trip —
   The backend contract requires NaN-last, not payload preservation. Argsort instead scatters a
   permutation of indices and MUST be stable; its comparator's index tie-break
   gives that (and determinism) without a stable-merge algorithm. */

#define NX_C_GEN_VSORT(sfx, kind, storage, compute, ld, st, cat)                \
  NX_C_DEFINE_INTROSORT(nx_c_vintro_##sfx, compute, nx_c_vlt_##sfx)               \
  static void nx_c_sort_slice_##sfx(char *o, int64_t os, const char *in,        \
                                   int64_t is, int64_t n, int desc,            \
                                   void *scr) {                                \
    compute *a = (compute *)scr;                                               \
    for (int64_t k = 0; k < n; k++) a[k] = nx_c_ld_##sfx(in + k * is);          \
    int64_t nf = 0;                                                            \
    for (int64_t k = 0; k < n; k++)                                            \
      if (!nx_c_isnan_##sfx(a[k])) {                                            \
        if (k != nf) {                                                         \
          compute t = a[k];                                                    \
          a[k] = a[nf];                                                        \
          a[nf] = t;                                                           \
        }                                                                      \
        nf++;                                                                  \
      }                                                                        \
    nx_c_vintro_##sfx(a, nf, NULL, 0);                                          \
    if (desc)                                                                  \
      for (int64_t i = 0, j = nf - 1; i < j; i++, j--) {                       \
        compute t = a[i];                                                      \
        a[i] = a[j];                                                           \
        a[j] = t;                                                              \
      }                                                                        \
    for (int64_t k = 0; k < n; k++) nx_c_st_##sfx(o + k * os, a[k]);            \
  }
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_GEN_VSORT)
#undef NX_C_GEN_VSORT

#define NX_C_GEN_ASORT(sfx, kind, storage, compute, ld, st, cat)                \
  NX_C_DEFINE_INTROSORT(nx_c_aintro_##sfx, int32_t, nx_c_abefore_##sfx)           \
  static void nx_c_argsort_slice_##sfx(char *o, int64_t os, const char *in,     \
                                      int64_t is, int64_t n, int desc,         \
                                      void *scr) {                             \
    compute *keys = (compute *)scr;                                            \
    for (int64_t k = 0; k < n; k++) keys[k] = nx_c_ld_##sfx(in + k * is);       \
    int32_t *idx = (int32_t *)((char *)scr +                                   \
                               nx_c_argsort_keys_bytes(n, (int64_t)sizeof(compute))); \
    int64_t f = 0;                                                             \
    for (int64_t k = 0; k < n; k++)                                            \
      if (!nx_c_isnan_##sfx(keys[k])) idx[f++] = (int32_t)k;                    \
    int64_t tail = f;                                                          \
    for (int64_t k = 0; k < n; k++)                                            \
      if (nx_c_isnan_##sfx(keys[k])) idx[tail++] = (int32_t)k;                  \
    nx_c_aintro_##sfx(idx, f, keys, desc);                                      \
    for (int64_t k = 0; k < n; k++) *(int32_t *)(o + k * os) = idx[k];         \
  }
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_GEN_ASORT)
#undef NX_C_GEN_ASORT

/* ── Dispatch tables ───────────────────────────────────────────────────────

   Indexed by nx_c_dtype; packed (int4/uint4) and any op-unsupported slot is NULL
   (the compute iterator skips packed rows). The driver is the single reader that
   turns NULL into a status before doing any work — kernels never index here. */

typedef void nx_c_sort_slice_fn(char *o, int64_t os, const char *in, int64_t is,
                               int64_t n, int desc, void *scr);
typedef struct {
  nx_c_sort_slice_fn *fn[NX_C_DTYPE_COUNT];
} nx_c_sort_table;

static const nx_c_sort_table nx_c_sort_vtable = {
    .fn = {
#define NX_C_ROW(sfx, kind, storage, compute, ld, st, cat)                      \
  [NX_C_DTYPE_##sfx] = nx_c_sort_slice_##sfx,
        NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_ROW)
#undef NX_C_ROW
    }};

static const nx_c_sort_table nx_c_argsort_vtable = {
    .fn = {
#define NX_C_ROW(sfx, kind, storage, compute, ld, st, cat)                      \
  [NX_C_DTYPE_##sfx] = nx_c_argsort_slice_##sfx,
        NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_ROW)
#undef NX_C_ROW
    }};

/* Compute-type byte size per dtype (differs from storage: f16 stores 2, computes
   in 4). The driver sizes scratch from this; packed rows stay 0 (never reached —
   the NULL dispatch slot is rejected first). */
static const int64_t nx_c_sort_csize[NX_C_DTYPE_COUNT] = {
#define NX_C_ROW(sfx, kind, storage, compute, ld, st, cat)                      \
  [NX_C_DTYPE_##sfx] = (int64_t)sizeof(compute),
    NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_ROW)
#undef NX_C_ROW
};

/* ── Driver ────────────────────────────────────────────────────────────────
   Sort each 1-D slice along `axis`; the non-axis dims index independent slices,
   parallelized across the engine pool (NX_C_COST_HEAVY). Scratch is one contiguous
   gather buffer PER THREAD, allocated once per call and indexed by `worker` — no
   per-slice malloc, no sharing between threads. A single huge slice stays serial
   (HEAVY returns one thread for one run). Returns a status; the stub raises. */

typedef struct {
  nx_c_sort_slice_fn *fn;
  int desc;
  int64_t n;               /* axis length */
  int64_t axis_in_stride;  /* byte */
  int64_t axis_out_stride; /* byte */
  char *in_base;
  char *out_base;
  int nk;
  int64_t kshape[NX_C_MAX_NDIM];
  int64_t k_in_stride[NX_C_MAX_NDIM];  /* byte */
  int64_t k_out_stride[NX_C_MAX_NDIM]; /* byte */
  char *scratch;      /* nthreads * slot_bytes, laid out contiguously */
  int64_t slot_bytes; /* one thread's private scratch */
} nx_c_sort_exec;

/* Sort slices [lo, hi) using worker `worker`'s private scratch slot. Pure C: no
   allocation, no runtime, no failure (nx_c_engine.h contract). */
static void nx_c_sort_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  const nx_c_sort_exec *e = vctx;
  void *scr = e->scratch + (int64_t)worker * e->slot_bytes;
  for (int64_t si = lo; si < hi; si++) {
    char *ip = e->in_base;
    char *op = e->out_base;
    int64_t rem = si;
    for (int d = e->nk - 1; d >= 0; d--) {
      int64_t c = rem % e->kshape[d];
      rem /= e->kshape[d];
      ip += c * e->k_in_stride[d];
      op += c * e->k_out_stride[d];
    }
    e->fn(op, e->axis_out_stride, ip, e->axis_in_stride, e->n, e->desc, scr);
  }
}

static nx_c_status nx_c_sort_drive(const nx_c_sort_table *tbl, nx_c_dtype dt,
                                 const nx_c_ndarray *in, int64_t in_elem,
                                 const nx_c_ndarray *out, int64_t out_elem,
                                 int axis, int desc, int is_arg) {
  nx_c_sort_slice_fn *fn = tbl->fn[dt];
  if (fn == NULL)
    return nx_c_dtype_is_packed(dt) ? NX_C_ERR_PACKED : NX_C_ERR_UNSUPPORTED_DTYPE;
  if (axis < 0 || axis >= in->ndim) return NX_C_ERR_AXIS;
  if (out->ndim != in->ndim) return NX_C_ERR_OUT_RANK;

  int64_t n = in->shape[axis];
  if (is_arg && n > INT32_MAX) return NX_C_ERR_SORT_CAP;

  nx_c_sort_exec e;
  e.fn = fn;
  e.desc = desc;
  e.n = n;
  e.in_base = (char *)in->data + in->offset * in_elem;
  e.out_base = (char *)out->data + out->offset * out_elem;
  e.axis_in_stride = in->strides[axis] * in_elem;
  e.axis_out_stride = out->strides[axis] * out_elem;
  e.nk = 0;
  int64_t nslices = 1;
  for (int a = 0; a < in->ndim; a++) {
    if (a == axis) continue;
    e.kshape[e.nk] = in->shape[a];
    e.k_in_stride[e.nk] = in->strides[a] * in_elem;
    e.k_out_stride[e.nk] = out->strides[a] * out_elem;
    nslices *= in->shape[a];
    e.nk++;
  }
  /* Empty axis or empty slice space: nothing to sort (backend_intf: a valid
     no-op). Checked before allocating so a zero-length slot is never requested. */
  if (nslices == 0 || n == 0) return NX_C_OK;

  /* The driver is the validation owner: a 0-stride output dim of extent > 1 makes
     distinct slices (or positions along the sorted axis) alias one cell — a data
     race once slices run in parallel. The frontend always allocates a fresh
     contiguous output, but verify rather than assume (checked after the empty
     short-circuit, so a harmless aliased-but-empty output is not rejected). */
  for (int a = 0; a < in->ndim; a++)
    if (in->shape[a] > 1 && out->strides[a] == 0) return NX_C_ERR_OUT_ALIASED;

  int64_t csize = nx_c_sort_csize[dt];
  int64_t slot_bytes =
      is_arg ? nx_c_argsort_keys_bytes(n, csize) + n * (int64_t)sizeof(int32_t)
             : n * csize;
  /* Round each slot to 16 bytes so every thread's slice base stays aligned for
     any compute type (complex64 wants 16) when the slots are laid end to end. */
  slot_bytes = (slot_bytes + 15) & ~(int64_t)15;
  e.slot_bytes = slot_bytes;

  /* Policy first: it sizes the scratch. HEAVY parallelizes once there is more
     than one slice; a lone slice returns one thread (serial in v1). */
  int64_t bytes = nslices * n * (in_elem + out_elem);
  int nth = nx_c_threads_for(NX_C_COST_HEAVY, nslices, n, bytes);
  if (nth > nslices) nth = (int)nslices;

  e.scratch = malloc((size_t)slot_bytes * (size_t)nth);
  if (e.scratch == NULL) return NX_C_ERR_ALLOC;
  /* Hand the scratch to the primitive as free_on_exit: it frees after the join
     but before re-acquiring the lock, so a raise from leave_blocking_section
     (pending signal/memprof) cannot longjmp past a free here and leak it. */
  nx_c_parallel_for(nth, nslices, bytes, nx_c_sort_body, &e, e.scratch);
  return NX_C_OK;
}

/* ── Stubs ──────────────────────────────────────────────────────────────────
   Marshal the FFI operands, dispatch on the INPUT dtype (argsort's output is
   int32; sort's output shares the input dtype), and raise on a non-NULL status
   with the op name. Runs with the runtime lock held; the lock handoff for the
   parallel region lives inside nx_c_parallel_for (the engine), so this file needs
   neither caml/fail.h nor caml/threads.h — it reaches the funnel raisers
   (nx_c.h) which the engine implements. */

static void nx_c_sort_stub(const char *op, const nx_c_sort_table *tbl, value vout,
                          value vin, int axis, int desc, int is_arg) {
  nx_c_ndarray in, out;
  nx_c_status s = nx_c_ndarray_of_value(vin, &in);
  if (s != NX_C_OK) nx_c_raise(op, s);
  s = nx_c_ndarray_of_value(vout, &out);
  if (s != NX_C_OK) nx_c_raise(op, s);

  nx_c_dtype dt = nx_c_dtype_of_value(vin);
  if (dt == NX_C_DTYPE_COUNT) nx_c_raise(op, NX_C_ERR_BAD_KIND);
  int64_t in_elem = nx_c_elem_size(dt);
  if (in_elem == 0) nx_c_raise(op, NX_C_ERR_PACKED);
  int64_t out_elem = is_arg ? (int64_t)sizeof(int32_t) : in_elem;

  s = nx_c_sort_drive(tbl, dt, &in, in_elem, &out, out_elem, axis, desc, is_arg);
  if (s != NX_C_OK) nx_c_raise_status(op, s);
}

CAMLprim value caml_nx_c_sort(value vout, value vin, value vaxis, value vdesc) {
  CAMLparam4(vout, vin, vaxis, vdesc);
  nx_c_sort_stub("sort", &nx_c_sort_vtable, vout, vin, Int_val(vaxis),
                Bool_val(vdesc), 0);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_nx_c_argsort(value vout, value vin, value vaxis,
                                value vdesc) {
  CAMLparam4(vout, vin, vaxis, vdesc);
  nx_c_sort_stub("argsort", &nx_c_argsort_vtable, vout, vin, Int_val(vaxis),
                Bool_val(vdesc), 1);
  CAMLreturn(Val_unit);
}
