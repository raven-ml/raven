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

   One kernel serves both ops. Each element becomes an unsigned integer key that
   orders like it, paired with its position, and the pairs are sorted by a
   stable LSD radix sort (a stable insertion sort on short slices). Argsort
   writes the positions; sort copies the input's elements at those positions,
   bit for bit. Stability comes from the algorithm, so equal elements keep their
   input order in either direction, -0 and +0 and NaNs included, and sort's
   values are exactly the input's elements at argsort's indices.

   Each slice is read once through its stride into contiguous per-thread
   scratch, sorted there, and written to the (C-contiguous) output.

   Threading: independent slices are split across the engine pool via
   nx_c_parallel_for (NX_C_COST_HEAVY), with a private scratch slot per worker; the
   engine owns the runtime-lock handshake, so this file touches neither
   caml/threads.h nor caml/fail.h. A single huge slice stays serial (HEAVY gives
   one thread for one run). */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_c_engine.h"

/* Positions are int32 (backend_intf), so a sorted axis longer than INT32_MAX
   cannot be indexed; reject up front rather than truncate. Maps to Failure. */
#define NX_C_ERR_SORT_CAP "sort axis length exceeds INT32_MAX"

/* ── Keys ──────────────────────────────────────────────────────────────────

   A key is one or more unsigned words of the element's storage width, the last
   word most significant, whose unsigned order is the element order:
   - an unsigned integer or bool is its own key;
   - a signed integer flips its sign bit;
   - a float's bits map through nx_c_fkey: -0 takes +0's key, a negative value
     complements its bits and a positive one sets its sign bit;
   - a complex value is two words, its real part's float key above its
     imaginary part's, which orders lexicographically.
   A descending sort complements the key. Every NaN, and every complex value
   with a NaN part, takes the greatest key in either direction, so NaNs tie at
   the end; no number takes that key (its complement would be all ones, a NaN's
   bits). The NaN test reads the element through the dtype's load, so every
   float format, fp8 included, is recognised by its own rules. */

#define NX_C_DEFINE_FKEY(W)                                                     \
  static inline uint##W##_t nx_c_fkey##W(uint##W##_t b) {                       \
    const uint##W##_t sign = (uint##W##_t)((uint##W##_t)1 << (W - 1));        \
    if ((uint##W##_t)(b << 1) == 0) return sign;                               \
    return b & sign ? (uint##W##_t)~b : (uint##W##_t)(b | sign);                \
  }
NX_C_DEFINE_FKEY(8)
NX_C_DEFINE_FKEY(16)
NX_C_DEFINE_FKEY(32)
NX_C_DEFINE_FKEY(64)
#undef NX_C_DEFINE_FKEY

/* NX_C_KEY_<cat>(sfx, W, p, flip, key) writes the key of the element at p into
   key[]; flip is all ones for a descending sort and 0 otherwise. */
#define NX_C_KEY_NX_C_CAT_BOOL(sfx, W, p, flip, key)                            \
  NX_C_KEY_NX_C_CAT_UINT(sfx, W, p, flip, key)
#define NX_C_KEY_NX_C_CAT_UINT(sfx, W, p, flip, key)                            \
  do {                                                                         \
    uint##W##_t b;                                                             \
    memcpy(&b, p, sizeof b);                                                   \
    key[0] = b ^ flip;                                                         \
  } while (0)
#define NX_C_KEY_NX_C_CAT_SINT(sfx, W, p, flip, key)                            \
  do {                                                                         \
    uint##W##_t b;                                                             \
    memcpy(&b, p, sizeof b);                                                   \
    key[0] = b ^ (uint##W##_t)((uint##W##_t)1 << (W - 1)) ^ flip;              \
  } while (0)
#define NX_C_KEY_NX_C_CAT_FLOAT(sfx, W, p, flip, key)                           \
  do {                                                                         \
    uint##W##_t b;                                                             \
    memcpy(&b, p, sizeof b);                                                   \
    key[0] = isnan(nx_c_ld_##sfx(p)) ? (uint##W##_t)~(uint##W##_t)0             \
                                    : (uint##W##_t)(nx_c_fkey##W(b) ^ flip);   \
  } while (0)
#define NX_C_KEY_NX_C_CAT_COMPLEX(sfx, W, p, flip, key)                         \
  do {                                                                         \
    uint##W##_t re, im;                                                        \
    memcpy(&re, p, sizeof re);                                                 \
    memcpy(&im, (const char *)(p) + sizeof re, sizeof im);                     \
    if (isnan(__real__ nx_c_ld_##sfx(p)) || isnan(__imag__ nx_c_ld_##sfx(p))) \
      key[1] = key[0] = (uint##W##_t)~(uint##W##_t)0;                          \
    else {                                                                     \
      key[1] = (uint##W##_t)(nx_c_fkey##W(re) ^ flip);                         \
      key[0] = (uint##W##_t)(nx_c_fkey##W(im) ^ flip);                         \
    }                                                                          \
  } while (0)

/* ── Slice kernel ──────────────────────────────────────────────────────────

   NX_C_SORT_KERNEL(sfx, W, NW, cat) stamps one dtype's kernel: its (key,
   position) pair of NW words of W bits, the order of a slice's pairs, and the
   slice kernel. The radix sort takes one 8-bit digit per pass, least significant
   first, from histograms of every digit built in one read of the pairs, and
   skips a pass whose digit is the same for every pair (the high bytes of small
   integers, the shared exponent bits of floats of one scale). Each pass
   scatters the pairs in order into the other half of the scratch slot, so it is
   stable, and a stable pass over each digit sorts by the whole key.

   Each pass costs a scan of its 256 counts whatever the slice's length, so a
   short slice takes a stable insertion sort instead: below 16 pairs per digit,
   where the two cost about the same (measured single-threaded on random u8,
   i32, f32 and f64 rows). */

#define NX_C_SORT_RADIX_MIN_PER_DIGIT 16

#define NX_C_SORT_DIGIT(pair, d, W)                                             \
  ((uint8_t)((pair).key[(d) / ((W) / 8)] >> (8 * ((d) % ((W) / 8)))))

#define NX_C_SORT_KERNEL(sfx, W, NW, cat)                                       \
  typedef struct {                                                             \
    uint##W##_t key[NW];                                                       \
    int32_t pos;                                                               \
  } nx_c_pair_##sfx;                                                            \
  static inline int nx_c_before_##sfx(const nx_c_pair_##sfx *x,                 \
                                     const nx_c_pair_##sfx *y) {                \
    for (int w = (NW) - 1; w > 0; w--)                                         \
      if (x->key[w] != y->key[w]) return x->key[w] < y->key[w];                \
    return x->key[0] < y->key[0];                                              \
  }                                                                            \
  static void nx_c_insert_##sfx(nx_c_pair_##sfx *a, int64_t n) {               \
    for (int64_t i = 1; i < n; i++) {                                          \
      nx_c_pair_##sfx v = a[i];                                                 \
      int64_t j = i;                                                           \
      for (; j > 0 && nx_c_before_##sfx(&v, &a[j - 1]); j--) a[j] = a[j - 1];   \
      a[j] = v;                                                                \
    }                                                                          \
  }                                                                            \
  /* Sorts the n pairs at a, using b as much again; returns whichever of the   \
     two holds the result. */                                                  \
  static const nx_c_pair_##sfx *nx_c_order_##sfx(nx_c_pair_##sfx *a,             \
                                                nx_c_pair_##sfx *b, int64_t n) { \
    enum { digits = (NW) * (W) / 8 };                                          \
    if (n < NX_C_SORT_RADIX_MIN_PER_DIGIT * digits) {                          \
      nx_c_insert_##sfx(a, n);                                                  \
      return a;                                                                \
    }                                                                          \
    uint32_t count[digits][256];                                               \
    memset(count, 0, sizeof count);                                            \
    for (int64_t i = 0; i < n; i++)                                            \
      for (int d = 0; d < digits; d++)                                         \
        count[d][NX_C_SORT_DIGIT(a[i], d, W)]++;                                \
    for (int d = 0; d < digits; d++) {                                         \
      uint32_t *at = count[d];                                                 \
      if (at[NX_C_SORT_DIGIT(a[0], d, W)] == (uint32_t)n) continue;             \
      uint32_t sum = 0;                                                        \
      for (int v = 0; v < 256; v++) {                                          \
        uint32_t c = at[v];                                                    \
        at[v] = sum;                                                           \
        sum += c;                                                              \
      }                                                                        \
      for (int64_t i = 0; i < n; i++) {                                        \
        nx_c_pair_##sfx p = a[i];                                               \
        b[at[NX_C_SORT_DIGIT(p, d, W)]++] = p;                                  \
      }                                                                        \
      nx_c_pair_##sfx *t = a;                                                   \
      a = b;                                                                   \
      b = t;                                                                   \
    }                                                                          \
    return a;                                                                  \
  }                                                                            \
  static void nx_c_sort_slice_##sfx(char *o, int64_t os, const char *in,        \
                                   int64_t is, int64_t n, int desc, int arg,   \
                                   void *scr) {                                \
    nx_c_pair_##sfx *a = (nx_c_pair_##sfx *)scr;                                 \
    const uint##W##_t flip = desc ? (uint##W##_t)~(uint##W##_t)0 : 0;          \
    for (int64_t k = 0; k < n; k++) {                                          \
      NX_C_KEY_##cat(sfx, W, in + k * is, flip, a[k].key);                      \
      a[k].pos = (int32_t)k;                                                   \
    }                                                                          \
    const nx_c_pair_##sfx *s = nx_c_order_##sfx(a, a + n, n);                   \
    if (arg)                                                                   \
      for (int64_t k = 0; k < n; k++) *(int32_t *)(o + k * os) = s[k].pos;     \
    else                                                                       \
      for (int64_t k = 0; k < n; k++)                                          \
        memcpy(o + k * os, in + (int64_t)s[k].pos * is, (NW) * (W) / 8);       \
  }

NX_C_SORT_KERNEL(f16, 16, 1, NX_C_CAT_FLOAT)
NX_C_SORT_KERNEL(f32, 32, 1, NX_C_CAT_FLOAT)
NX_C_SORT_KERNEL(f64, 64, 1, NX_C_CAT_FLOAT)
NX_C_SORT_KERNEL(bf16, 16, 1, NX_C_CAT_FLOAT)
NX_C_SORT_KERNEL(f8e4m3, 8, 1, NX_C_CAT_FLOAT)
NX_C_SORT_KERNEL(f8e5m2, 8, 1, NX_C_CAT_FLOAT)
NX_C_SORT_KERNEL(i8, 8, 1, NX_C_CAT_SINT)
NX_C_SORT_KERNEL(u8, 8, 1, NX_C_CAT_UINT)
NX_C_SORT_KERNEL(i16, 16, 1, NX_C_CAT_SINT)
NX_C_SORT_KERNEL(u16, 16, 1, NX_C_CAT_UINT)
NX_C_SORT_KERNEL(i32, 32, 1, NX_C_CAT_SINT)
NX_C_SORT_KERNEL(u32, 32, 1, NX_C_CAT_UINT)
NX_C_SORT_KERNEL(i64, 64, 1, NX_C_CAT_SINT)
NX_C_SORT_KERNEL(u64, 64, 1, NX_C_CAT_UINT)
NX_C_SORT_KERNEL(c32, 32, 2, NX_C_CAT_COMPLEX)
NX_C_SORT_KERNEL(c64, 64, 2, NX_C_CAT_COMPLEX)
NX_C_SORT_KERNEL(bool_, 8, 1, NX_C_CAT_BOOL)
#undef NX_C_SORT_KERNEL

/* ── Dispatch tables ───────────────────────────────────────────────────────

   Indexed by nx_c_dtype; packed (int4/uint4) slots are NULL (the compute
   iterator skips packed rows), and a compute dtype without a kernel above fails
   to compile. The driver is the single reader that turns NULL into a status
   before doing any work — kernels never index here. The pair size sizes the
   scratch. */

typedef void nx_c_sort_slice_fn(char *o, int64_t os, const char *in, int64_t is,
                               int64_t n, int desc, int arg, void *scr);

static nx_c_sort_slice_fn *const nx_c_sort_fn[NX_C_DTYPE_COUNT] = {
#define NX_C_ROW(sfx, kind, storage, compute, ld, st, cat)                      \
  [NX_C_DTYPE_##sfx] = nx_c_sort_slice_##sfx,
    NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_ROW)
#undef NX_C_ROW
};

static const int64_t nx_c_sort_pair_size[NX_C_DTYPE_COUNT] = {
#define NX_C_ROW(sfx, kind, storage, compute, ld, st, cat)                      \
  [NX_C_DTYPE_##sfx] = (int64_t)sizeof(nx_c_pair_##sfx),
    NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_ROW)
#undef NX_C_ROW
};

/* ── Driver ────────────────────────────────────────────────────────────────
   Sort each 1-D slice along `axis`; the non-axis dims index independent slices,
   parallelized across the engine pool (NX_C_COST_HEAVY). Scratch is one slot of
   2n pairs PER THREAD, allocated once per call and indexed by `worker` — no
   per-slice malloc, no sharing between threads. A single huge slice stays serial
   (HEAVY returns one thread for one run). Returns a status; the stub raises. */

typedef struct {
  nx_c_sort_slice_fn *fn;
  int desc;
  int arg;                 /* write positions (argsort) or elements (sort) */
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
    e->fn(op, e->axis_out_stride, ip, e->axis_in_stride, e->n, e->desc, e->arg,
          scr);
  }
}

static nx_c_status nx_c_sort_drive(nx_c_dtype dt, const nx_c_ndarray *in,
                                 int64_t in_elem, const nx_c_ndarray *out,
                                 int64_t out_elem, int axis, int desc,
                                 int is_arg) {
  nx_c_sort_slice_fn *fn = nx_c_sort_fn[dt];
  if (fn == NULL)
    return nx_c_dtype_is_packed(dt) ? NX_C_ERR_PACKED : NX_C_ERR_UNSUPPORTED_DTYPE;
  if (axis < 0 || axis >= in->ndim) return NX_C_ERR_AXIS;
  if (out->ndim != in->ndim) return NX_C_ERR_OUT_RANK;

  int64_t n = in->shape[axis];
  if (n > INT32_MAX) return NX_C_ERR_SORT_CAP;

  nx_c_sort_exec e;
  e.fn = fn;
  e.desc = desc;
  e.arg = is_arg;
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

  /* Round each slot to 16 bytes so every thread's pairs stay aligned when the
     slots are laid end to end. */
  int64_t slot_bytes = (2 * n * nx_c_sort_pair_size[dt] + 15) & ~(int64_t)15;
  e.slot_bytes = slot_bytes;

  /* Policy first: it sizes the scratch. HEAVY parallelizes once there is more
     than one slice; a lone slice returns one thread (serial in v1). */
  int64_t bytes = nslices * n * (in_elem + out_elem);
  int nth = nx_c_threads_for(NX_C_COST_HEAVY, nslices, n, bytes);
  if (nth > nslices) nth = (int)nslices;

  e.scratch = nx_c_aligned_alloc((size_t)slot_bytes * (size_t)nth);
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

static void nx_c_sort_stub(const char *op, value vout, value vin, int axis,
                          int desc, int is_arg) {
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

  s = nx_c_sort_drive(dt, &in, in_elem, &out, out_elem, axis, desc, is_arg);
  if (s != NX_C_OK) nx_c_raise_status(op, s);
}

CAMLprim value caml_nx_c_sort(value vout, value vin, value vaxis, value vdesc) {
  CAMLparam4(vout, vin, vaxis, vdesc);
  nx_c_sort_stub("sort", vout, vin, Int_val(vaxis), Bool_val(vdesc), 0);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_nx_c_argsort(value vout, value vin, value vaxis,
                                value vdesc) {
  CAMLparam4(vout, vin, vaxis, vdesc);
  nx_c_sort_stub("argsort", vout, vin, Int_val(vaxis), Bool_val(vdesc), 1);
  CAMLreturn(Val_unit);
}
