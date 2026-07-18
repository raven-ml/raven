/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_move.c — the data-movement family: copy/assign/contiguous, pad, cat,
   gather/scatter, unfold/fold.

   Two shapes of op live here. The layout-preserving movers (copy, pad, cat)
   are just identity copies through the engine's map driver: one generated,
   bit-exact identity kernel per compute dtype (storage-typed, so it preserves
   NaN payloads and -0.0 that a load/store round-trip would not), with a memcpy
   fast path inside the kernel's contiguous run. pad and cat build sub-views and
   drive the SAME copy table through nx_c_map_run, so they inherit coalescing,
   threading, and the lock handshake for free, and cat releases the lock per
   member rather than across the whole concatenation.

   The index-/structure-dependent movers (gather, unfold, fold) cannot ride a
   fixed-stride run, so they own thin drivers that parallelize over independent
   output elements through the engine's one pool (nx_c_threads_for policy +
   nx_c_parallel_for executor); scatter is serial (see its note). They still reuse
   the engine's extraction, dtype table, parallel policy, and status protocol.
   Every driver returns a status; only the stub raises, via the engine funnel's
   nx_c_raise/nx_c_raise_invalid. This TU never includes
   caml/fail.h or caml/threads.h: it cannot raise or touch the runtime lock
   except through the engine, exactly like every other kernel-family file.

   Packed int4/uint4 reach exactly one op here — contiguous copy — as a byte-
   level memcpy reinterpreted through the u8 identity kernel (so even that gets
   the engine's threading and lock handling). Every other op rejects packed via
   the copy table's NULL slots (NX_C_ERR_PACKED) or an explicit guard. */

#include <string.h>

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_c_engine.h"

/* Spatial-dimension bound for unfold/fold stack arrays. A tensor's total rank
   is already capped at NX_C_MAX_NDIM, and leading_ndim + K <= ndim, so K can
   never exceed this. */
#define NX_C_MAX_SPATIAL NX_C_MAX_NDIM

/* Status strings this family owns (the shared set in nx_c.h/nx_c_engine.h has no
   index-domain error). Static, never freed — the status-protocol contract. An
   out-of-bounds gather/scatter index is a data fault, surfaced as Failure by
   the stub, never an out-of-bounds access or abort. */
#define NX_C_ERR_INDEX_OOB "index out of bounds for the gathered/scattered axis"

/* ── Identity copy table ───────────────────────────────────────────────────

   One kernel per compute dtype, keyed by the dtype's STORAGE type so the copy
   is bit-exact (no compute-type round trip). The engine hands the kernel one
   1-D run with byte steps; when both steps equal the element size the run is
   contiguous and collapses to a single memcpy — the one fast path, kept in the
   kernel, with the policy (which runs, how threaded) owned by the engine. A
   strided run (transpose materialize, pad interior, cat slice) is a typed
   word-copy loop. Packed rows are absent from NX_C_FOR_EACH_COMPUTE_DTYPE, so their table
   slots stay NULL and the map driver reports NX_C_ERR_PACKED. */
#define NX_C_COPY_KERNEL(sfx, kind, storage, compute, ld, st, cat)              \
  static void nx_c_copy_##sfx(char *const *ptrs, const int64_t *steps,          \
                             int64_t n, void *ctx) {                           \
    (void)ctx;                                                                 \
    char *o = ptrs[0];                                                         \
    const char *a = ptrs[1];                                                   \
    int64_t so = steps[0], sa = steps[1];                                      \
    if (so == (int64_t)sizeof(storage) && sa == (int64_t)sizeof(storage)) {    \
      memcpy(o, a, (size_t)n * sizeof(storage));                               \
      return;                                                                  \
    }                                                                          \
    for (int64_t i = 0; i < n; i++)                                            \
      *(storage *)(o + i * so) = *(const storage *)(a + i * sa);               \
  }
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_COPY_KERNEL)
#undef NX_C_COPY_KERNEL

static const nx_c_map_table nx_c_copy_table = {
    .fn = {
#define NX_C_COPY_ROW(sfx, kind, storage, compute, ld, st, cat)                 \
  [NX_C_DTYPE_##sfx] = nx_c_copy_##sfx,
        NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_COPY_ROW)
#undef NX_C_COPY_ROW
    }};

/* ── Small geometry helpers (custom drivers only) ──────────────────────────
   The map/fold engine walks hot runs with incremental odometers; the index-
   dependent movers below touch each output once and do scattered reads, so a
   plain unravel + dot per output is negligible against the copy/accumulate and
   keeps the drivers legible. All strides/offsets are in ELEMENTS until the
   final byte address, matching the ndarray convention. */

static inline int64_t nx_c_prod(int ndim, const int64_t *shape) {
  int64_t p = 1;
  for (int d = 0; d < ndim; d++) p *= shape[d];
  return p;
}

static inline void nx_c_unravel(int64_t idx, int ndim, const int64_t *shape,
                               int64_t *coord) {
  for (int d = ndim - 1; d >= 0; d--) {
    coord[d] = idx % shape[d];
    idx /= shape[d];
  }
}

static inline int64_t nx_c_dot(int ndim, const int64_t *coord,
                              const int64_t *stride) {
  int64_t s = 0;
  for (int d = 0; d < ndim; d++) s += coord[d] * stride[d];
  return s;
}

static bool nx_c_is_contiguous_off0(const nx_c_ndarray *a) {
  if (a->offset != 0) return false;
  int64_t s = 1;
  for (int d = a->ndim - 1; d >= 0; d--) {
    if (a->shape[d] == 1) continue; /* size-1 dims add no iteration */
    if (a->strides[d] != s) return false;
    s *= a->shape[d];
  }
  return true;
}

/* Policy + execution for the custom drivers: pick the thread count for `total`
   independent output units of `run_len` work each, cap it by the unit count
   (never more threads than units), and run the body through the engine's one
   pool. None of these bodies keep per-worker scratch, so the worker index is
   unused and free_on_exit is NULL; the policy-first ordering still matches the
   driver contract. */
static void nx_c_move_dispatch(nx_c_cost_class cls, int64_t total, int64_t run_len,
                              int64_t bytes, nx_c_range_body body, void *ctx) {
  int nth = nx_c_threads_for(cls, total, run_len, bytes);
  if (nth > total) nth = (int)total;
  nx_c_parallel_for(nth, total, bytes, body, ctx, NULL);
}

/* ── copy / assign / contiguous ────────────────────────────────────────────

   One stub serves all three: the frontend/binding decides the destination
   (fresh buffer for copy/contiguous, caller's buffer for assign) and passes it
   as vout; the C job is identical — write vin's logical content into vout,
   honoring both strides. Assign to a transposed dst is a strided-output map the
   engine already handles. Dispatch is on the output dtype (== input dtype). */

/* Packed contiguous copy: reinterpret the nibble stream as bytes and run it
   through the u8 identity kernel, so the byte memcpy still rides the engine's
   threading and lock handshake (no raw memcpy under the runtime lock). Only the
   contiguous, offset-0 case is in scope for packed dtypes (the dtype policy);
   anything else is NX_C_ERR_PACKED. */
static void nx_c_copy_packed(value vout, value vin, nx_c_dtype dt) {
  nx_c_ndarray out, in;
  nx_c_status s = nx_c_ndarray_of_value(vout, &out);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vin, &in);
  if (s != NX_C_OK) nx_c_raise("copy", s);

  int64_t total = nx_c_prod(out.ndim, out.shape);
  if (total != nx_c_prod(in.ndim, in.shape) || !nx_c_is_contiguous_off0(&out) ||
      !nx_c_is_contiguous_off0(&in))
    nx_c_raise("copy", NX_C_ERR_PACKED);
  (void)dt; /* packedness already established by the caller */

  /* Copy the WHOLE bytes both operands own outright through the u8 kernel
     (threaded + lock-handled for large buffers). An odd element count leaves a
     final byte whose HIGH nibble is the neighbor of element `total` — outside
     this tensor — while the last element sits in that byte's LOW nibble.
     memcpy-ing the whole byte would clobber the neighbor: harmless for
     copy/contiguous (fresh full-buffer dst, private padding) but a lost update
     when this shared stub backs an assign into an odd-length contiguous prefix
     sub-view. So merge only the low nibble, preserving the dst's high nibble. */
  int64_t whole = total / 2;
  if (whole > 0) {
    nx_c_ndarray bout = out, bin = in;
    bout.ndim = bin.ndim = 1;
    bout.shape[0] = bin.shape[0] = whole;
    bout.strides[0] = bin.strides[0] = 1;
    bout.offset = bin.offset = 0;
    int64_t e2[2] = {1, 1};
    nx_c_ndarray ops[2] = {bout, bin};
    s = nx_c_map_run(&nx_c_copy_table, NX_C_DTYPE_u8, 1, ops, e2,
                    NX_C_COST_BANDWIDTH, NULL);
    if (s != NX_C_OK) nx_c_raise("copy", s);
  }
  if (total & 1) {
    uint8_t *d = (uint8_t *)out.data + whole;
    const uint8_t *sp = (const uint8_t *)in.data + whole;
    *d = (uint8_t)((*d & 0xF0) | (*sp & 0x0F));
  }
}

CAMLprim value caml_nx_c_copy(value vout, value vin) {
  CAMLparam2(vout, vin);
  nx_c_dtype dt = nx_c_dtype_of_value(vout);
  if (dt != NX_C_DTYPE_COUNT && nx_c_dtype_is_packed(dt)) {
    nx_c_copy_packed(vout, vin, dt);
  } else {
    value vals[2] = {vout, vin};
    nx_c_map_funnel("copy", &nx_c_copy_table, NX_C_COST_BANDWIDTH, 1, vals, NULL);
  }
  CAMLreturn(Val_unit);
}

/* ── pad ────────────────────────────────────────────────────────────────────

   Border fills + interior copy, all engine-driven, no bespoke iterator — each
   output element is written exactly once (~output-size traffic, not the 2× of
   a fill-everything-then-copy). The border is tiled into disjoint slabs: slab
   d confines axes < d to the interior, puts axis d in one of its two pad
   bands, and leaves axes > d full; the union over d is exactly
   output ∖ interior. Every slab is a sub-view filled by broadcasting the fill
   scalar (a same-dtype 1-element operand with all strides zeroed) through the
   copy kernel, so each pass inherits coalescing, threading, and the lock
   handshake. The fill value crosses the FFI as a scalar tensor rather than a
   per-dtype value: the binding, which knows the OCaml element type statically,
   sets it with a typed Bigarray store, so C needs no per-dtype value-extraction
   switch — and every pass reuses the one copy table. Packed dtypes are rejected
   by the copy table's NULL slots. */

static nx_c_status nx_c_pad_fill(const nx_c_ndarray *slab, const nx_c_ndarray *fill,
                               nx_c_dtype dt, const int64_t *e2) {
  nx_c_ndarray fs = *slab; /* borrows the slab's shape; strides zeroed */
  fs.data = fill->data;
  fs.offset = fill->offset;
  for (int d = 0; d < fs.ndim; d++) fs.strides[d] = 0;
  nx_c_ndarray ops[2] = {*slab, fs};
  return nx_c_map_run(&nx_c_copy_table, dt, 1, ops, e2, NX_C_COST_BANDWIDTH, NULL);
}

CAMLprim value caml_nx_c_pad(value vout, value vin, value vfill,
                            value vpad_before) {
  CAMLparam4(vout, vin, vfill, vpad_before);
  nx_c_ndarray out, in, fill;
  nx_c_status s = nx_c_ndarray_of_value(vout, &out);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vin, &in);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vfill, &fill);
  if (s != NX_C_OK) nx_c_raise("pad", s);

  nx_c_dtype dt = nx_c_dtype_of_value(vout);
  if (dt == NX_C_DTYPE_COUNT) nx_c_raise("pad", NX_C_ERR_BAD_KIND);
  if (out.ndim != in.ndim || (int)Wosize_val(vpad_before) != out.ndim)
    nx_c_raise_invalid("pad", NX_C_ERR_SHAPE);
  int64_t esize = nx_c_elem_size(dt);
  int64_t e2[2] = {esize, esize};

  /* `slab` narrows toward the interior: entering iteration d, axes < d carry
     the interior shape/offset and axes >= d are still full. The frontend
     rejects negative padding, so 0 <= before and 0 <= after. */
  nx_c_ndarray slab = out;
  for (int d = 0; d < out.ndim; d++) {
    int64_t before = Long_val(Field(vpad_before, d));
    int64_t after = out.shape[d] - in.shape[d] - before;
    if (before > 0) {
      slab.shape[d] = before;
      s = nx_c_pad_fill(&slab, &fill, dt, e2);
      if (s != NX_C_OK) nx_c_raise("pad", s);
    }
    if (after > 0) {
      slab.shape[d] = after;
      slab.offset += (before + in.shape[d]) * out.strides[d];
      s = nx_c_pad_fill(&slab, &fill, dt, e2);
      if (s != NX_C_OK) nx_c_raise("pad", s);
      slab.offset -= (before + in.shape[d]) * out.strides[d];
    }
    slab.shape[d] = in.shape[d];
    slab.offset += before * out.strides[d];
  }
  /* slab is now exactly the interior sub-view */
  nx_c_ndarray copy_ops[2] = {slab, in};
  s = nx_c_map_run(&nx_c_copy_table, dt, 1, copy_ops, e2, NX_C_COST_BANDWIDTH,
                  NULL);
  if (s != NX_C_OK) nx_c_raise("pad", s);
  CAMLreturn(Val_unit);
}

/* ── cat ────────────────────────────────────────────────────────────────────

   N strided copies, one per member, into successive slices of the output along
   the concat axis. Each member copy is its own nx_c_map_run, so the runtime lock
   is released and re-acquired per member per the engine's size policy — never
   held across the whole concatenation. */
CAMLprim value caml_nx_c_cat(value vout, value vinputs, value vaxis) {
  CAMLparam3(vout, vinputs, vaxis);
  int axis = Int_val(vaxis);
  nx_c_ndarray out;
  nx_c_status s = nx_c_ndarray_of_value(vout, &out);
  if (s != NX_C_OK) nx_c_raise("cat", s);
  nx_c_dtype dt = nx_c_dtype_of_value(vout);
  if (dt == NX_C_DTYPE_COUNT) nx_c_raise("cat", NX_C_ERR_BAD_KIND);
  if (axis < 0 || axis >= out.ndim) nx_c_raise_invalid("cat", NX_C_ERR_AXIS);
  int64_t esize = nx_c_elem_size(dt);
  int64_t e2[2] = {esize, esize};

  int n = (int)Wosize_val(vinputs);
  int64_t pos = 0;
  for (int m = 0; m < n; m++) {
    nx_c_ndarray in;
    s = nx_c_ndarray_of_value(Field(vinputs, m), &in);
    if (s != NX_C_OK) nx_c_raise("cat", s);
    if (in.ndim != out.ndim) nx_c_raise_invalid("cat", NX_C_ERR_SHAPE);
    nx_c_ndarray slice = out;
    slice.offset = out.offset + pos * out.strides[axis];
    for (int d = 0; d < out.ndim; d++) {
      slice.shape[d] = in.shape[d];
      slice.strides[d] = out.strides[d];
    }
    nx_c_ndarray ops[2] = {slice, in};
    s = nx_c_map_run(&nx_c_copy_table, dt, 1, ops, e2, NX_C_COST_BANDWIDTH, NULL);
    if (s != NX_C_OK) nx_c_raise("cat", s);
    pos += in.shape[axis];
  }
  CAMLreturn(Val_unit);
}

/* ── gather ──────────────────────────────────────────────────────────────────

   out[c] = data[c with axis -> indices[c]], over the output/index space (they
   share a shape). Indices are int32, Python-wrapped then bounds-checked. Reads
   are all disjoint across outputs, so the copy parallelizes freely over output
   elements. The bounds check is folded into the copy body: a worker wraps and
   range-checks each index immediately before the read it guards, so a bad index
   is caught before it can fault. Faults are reported through per-worker status
   slots (race-free — each worker writes only its own) and aggregated after the
   join. This replaces a serial pre-scan of the whole index space — pure overhead,
   and (on a column-broadcast index) redundant by the column count, since the row
   fast path consumes one index per row, not one per output element. */

/* Per-worker fault slots live on the driver stack. MUST be >= the engine pool cap
   NX_C_MAX_THREADS (nx_c_engine.c): a worker index reaches nthreads-1 and the pool
   clamps nthreads to that cap. Same literal-64 bound as nx_c_linalg.c's
   LA_MAX_WORKERS. */
#define MOVE_MAX_WORKERS 64

typedef struct {
  const nx_c_ndarray *data;
  const nx_c_ndarray *indices;
  const nx_c_ndarray *out;
  int axis;
  int64_t esize;
  nx_c_status *status;
} nx_c_gather_ctx;

static void nx_c_gather_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  const nx_c_gather_ctx *g = vctx;
  const nx_c_ndarray *data = g->data;
  const nx_c_ndarray *idx = g->indices;
  const nx_c_ndarray *out = g->out;
  int nd = out->ndim, axis = g->axis;
  int64_t esize = g->esize, axis_len = data->shape[axis];
  int64_t coord[NX_C_MAX_NDIM], dcoord[NX_C_MAX_NDIM];
  for (int64_t it = lo; it < hi; it++) {
    nx_c_unravel(it, nd, out->shape, coord);
    int64_t idx_off = idx->offset + nx_c_dot(nd, coord, idx->strides);
    int64_t index = *(const int32_t *)((const char *)idx->data +
                                        idx_off * (int64_t)sizeof(int32_t));
    if (index < 0) index += axis_len;
    if (index < 0 || index >= axis_len) {
      if (g->status[worker] == NX_C_OK) g->status[worker] = NX_C_ERR_INDEX_OOB;
      return;
    }
    for (int d = 0; d < nd; d++) dcoord[d] = (d == axis) ? index : coord[d];
    int64_t data_off = data->offset + nx_c_dot(nd, dcoord, data->strides);
    int64_t out_off = out->offset + nx_c_dot(nd, coord, out->strides);
    memcpy((char *)out->data + out_off * esize,
           (const char *)data->data + data_off * esize, (size_t)esize);
  }
}

/* Fast path: axis-0 2-D gather with a column-broadcast index (indices stride 1
   == 0) over contiguous data/out — every output row is a whole source row, so
   copy rows, not elements. The stride-0 column axis makes the index constant
   across a row, so the body reads and validates one index per row (`rows` checks,
   not the broadcast index space). */
typedef struct {
  const nx_c_ndarray *data;
  const nx_c_ndarray *indices;
  const nx_c_ndarray *out;
  int64_t esize;
  int64_t row_elems;
  nx_c_status *status;
} nx_c_gather_rows_ctx;

static void nx_c_gather_rows_body(int64_t lo, int64_t hi, int worker,
                                 void *vctx) {
  const nx_c_gather_rows_ctx *g = vctx;
  const nx_c_ndarray *data = g->data;
  const nx_c_ndarray *idx = g->indices;
  const nx_c_ndarray *out = g->out;
  int64_t axis_len = data->shape[0];
  size_t row_bytes = (size_t)g->row_elems * g->esize;
  for (int64_t i = lo; i < hi; i++) {
    int64_t idx_off = idx->offset + i * idx->strides[0];
    int64_t index = *(const int32_t *)((const char *)idx->data +
                                       idx_off * (int64_t)sizeof(int32_t));
    if (index < 0) index += axis_len;
    if (index < 0 || index >= axis_len) {
      if (g->status[worker] == NX_C_OK) g->status[worker] = NX_C_ERR_INDEX_OOB;
      return;
    }
    int64_t src = data->offset + index * data->strides[0];
    int64_t dst = out->offset + i * out->strides[0];
    memcpy((char *)out->data + dst * g->esize,
           (const char *)data->data + src * g->esize, row_bytes);
  }
}

static nx_c_status nx_c_gather_run(const nx_c_ndarray *data,
                                 const nx_c_ndarray *indices,
                                 const nx_c_ndarray *out, int axis,
                                 int64_t esize) {
  if (axis < 0 || axis >= data->ndim) return NX_C_ERR_AXIS;
  if (data->ndim != indices->ndim || data->ndim != out->ndim)
    return NX_C_ERR_SHAPE;
  for (int d = 0; d < out->ndim; d++)
    if (out->shape[d] != indices->shape[d]) return NX_C_ERR_SHAPE;

  int64_t total = nx_c_prod(out->ndim, out->shape);
  if (total == 0) return NX_C_OK;

  /* The bodies fault-check inline and report OOB indices through these slots; the
     pool caps workers at MOVE_MAX_WORKERS, so every slot a body touches is
     initialized here and read back after the join. */
  nx_c_status status[MOVE_MAX_WORKERS];
  for (int w = 0; w < MOVE_MAX_WORKERS; w++) status[w] = NX_C_OK;

  if (axis == 0 && data->ndim == 2 && indices->strides[1] == 0 &&
      data->shape[1] == out->shape[1] && nx_c_is_contiguous_off0(data) &&
      nx_c_is_contiguous_off0(out)) {
    int64_t rows = out->shape[0], row_elems = out->shape[1];
    nx_c_gather_rows_ctx g = {data, indices, out, esize, row_elems, status};
    int64_t bytes = 2 * rows * row_elems * esize;
    nx_c_move_dispatch(NX_C_COST_BANDWIDTH, rows, row_elems, bytes,
                      nx_c_gather_rows_body, &g);
  } else {
    nx_c_gather_ctx g = {data, indices, out, axis, esize, status};
    int64_t bytes = 2 * total * esize;
    nx_c_move_dispatch(NX_C_COST_BANDWIDTH, total, 1, bytes, nx_c_gather_body, &g);
  }

  for (int w = 0; w < MOVE_MAX_WORKERS; w++)
    if (status[w] != NX_C_OK) return status[w];
  return NX_C_OK;
}

CAMLprim value caml_nx_c_gather(value vout, value vdata, value vindices,
                               value vaxis) {
  CAMLparam4(vout, vdata, vindices, vaxis);
  nx_c_ndarray out, data, indices;
  nx_c_status s = nx_c_ndarray_of_value(vout, &out);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vdata, &data);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vindices, &indices);
  if (s != NX_C_OK) nx_c_raise("gather", s);
  nx_c_dtype dt = nx_c_dtype_of_value(vout);
  if (dt == NX_C_DTYPE_COUNT) nx_c_raise("gather", NX_C_ERR_BAD_KIND);
  if (nx_c_dtype_is_packed(dt)) nx_c_raise("gather", NX_C_ERR_PACKED);
  s = nx_c_gather_run(&data, &indices, &out, Int_val(vaxis), nx_c_elem_size(dt));
  if (s != NX_C_OK) nx_c_raise_status("gather", s);
  CAMLreturn(Val_unit);
}

/* ── scatter ─────────────────────────────────────────────────────────────────

   out (already initialized to the template by the binding) receives updates at
   out[c with axis -> indices[c]] for each index-space point c. `Set overwrites
   (last write in row-major scan order wins); `Add accumulates. Scatter runs
   SERIALLY: `Set's last-wins is only well defined under a fixed order, and a
   parallel `Add over duplicate targets is an unsynchronized read-modify-write
   race. A serial row-major walk makes both modes
   deterministic and race-free with no partitioning or atomics; unique_indices
   could unlock a parallel path but is not needed for correctness and buys
   nothing on the ops that use scatter, so it is accepted and ignored. Add uses
   a per-dtype accumulate (compute-typed load/add/store); Set is a bit-exact
   byte copy. Serial does NOT mean under the runtime lock: the walk is a
   one-worker body driven through nx_c_parallel_for, which runs it in order on
   the calling thread and releases/re-acquires the lock around it per the
   engine's size cutoff, like every other kernel. */

/* Compute-typed add for scatter-add and fold's accumulate. The signed form
   runs in the unsigned width: the contract is modular wrap (SINT compute is
   always int64_t — dtype table), defined without -fwrapv, which stays in the
   flags as belt and suspenders. */
#define NX_C_MOVE_ADD_NX_C_CAT_SINT(a, b)                                        \
  ((int64_t)((uint64_t)(a) + (uint64_t)(b)))
#define NX_C_MOVE_ADD_NX_C_CAT_UINT(a, b) ((a) + (b))
#define NX_C_MOVE_ADD_NX_C_CAT_FLOAT(a, b) ((a) + (b))
#define NX_C_MOVE_ADD_NX_C_CAT_COMPLEX(a, b) ((a) + (b))
#define NX_C_MOVE_ADD_NX_C_CAT_BOOL(a, b) ((a) + (b))

typedef void nx_c_scatter_add_fn(char *out, const char *upd);
#define NX_C_SCATTER_ADD(sfx, kind, storage, compute, ld, st, cat)             \
  static void nx_c_scatter_add_##sfx(char *out, const char *upd) {             \
    nx_c_st_##sfx(out,                                                         \
                 NX_C_MOVE_ADD_##cat(nx_c_ld_##sfx(out), nx_c_ld_##sfx(upd)));   \
  }
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_SCATTER_ADD)
#undef NX_C_SCATTER_ADD

static nx_c_scatter_add_fn *const nx_c_scatter_add_tbl[NX_C_DTYPE_COUNT] = {
#define NX_C_SCATTER_ADD_ROW(sfx, kind, storage, compute, ld, st, cat)         \
  [NX_C_DTYPE_##sfx] = nx_c_scatter_add_##sfx,
    NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_SCATTER_ADD_ROW)
#undef NX_C_SCATTER_ADD_ROW
};

typedef struct {
  const nx_c_ndarray *out;
  const nx_c_ndarray *indices;
  const nx_c_ndarray *updates;
  nx_c_scatter_add_fn *add; /* NULL = Set (bit-exact byte copy) */
  int axis;
  int64_t esize;
  nx_c_status *status; /* one slot: the walk runs on a single worker */
} nx_c_scatter_ctx;

static void nx_c_scatter_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  (void)worker;
  const nx_c_scatter_ctx *sc = vctx;
  const nx_c_ndarray *out = sc->out;
  const nx_c_ndarray *indices = sc->indices;
  const nx_c_ndarray *updates = sc->updates;
  int nd = indices->ndim, axis = sc->axis;
  int64_t esize = sc->esize, axis_len = out->shape[axis];
  int64_t coord[NX_C_MAX_NDIM], ocoord[NX_C_MAX_NDIM];
  for (int64_t it = lo; it < hi; it++) {
    nx_c_unravel(it, nd, indices->shape, coord);
    int64_t idx_off = indices->offset + nx_c_dot(nd, coord, indices->strides);
    int64_t index = *(const int32_t *)((const char *)indices->data +
                                        idx_off * (int64_t)sizeof(int32_t));
    if (index < 0) index += axis_len;
    if (index < 0 || index >= axis_len) {
      *sc->status = NX_C_ERR_INDEX_OOB;
      return;
    }
    for (int d = 0; d < nd; d++) ocoord[d] = (d == axis) ? index : coord[d];
    int64_t out_off = out->offset + nx_c_dot(nd, ocoord, out->strides);
    int64_t upd_off = updates->offset + nx_c_dot(nd, coord, updates->strides);
    char *op = (char *)out->data + out_off * esize;
    const char *up = (const char *)updates->data + upd_off * esize;
    if (sc->add)
      sc->add(op, up);
    else
      memcpy(op, up, (size_t)esize);
  }
}

static nx_c_status nx_c_scatter_run(const nx_c_ndarray *out,
                                  const nx_c_ndarray *indices,
                                  const nx_c_ndarray *updates, int axis, int mode,
                                  nx_c_dtype dt, int64_t esize) {
  if (axis < 0 || axis >= out->ndim) return NX_C_ERR_AXIS;
  if (out->ndim != indices->ndim || out->ndim != updates->ndim)
    return NX_C_ERR_SHAPE;
  for (int d = 0; d < out->ndim; d++) {
    if (indices->shape[d] != updates->shape[d]) return NX_C_ERR_SHAPE;
    if (d != axis && indices->shape[d] != out->shape[d]) return NX_C_ERR_SHAPE;
  }
  nx_c_scatter_add_fn *add = (mode == 1) ? nx_c_scatter_add_tbl[dt] : NULL;
  if (mode == 1 && add == NULL) return NX_C_ERR_UNSUPPORTED_DTYPE;

  int nd = indices->ndim;
  int64_t total = nx_c_prod(nd, indices->shape);
  if (total == 0) return NX_C_OK;
  /* ONE worker keeps the row-major order (Set last-wins, Add accumulation
     order) deterministic; nx_c_parallel_for still owns the lock handshake. */
  nx_c_status status = NX_C_OK;
  nx_c_scatter_ctx sc = {out, indices, updates, add, axis, esize, &status};
  int64_t bytes = total * (2 * esize + (int64_t)sizeof(int32_t));
  nx_c_parallel_for(1, total, bytes, nx_c_scatter_body, &sc, NULL);
  return status;
}

CAMLprim value caml_nx_c_scatter(value vout, value vindices, value vupdates,
                                value vaxis, value vmode) {
  CAMLparam5(vout, vindices, vupdates, vaxis, vmode);
  nx_c_ndarray out, indices, updates;
  nx_c_status s = nx_c_ndarray_of_value(vout, &out);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vindices, &indices);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vupdates, &updates);
  if (s != NX_C_OK) nx_c_raise("scatter", s);
  nx_c_dtype dt = nx_c_dtype_of_value(vout);
  if (dt == NX_C_DTYPE_COUNT) nx_c_raise("scatter", NX_C_ERR_BAD_KIND);
  if (nx_c_dtype_is_packed(dt)) nx_c_raise("scatter", NX_C_ERR_PACKED);
  s = nx_c_scatter_run(&out, &indices, &updates, Int_val(vaxis), Int_val(vmode),
                      dt, nx_c_elem_size(dt));
  if (s != NX_C_OK) nx_c_raise_status("scatter", s);
  CAMLreturn(Val_unit);
}

/* ── Window geometry shared by unfold/fold ─────────────────────────────────*/

typedef struct {
  int leading_ndim;
  int K;
  int64_t esize;
  int64_t kernel[NX_C_MAX_SPATIAL];
  int64_t stride[NX_C_MAX_SPATIAL];
  int64_t dilation[NX_C_MAX_SPATIAL];
  int64_t pad_before[NX_C_MAX_SPATIAL];
  int64_t pad_after[NX_C_MAX_SPATIAL];
  int64_t win[NX_C_MAX_SPATIAL];            /* windows per spatial dim */
  int64_t win_cumprod[NX_C_MAX_SPATIAL];    /* row-major cumprod of win */
  int64_t kernel_cumprod[NX_C_MAX_SPATIAL]; /* row-major cumprod of kernel */
  int64_t output_size[NX_C_MAX_SPATIAL];    /* fold only */
  int64_t kernel_prod;
  int64_t L;
  int64_t leading_size;
} nx_c_window;

/* Read the K-long parameter arrays. `padding` is flat [before0, after0, ...] of
   length 2K. */
static void nx_c_window_params(nx_c_window *w, value vkernel, value vstride,
                              value vdilation, value vpadding, int K,
                              int leading_ndim, int64_t esize) {
  w->K = K;
  w->leading_ndim = leading_ndim;
  w->esize = esize;
  for (int d = 0; d < K; d++) {
    w->kernel[d] = Long_val(Field(vkernel, d));
    w->stride[d] = Long_val(Field(vstride, d));
    w->dilation[d] = Long_val(Field(vdilation, d));
    w->pad_before[d] = Long_val(Field(vpadding, 2 * d));
    w->pad_after[d] = Long_val(Field(vpadding, 2 * d + 1));
  }
}

/* Per-dim window count from the (unpadded) spatial extents, plus the row-major
   cumprods used to decompose a flat window / kernel index. win[d] is exactly the
   reference's L factor: how many strided placements of the effective (dilated)
   kernel fit in the padded extent. spatial_extent is the input extent for unfold
   and the output extent for fold. */
static void nx_c_window_counts(nx_c_window *w, const int64_t *spatial_extent) {
  int K = w->K;
  for (int d = 0; d < K; d++) {
    int64_t eff = w->dilation[d] * (w->kernel[d] - 1) + 1;
    int64_t padded = spatial_extent[d] + w->pad_before[d] + w->pad_after[d];
    int64_t win = (padded - eff) / w->stride[d] + 1;
    w->win[d] = win < 1 ? 1 : win;
  }
  w->win_cumprod[K - 1] = 1;
  for (int d = K - 2; d >= 0; d--)
    w->win_cumprod[d] = w->win_cumprod[d + 1] * w->win[d + 1];
  w->kernel_cumprod[K - 1] = 1;
  for (int d = K - 2; d >= 0; d--)
    w->kernel_cumprod[d] = w->kernel_cumprod[d + 1] * w->kernel[d + 1];
}

/* ── unfold ──────────────────────────────────────────────────────────────────

   (leading..., spatial...) -> (leading..., kernel_prod, L). Each output element
   (lead, kernel-tap kf, window w) reads one input element, or zero when the tap
   falls in the pad. Every output is written exactly once, so the whole thing
   parallelizes over output elements with no coordination; zeroing a padded tap
   is memset of one element (esize > 0 because packed storage is rejected). */

typedef struct {
  const nx_c_window *w;
  const nx_c_ndarray *in;  /* (leading..., spatial...) */
  const nx_c_ndarray *out; /* (leading..., kernel_prod, L) */
} nx_c_unfold_ctx;

static void nx_c_unfold_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  (void)worker;
  const nx_c_unfold_ctx *u = vctx;
  const nx_c_window *w = u->w;
  const nx_c_ndarray *in = u->in, *out = u->out;
  int ld = w->leading_ndim, K = w->K;
  int64_t esize = w->esize;
  int64_t lead_coord[NX_C_MAX_NDIM];
  for (int64_t it = lo; it < hi; it++) {
    int64_t win_flat = it % w->L;
    int64_t rem = it / w->L;
    int64_t kf = rem % w->kernel_prod;
    int64_t lead = rem / w->kernel_prod;

    nx_c_unravel(lead, ld, in->shape, lead_coord);
    int64_t out_off = out->offset + nx_c_dot(ld, lead_coord, out->strides) +
                      kf * out->strides[ld] + win_flat * out->strides[ld + 1];
    int64_t in_off = in->offset + nx_c_dot(ld, lead_coord, in->strides);
    bool valid = true;
    for (int d = 0; d < K; d++) {
      int64_t wc = (win_flat / w->win_cumprod[d]) % w->win[d];
      int64_t kc = (kf / w->kernel_cumprod[d]) % w->kernel[d];
      int64_t sp = wc * w->stride[d] + kc * w->dilation[d] - w->pad_before[d];
      if (sp < 0 || sp >= in->shape[ld + d]) {
        valid = false;
        break;
      }
      in_off += sp * in->strides[ld + d];
    }
    char *dst = (char *)out->data + out_off * esize;
    if (valid)
      memcpy(dst, (const char *)in->data + in_off * esize, (size_t)esize);
    else
      memset(dst, 0, (size_t)esize);
  }
}

/* ── fold (col2im) ───────────────────────────────────────────────────────────

   (leading..., kernel_prod, L) -> (leading..., output...). Overlapping windows
   sum. Parallelized over OUTPUT elements: each output computes a fresh sum of
   the input taps that map onto it (inverse of unfold's window placement), so
   two threads never touch the same output — race-free by construction, with no
   separate zeroing pass, since an output with
   no contributing tap simply stores its zero-seeded accumulator. The
   accumulate/store is per-dtype (compute-typed, so low-precision sums in float
   and small ints in 64-bit); the tap search is dtype-blind. */

typedef struct {
  void (*zero)(nx_c_acc *acc);
  void (*accum)(nx_c_acc *acc, const char *in);
  void (*store)(char *out, const nx_c_acc *acc);
} nx_c_foldelem;

#define NX_C_FOLD_OPS(sfx, kind, storage, compute, ld, st, cat)                \
  static void nx_c_fold_zero_##sfx(nx_c_acc *a) { *(compute *)a = (compute)0; } \
  static void nx_c_fold_accum_##sfx(nx_c_acc *a, const char *in) {              \
    *(compute *)a =                                                           \
        (compute)NX_C_MOVE_ADD_##cat(*(compute *)a, nx_c_ld_##sfx(in));         \
  }                                                                           \
  static void nx_c_fold_store_##sfx(char *out, const nx_c_acc *a) {             \
    nx_c_st_##sfx(out, *(const compute *)a);                                   \
  }
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_FOLD_OPS)
#undef NX_C_FOLD_OPS

static const nx_c_foldelem nx_c_fold_tbl[NX_C_DTYPE_COUNT] = {
#define NX_C_FOLD_ROW(sfx, kind, storage, compute, ld, st, cat)                \
  [NX_C_DTYPE_##sfx] = {nx_c_fold_zero_##sfx, nx_c_fold_accum_##sfx,             \
                       nx_c_fold_store_##sfx},
    NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_FOLD_ROW)
#undef NX_C_FOLD_ROW
};

typedef struct {
  const nx_c_window *w;
  const nx_c_foldelem *ops;
  const nx_c_ndarray *in;  /* (leading..., kernel_prod, L) */
  const nx_c_ndarray *out; /* (leading..., output...) */
} nx_c_fold_ctx;

static void nx_c_fold_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  (void)worker;
  const nx_c_fold_ctx *f = vctx;
  const nx_c_window *w = f->w;
  const nx_c_ndarray *in = f->in, *out = f->out;
  int ld = w->leading_ndim, K = w->K;
  int64_t esize = w->esize;
  int64_t out_spatial = nx_c_prod(K, w->output_size);
  int64_t lead_coord[NX_C_MAX_NDIM], ocoord[NX_C_MAX_SPATIAL];
  for (int64_t it = lo; it < hi; it++) {
    int64_t o_lin = it % out_spatial;
    int64_t lead = it / out_spatial;
    nx_c_unravel(o_lin, K, w->output_size, ocoord);
    nx_c_unravel(lead, ld, out->shape, lead_coord);

    int64_t out_off = out->offset + nx_c_dot(ld, lead_coord, out->strides);
    for (int d = 0; d < K; d++) out_off += ocoord[d] * out->strides[ld + d];
    int64_t in_lead = in->offset + nx_c_dot(ld, lead_coord, in->strides);

    nx_c_acc acc;
    f->ops->zero(&acc);
    for (int64_t kf = 0; kf < w->kernel_prod; kf++) {
      bool valid = true;
      int64_t win_flat = 0;
      for (int d = 0; d < K; d++) {
        int64_t kc = (kf / w->kernel_cumprod[d]) % w->kernel[d];
        int64_t num = ocoord[d] + w->pad_before[d] - kc * w->dilation[d];
        if (num < 0 || num % w->stride[d] != 0) {
          valid = false;
          break;
        }
        int64_t wc = num / w->stride[d];
        if (wc >= w->win[d]) {
          valid = false;
          break;
        }
        win_flat += wc * w->win_cumprod[d];
      }
      if (!valid) continue;
      int64_t in_off =
          in_lead + kf * in->strides[ld] + win_flat * in->strides[ld + 1];
      f->ops->accum(&acc, (const char *)in->data + in_off * esize);
    }
    f->ops->store((char *)out->data + out_off * esize, &acc);
  }
}

CAMLprim value caml_nx_c_unfold(value vout, value vin, value vkernel,
                               value vstride, value vdilation, value vpadding) {
  CAMLparam5(vout, vin, vkernel, vstride, vdilation);
  CAMLxparam1(vpadding);
  nx_c_ndarray out, in;
  nx_c_status s = nx_c_ndarray_of_value(vout, &out);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vin, &in);
  if (s != NX_C_OK) nx_c_raise("unfold", s);
  nx_c_dtype dt = nx_c_dtype_of_value(vout);
  if (dt == NX_C_DTYPE_COUNT) nx_c_raise("unfold", NX_C_ERR_BAD_KIND);
  if (nx_c_dtype_is_packed(dt)) nx_c_raise("unfold", NX_C_ERR_PACKED);

  int K = (int)Wosize_val(vkernel);
  if (K < 1 || K > NX_C_MAX_SPATIAL || in.ndim < K)
    nx_c_raise_invalid("unfold", NX_C_ERR_SHAPE);
  int ld = in.ndim - K;
  nx_c_window w;
  nx_c_window_params(&w, vkernel, vstride, vdilation, vpadding, K, ld,
                    nx_c_elem_size(dt));
  w.kernel_prod = out.shape[ld];
  w.L = out.shape[ld + 1];
  w.leading_size = nx_c_prod(ld, in.shape);
  nx_c_window_counts(&w, &in.shape[ld]); /* windows per input spatial dim */

  int64_t total = w.leading_size * w.kernel_prod * w.L;
  if (total > 0) {
    nx_c_unfold_ctx u = {&w, &in, &out};
    int64_t bytes = 2 * total * w.esize;
    nx_c_move_dispatch(NX_C_COST_BANDWIDTH, total, 1, bytes, nx_c_unfold_body, &u);
  }
  CAMLreturn(Val_unit);
}

CAMLprim value caml_nx_c_unfold_bc(value *argv, int argn) {
  (void)argn;
  return caml_nx_c_unfold(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

CAMLprim value caml_nx_c_fold(value vout, value vin, value voutput_size,
                             value vkernel, value vstride, value vdilation,
                             value vpadding) {
  CAMLparam5(vout, vin, voutput_size, vkernel, vstride);
  CAMLxparam2(vdilation, vpadding);
  nx_c_ndarray out, in;
  nx_c_status s = nx_c_ndarray_of_value(vout, &out);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vin, &in);
  if (s != NX_C_OK) nx_c_raise("fold", s);
  nx_c_dtype dt = nx_c_dtype_of_value(vout);
  if (dt == NX_C_DTYPE_COUNT) nx_c_raise("fold", NX_C_ERR_BAD_KIND);
  const nx_c_foldelem *ops = &nx_c_fold_tbl[dt];
  if (ops->accum == NULL)
    nx_c_raise("fold", nx_c_dtype_is_packed(dt) ? NX_C_ERR_PACKED
                                              : NX_C_ERR_UNSUPPORTED_DTYPE);

  int K = (int)Wosize_val(vkernel);
  if (K < 1 || K > NX_C_MAX_SPATIAL || in.ndim < 2)
    nx_c_raise_invalid("fold", NX_C_ERR_SHAPE);
  int ld = in.ndim - 2;
  nx_c_window w;
  nx_c_window_params(&w, vkernel, vstride, vdilation, vpadding, K, ld,
                    nx_c_elem_size(dt));
  w.kernel_prod = in.shape[ld];
  w.L = in.shape[ld + 1];
  w.leading_size = nx_c_prod(ld, in.shape);
  for (int d = 0; d < K; d++) w.output_size[d] = Long_val(Field(voutput_size, d));
  nx_c_window_counts(&w, w.output_size); /* windows per output spatial dim */

  int64_t out_spatial = nx_c_prod(K, w.output_size);
  int64_t total = w.leading_size * out_spatial;
  if (total > 0) {
    nx_c_fold_ctx f = {&w, ops, &in, &out};
    int64_t bytes = total * w.kernel_prod * w.esize;
    nx_c_move_dispatch(NX_C_COST_COMPUTE, total, w.kernel_prod, bytes,
                      nx_c_fold_body, &f);
  }
  CAMLreturn(Val_unit);
}

CAMLprim value caml_nx_c_fold_bc(value *argv, int argn) {
  (void)argn;
  return caml_nx_c_fold(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                       argv[6]);
}
