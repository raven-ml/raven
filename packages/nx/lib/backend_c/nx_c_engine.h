/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_engine.h — the engine's contract with the kernel-family stub files.

   nx_c.h owns the dtype table, the kernel ABIs, the dispatch-table types, the
   status protocol, and the parallel-policy declaration. This header owns the
   layer above the kernels: the drivers a family stub calls once it has
   extracted its operands. A driver is the single owner of dispatch null-checks,
   dimension coalescing, the iteration ladder, thread splitting over the engine
   pool, and the runtime-lock handshake. A driver never raises: it returns a
   status, and the family stub turns a non-NULL status into an OCaml exception
   with the op name via nx_c_raise / nx_c_raise_invalid (nx_c.h). Kept separate
   from nx_c.h so the reviewed header stays untouched; the two are the whole
   below-frontend contract. */

#ifndef NX_C_ENGINE_H
#define NX_C_ENGINE_H

#include "nx_c.h"

/* Contract for every driver and funnel below (family authors, read this):
   - Call only with the OCaml runtime lock held — as a CAMLprim naturally does.
     The driver releases and re-acquires it internally around large parallel
     work; a caller must never invoke one from a pool worker thread.
   - The engine runs ONE parallel region at a time. Drivers are not re-entrant
     and kernels must not call back into a driver: nested parallelism is
     unsupported (it would deadlock on the pool's single job slot). */

/* Most operands any generated family takes: where() is 3 inputs + 1 output. */
#define NX_C_MAX_OPERANDS 4

/* Argmax/argmin axis longer than this cannot be indexed by the int32 result
   (nx_c_arg_fini truncates to int32); the driver rejects it up front rather than
   returning a truncated index. Maps to Failure in the binding. */
#define NX_C_ERR_ARGREDUCE_CAP "argreduce axis length exceeds INT32_MAX"

/* Preconditions the drivers verify rather than assume (a binding that forgets
   to sort axes, squeeze the output descriptor, or bounds-check the axis gets a
   loud status, never silent corruption). The funnels map these to
   Invalid_argument. */
#define NX_C_ERR_AXES "reduce axes must be strictly increasing and in range"
#define NX_C_ERR_OUT_RANK "output rank inconsistent with the operation"
#define NX_C_ERR_AXIS "axis out of range"
#define NX_C_ERR_OUT_ALIASED "output has a broadcast (zero) stride"
/* A programming error in a family stub, not user data: more operands than the
   engine's fixed metadata arrays hold. Maps to Failure. */
#define NX_C_ERR_ARITY "operand count exceeds NX_C_MAX_OPERANDS"

/* The single status -> exception-kind classifier, so every family (map, move,
   sort, ...) raises identically: Invalid_argument for precondition and empty-
   axis violations (a bad user/binding argument), Failure for everything else
   (unsupported dtype, packed, bad kind, argreduce cap, allocation). A family
   stub that catches a driver's non-NULL status routes it here rather than
   hand-rolling its own map — that divergence is exactly what this forecloses.
   Same contract as nx_c_raise/nx_c_raise_invalid (nx_c.h): call ONLY with the
   runtime lock held. Implemented in nx_c_engine.c. */
NX_C_NORETURN void nx_c_raise_status(const char *op, nx_c_status status);

/* ── Parallel execution primitive (custom families) ────────────────────────

   The generated families (map/fold/argreduce/scan) ride their drivers above and
   never touch this. The CUSTOM families whose access is data- or structure-
   dependent — sort, gather/scatter, matmul, linalg — own their traversal but
   still parallelize through this one primitive, so there is exactly one thread
   pool and one parallel policy in the backend.

   `nx_c_parallel_for` cuts [0, total) into contiguous chunks (granularity is
   engine policy, from nthreads/total/bytes — nx_c_engine.c, beside the thread
   policy); the calling thread (worker 0) and the spawned pool workers then
   claim chunks from a shared counter until none remain, calling `body` once
   per claimed chunk — so a fast thread absorbs the tail a slow one would
   otherwise drag through the join. `body` receives the chunk [lo, hi), its
   thread's `worker` index (0 <= worker < nthreads, STABLE across every chunk
   that thread claims), and `ctx`. The worker index is the whole point of the
   export: it lets a body address a private per-thread scratch slot with no
   sharing and no locking. What a body may NOT assume: that it runs once per
   worker, that chunk c runs on worker c, that every worker index occurs, or
   any claim order — only that the chunks it is handed partition [0, total) and
   that per-worker slots (scratch, sticky error status) stay exclusive because
   the index is per-thread. nthreads <= 1 is a single whole-range call.

   The runtime-lock handshake lives INSIDE this call, so a family TU that only
   calls it has no reason to do the enter/leave itself. nx_c_engine.h transitively
   includes neither caml/threads.h nor caml/fail.h (verified), so family kernel
   code cannot raise an OCaml exception — the load-bearing safety property, since
   a raise under a released lock is the bug class the layering forecloses. Call
   this with the OCaml runtime lock HELD: it releases the lock internally iff the
   work warrants (nthreads > 1, or `bytes` over the engine's lock-release
   cutoff), runs the split, re-acquires, and returns with the lock held. `bytes`
   is the op's total traffic, the same figure fed to nx_c_threads_for.

   `free_on_exit` (nullable) is a heap block the primitive free()s AFTER the
   parallel join but BEFORE it re-acquires the runtime lock. The re-acquire is
   the one point that can raise an async exception (a signal handler or memprof
   callback runs there) and longjmp past the driver's own cleanup; freeing the
   driver's scratch here, before that point, makes the leak impossible on every
   path. Pass NULL when there is no scratch (the generated drivers do).

   Standard driver pattern for a custom family (this IS the contract):
     1. validate operands.
     2. nthreads = nx_c_threads_for(cls, ...)   — policy stays an explicit step,
        and it must come first: scratch is sized by nthreads.
     3. Allocate nthreads * per_slot_bytes of scratch. A failed allocation is the
        only place the driver reports a failure STATUS (NX_C_ERR_ALLOC; the funnel
        raises it).
     4. nx_c_parallel_for(nthreads, total, bytes, body, &ctx, scratch) — hand the
        scratch to the primitive as free_on_exit. Do NOT free it yourself: the
        primitive frees it leak-safely across the re-acquire's possible raise.
     5. return status.
   Bodies are pure C: they never touch the OCaml runtime, never raise, never
   allocate, and each indexes its private scratch by `worker`.

   Not re-entrant: the pool runs ONE parallel region at a time, so a body must
   never call back into a driver or into nx_c_parallel_for (it would deadlock on
   the single job slot). nthreads is clamped internally to the pool size, so the
   max worker index never exceeds the nthreads you pass — size scratch by that. */
typedef void (*nx_c_range_body)(int64_t lo, int64_t hi, int worker, void *ctx);
void nx_c_parallel_for(int nthreads, int64_t total, int64_t bytes,
                      nx_c_range_body body, void *ctx, void *free_on_exit);

/* ── Map driver ────────────────────────────────────────────────────────────
   Elementwise, one broadcast-resolved shape for every operand. ops[0] is the
   output, ops[1..nin] the inputs in argument order; all carry ops[0].ndim dims
   with the shared shape (ops[0].shape). Input strides MAY be 0 (broadcast);
   the output's never are. elem_size[k] is operand k's element byte size — they
   differ under cast (out dtype != in dtype) and where (bool condition). `dt`
   selects the kernel slot; a NULL slot becomes NX_C_ERR_UNSUPPORTED_DTYPE, or
   NX_C_ERR_PACKED for a packed dtype. `ctx` is the op's parameter block or
   NULL. */
nx_c_status nx_c_map_run(const nx_c_map_table *tbl, nx_c_dtype dt, int nin,
                       const nx_c_ndarray *ops, const int64_t *elem_size,
                       nx_c_cost_class cls, void *ctx);

/* ── Streaming reduction kernels (the fold driver's contiguous-input path) ───

   When a KEPT (output) axis is more contiguous than every reduced axis, the
   per-output path gathers each output's reduced run with a large stride — a
   fresh cache line per element for an axis-0 sum of a C-contiguous matrix. The
   driver's streaming path instead walks the input
   contiguously, folding each reduced "row" of `n` lane elements into an array of
   `n` accumulators, then converting the accumulators to the output slice. The
   lane is the most-contiguous kept axis, so the per-row fold vectorizes across
   lanes; reads are sequential over the whole input.

   `accs` is a driver-owned scratch of `n` accumulators in the op's COMPUTE type
   (so f16/bf16/fp8 keep their float accumulation and small ints their 64-bit
   accumulation, exactly as the per-output path). One accumulator per lane runs
   over the reduced extent sequentially — the multi-accumulator unrolling of the
   contiguous per-output step (nx_c_fold.c) is a property of THAT traversal;
   streaming's vectorization comes from the lanes instead. Each lane folds the
   reduced extent in a fixed order (input order over the reduced axes); for a
   single reduced axis that is exactly the per-output order, and sums stay within
   the conformance suite's reassociation tolerance in every case.

   stream: fold one reduced row into the `n` accumulators. `first != 0` seeds
     them from this row (accs[j] = load(in_row + j*lane_step)); `first == 0`
     folds it in (accs[j] <combine>= load(in_row + j*lane_step)). Op-specific.
   scatter: convert the `n` accumulators to storage and write the output slice
     (out[j*out_step] = accs[j]). Depends only on the dtype, so all four
     reduction tables share one instance per dtype.

   A NULL stream/scatter slot (packed dtype, or an op with no streaming form)
   makes the driver fall back to the per-output path — always correct. */
typedef void nx_c_fold_stream(void *accs, const char *in_row, int64_t lane_step,
                             int64_t n, int first, void *ctx);
typedef void nx_c_fold_scatter(char *out, int64_t out_step, const void *accs,
                              int64_t n, void *ctx);
typedef struct {
  nx_c_fold_stream *stream[NX_C_DTYPE_COUNT];
  nx_c_fold_scatter *scatter[NX_C_DTYPE_COUNT];
} nx_c_stream_table;

/* ── Fold driver ───────────────────────────────────────────────────────────
   Reduce `in` over `reduce_axes` (n_reduce input-axis indices, strictly
   increasing, each in [0, in->ndim)) into `out`. `out` has rank
   in->ndim - n_reduce with its axes aligned, in order, to the kept (non-reduced)
   input axes — the binding squeezes any keepdims size-1 dims out of the
   descriptor it passes here. in_elem / out_elem are the operands' element byte
   sizes (they differ when the reduce changes dtype, e.g. a widening sum).
   `no_identity` states the op's empty-axis policy, which only the op knows:
   false for sum/prod (init seeds the neutral identity, so an empty reduced
   extent stores it); true for max/min (no neutral identity). When it is true and
   the reduced extent is empty (product 0) with at least one output element, the
   driver returns NX_C_ERR_EMPTY_REDUCE before any kernel runs — so the -inf/
   INT64_MIN sentinel a max kernel would otherwise store can never escape, no
   matter which caller (funnel or raw) drives the fold.

   `stbl` (nullable) supplies the streaming kernels above; when it is non-NULL
   and a kept axis out-contiguities every reduced axis, the driver takes the
   streaming path. NULL forces the per-output path (a caller with no streaming
   form, e.g. an experimental op). */
nx_c_status nx_c_fold_run(const nx_c_fold_table *tbl, const nx_c_stream_table *stbl,
                        nx_c_dtype dt, const nx_c_ndarray *in, int64_t in_elem,
                        const nx_c_ndarray *out, int64_t out_elem,
                        const int *reduce_axes, int n_reduce, bool no_identity,
                        nx_c_cost_class cls, void *ctx);

/* ── Argreduce driver ──────────────────────────────────────────────────────
   Argmax/argmin over exactly one axis. `out` is int32, rank in->ndim - 1, its
   axes aligned in order to the non-`axis` input axes. Rejects an empty axis
   (NX_C_ERR_EMPTY_REDUCE) and an axis longer than INT32_MAX
   (NX_C_ERR_ARGREDUCE_CAP) before any work. */
nx_c_status nx_c_argreduce_run(const nx_c_arg_table *tbl, nx_c_dtype dt,
                             const nx_c_ndarray *in, int64_t in_elem,
                             const nx_c_ndarray *out, int axis,
                             nx_c_cost_class cls, void *ctx);

/* Validate one argreduce axis length without touching operands, so the binding
   (and tests) can reject an oversized axis without materializing it. */
nx_c_status nx_c_argreduce_validate(int64_t axis_len);

/* ── Scan driver ───────────────────────────────────────────────────────────
   Inclusive prefix scan over one axis; `out` has the same shape as `in`. The
   slices (the odometer over every non-`axis` dim) are independent and run in
   parallel; within a slice the walk is sequential. in_elem / out_elem are the
   operands' element byte sizes. */
nx_c_status nx_c_scan_run(const nx_c_scan_table *tbl, nx_c_dtype dt,
                        const nx_c_ndarray *in, int64_t in_elem,
                        const nx_c_ndarray *out, int64_t out_elem, int axis,
                        nx_c_cost_class cls, void *ctx);

/* ── The funnel: the ONE way a family stub reaches a driver ─────────────────

   A family file never hand-rolls the extract -> validate -> dispatch ->
   run -> raise sequence; it calls one of these. Each extracts every operand
   from its FFI record (nx_c_ndarray_of_value), derives per-operand element sizes
   and the dispatch dtype, squeezes the reduction output descriptor where
   applicable, runs the driver, and on a non-NULL status raises the right OCaml
   exception with the op name (Invalid_argument for precondition/empty-axis
   violations, Failure otherwise). They run with the runtime lock held (as any
   CAMLprim does) and perform no OCaml allocation before extraction, so the
   caller's CAMLparam roots suffice — these helpers take no roots of their own.

   `vals` for the map funnel is {output, input0, input1, ...} in argument order,
   length nin+1. The reduction funnels read `vaxes` (an OCaml int array, sorted
   ascending — the binding's responsibility, verified here) / `axis` and infer
   keepdims from the output rank. `no_identity` is the fold op's empty-axis
   policy (true for max/min, false for sum/prod), forwarded to nx_c_fold_run.
   `ctx` is the op's parameter block or NULL. */
void nx_c_map_funnel(const char *op, const nx_c_map_table *tbl,
                    nx_c_cost_class cls, int nin, const value *vals, void *ctx);
void nx_c_fold_funnel(const char *op, const nx_c_fold_table *tbl,
                     const nx_c_stream_table *stbl, nx_c_cost_class cls,
                     value vout, value vin, value vaxes, bool no_identity,
                     void *ctx);
void nx_c_argreduce_funnel(const char *op, const nx_c_arg_table *tbl,
                          nx_c_cost_class cls, value vout, value vin, int axis,
                          void *ctx);
void nx_c_scan_funnel(const char *op, const nx_c_scan_table *tbl,
                     nx_c_cost_class cls, value vout, value vin, int axis,
                     void *ctx);

/* Map-family CAMLprim generators, for ops whose kernel is keyed by the OUTPUT
   dtype — the elementwise ops where output, inputs, and kernel share one dtype
   (add, mul, neg, ...). A stub is then fully determined by its C name, display
   name, kernel table, and cost class, so it is one macro, not a hand-copied
   funnel. The output is the first OCaml argument; inputs follow in order (the
   OCaml binding allocates the output and passes it in).

   NOT for ops whose kernel dtype differs from the output dtype: nx_c_map_funnel
   dispatches on the output (vals[0]) dtype, but a COMPARISON outputs bool while
   its kernel is keyed by the INPUT dtype, and CAST's output dtype differs from
   its input. Those call nx_c_map_run directly, passing the input/compute dtype as
   `dt` and the per-operand elem_size[] (see the cast case in nx_c_selftest.c).
   Same-dtype ctx-carrying ops (fill) may still use nx_c_map_funnel with a ctx.

   The including .c must pull in caml/memory.h and caml/mlvalues.h. */
#define NX_C_MAP1_STUB(cname, opname, table, cls)                               \
  CAMLprim value caml_nx_c_##cname(value vout, value va) {                      \
    CAMLparam2(vout, va);                                                      \
    value vals[2] = {vout, va};                                               \
    nx_c_map_funnel((opname), &(table), (cls), 1, vals, NULL);                  \
    CAMLreturn(Val_unit);                                                      \
  }
#define NX_C_MAP2_STUB(cname, opname, table, cls)                               \
  CAMLprim value caml_nx_c_##cname(value vout, value va, value vb) {            \
    CAMLparam3(vout, va, vb);                                                  \
    value vals[3] = {vout, va, vb};                                           \
    nx_c_map_funnel((opname), &(table), (cls), 2, vals, NULL);                  \
    CAMLreturn(Val_unit);                                                      \
  }
#define NX_C_MAP3_STUB(cname, opname, table, cls)                               \
  CAMLprim value caml_nx_c_##cname(value vout, value va, value vb, value vc) {  \
    CAMLparam4(vout, va, vb, vc);                                              \
    value vals[4] = {vout, va, vb, vc};                                       \
    nx_c_map_funnel((opname), &(table), (cls), 3, vals, NULL);                  \
    CAMLreturn(Val_unit);                                                      \
  }

#endif /* NX_C_ENGINE_H */
