/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_engine.c — the backend's single owner of iteration, threading, dispatch,
   and the error funnel.

   Layering: this is the ONLY translation unit that includes caml/fail.h and
   caml/threads.h, so it is the only place that can raise an OCaml exception or
   hand off the runtime lock. Kernels (which include only nx_c.h) therefore
   cannot do either — the rule is enforced by what each file can reach.

   Contents, top to bottom: the funnel raisers; the thread pool's policy and
   its parallel-for; the one parallel-policy table (nx_c_threads_for) and the
   plan's thread count (nx_c_plan_threads); dimension coalescing; the four
   generated-family drivers (map, fold, argreduce, scan); and the funnels.
   Every driver returns a status; the funnels raise on non-NULL. */

#include <caml/fail.h>
#include <caml/threads.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nx_c_engine.h"

/* ── Funnel raisers ────────────────────────────────────────────────────────
   The one place a status becomes an exception. Called only with the runtime
   lock held (before enter_blocking_section, or after leave). caml_failwith /
   caml_invalid_argument copy the message into an OCaml string, so a stack
   buffer is sufficient and nothing leaks. */

_Thread_local char nx_c_consumed[256];

void nx_c_raise(const char *op, nx_c_status status) {
  if (status && strcmp(status, NX_C_ERR_CONSUMED) == 0)
    caml_invalid_argument(nx_c_consumed);
  /* A dtype an operation does not take is the caller's to avoid. */
  if (status && (strcmp(status, NX_C_ERR_UNSUPPORTED_DTYPE) == 0 ||
                 strcmp(status, NX_C_ERR_PACKED) == 0))
    nx_c_raise_invalid(op, status);
  char buf[256];
  snprintf(buf, sizeof buf, "%s: %s", op, status ? status : "unknown error");
  caml_failwith(buf);
}

void nx_c_raise_invalid(const char *op, nx_c_status status) {
  if (status && strcmp(status, NX_C_ERR_CONSUMED) == 0)
    caml_invalid_argument(nx_c_consumed);
  char buf[256];
  snprintf(buf, sizeof buf, "%s: %s", op, status ? status : "invalid argument");
  caml_invalid_argument(buf);
}

/* ── Thread pool ───────────────────────────────────────────────────────────

   The host's one pool is nx.device's (nx_device.h): Nx_cpu hands it to the
   engine as the module initialises (caml_nx_c_set_pool). This file keeps the
   policy: how many threads a job takes (nx_c_threads_for) and how finely it is
   cut (nx_c_chunks_for). */

static const nx_device_pool *g_pool;

value caml_nx_c_set_pool(value v_pool) {
  g_pool = (const nx_device_pool *)Nativeint_val(v_pool);
  return Val_unit;
}

static int nx_c_ncores(void) { return g_pool->workers(); }
static int nx_c_pcores(void) { return g_pool->compute_workers(); }

/* Chunk count for a published job — the dispatch-granularity half of the
   parallel policy (nx_c_threads_for below is the thread-count half). The cut
   trades tail absorption against claim traffic and streaming locality:

   - A job with few units (a HEAVY batch, sort slices) is claimed per unit up
     to NX_C_CLAIM_CHUNKS_PER_WORKER per thread: the 9-panels-on-8-workers tail
     goes to whichever thread frees first, and a costlier unit (an eig matrix
     that iterates longer, a sort slice with more disorder) stops dragging the
     join since only its claimant is committed to it.
   - An element-granular job is cut into NX_C_CLAIM_CHUNKS_PER_WORKER chunks
     per thread, bounding the straggle to one chunk (~1/8 of a static share)
     for at most a few hundred uncontended fetch-adds per job — nanoseconds
     against the tens-of-microseconds jobs the serial floors admit.
   - The byte floor keeps chunks streaming-sized: never cut finer than
     NX_C_CLAIM_CHUNK_BYTES of traffic per chunk (but never coarser than one
     chunk per thread, or a thread would have nothing to claim). Measured, not
     guessed: interleaved A/B on the 64 MB and 128 MB f32 add rows put floors
     of 4/8/32 MB and the static split all within run-to-run noise (DRAM-bound
     work does not care where the cuts fall), so BANDWIDTH keeps nothing by
     being special-cased and the engine keeps one dispatch path; 8 MB sits in
     the measured-flat band while keeping tail-absorption granularity. */
#define NX_C_CLAIM_CHUNKS_PER_WORKER 8
#define NX_C_CLAIM_CHUNK_BYTES (8 * 1024 * 1024)

static int64_t nx_c_chunks_for(int nthreads, int64_t total, int64_t bytes) {
  int64_t n = (int64_t)nthreads * NX_C_CLAIM_CHUNKS_PER_WORKER;
  if (total <= n) return total; /* unit-granular: one claim per unit */
  int64_t fat = bytes / NX_C_CLAIM_CHUNK_BYTES;
  if (fat < nthreads) fat = nthreads;
  return n < fat ? n : fat;
}

/* Below this much traffic a SERIAL op keeps the runtime lock and runs inline
   rather than pay the enter/leave-blocking-section handshake. Parallel work
   always releases (its workers need the lock free regardless), so this gates
   only the single-thread path.

   Releasing the lock on a serial op buys nothing for the common single-domain
   caller (no other domain is waiting) and costs the handshake; it only helps a
   program running OCaml on several domains at once, and then only in proportion
   to how long the lock is held. For the L2-resident serial band (a 512 KiB
   three-operand add, about 1.5 MB), releasing can cost more than the work itself
   because the scheduler may park the thread under concurrent load.

   4 MiB is that boundary: the L2-resident short-serial band (≲60 µs) keeps the
   lock and skips the handshake, while a longer serial op (a multi-MB reduction
   or an elementwise op below the 16M parallel floor — up to ~700 µs of held
   lock) still releases, where the handshake is negligible and holding the lock
   would actually delay other domains. */
#define NX_C_LOCK_RELEASE_BYTES (4 * 1024 * 1024)

/* Exported for custom families (nx_c_engine.h documents the full contract).
   Called with the runtime lock HELD; releases it internally iff the work
   warrants (nthreads>1, or bytes over the cutoff), runs the split via the pool,
   re-acquires, and returns with the lock held. The handshake lives here so a
   family TU that only calls this never gains caml/threads.h reachability.

   free_on_exit (nullable) is freed after the join but before the re-acquire:
   caml_leave_blocking_section can process pending actions and raise, longjmp-ing
   past the caller's cleanup, so freeing the driver's scratch here — the last
   instruction before that raise can occur — closes the leak on every path.
   caml_enter_blocking_section only releases the lock (no action processing, no
   raise), so the scratch is safe from allocation through the join. Releasing
   NULL is a no-op, so NULL (the generated drivers) costs nothing. */
void nx_c_parallel_for(int nthreads, int64_t total, int64_t bytes,
                      nx_c_range_body body, void *ctx, void *free_on_exit) {
  int release = (nthreads > 1) || (bytes >= NX_C_LOCK_RELEASE_BYTES);
  if (release) caml_enter_blocking_section();
  if (total > 0)
    g_pool->run(nthreads, total, nx_c_chunks_for(nthreads, total, bytes), body,
                ctx);
  nx_c_aligned_free(free_on_exit);
  if (release) caml_leave_blocking_section();
}

/* ── Parallel policy ───────────────────────────────────────────────────────

   One table, keyed by cost class. The engine caps the
   returned count by the number of independent work units it can actually split,
   so a policy that "wants" more threads than there is parallelism costs nothing.

   On Apple Silicon, a single core with serial SIMD saturates
   DRAM for bandwidth-bound work, so parallelizing below ~16M elements only adds
   fork/join cost (serial SIMD beats parallel-for decisively below that floor).
   Compute-bound work does
   not saturate DRAM, so it pays far sooner. Heavy per-run work parallelizes as
   soon as there is more than one run. */

#define NX_C_BW_SERIAL_ELEMS (16 * 1024 * 1024)
#define NX_C_BW_BYTES_PER_THREAD (32 * 1024 * 1024) /* saturate DRAM, not spam */
#define NX_C_COMPUTE_SERIAL_ELEMS (64 * 1024)
#define NX_C_COMPUTE_ELEMS_PER_THREAD (64 * 1024)
#define NX_C_HEAVY_MIN_RUNS 2

int nx_c_threads_for(nx_c_cost_class cls, int64_t runs, int64_t run_len,
                    int64_t bytes) {
  int64_t total = runs * run_len;
  int64_t want;
  int cap;
  switch (cls) {
    case NX_C_COST_BANDWIDTH:
      if (total < NX_C_BW_SERIAL_ELEMS) return 1;
      want = bytes / NX_C_BW_BYTES_PER_THREAD;
      /* Bandwidth keeps the full pool, not the P-core cap. Two reasons: the
         E-core-drag evidence is compute-bound (GEMM), and bandwidth work is
         memory-bound — an E-core issues memory requests toward the same DRAM
         wall rather than running a slow arithmetic chunk. Representative
         bandwidth workloads land below both caps anyway (a 64 MB f32 map is
         192 MB of traffic ÷ 32 MB = 6 threads < 8 P-cores), so this only differs
         for larger arrays. */
      cap = nx_c_ncores();
      break;
    case NX_C_COST_COMPUTE:
      if (total < NX_C_COMPUTE_SERIAL_ELEMS) return 1;
      want = total / NX_C_COMPUTE_ELEMS_PER_THREAD;
      cap = nx_c_pcores();
      break;
    case NX_C_COST_HEAVY:
      if (runs < NX_C_HEAVY_MIN_RUNS) return 1;
      want = runs;
      cap = nx_c_pcores();
      break;
    default:
      return 1;
  }
  if (want < 1) want = 1;
  if (want > cap) want = cap;
  return (int)want;
}

int nx_c_plan_threads(int threads, nx_c_cost_class cls, int64_t runs,
                      int64_t run_len, int64_t bytes) {
  int64_t n =
      threads > 0 ? threads : nx_c_threads_for(cls, runs, run_len, bytes);
  if (n > nx_c_ncores()) n = nx_c_ncores();
  if (n > runs) n = runs;
  if (n < 1) n = 1;
  return (int)n;
}

/* ── Dimension coalescing ──────────────────────────────────────────────────

   The map-family iteration plan: K operands sharing one shape, dropped of their
   size-1 dims and merged wherever adjacent dims compose on EVERY operand, then
   converted from element strides to byte strides (once) with `offset` folded
   into a base pointer. A stride-0 (broadcast) dim merges with a neighbour only
   when both sides are 0-stride over the merge; the general condition
   stride[outer] == stride[inner] * shape[inner] yields exactly that (0 == 0*s),
   so no special case is needed. The result always has rank >= 1: an all-size-1
   operand collapses to a single element. */

typedef struct {
  int nop;
  int ndim; /* coalesced rank, >= 1 */
  int64_t shape[NX_C_MAX_NDIM];
  int64_t bstride[NX_C_MAX_OPERANDS][NX_C_MAX_NDIM]; /* byte strides */
  char *base[NX_C_MAX_OPERANDS];
  int64_t total;
} nx_c_plan;

static void nx_c_coalesce_map(const nx_c_ndarray *ops, int nop,
                             const int64_t *elem_size, nx_c_plan *p) {
  int ndim = ops[0].ndim;
  p->nop = nop;
  for (int k = 0; k < nop; k++)
    p->base[k] = (char *)ops[k].data + ops[k].offset * elem_size[k];

  int64_t total = 1;
  for (int i = 0; i < ndim; i++) total *= ops[0].shape[i];
  p->total = total;

  int nd = 0;
  int64_t cshape[NX_C_MAX_NDIM];
  int64_t cstride[NX_C_MAX_OPERANDS][NX_C_MAX_NDIM]; /* element strides */
  for (int i = 0; i < ndim; i++) {
    int64_t s = ops[0].shape[i];
    if (s == 1) continue; /* size-1 dims carry no iteration */
    int merged = 0;
    if (nd > 0) {
      merged = 1;
      for (int k = 0; k < nop; k++) {
        if (cstride[k][nd - 1] != ops[k].strides[i] * s) {
          merged = 0;
          break;
        }
      }
    }
    if (merged) {
      cshape[nd - 1] *= s;
      for (int k = 0; k < nop; k++) cstride[k][nd - 1] = ops[k].strides[i];
    } else {
      cshape[nd] = s;
      for (int k = 0; k < nop; k++) cstride[k][nd] = ops[k].strides[i];
      nd++;
    }
  }
  if (nd == 0) { /* every dim was size 1: one element */
    cshape[0] = 1;
    for (int k = 0; k < nop; k++) cstride[k][0] = 0;
    nd = 1;
  }

  p->ndim = nd;
  for (int i = 0; i < nd; i++) {
    p->shape[i] = cshape[i];
    for (int k = 0; k < nop; k++)
      p->bstride[k][i] = cstride[k][i] * elem_size[k];
  }
}

/* ── Shared 2-stream odometer ──────────────────────────────────────────────
   fold/argreduce/scan all iterate a nest of kept dims carrying one input and
   one output pointer. seek positions both pointers at a linear index (once per
   thread chunk); next advances them incrementally — add on increment, subtract
   shape*stride on carry — never a per-element dot product. */

static void nx_c_seek2(int nk, const int64_t *shape, const int64_t *s_in,
                      const int64_t *s_out, int64_t idx, int64_t *coord,
                      char *in_base, char *out_base, char **ip, char **op) {
  char *a = in_base;
  char *b = out_base;
  int64_t rem = idx;
  for (int d = nk - 1; d >= 0; d--) {
    int64_t c = rem % shape[d];
    rem /= shape[d];
    coord[d] = c;
    a += c * s_in[d];
    b += c * s_out[d];
  }
  *ip = a;
  *op = b;
}

static void nx_c_next2(int nk, const int64_t *shape, const int64_t *s_in,
                      const int64_t *s_out, int64_t *coord, char **ip,
                      char **op) {
  for (int d = nk - 1; d >= 0; d--) {
    if (++coord[d] < shape[d]) {
      *ip += s_in[d];
      *op += s_out[d];
      return;
    }
    coord[d] = 0;
    *ip -= (shape[d] - 1) * s_in[d];
    *op -= (shape[d] - 1) * s_out[d];
  }
}

/* ── Map driver ────────────────────────────────────────────────────────────
   After coalescing, either the whole thing is one run (rank 1) split across
   threads by element range, or dims [0, ndim-1) form an outer odometer split
   across threads by run, with the innermost dim handed to the kernel as the
   strided run. */

typedef struct {
  const nx_c_plan *p;
  nx_c_map_loop *kernel;
  void *ctx;
} nx_c_map_exec;

static void nx_c_map_run_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  (void)worker;
  const nx_c_map_exec *e = vctx;
  const nx_c_plan *p = e->p;
  char *ptrs[NX_C_MAX_OPERANDS];
  int64_t steps[NX_C_MAX_OPERANDS];
  for (int k = 0; k < p->nop; k++) {
    steps[k] = p->bstride[k][0];
    ptrs[k] = p->base[k] + lo * steps[k];
  }
  e->kernel(ptrs, steps, hi - lo, e->ctx);
}

static void nx_c_map_outer_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  (void)worker;
  const nx_c_map_exec *e = vctx;
  const nx_c_plan *p = e->p;
  int od = p->ndim - 1; /* number of odometer dims */
  int64_t run = p->shape[od];

  char *ptr[NX_C_MAX_OPERANDS];
  int64_t step[NX_C_MAX_OPERANDS];
  int64_t coord[NX_C_MAX_NDIM];
  int64_t rem = lo;
  for (int k = 0; k < p->nop; k++) {
    ptr[k] = p->base[k];
    step[k] = p->bstride[k][od];
  }
  for (int d = od - 1; d >= 0; d--) {
    int64_t c = rem % p->shape[d];
    rem /= p->shape[d];
    coord[d] = c;
    for (int k = 0; k < p->nop; k++) ptr[k] += c * p->bstride[k][d];
  }

  for (int64_t it = lo; it < hi; it++) {
    char *runptr[NX_C_MAX_OPERANDS];
    for (int k = 0; k < p->nop; k++) runptr[k] = ptr[k];
    e->kernel(runptr, step, run, e->ctx);
    for (int d = od - 1; d >= 0; d--) {
      if (++coord[d] < p->shape[d]) {
        for (int k = 0; k < p->nop; k++) ptr[k] += p->bstride[k][d];
        break;
      }
      coord[d] = 0;
      for (int k = 0; k < p->nop; k++)
        ptr[k] -= (p->shape[d] - 1) * p->bstride[k][d];
    }
  }
}

nx_c_status nx_c_map_run(const nx_c_map_table *tbl, nx_c_dtype dt, int nin,
                       const nx_c_ndarray *ops, const int64_t *elem_size,
                       nx_c_cost_class cls, void *ctx) {
  int nop = nin + 1;
  if (nop > NX_C_MAX_OPERANDS) return NX_C_ERR_ARITY;

  nx_c_map_loop *kernel = tbl->fn[dt];
  if (kernel == NULL)
    return nx_c_dtype_is_packed(dt) ? NX_C_ERR_PACKED : NX_C_ERR_UNSUPPORTED_DTYPE;

  nx_c_plan p;
  nx_c_coalesce_map(ops, nop, elem_size, &p);
  if (p.total == 0) return NX_C_OK; /* empty tensor: kernels are no-ops */

  /* The binding allocates a fresh output, so no dim of extent > 1 has a 0
     stride: one would put parallel threads racing on one cell. Asserted on the
     coalesced output (index 0), after the empty short-circuit, since an empty
     tensor writes nothing. */
  for (int i = 0; i < p.ndim; i++)
    if (p.shape[i] > 1 && p.bstride[0][i] == 0) return NX_C_ERR_OUT_ALIASED;

  /* Traffic for the bandwidth heuristic: an operand only touches the elements
     it actually holds, so a 0-stride (broadcast) dim contributes one element,
     not shape[i] of them. Counting p.total for every operand would bill a
     splat operand — one cell the kernel hoists into a register — as a full
     stream and hand the run more threads than its bandwidth warrants. The
     output has no 0-stride dim (rejected just above), so it still counts
     p.total. */
  int64_t bytes = 0;
  for (int k = 0; k < nop; k++) {
    int64_t touched = 1;
    for (int i = 0; i < p.ndim; i++)
      if (p.bstride[k][i] != 0) touched *= p.shape[i];
    bytes += touched * elem_size[k];
  }

  nx_c_map_exec e = {&p, kernel, ctx};
  if (p.ndim == 1) {
    int nth = nx_c_threads_for(cls, 1, p.total, bytes);
    if (nth > p.total) nth = (int)p.total;
    nx_c_parallel_for(nth, p.total, bytes, nx_c_map_run_body, &e, NULL);
  } else {
    int od = p.ndim - 1;
    int64_t runs = 1;
    for (int d = 0; d < od; d++) runs *= p.shape[d];
    int nth = nx_c_threads_for(cls, runs, p.shape[od], bytes);
    if (nth > runs) nth = (int)runs;
    nx_c_parallel_for(nth, runs, bytes, nx_c_map_outer_body, &e, NULL);
  }
  return NX_C_OK;
}

/* ── Fold driver ───────────────────────────────────────────────────────────
   Both paths fold an output's terms in blocks and combine the blocks by
   nx_c.h's tree (Associations); nx_c_tree_push and nx_c_tree_close compute it,
   on one accumulator per slot on the per-output path and on a row of them on
   the streaming path. */

/* The left-complete tree of nx_c.h's Associations: push the result of block
   `done` (counted from 1) at slot `top`, then combine newest into older once
   per trailing zero of `done`; return the slots in use. A slot is one value
   (per-output) or a row of n (streaming). */
static int nx_c_tree_push(nx_c_fold_combine *combine, char *slot, int64_t pitch,
                          int64_t n, int top, int64_t done, void *ctx) {
  top++;
  for (int64_t c = done; (c & 1) == 0; c >>= 1, top--)
    combine(slot + (top - 2) * pitch, slot + (top - 1) * pitch, n, ctx);
  return top;
}

static void nx_c_tree_close(nx_c_fold_combine *combine, char *slot,
                            int64_t pitch, int64_t n, int top, void *ctx) {
  for (; top > 1; top--)
    combine(slot + (top - 2) * pitch, slot + (top - 1) * pitch, n, ctx);
}

/* The tree holds at most one result per bit of the blocks done, plus the open
   block: 64 covers any int64 count. */
#define NX_C_FOLD_STACK 64

/* Advance a reduced-axis odometer by one point (one term per output). */
static void nx_c_next1(int n, const int64_t *shape, const int64_t *stride,
                       int64_t *coord, char **p) {
  for (int d = n - 1; d >= 0; d--) {
    if (++coord[d] < shape[d]) {
      *p += stride[d];
      return;
    }
    coord[d] = 0;
    *p -= (shape[d] - 1) * stride[d];
  }
}

typedef struct {
  nx_c_fold_init *init;
  nx_c_fold_step *step;
  nx_c_fold_combine *combine;
  nx_c_fold_fini *fini;
  void *ctx;
  char *in_base;
  char *out_base;
  int nk; /* kept (non-reduced) dims */
  int64_t kshape[NX_C_MAX_NDIM];
  int64_t k_in_stride[NX_C_MAX_NDIM];  /* byte */
  int64_t k_out_stride[NX_C_MAX_NDIM]; /* byte */
  int nr; /* ordered reduced dims; the last is the run */
  int64_t rshape[NX_C_MAX_NDIM];
  int64_t r_in_stride[NX_C_MAX_NDIM]; /* byte */
  int64_t reduced_len;                /* terms per output */
} nx_c_fold_exec;

/* One output: its terms, run by run, cut into blocks of NX_C_FOLD_BLOCK; a
   block spanning runs folds each piece with its own step call. The kernels
   and the context are read into locals once: an indirect call could change
   memory `e` points into, so the compiler would reload them after each. */
static void nx_c_fold_reduce_one(const nx_c_fold_exec *e, char *ip, char *op) {
  nx_c_fold_init *init = e->init;
  nx_c_fold_step *step = e->step;
  nx_c_fold_combine *combine = e->combine;
  nx_c_fold_fini *fini = e->fini;
  void *ctx = e->ctx;
  /* st[0, top) are the tree's results; the open block folds in `acc`, which
     an output with no term leaves at the identity. */
  nx_c_acc st[NX_C_FOLD_STACK];
  char *slot = (char *)st;
  int64_t pitch = (int64_t)sizeof(nx_c_acc);
  int top = 0;
  int64_t done = 0;
  int64_t room = NX_C_FOLD_BLOCK; /* terms the open block takes */
  nx_c_acc acc;
  init(&acc, ctx);
  if (e->nr == 0) { /* every reduced axis had extent 1: one term */
    step(&acc, ip, 0, 1, ctx);
  } else if (e->reduced_len > 0) {
    int rod = e->nr - 1; /* reduced odometer dims; dim rod is the run */
    int64_t run_len = e->rshape[rod];
    int64_t run_stride = e->r_in_stride[rod];
    int64_t outer = e->reduced_len / run_len;
    int64_t rcoord[NX_C_MAX_NDIM];
    for (int d = 0; d < rod; d++) rcoord[d] = 0;
    char *rp = ip;
    for (int64_t o = 0; o < outer; o++) {
      /* The general branch also handles a run that fits; this one measured
         up to 12% faster on short runs ([65536; 16; 2] over axes 0 and 2,
         [512; 512] over axis 1). */
      if (run_len < room) { /* the run fits in the open block */
        step(&acc, rp, run_stride, run_len, ctx);
        room -= run_len;
      } else {
        char *p = rp;
        int64_t left = run_len;
        while (left >= room) { /* the run fills the open block */
          step(&acc, p, run_stride, room, ctx);
          p += room * run_stride;
          left -= room;
          st[top] = acc;
          top = nx_c_tree_push(combine, slot, pitch, 1, top, ++done, ctx);
          init(&acc, ctx);
          room = NX_C_FOLD_BLOCK;
        }
        if (left > 0) {
          step(&acc, p, run_stride, left, ctx);
          room -= left;
        }
      }
      nx_c_next1(rod, e->rshape, e->r_in_stride, rcoord, &rp);
    }
  }
  if (top == 0) { /* one block: no tree */
    fini(op, 0, &acc, 1, ctx);
    return;
  }
  /* Closing over the open block combines it as pushing it would. */
  if (room < NX_C_FOLD_BLOCK) st[top++] = acc;
  nx_c_tree_close(combine, slot, pitch, 1, top, ctx);
  fini(op, 0, &st[0], 1, ctx);
}

static void nx_c_fold_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  (void)worker;
  const nx_c_fold_exec *e = vctx;
  int64_t coord[NX_C_MAX_NDIM];
  char *ip;
  char *op;
  nx_c_seek2(e->nk, e->kshape, e->k_in_stride, e->k_out_stride, lo, coord,
            e->in_base, e->out_base, &ip, &op);
  for (int64_t it = lo; it < hi; it++) {
    nx_c_fold_reduce_one(e, ip, op);
    nx_c_next2(e->nk, e->kshape, e->k_in_stride, e->k_out_stride, coord, &ip,
              &op);
  }
}

/* ── Streaming fold path ────────────────────────────────────────────────────
   A unit is one tile of at most NX_C_FOLD_TILE lanes of one panel. It folds
   the rows in blocks of NX_C_FOLD_BLOCK into a row of accumulators per block,
   combines the block rows by the tree and stores the result by fini. A worker's
   scratch is the tree's rows for one tile, so it is bounded whatever the lane
   length and the thread count.

   NX_C_FOLD_TILE keeps a tile's top row cache-resident (64 KiB at float32,
   256 KiB at complex128) while a row's read stays long enough to stream:
   measured against whole-lane rows it is level on [512; 512] to
   [20000; 2000] and faster on [2; 10^6], where a tile of 1024 is 2.7x slower
   on [20000; 2000]. No bit depends on it: lanes are independent outputs. */
#define NX_C_FOLD_TILE 16384

typedef struct {
  nx_c_fold_stream *stream;
  nx_c_fold_combine *combine;
  nx_c_fold_fini *fini;
  void *ctx;
  char *in_base;
  char *out_base;
  int64_t lane_len;        /* the vectorized inner (most contiguous kept) axis */
  int64_t lane_in_stride;  /* byte */
  int64_t lane_out_stride; /* byte */
  int64_t tile;            /* lanes per unit, the last tile of a lane shorter */
  int64_t ntiles;          /* tiles per lane */
  int np;                  /* panel dims (kept dims other than the lane) */
  int64_t pshape[NX_C_MAX_NDIM];
  int64_t p_in_stride[NX_C_MAX_NDIM];  /* byte */
  int64_t p_out_stride[NX_C_MAX_NDIM]; /* byte */
  int nr;                             /* ordered reduced dims */
  int64_t rshape[NX_C_MAX_NDIM];
  int64_t r_in_stride[NX_C_MAX_NDIM]; /* byte */
  int64_t rows;                       /* points of the reduced dims */
  char *scratch;                      /* nthreads slots of slot_bytes each */
  int64_t pitch;                      /* bytes of one row of accumulators */
  int64_t slot_bytes;                 /* the tree's rows for one tile */
} nx_c_fold_stream_exec;

static void nx_c_fold_stream_body(int64_t lo, int64_t hi, int worker,
                                 void *vctx) {
  const nx_c_fold_stream_exec *e = vctx;
  char *slot = e->scratch + (int64_t)worker * e->slot_bytes;
  int64_t coord[NX_C_MAX_NDIM];
  int64_t rcoord[NX_C_MAX_NDIM];
  for (int64_t u = lo; u < hi; u++) {
    int64_t tile = u % e->ntiles;
    char *ip;
    char *op;
    nx_c_seek2(e->np, e->pshape, e->p_in_stride, e->p_out_stride,
               u / e->ntiles, coord, e->in_base, e->out_base, &ip, &op);
    ip += tile * e->tile * e->lane_in_stride;
    op += tile * e->tile * e->lane_out_stride;
    int64_t tn = e->lane_len - tile * e->tile;
    if (tn > e->tile) tn = e->tile;
    int top = 0;
    int64_t done = 0;
    for (int d = 0; d < e->nr; d++) rcoord[d] = 0;
    char *rp = ip;
    for (int64_t r = 0; r < e->rows;) {
      char *accs = slot + top * e->pitch;
      int64_t end = r + NX_C_FOLD_BLOCK;
      if (end > e->rows) end = e->rows;
      for (int64_t k = r; k < end; k++) {
        e->stream(accs, rp, e->lane_in_stride, tn, k == r, e->ctx);
        nx_c_next1(e->nr, e->rshape, e->r_in_stride, rcoord, &rp);
      }
      r = end;
      top = nx_c_tree_push(e->combine, slot, e->pitch, tn, top, ++done, e->ctx);
    }
    nx_c_tree_close(e->combine, slot, e->pitch, tn, top, e->ctx);
    e->fini(op, e->lane_out_stride, slot, tn, e->ctx);
  }
}

/* Build the streaming exec from the plan, allocate the per-thread tree
   scratch and drive the units. `lane` indexes the kept-axis arrays. */
static nx_c_status nx_c_fold_stream_run(const nx_c_fold_table *tbl,
                                        nx_c_dtype dt, const nx_c_fold_exec *fe,
                                        int lane, nx_c_cost_class cls,
                                        int threads, int64_t bytes) {
  nx_c_fold_stream_exec e;
  e.stream = tbl->stream[dt];
  e.combine = tbl->combine[dt];
  e.fini = tbl->fini[dt];
  e.ctx = fe->ctx;
  e.in_base = fe->in_base;
  e.out_base = fe->out_base;
  e.lane_len = fe->kshape[lane];
  e.lane_in_stride = fe->k_in_stride[lane];
  e.lane_out_stride = fe->k_out_stride[lane];
  e.tile = e.lane_len < NX_C_FOLD_TILE ? e.lane_len : NX_C_FOLD_TILE;
  e.ntiles = (e.lane_len + e.tile - 1) / e.tile;
  e.np = 0;
  for (int j = 0; j < fe->nk; j++) {
    if (j == lane) continue;
    e.pshape[e.np] = fe->kshape[j];
    e.p_in_stride[e.np] = fe->k_in_stride[j];
    e.p_out_stride[e.np] = fe->k_out_stride[j];
    e.np++;
  }
  e.nr = fe->nr;
  for (int d = 0; d < fe->nr; d++) {
    e.rshape[d] = fe->rshape[d];
    e.r_in_stride[d] = fe->r_in_stride[d];
  }
  e.rows = fe->reduced_len;

  /* While a block folds, the tree holds at most floor(log2(blocks)) results
     below it. */
  int64_t blocks = (e.rows + NX_C_FOLD_BLOCK - 1) / NX_C_FOLD_BLOCK;
  int64_t rows_held = 1;
  for (int64_t b = blocks; b > 1; b >>= 1) rows_held++;
  e.pitch = e.tile * (int64_t)sizeof(nx_c_acc);
  e.slot_bytes = rows_held * e.pitch;

  int64_t panels = 1;
  for (int j = 0; j < e.np; j++) panels *= e.pshape[j];
  int64_t nunits = panels * e.ntiles;
  int nth = nx_c_plan_threads(threads, cls, nunits, e.tile * e.rows, bytes);
  void *scratch = nx_c_aligned_alloc((size_t)nth * (size_t)e.slot_bytes);
  if (scratch == NULL) return NX_C_ERR_ALLOC;
  e.scratch = scratch;
  /* scratch is freed by nx_c_parallel_for after the join, leak-safe across the
     re-acquire's possible raise (nx_c_engine.h free_on_exit contract). */
  nx_c_parallel_for(nth, nunits, bytes, nx_c_fold_stream_body, &e, scratch);
  return NX_C_OK;
}

nx_c_status nx_c_fold_run(const nx_c_fold_table *tbl, nx_c_dtype dt,
                        const nx_c_ndarray *in, int64_t in_elem,
                        const nx_c_ndarray *out, int64_t out_elem,
                        const int *reduce_axes, int n_reduce, bool no_identity,
                        nx_c_cost_class cls, int threads, void *ctx) {
  if (tbl->init[dt] == NULL || tbl->step[dt] == NULL ||
      tbl->combine[dt] == NULL || tbl->fini[dt] == NULL ||
      tbl->stream[dt] == NULL)
    return nx_c_dtype_is_packed(dt) ? NX_C_ERR_PACKED : NX_C_ERR_UNSUPPORTED_DTYPE;

  nx_c_fold_exec e;
  e.init = tbl->init[dt];
  e.step = tbl->step[dt];
  e.combine = tbl->combine[dt];
  e.fini = tbl->fini[dt];
  e.ctx = ctx;
  e.in_base = (char *)in->data + in->offset * in_elem;
  e.out_base = (char *)out->data + out->offset * out_elem;

  /* Split input axes into kept (output-indexing) and reduced. reduce_axes is
     strictly increasing, so a single merge pass classifies each axis. */
  int nk = 0;
  int nr = 0;
  int ra = 0;
  for (int a = 0; a < in->ndim; a++) {
    if (ra < n_reduce && reduce_axes[ra] == a) {
      e.rshape[nr] = in->shape[a];
      e.r_in_stride[nr] = in->strides[a] * in_elem;
      nr++;
      ra++;
    } else {
      e.kshape[nk] = in->shape[a];
      e.k_in_stride[nk] = in->strides[a] * in_elem;
      nk++;
    }
  }
  /* The forward scan consumes reduce_axes iff they are strictly increasing and
     in range; any unsorted/duplicate/out-of-range axis leaves ra < n_reduce.
     Verified, not assumed — a binding that forgets to sort gets a loud status
     rather than a silently-wrong partial reduction. */
  if (ra != n_reduce) return NX_C_ERR_AXES;
  /* out is aligned, one axis per kept input axis, as the binding allocates it;
     a short/long descriptor would read unspecified stride slots, so the rank is
     asserted before the out strides are paired. */
  if (out->ndim != nk) return NX_C_ERR_OUT_RANK;
  for (int j = 0; j < nk; j++) e.k_out_stride[j] = out->strides[j] * out_elem;

  /* Axes of extent 1 carry no iteration: drop them, a kept one with its out
     stride, so that no arbitrary stride of theirs steers the plan. */
  e.nk = 0;
  for (int j = 0; j < nk; j++) {
    if (e.kshape[j] == 1) continue;
    e.kshape[e.nk] = e.kshape[j];
    e.k_in_stride[e.nk] = e.k_in_stride[j];
    e.k_out_stride[e.nk] = e.k_out_stride[j];
    e.nk++;
  }
  e.nr = 0;
  for (int d = 0; d < nr; d++) {
    if (e.rshape[d] == 1) continue;
    e.rshape[e.nr] = e.rshape[d];
    e.r_in_stride[e.nr] = e.r_in_stride[d];
    e.nr++;
  }

  int64_t out_total = 1;
  for (int d = 0; d < e.nk; d++) out_total *= e.kshape[d];
  if (out_total == 0) return NX_C_OK; /* no output elements */
  int64_t reduced_len = 1;
  for (int d = 0; d < e.nr; d++) reduced_len *= e.rshape[d];
  e.reduced_len = reduced_len;
  /* max/min have no identity for an empty reduced extent: with outputs to fill
     (out_total > 0 here) an empty extent would store the init sentinel
     (-inf / INT64_MIN). Reject before any kernel runs — the check lives here, in
     the shared driver, so no caller (funnel or raw) can leak the sentinel. */
  if (no_identity && reduced_len == 0) return NX_C_ERR_EMPTY_REDUCE;
  int64_t bytes = out_total * reduced_len * in_elem;

  /* Order the reduced axes from the largest |stride| to the smallest, ties in
     axis order (a stable insertion sort), then merge each into its outer
     neighbour where the two are contiguous with each other. */
  for (int d = 1; d < e.nr; d++) {
    int64_t sh = e.rshape[d], st = e.r_in_stride[d];
    int i = d;
    for (; i > 0 && llabs(e.r_in_stride[i - 1]) < llabs(st); i--) {
      e.rshape[i] = e.rshape[i - 1];
      e.r_in_stride[i] = e.r_in_stride[i - 1];
    }
    e.rshape[i] = sh;
    e.r_in_stride[i] = st;
  }
  if (e.nr > 1) {
    int m = 1;
    for (int d = 1; d < e.nr; d++) {
      if (e.r_in_stride[m - 1] == e.r_in_stride[d] * e.rshape[d]) {
        e.rshape[m - 1] *= e.rshape[d];
        e.r_in_stride[m - 1] = e.r_in_stride[d];
      } else {
        e.rshape[m] = e.rshape[d];
        e.r_in_stride[m] = e.r_in_stride[d];
        m++;
      }
    }
    e.nr = m;
  }

  /* Streaming when a kept axis is strictly more contiguous than every reduced
     axis, the run's being the smallest. An empty reduced extent has no first
     row to seed the accumulators and stays on the per-output path, whose
     identity fill needs none. A broadcast (0-stride) kept axis can win the
     lane: every lane then folds the same terms, and the equal outputs are the
     broadcast result. */
  if (e.nk >= 1 && e.nr >= 1 && reduced_len >= 1) {
    int lane = 0;
    for (int j = 1; j < e.nk; j++)
      if (llabs(e.k_in_stride[j]) < llabs(e.k_in_stride[lane])) lane = j;
    if (llabs(e.k_in_stride[lane]) < llabs(e.r_in_stride[e.nr - 1]))
      return nx_c_fold_stream_run(tbl, dt, &e, lane, cls, threads, bytes);
  }

  int nth = nx_c_plan_threads(threads, cls, out_total, reduced_len, bytes);
  nx_c_parallel_for(nth, out_total, bytes, nx_c_fold_body, &e, NULL);
  return NX_C_OK;
}

/* ── Argreduce driver ──────────────────────────────────────────────────────
   Argmax/argmin over one axis into an int64 output, parallelized over the
   non-axis nest. One run per output (the axis); the kernel carries the running
   extreme and its index. */

typedef struct {
  nx_c_arg_step *step;
  void *ctx;
  char *in_base;
  char *out_base;
  int nk;
  int64_t kshape[NX_C_MAX_NDIM];
  int64_t k_in_stride[NX_C_MAX_NDIM];  /* byte */
  int64_t k_out_stride[NX_C_MAX_NDIM]; /* byte */
  int64_t axis_stride;                /* byte */
  int64_t axis_len;
} nx_c_arg_exec;

static void nx_c_arg_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  (void)worker;
  const nx_c_arg_exec *e = vctx;
  int64_t coord[NX_C_MAX_NDIM];
  char *ip;
  char *op;
  nx_c_seek2(e->nk, e->kshape, e->k_in_stride, e->k_out_stride, lo, coord,
            e->in_base, e->out_base, &ip, &op);
  for (int64_t it = lo; it < hi; it++) {
    nx_c_arg_acc acc;
    nx_c_arg_init(&acc);
    e->step(&acc, ip, e->axis_stride, e->axis_len, e->ctx);
    nx_c_arg_fini(op, &acc);
    nx_c_next2(e->nk, e->kshape, e->k_in_stride, e->k_out_stride, coord, &ip,
              &op);
  }
}

nx_c_status nx_c_argreduce_run(const nx_c_arg_table *tbl, nx_c_dtype dt,
                             const nx_c_ndarray *in, int64_t in_elem,
                             const nx_c_ndarray *out, int axis,
                             nx_c_cost_class cls, int threads, void *ctx) {
  if (tbl->step[dt] == NULL)
    return nx_c_dtype_is_packed(dt) ? NX_C_ERR_PACKED : NX_C_ERR_UNSUPPORTED_DTYPE;

  /* The frontend passes a valid axis and the binding allocates out: shape and
     stride reads rely on both. */
  if (axis < 0 || axis >= in->ndim) return NX_C_ERR_AXIS;
  if (out->ndim != in->ndim - 1) return NX_C_ERR_OUT_RANK;

  int64_t axis_len = in->shape[axis];
  if (axis_len == 0) return NX_C_ERR_EMPTY_REDUCE;

  nx_c_arg_exec e;
  e.step = tbl->step[dt];
  e.ctx = ctx;
  e.in_base = (char *)in->data + in->offset * in_elem;
  e.out_base = (char *)out->data + out->offset * (int64_t)sizeof(int64_t);
  e.axis_stride = in->strides[axis] * in_elem;
  e.axis_len = axis_len;

  e.nk = 0;
  for (int a = 0; a < in->ndim; a++) {
    if (a == axis) continue;
    e.kshape[e.nk] = in->shape[a];
    e.k_in_stride[e.nk] = in->strides[a] * in_elem;
    e.k_out_stride[e.nk] = out->strides[e.nk] * (int64_t)sizeof(int64_t);
    e.nk++;
  }

  int64_t out_total = 1;
  for (int d = 0; d < e.nk; d++) out_total *= e.kshape[d];
  if (out_total == 0) return NX_C_OK;
  int64_t bytes = out_total * axis_len * in_elem;

  int nth = nx_c_plan_threads(threads, cls, out_total, axis_len, bytes);
  nx_c_parallel_for(nth, out_total, bytes, nx_c_arg_body, &e, NULL);
  return NX_C_OK;
}

/* ── Scan driver ───────────────────────────────────────────────────────────
   Inclusive scan over one axis. The non-axis nest indexes independent slices
   (parallelized); a slice runs in nx_c.h's chunks, each from its carry. */

typedef struct {
  nx_c_fold_init *init;
  nx_c_fold_combine *combine;
  nx_c_scan_step *step;
  void *ctx;
  char *in_base;
  char *out_base;
  int nk;
  int64_t kshape[NX_C_MAX_NDIM];
  int64_t k_in_stride[NX_C_MAX_NDIM];  /* byte */
  int64_t k_out_stride[NX_C_MAX_NDIM]; /* byte */
  int64_t axis_in_stride;             /* byte */
  int64_t axis_out_stride;            /* byte */
  int64_t axis_len;
  int64_t chunk; /* NX_C_SCAN_CHUNK, or the axis length for an exact scan */
} nx_c_scan_exec;

static void nx_c_scan_slice(const nx_c_scan_exec *e, char *ip, char *op) {
  int64_t len = e->axis_len;
  int64_t is = e->axis_in_stride;
  int64_t os = e->axis_out_stride;
  int64_t chunk = e->chunk;
  nx_c_acc id, state, total, carry;
  e->init(&id, e->ctx);
  state = id;
  int64_t first = len < chunk ? len : chunk;
  e->step(op, os, ip, is, first, &state, NULL, e->ctx);
  carry = state;
  for (int64_t lo = chunk; lo < len; lo += chunk) {
    int64_t m = len - lo < chunk ? len - lo : chunk;
    state = carry;
    if (lo + m < len) {
      total = id;
      e->step(op + lo * os, os, ip + lo * is, is, m, &state, &total, e->ctx);
      e->combine(&carry, &total, 1, e->ctx);
    } else {
      e->step(op + lo * os, os, ip + lo * is, is, m, &state, NULL, e->ctx);
    }
  }
}

static void nx_c_scan_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  (void)worker;
  const nx_c_scan_exec *e = vctx;
  int64_t coord[NX_C_MAX_NDIM];
  char *ip;
  char *op;
  nx_c_seek2(e->nk, e->kshape, e->k_in_stride, e->k_out_stride, lo, coord,
            e->in_base, e->out_base, &ip, &op);
  for (int64_t it = lo; it < hi; it++) {
    nx_c_scan_slice(e, ip, op);
    nx_c_next2(e->nk, e->kshape, e->k_in_stride, e->k_out_stride, coord, &ip,
              &op);
  }
}

nx_c_status nx_c_scan_run(const nx_c_scan_table *tbl, nx_c_dtype dt,
                        const nx_c_ndarray *in, int64_t in_elem,
                        const nx_c_ndarray *out, int64_t out_elem, int axis,
                        nx_c_cost_class cls, int threads, void *ctx) {
  if (tbl->op->init[dt] == NULL || tbl->op->combine[dt] == NULL ||
      tbl->step[dt] == NULL)
    return nx_c_dtype_is_packed(dt) ? NX_C_ERR_PACKED : NX_C_ERR_UNSUPPORTED_DTYPE;

  /* The frontend passes a valid axis and the binding allocates out: shape and
     stride reads rely on both. */
  if (axis < 0 || axis >= in->ndim) return NX_C_ERR_AXIS;
  if (out->ndim != in->ndim) return NX_C_ERR_OUT_RANK;

  nx_c_scan_exec e;
  e.init = tbl->op->init[dt];
  e.combine = tbl->op->combine[dt];
  e.step = tbl->step[dt];
  e.ctx = ctx;
  e.in_base = (char *)in->data + in->offset * in_elem;
  e.out_base = (char *)out->data + out->offset * out_elem;
  e.axis_in_stride = in->strides[axis] * in_elem;
  e.axis_out_stride = out->strides[axis] * out_elem;
  e.axis_len = in->shape[axis];
  /* An integer or bool scan gives the same bits under every grouping (nx_c.h,
     Associations), so one walk over the slice computes its chunks: a second
     chain for the totals costs an integer scan a tenth of its time. */
  e.chunk = nx_c_dtype_is_int(dt) || nx_c_dtype_is_bool(dt) ? e.axis_len
                                                            : NX_C_SCAN_CHUNK;

  e.nk = 0;
  for (int a = 0; a < in->ndim; a++) {
    if (a == axis) continue;
    e.kshape[e.nk] = in->shape[a];
    e.k_in_stride[e.nk] = in->strides[a] * in_elem;
    e.k_out_stride[e.nk] = out->strides[a] * out_elem;
    e.nk++;
  }

  int64_t slices = 1;
  for (int d = 0; d < e.nk; d++) slices *= e.kshape[d];
  if (slices == 0 || e.axis_len == 0) return NX_C_OK;
  int64_t bytes = slices * e.axis_len * (in_elem + out_elem);

  int nth = nx_c_plan_threads(threads, cls, slices, e.axis_len, bytes);
  nx_c_parallel_for(nth, slices, bytes, nx_c_scan_body, &e, NULL);
  return NX_C_OK;
}

/* ── The funnel ────────────────────────────────────────────────────────────

   The single extract -> validate -> dispatch -> run -> raise path a family stub
   uses (nx_c_engine.h). These read OCaml operand records but perform no OCaml
   allocation before extraction and never touch a value again after it, so the
   caller's CAMLparam roots suffice — no local rooting here. The drivers they
   call own the runtime-lock handshake internally. */

/* One place maps a status to an exception kind. Precondition and empty-axis
   violations are the caller's bad argument (Invalid_argument); everything else
   (unsupported dtype, packed, allocation) is a Failure. Runs only on the cold
   error path, so strcmp is free. */
NX_C_NORETURN void nx_c_raise_status(const char *op, nx_c_status s) {
  if (strcmp(s, NX_C_ERR_EMPTY_REDUCE) == 0 || strcmp(s, NX_C_ERR_AXES) == 0 ||
      strcmp(s, NX_C_ERR_AXIS) == 0 || strcmp(s, NX_C_ERR_OUT_RANK) == 0 ||
      strcmp(s, NX_C_ERR_OUT_ALIASED) == 0 || strcmp(s, NX_C_ERR_SHAPE) == 0)
    nx_c_raise_invalid(op, s);
  nx_c_raise(op, s);
}

void nx_c_map_funnel(const char *op, const nx_c_map_table *tbl, nx_c_cost_class cls,
                    int nin, const value *vals, void *ctx) {
  int nop = nin + 1;
  if (nop > NX_C_MAX_OPERANDS) nx_c_raise(op, NX_C_ERR_ARITY);

  nx_c_ndarray ops[NX_C_MAX_OPERANDS];
  int64_t elem[NX_C_MAX_OPERANDS];
  nx_c_dtype dt = NX_C_DTYPE_COUNT;
  for (int k = 0; k < nop; k++) {
    nx_c_status s = nx_c_ndarray_of_value(vals[k], &ops[k]);
    if (s != NX_C_OK) nx_c_raise(op, s);
    nx_c_dtype dk = nx_c_dtype_of_value(vals[k]);
    elem[k] = nx_c_elem_size(dk);
    /* A packed operand of a compute op yields a poison 0 element size; reject it
       here rather than let coalescing build zero-length runs. */
    if (elem[k] == 0) nx_c_raise(op, NX_C_ERR_PACKED);
    if (k == 0) dt = dk; /* dispatch on the output (compute) dtype */
  }

  nx_c_status s = nx_c_map_run(tbl, dt, nin, ops, elem, cls, ctx);
  if (s != NX_C_OK) nx_c_raise_status(op, s);
}

/* Build the squeezed output descriptor the reduction drivers want: rank equal
   to the kept (non-reduced) input axes, aligned to them in order. Accepts an
   output already squeezed (keepdims=false) or full-rank with size-1 reduced dims
   (keepdims=true), inferred from its rank. The frontend passes valid,
   deduplicated axes and nx allocates out, which the mask indexing and the
   stride copy rely on: asserted, the strict ordering the fold driver checks. */
static nx_c_status nx_c_squeeze_out(const nx_c_ndarray *in, const nx_c_ndarray *out,
                                  const int *axes, int n_reduce,
                                  nx_c_ndarray *sq) {
  if (n_reduce < 0 || n_reduce > in->ndim) return NX_C_ERR_AXES;
  bool reduced[NX_C_MAX_NDIM];
  for (int a = 0; a < in->ndim; a++) reduced[a] = false;
  for (int i = 0; i < n_reduce; i++) {
    int a = axes[i];
    if (a < 0 || a >= in->ndim || reduced[a]) return NX_C_ERR_AXES;
    reduced[a] = true;
  }
  int kept = in->ndim - n_reduce;
  if (out->ndim == kept) {
    *sq = *out; /* already squeezed */
    return NX_C_OK;
  }
  if (out->ndim != in->ndim) return NX_C_ERR_OUT_RANK;
  sq->data = out->data;
  sq->offset = out->offset;
  sq->ndim = kept;
  int j = 0;
  for (int a = 0; a < in->ndim; a++)
    if (!reduced[a]) {
      sq->shape[j] = out->shape[a];
      sq->strides[j] = out->strides[a];
      j++;
    }
  return NX_C_OK;
}

void nx_c_fold_funnel(const char *op, const nx_c_fold_table *tbl,
                     nx_c_cost_class cls, value vout, value vin, value vaxes,
                     bool no_identity, int threads, void *ctx) {
  nx_c_ndarray in, out;
  nx_c_status s = nx_c_ndarray_of_value(vin, &in);
  if (s != NX_C_OK) nx_c_raise(op, s);
  s = nx_c_ndarray_of_value(vout, &out);
  if (s != NX_C_OK) nx_c_raise(op, s);

  nx_c_dtype dt = nx_c_dtype_of_value(vin);
  int64_t in_elem = nx_c_elem_size(dt);
  nx_c_dtype odt = nx_c_dtype_of_value(vout);
  int64_t out_elem = nx_c_elem_size(odt);

  int n_reduce = (int)Wosize_val(vaxes);
  if (n_reduce > NX_C_MAX_NDIM) nx_c_raise(op, NX_C_ERR_NDIM);
  int axes[NX_C_MAX_NDIM];
  for (int i = 0; i < n_reduce; i++) axes[i] = (int)Long_val(Field(vaxes, i));

  nx_c_ndarray sq;
  s = nx_c_squeeze_out(&in, &out, axes, n_reduce, &sq);
  if (s != NX_C_OK) nx_c_raise_status(op, s);
  s = nx_c_fold_run(tbl, dt, &in, in_elem, &sq, out_elem, axes, n_reduce,
                   no_identity, cls, threads, ctx);
  if (s != NX_C_OK) nx_c_raise_status(op, s);
}

void nx_c_argreduce_funnel(const char *op, const nx_c_arg_table *tbl,
                          nx_c_cost_class cls, value vout, value vin, int axis,
                          int threads, void *ctx) {
  nx_c_ndarray in, out;
  nx_c_status s = nx_c_ndarray_of_value(vin, &in);
  if (s != NX_C_OK) nx_c_raise(op, s);
  s = nx_c_ndarray_of_value(vout, &out);
  if (s != NX_C_OK) nx_c_raise(op, s);

  nx_c_dtype dt = nx_c_dtype_of_value(vin);
  int64_t in_elem = nx_c_elem_size(dt);

  nx_c_ndarray sq;
  s = nx_c_squeeze_out(&in, &out, &axis, 1, &sq);
  if (s != NX_C_OK) nx_c_raise_status(op, s);
  s = nx_c_argreduce_run(tbl, dt, &in, in_elem, &sq, axis, cls, threads, ctx);
  if (s != NX_C_OK) nx_c_raise_status(op, s);
}

void nx_c_scan_funnel(const char *op, const nx_c_scan_table *tbl,
                     nx_c_cost_class cls, value vout, value vin, int axis,
                     int threads, void *ctx) {
  nx_c_ndarray in, out;
  nx_c_status s = nx_c_ndarray_of_value(vin, &in);
  if (s != NX_C_OK) nx_c_raise(op, s);
  s = nx_c_ndarray_of_value(vout, &out);
  if (s != NX_C_OK) nx_c_raise(op, s);

  nx_c_dtype dt = nx_c_dtype_of_value(vin);
  int64_t in_elem = nx_c_elem_size(dt);
  nx_c_dtype odt = nx_c_dtype_of_value(vout);
  int64_t out_elem = nx_c_elem_size(odt);

  s = nx_c_scan_run(tbl, dt, &in, in_elem, &out, out_elem, axis, cls, threads,
                   ctx);
  if (s != NX_C_OK) nx_c_raise_status(op, s);
}
