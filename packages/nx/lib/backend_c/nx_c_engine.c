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

   Contents, top to bottom: the funnel raisers; the persistent thread pool and
   its parallel-for; the one parallel-policy table (nx_c_threads_for); dimension
   coalescing; and the four generated-family drivers (map, fold, argreduce,
   scan). Every driver returns a status; the binding raises on non-NULL. */

#include <caml/fail.h>
#include <caml/threads.h>

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__) || defined(_SC_NPROCESSORS_ONLN)
#include <unistd.h>
#endif

#include "nx_c_engine.h"

/* ── Funnel raisers ────────────────────────────────────────────────────────
   The one place a status becomes an exception. Called only with the runtime
   lock held (before enter_blocking_section, or after leave). caml_failwith /
   caml_invalid_argument copy the message into an OCaml string, so a stack
   buffer is sufficient and nothing leaks. */

void nx_c_raise(const char *op, nx_c_status status) {
  char buf[256];
  snprintf(buf, sizeof buf, "%s: %s", op, status ? status : "unknown error");
  caml_failwith(buf);
}

void nx_c_raise_invalid(const char *op, nx_c_status status) {
  char buf[256];
  snprintf(buf, sizeof buf, "%s: %s", op, status ? status : "invalid argument");
  caml_invalid_argument(buf);
}

/* ── Thread pool ───────────────────────────────────────────────────────────

   A fixed set of persistent workers created lazily on first parallel use and
   sized to the physical core count. Work is a single integer range [0, total)
   cut into `nchunks` contiguous chunks (nx_c_chunks_for below); the `active`
   participating threads — the calling (main) thread as worker 0 plus spawned
   workers 1..active-1 — loop claiming the next unclaimed chunk index from a
   shared atomic counter until the chunks run out. Dynamic claiming, not a
   deque: a chunk's [lo, hi) is a pure function of its index, so one relaxed
   fetch-add is the entire scheduler. Faster threads absorb the tail a slower
   one (an OS-preempted core, a costlier unit) would otherwise drag through the
   join barrier; the P-core cap in nx_c_threads_for removes only the class-level
   half of that heterogeneity, this removes the per-job half. A thread's worker
   index is its pool identity, stable across every chunk it claims, so
   per-worker scratch and error slots stay exclusive. There is no per-job
   allocation: a job is published under the pool mutex behind a monotonically
   increasing generation counter, workers wake on a condition variable, and
   completion is a countdown signalled back to main.

   Workers run pure C kernels and never touch the OCaml runtime, so they are not
   registered with it. Teardown policy: the pool lives until process exit. The
   workers block forever on the wake condition between jobs; the OS reclaims them
   at exit. There is no join and no destroy — a shared numeric pool has no
   well-defined shutdown point, and leaking a handful of parked threads to
   process teardown is the correct trade. */

#define NX_C_MAX_THREADS 64

/* nx_c_range_body is declared in nx_c_engine.h (custom families use it). */

typedef struct nx_c_pool nx_c_pool;

typedef struct {
  nx_c_pool *pool;
  int id;
} nx_c_worker_arg;

struct nx_c_pool {
  pthread_t threads[NX_C_MAX_THREADS]; /* [1, nworkers); slot 0 is the caller */
  nx_c_worker_arg worker_args[NX_C_MAX_THREADS];
  int nworkers;                       /* physical cores, clamped [1, MAX] */
  pthread_mutex_t drive; /* one published parallel region at a time */
  pthread_mutex_t mtx;
  pthread_cond_t wake; /* workers wait here for a new generation */
  pthread_cond_t done; /* main waits here for the job to finish */
  uint64_t generation; /* bumped once per published job */
  int active;          /* workers participating in the current job */
  int pending;         /* participating workers not yet finished */
  nx_c_range_body body;
  void *body_ctx;
  int64_t total;
  int64_t nchunks;      /* chunks in the current job, in [1, total] */
  _Atomic int64_t next; /* next unclaimed chunk index; reset per job */
};

/* The pool is heap-owned so a fork child can abandon the inherited object and
   lazily build a fresh one. Only the thread that called fork survives in the
   child; the copied worker thread ids, mutex state, and condition waiters are
   therefore unusable. Reinitializing those live pthread objects in place would
   be undefined; the child instead leaves only the pool object inherited at that
   fork unreachable, matching the process-lifetime teardown policy above.

   g_pool_init_mtx serializes lazy creation. The atfork prepare handler holds it
   and the current pool's drive/job locks, so fork observes no active region and
   cannot race creation. The parent releases those locks. The child clears the
   pointer and releases only the still-valid init mutex; the abandoned pool stays
   unreachable and its locked pthread objects are never touched again. */
static _Atomic(nx_c_pool *) g_pool;
static pthread_mutex_t g_pool_init_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t g_pool_atfork_once = PTHREAD_ONCE_INIT;
static int g_pool_atfork_ok;

/* Hardware CPU count, computed once. Apple: physical cores (hw.physicalcpu) —
   Apple Silicon has no SMT, so this equals the online count. Elsewhere: online
   logical CPUs (_SC_NPROCESSORS_ONLN), which on SMT x86 exceeds physical; the
   bandwidth policy caps effective threads regardless.
*/
static int g_ncores;
static pthread_once_t g_ncores_once = PTHREAD_ONCE_INIT;

static int nx_c_cpu_count(void) {
#if defined(__APPLE__)
  int n = 0;
  size_t sz = sizeof n;
  if (sysctlbyname("hw.physicalcpu", &n, &sz, NULL, 0) == 0 && n > 0) return n;
  return 1;
#elif defined(_SC_NPROCESSORS_ONLN)
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return (n > 0) ? (int)n : 1;
#else
  return 1;
#endif
}

static void nx_c_ncores_init(void) {
  int n = nx_c_cpu_count();
  if (n < 1) n = 1;
  if (n > NX_C_MAX_THREADS) n = NX_C_MAX_THREADS;
  g_ncores = n;
}

static int nx_c_ncores(void) {
  pthread_once(&g_ncores_once, nx_c_ncores_init);
  return g_ncores;
}

/* Performance-core count, for the compute/heavy split. Apple Silicon is
   heterogeneous (P + E cores), and an E-core is a net loss for compute-bound
   work even under the claim dispatch: any chunk it claims runs ~2-3x slower,
   and a coarse chunk (a whole GEMM panel of a unit-granular HEAVY job) still
   drags the join once claimed, while the E-cores add little compute in return.
   gemm-accel measured it directly under the static split: f32 GEMM at nth=8
   (P-cores) hit 418 GFLOP/s vs 390 at nth=10 (all cores) — the two E-core
   shares cost more than they added. So COMPUTE/HEAVY are capped at the P-core
   count and the claim loop then balances within that homogeneous set; BANDWIDTH
   keeps the full pool (memory-bound work tolerates E-cores — nx_c_threads_for).
   macOS reports the top performance level as hw.perflevel0.physicalcpu; a
   homogeneous machine (or any platform without the query) has no slow tier, so
   P-cores degrades to the full count and the cap is a no-op. */
static int g_pcores;
static pthread_once_t g_pcores_once = PTHREAD_ONCE_INIT;

static void nx_c_pcores_init(void) {
  int p = 0;
#if defined(__APPLE__)
  size_t sz = sizeof p;
  if (sysctlbyname("hw.perflevel0.physicalcpu", &p, &sz, NULL, 0) != 0 || p <= 0)
    p = 0;
#endif
  if (p < 1) p = nx_c_ncores(); /* homogeneous / unknown: no cap below the pool */
  if (p > nx_c_ncores()) p = nx_c_ncores(); /* never exceed the pool we built */
  g_pcores = p;
}

static int nx_c_pcores(void) {
  pthread_once(&g_pcores_once, nx_c_pcores_init);
  return g_pcores;
}

/* Proportional cut: chunk idx of `parts` over [0, total). Balanced to one unit
   and never empty when parts <= total — which nx_c_chunks_for guarantees, so the
   claim loops skip no index. */
static void nx_c_chunk(int64_t total, int64_t parts, int64_t idx, int64_t *lo,
                      int64_t *hi) {
  *lo = idx * total / parts;
  *hi = (idx + 1) * total / parts;
}

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

static void *nx_c_worker(void *arg) {
  const nx_c_worker_arg *worker_arg = arg;
  nx_c_pool *pool = worker_arg->pool;
  int id = worker_arg->id; /* 1 .. nworkers-1 */
  /* Workers are created before the first job, when generation is 0. Seeding
     `seen` to 0 (not a read of the live generation) closes the startup race: a
     worker that first runs *after* main has already published job 1 sees
     generation != seen and processes it, rather than waiting for a job 2 that
     main is blocked awaiting the completion of job 1 to send. */
  uint64_t seen = 0;
  pthread_mutex_lock(&pool->mtx);
  for (;;) {
    while (pool->generation == seen)
      pthread_cond_wait(&pool->wake, &pool->mtx);
    seen = pool->generation;
    int active = pool->active;
    nx_c_range_body body = pool->body;
    void *ctx = pool->body_ctx;
    int64_t total = pool->total;
    int64_t nchunks = pool->nchunks;
    int participates = (id < active);
    pthread_mutex_unlock(&pool->mtx);

    if (participates) {
      /* Claim loop. Relaxed fetch-add suffices: the RMW's atomicity alone
         makes every claimed index unique, and all the ordering this job needs
         — fields and input data visible before work, body writes visible to
         the joiner — rides the mutex handshake at publish and at the pending
         countdown below. The final (losing) fetch-add merely overshoots. */
      for (;;) {
        int64_t c =
            atomic_fetch_add_explicit(&pool->next, 1, memory_order_relaxed);
        if (c >= nchunks) break;
        int64_t lo, hi;
        nx_c_chunk(total, nchunks, c, &lo, &hi);
        body(lo, hi, id, ctx); /* worker index: thread id, not chunk */
      }
    }

    pthread_mutex_lock(&pool->mtx);
    if (participates && --pool->pending == 0) pthread_cond_signal(&pool->done);
    /* hold the lock across the loop edge so the generation re-check is atomic
       with respect to the next published job */
  }
  return NULL;
}

static nx_c_pool *nx_c_pool_create(void) {
  nx_c_pool *pool = calloc(1, sizeof(*pool));
  if (!pool) return NULL;
  int n = nx_c_ncores();
  pool->nworkers = n;
  atomic_init(&pool->next, 0);
  if (pthread_mutex_init(&pool->drive, NULL) != 0) goto fail_pool;
  if (pthread_mutex_init(&pool->mtx, NULL) != 0) goto fail_drive;
  if (pthread_cond_init(&pool->wake, NULL) != 0) goto fail_mtx;
  if (pthread_cond_init(&pool->done, NULL) != 0) goto fail_wake;
  for (intptr_t i = 1; i < n; i++) {
    pool->worker_args[i].pool = pool;
    pool->worker_args[i].id = (int)i;
    if (pthread_create(&pool->threads[i], NULL, nx_c_worker,
                       &pool->worker_args[i]) != 0) {
      /* Spawn failure is not fatal: run with the workers we have (main alone,
         if none), which is correct, only slower. */
      pool->nworkers = (int)i;
      break;
    }
  }
  return pool;

fail_wake:
  pthread_cond_destroy(&pool->wake);
fail_mtx:
  pthread_mutex_destroy(&pool->mtx);
fail_drive:
  pthread_mutex_destroy(&pool->drive);
fail_pool:
  free(pool);
  return NULL;
}

static void nx_c_pool_atfork_prepare(void) {
  pthread_mutex_lock(&g_pool_init_mtx);
  nx_c_pool *pool = atomic_load_explicit(&g_pool, memory_order_acquire);
  if (pool) {
    pthread_mutex_lock(&pool->drive);
    pthread_mutex_lock(&pool->mtx);
  }
}

static void nx_c_pool_atfork_parent(void) {
  nx_c_pool *pool = atomic_load_explicit(&g_pool, memory_order_acquire);
  if (pool) {
    pthread_mutex_unlock(&pool->mtx);
    pthread_mutex_unlock(&pool->drive);
  }
  pthread_mutex_unlock(&g_pool_init_mtx);
}

static void nx_c_pool_atfork_child(void) {
  /* The inherited pool has no worker threads in this process. Abandon it; the
     first child dispatch builds a fresh pool with fresh pthread objects. */
  atomic_store_explicit(&g_pool, NULL, memory_order_release);
  pthread_mutex_unlock(&g_pool_init_mtx);
}

static void nx_c_pool_register_atfork(void) {
  g_pool_atfork_ok =
      pthread_atfork(nx_c_pool_atfork_prepare, nx_c_pool_atfork_parent,
                     nx_c_pool_atfork_child) == 0;
}

static nx_c_pool *nx_c_pool_get(void) {
  pthread_once(&g_pool_atfork_once, nx_c_pool_register_atfork);
  /* If atfork registration fails, persistent workers cannot be made safe for a
     later fork. Correctly degrade to caller-only execution. */
  if (!g_pool_atfork_ok) return NULL;
  nx_c_pool *pool = atomic_load_explicit(&g_pool, memory_order_acquire);
  if (pool) return pool;
  pthread_mutex_lock(&g_pool_init_mtx);
  pool = atomic_load_explicit(&g_pool, memory_order_relaxed);
  if (!pool) {
    pool = nx_c_pool_create();
    atomic_store_explicit(&g_pool, pool, memory_order_release);
  }
  pthread_mutex_unlock(&g_pool_init_mtx);
  return pool;
}

/* Raw pool dispatch, runtime-lock agnostic: cut [0, total) into policy-sized
   chunks (nx_c_chunks_for) and let `nthreads` threads claim them until none
   remain, the calling thread as worker 0. Internal — nx_c_parallel_for wraps it
   with the lock handshake so no caller (and no family TU) needs the runtime
   headers. For nthreads>1 the lock must already be released; nthreads<=1 runs
   body inline on the caller as one whole-range call.

   Counter lifetime: `next` is reset under both mutexes below and claimed only
   by threads participating in the current generation. A participant's last
   fetch-add (the losing one) happens before its pending decrement, and main
   returns only after pending reaches 0 — so once the join completes no thread
   can touch the counter again until the next publish resets it. */
static void nx_c_pool_dispatch(int nthreads, int64_t total, int64_t bytes,
                              nx_c_range_body body, void *ctx) {
  if (total <= 0) return;
  if (nthreads <= 1) {
    body(0, total, 0, ctx);
    return;
  }
  nx_c_pool *pool = nx_c_pool_get();
  if (!pool) {
    body(0, total, 0, ctx);
    return;
  }
  if (nthreads > pool->nworkers) nthreads = pool->nworkers;
  if (nthreads <= 1) {
    body(0, total, 0, ctx);
    return;
  }

  int64_t nchunks = nx_c_chunks_for(nthreads, total, bytes);

  pthread_mutex_lock(&pool->drive); /* one parallel region at a time */
  pthread_mutex_lock(&pool->mtx);
  pool->body = body;
  pool->body_ctx = ctx;
  pool->total = total;
  pool->nchunks = nchunks;
  /* Relaxed store: the mutex hand-off publishes it with the other job fields
     (workers touch the counter only after acquiring mtx and reading the new
     generation). */
  atomic_store_explicit(&pool->next, 0, memory_order_relaxed);
  pool->active = nthreads;
  pool->pending = nthreads - 1;
  pool->generation++;
  pthread_cond_broadcast(&pool->wake);
  pthread_mutex_unlock(&pool->mtx);

  for (;;) { /* main claims as worker 0 (same loop as nx_c_worker) */
    int64_t c =
        atomic_fetch_add_explicit(&pool->next, 1, memory_order_relaxed);
    if (c >= nchunks) break;
    int64_t lo, hi;
    nx_c_chunk(total, nchunks, c, &lo, &hi);
    body(lo, hi, 0, ctx);
  }

  pthread_mutex_lock(&pool->mtx);
  while (pool->pending != 0) pthread_cond_wait(&pool->done, &pool->mtx);
  pthread_mutex_unlock(&pool->mtx);
  pthread_mutex_unlock(&pool->drive);
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
   raise), so the scratch is safe from allocation through the join. free(NULL) is
   a no-op, so NULL (the generated drivers) costs nothing. */
void nx_c_parallel_for(int nthreads, int64_t total, int64_t bytes,
                      nx_c_range_body body, void *ctx, void *free_on_exit) {
  int release = (nthreads > 1) || (bytes >= NX_C_LOCK_RELEASE_BYTES);
  if (release) caml_enter_blocking_section();
  nx_c_pool_dispatch(nthreads, total, bytes, body, ctx);
  free(free_on_exit);
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

/* Test hook: the coalesced rank for the given operands (see nx_c_selftest.c). */
int nx_c_selftest_coalesce_rank(const nx_c_ndarray *ops, int nop,
                               const int64_t *elem_size, int64_t *total) {
  nx_c_plan p;
  nx_c_coalesce_map(ops, nop, elem_size, &p);
  if (total) *total = p.total;
  return p.ndim;
}

/* Test hook: hardware CPU count, so the self-test can assert the policy splits
   when hardware allows. */
int nx_c_selftest_ncores(void) { return nx_c_ncores(); }

/* Test hook: exercise the exported nx_c_parallel_for under the claim dispatch
   and report what the contract actually promises: every chunk carries a worker
   index in [0, nth), and the claimed chunks partition [0, total) exactly (no
   gap, no overlap, verified by per-unit marks plus a covered-count). WHICH
   worker ids appear beyond that is a race by design — a fast thread may claim
   everything — so the static split's all-indices-contiguous guarantee is gone,
   and `distinct` reports what happened rather than a promise.

   `gate` makes multi-thread fan-out deterministic, not probabilistic: every
   body entry bumps `arrivals`, and a body holding a PARTIAL chunk (hi-lo <
   total, i.e. the range was actually cut) spins until a second entry lands.
   The spinner holds its chunk, so it cannot claim the rest of the range
   itself; the parallel path only cuts when >= 2 threads participate, so
   another thread must claim a remaining chunk and bump the counter — the spin
   terminates and >= 2 distinct worker ids are recorded. The serial path is one
   whole-range call, which the partial-chunk test exempts; pass gate only when
   the pool can actually go parallel (cores > 1).

   Also drives the free_on_exit path: a heap per-worker scratch is written by
   the body (live during the run) and handed to nx_c_parallel_for to free — a
   double-free or leak here would show under ASan. Honors the contract by
   holding the lock on entry. */
typedef struct {
  pthread_mutex_t m;
  unsigned char seen[NX_C_MAX_THREADS];
  int distinct;
  int max_worker;
  int bad_worker;      /* saw an index outside [0, NX_C_MAX_THREADS) */
  int bad_range;       /* saw a chunk outside [0, total), or hi < lo */
  int overlap;         /* some unit claimed twice */
  int64_t covered;     /* sum of (hi - lo) over claimed chunks */
  int64_t total;
  unsigned char *mark; /* one flag per unit, or NULL if OOM */
  int gate;
  _Atomic int arrivals;
  int *scratch; /* per-worker heap scratch; freed by nx_c_parallel_for */
} nx_c_widx_probe;

static void nx_c_widx_body(int64_t lo, int64_t hi, int worker, void *ctx) {
  nx_c_widx_probe *p = ctx;
  atomic_fetch_add_explicit(&p->arrivals, 1, memory_order_relaxed);
  if (p->scratch && worker >= 0 && worker < NX_C_MAX_THREADS)
    p->scratch[worker] = worker; /* scratch is live here, before the free */
  /* Clamp before touching mark: a cut mutant handing an out-of-range chunk must
     FAIL the partition check, never scribble past the test heap. `covered`
     keeps the raw extent so an oversized cut also shows as covered != total. */
  int64_t mlo = lo < 0 ? 0 : lo;
  int64_t mhi = hi > p->total ? p->total : hi;
  pthread_mutex_lock(&p->m);
  if (lo < 0 || hi > p->total || hi < lo) p->bad_range = 1;
  if (worker >= 0 && worker < NX_C_MAX_THREADS) {
    if (!p->seen[worker]) {
      p->seen[worker] = 1;
      p->distinct++;
    }
  } else {
    p->bad_worker = 1;
  }
  if (worker > p->max_worker) p->max_worker = worker;
  p->covered += hi - lo;
  if (p->mark)
    for (int64_t i = mlo; i < mhi; i++) {
      if (p->mark[i]) p->overlap = 1;
      p->mark[i] = 1;
    }
  pthread_mutex_unlock(&p->m);
  if (p->gate && hi - lo < p->total)
    while (atomic_load_explicit(&p->arrivals, memory_order_relaxed) < 2)
      ;
}

int nx_c_selftest_worker_indices(int nth, int64_t total, int gate, int *out_max,
                                int *out_partition_ok,
                                int *out_pool_workers) {
  nx_c_widx_probe p;
  pthread_mutex_init(&p.m, NULL);
  memset(p.seen, 0, sizeof p.seen);
  p.distinct = 0;
  p.max_worker = -1;
  p.bad_worker = 0;
  p.bad_range = 0;
  p.overlap = 0;
  p.covered = 0;
  p.total = total;
  p.mark = calloc((size_t)total, 1);
  p.gate = gate;
  atomic_store_explicit(&p.arrivals, 0, memory_order_relaxed);
  p.scratch = malloc((size_t)(nth > 0 ? nth : 1) * sizeof(int));
  /* Report the pool's REAL worker count, so callers gate fan-out assertions on
     a pool that actually has >= 2 threads — if allocation, atfork registration,
     or every spawn failed, dispatch legally degrades to serial and a
     distinct>=2 assertion would false-fail. */
  nx_c_pool *pool = nx_c_pool_get();
  if (out_pool_workers) *out_pool_workers = pool ? pool->nworkers : 1;
  /* Effectively unbounded `bytes` keeps the chunk policy off its byte floor, so
     total > nth * chunks-per-worker exercises the multi-chunk claim loop and a
     small total the unit-granular tail shape. The scratch is handed to the
     primitive to free — not freed here (it would be a leak on the
     re-acquire-raises path; that is exactly what free_on_exit fixes). */
  nx_c_parallel_for(nth, total, INT64_MAX / 2, nx_c_widx_body, &p, p.scratch);
  pthread_mutex_destroy(&p.m);
  int partition_ok = p.covered == total && !p.overlap && !p.bad_worker &&
                     !p.bad_range && p.max_worker >= 0 &&
                     p.max_worker < (nth > 0 ? nth : 1);
  free(p.mark);
  if (out_max) *out_max = p.max_worker;
  if (out_partition_ok) *out_partition_ok = partition_ok;
  return p.distinct;
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

/* Test hook: seek2's positioning for a linear index, as byte offsets from a
   caller-supplied base. Lets the self-test check a mid-nest (idx>0) thread start
   directly — the chunk-start seek every parallel driver depends on — rather than
   only through a driver. The caller sizes `base` to cover the offsets. */
void nx_c_selftest_seek2(int nk, const int64_t *shape, const int64_t *s_in,
                        const int64_t *s_out, int64_t idx, char *base,
                        int64_t *in_off, int64_t *out_off) {
  int64_t coord[NX_C_MAX_NDIM];
  char *ip;
  char *op;
  nx_c_seek2(nk, shape, s_in, s_out, idx, coord, base, base, &ip, &op);
  *in_off = ip - base;
  *out_off = op - base;
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

  /* On a non-empty output, a 0-stride dim of extent > 1 would alias elements
     (parallel threads racing one cell, wrong even serially). Checked on the
     coalesced output (index 0), after the empty short-circuit — an empty tensor
     writes nothing, so a stride-0 dim there is harmless. Verified, not assumed. */
  for (int i = 0; i < p.ndim; i++)
    if (p.shape[i] > 1 && p.bstride[0][i] == 0) return NX_C_ERR_OUT_ALIASED;

  int64_t bytes = 0;
  for (int k = 0; k < nop; k++) bytes += p.total * elem_size[k];

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
   Parallelize over output elements (the kept-axis nest); each output seeds an
   accumulator once, streams every contributing input run through step (the
   innermost reduced axis is the run so it streams over the most contiguous
   input dim), then converts through fini. */

typedef struct {
  nx_c_fold_init *init;
  nx_c_fold_step *step;
  nx_c_fold_fini *fini;
  void *ctx;
  char *in_base;
  char *out_base;
  int nk; /* kept (non-reduced) dims */
  int64_t kshape[NX_C_MAX_NDIM];
  int64_t k_in_stride[NX_C_MAX_NDIM];  /* byte */
  int64_t k_out_stride[NX_C_MAX_NDIM]; /* byte */
  int nr;                             /* reduced dims; last is the run axis */
  int64_t rshape[NX_C_MAX_NDIM];
  int64_t r_in_stride[NX_C_MAX_NDIM]; /* byte */
} nx_c_fold_exec;

static void nx_c_fold_reduce_one(const nx_c_fold_exec *e, char *ip, char *op) {
  nx_c_acc acc;
  e->init(&acc, e->ctx);
  if (e->nr == 0) {
    e->step(&acc, ip, 0, 1, e->ctx);
  } else {
    int rod = e->nr - 1; /* reduced odometer dims; dim rod is the run */
    int64_t run_len = e->rshape[rod];
    int64_t run_stride = e->r_in_stride[rod];
    int64_t outer = 1;
    for (int d = 0; d < rod; d++) outer *= e->rshape[d];
    int64_t rcoord[NX_C_MAX_NDIM];
    char *rp = ip;
    for (int d = 0; d < rod; d++) rcoord[d] = 0;
    for (int64_t o = 0; o < outer; o++) {
      e->step(&acc, rp, run_stride, run_len, e->ctx);
      for (int d = rod - 1; d >= 0; d--) {
        if (++rcoord[d] < e->rshape[d]) {
          rp += e->r_in_stride[d];
          break;
        }
        rcoord[d] = 0;
        rp -= (e->rshape[d] - 1) * e->r_in_stride[d];
      }
    }
  }
  e->fini(op, &acc, e->ctx);
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
   Chosen when a kept axis out-contiguities every reduced axis: reduce over an
   outer (large-stride) axis while a kept axis is inner. The per-output path
   would gather each output's reduced run one cache line per element; streaming
   instead walks the input contiguously and folds each reduced row of `lane_len`
   elements into a per-lane accumulator array (in the compute type, so wide
   accumulation is preserved), vectorizing across lanes. Parallelizes over the
   panels — the odometer over the kept dims OTHER than the lane, which index
   independent output slices. Each thread reuses one accumulator slot of
   lane_len accumulators (nx_c_engine.h streaming contract). */
typedef struct {
  nx_c_fold_stream *stream;
  nx_c_fold_scatter *scatter;
  void *ctx;
  char *in_base;
  char *out_base;
  int64_t lane_len;        /* the vectorized inner (most contiguous kept) axis */
  int64_t lane_in_stride;  /* byte */
  int64_t lane_out_stride; /* byte */
  int np;                  /* panel dims (kept dims other than the lane) */
  int64_t pshape[NX_C_MAX_NDIM];
  int64_t p_in_stride[NX_C_MAX_NDIM];  /* byte */
  int64_t p_out_stride[NX_C_MAX_NDIM]; /* byte */
  int nr;                             /* reduced dims (all in the odometer) */
  int64_t rshape[NX_C_MAX_NDIM];
  int64_t r_in_stride[NX_C_MAX_NDIM]; /* byte */
  char *scratch;                     /* nthreads slots of slot_bytes each */
  int64_t slot_bytes;                /* lane_len * sizeof(nx_c_acc) */
} nx_c_fold_stream_exec;

static void nx_c_fold_stream_body(int64_t lo, int64_t hi, int worker,
                                 void *vctx) {
  const nx_c_fold_stream_exec *e = vctx;
  void *accs = e->scratch + (int64_t)worker * e->slot_bytes;
  int64_t coord[NX_C_MAX_NDIM];
  char *ip;
  char *op;
  nx_c_seek2(e->np, e->pshape, e->p_in_stride, e->p_out_stride, lo, coord,
            e->in_base, e->out_base, &ip, &op);
  int64_t rtotal = 1;
  for (int d = 0; d < e->nr; d++) rtotal *= e->rshape[d];
  for (int64_t it = lo; it < hi; it++) {
    int64_t rcoord[NX_C_MAX_NDIM];
    for (int d = 0; d < e->nr; d++) rcoord[d] = 0;
    char *rp = ip;
    for (int64_t r = 0; r < rtotal; r++) {
      e->stream(accs, rp, e->lane_in_stride, e->lane_len, r == 0, e->ctx);
      for (int d = e->nr - 1; d >= 0; d--) {
        if (++rcoord[d] < e->rshape[d]) {
          rp += e->r_in_stride[d];
          break;
        }
        rcoord[d] = 0;
        rp -= (e->rshape[d] - 1) * e->r_in_stride[d];
      }
    }
    e->scatter(op, e->lane_out_stride, accs, e->lane_len, e->ctx);
    nx_c_next2(e->np, e->pshape, e->p_in_stride, e->p_out_stride, coord, &ip,
              &op);
  }
}

/* Upper bound on the streaming accumulator scratch (nth * lane_len *
   sizeof(nx_c_acc)). The per-output path allocates nothing, so a shape with a
   short reduced axis and a huge lane (e.g. reducing axis 0 of [2, 1e8] wants
   ~1.6 GB of accumulators) must NOT turn a working reduction into an allocation
   failure — over this cap the driver falls back to the per-output path. That is
   the right call on the merits, not just a safety valve: streaming only wins by
   turning a strided reduced-axis gather into contiguous reads, and a lane that
   wide means the kept axis is already contiguous and dominates the bandwidth, so
   the per-output path is close anyway. 64 MiB covers every realistic reduction
   (a 4M-element single-thread lane) while bounding the transient allocation. */
#define NX_C_FOLD_STREAM_SCRATCH_CAP (64 * 1024 * 1024)

/* Build the streaming exec from the shared classification, allocate the
   per-thread accumulator scratch, and drive the panels. `lane` is the index of
   the chosen lane within the kept-axis arrays; `nth`/`panels` are the caller's
   already-clamped split (the caller sized the scratch against the cap from
   them). */
static nx_c_status nx_c_fold_stream_run(const nx_c_stream_table *stbl, nx_c_dtype dt,
                                      const nx_c_fold_exec *fe, int lane, int nth,
                                      int64_t panels, int64_t bytes) {
  nx_c_fold_stream_exec e;
  e.stream = stbl->stream[dt];
  e.scatter = stbl->scatter[dt];
  e.ctx = fe->ctx;
  e.in_base = fe->in_base;
  e.out_base = fe->out_base;
  e.lane_len = fe->kshape[lane];
  e.lane_in_stride = fe->k_in_stride[lane];
  e.lane_out_stride = fe->k_out_stride[lane];
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

  e.slot_bytes = e.lane_len * (int64_t)sizeof(nx_c_acc);
  void *scratch = malloc((size_t)nth * (size_t)e.slot_bytes);
  if (scratch == NULL) return NX_C_ERR_ALLOC;
  e.scratch = scratch;
  /* scratch is freed by nx_c_parallel_for after the join, leak-safe across the
     re-acquire's possible raise (nx_c_engine.h free_on_exit contract). */
  nx_c_parallel_for(nth, panels, bytes, nx_c_fold_stream_body, &e, scratch);
  return NX_C_OK;
}

nx_c_status nx_c_fold_run(const nx_c_fold_table *tbl, const nx_c_stream_table *stbl,
                        nx_c_dtype dt, const nx_c_ndarray *in, int64_t in_elem,
                        const nx_c_ndarray *out, int64_t out_elem,
                        const int *reduce_axes, int n_reduce, bool no_identity,
                        nx_c_cost_class cls, void *ctx) {
  if (tbl->init[dt] == NULL || tbl->step[dt] == NULL || tbl->fini[dt] == NULL)
    return nx_c_dtype_is_packed(dt) ? NX_C_ERR_PACKED : NX_C_ERR_UNSUPPORTED_DTYPE;

  nx_c_fold_exec e;
  e.init = tbl->init[dt];
  e.step = tbl->step[dt];
  e.fini = tbl->fini[dt];
  e.ctx = ctx;
  e.in_base = (char *)in->data + in->offset * in_elem;
  e.out_base = (char *)out->data + out->offset * out_elem;

  /* Split input axes into kept (output-indexing) and reduced. reduce_axes is
     strictly increasing, so a single merge pass classifies each axis. */
  e.nk = 0;
  e.nr = 0;
  int ra = 0;
  for (int a = 0; a < in->ndim; a++) {
    if (ra < n_reduce && reduce_axes[ra] == a) {
      e.rshape[e.nr] = in->shape[a];
      e.r_in_stride[e.nr] = in->strides[a] * in_elem;
      e.nr++;
      ra++;
    } else {
      e.kshape[e.nk] = in->shape[a];
      e.k_in_stride[e.nk] = in->strides[a] * in_elem;
      e.nk++;
    }
  }
  /* The forward scan consumes reduce_axes iff they are strictly increasing and
     in range; any unsorted/duplicate/out-of-range axis leaves ra < n_reduce.
     Verified, not assumed — a binding that forgets to sort gets a loud status
     rather than a silently-wrong partial reduction. */
  if (ra != n_reduce) return NX_C_ERR_AXES;
  /* out is aligned, one axis per kept input axis; a short/long descriptor would
     read unspecified stride slots — so pair the out strides only now that the
     rank is verified. */
  if (out->ndim != e.nk) return NX_C_ERR_OUT_RANK;
  for (int j = 0; j < e.nk; j++) e.k_out_stride[j] = out->strides[j] * out_elem;

  int64_t out_total = 1;
  for (int d = 0; d < e.nk; d++) out_total *= e.kshape[d];
  if (out_total == 0) return NX_C_OK; /* no output elements */
  int64_t reduced_len = 1;
  for (int d = 0; d < e.nr; d++) reduced_len *= e.rshape[d];
  /* max/min have no identity for an empty reduced extent: with outputs to fill
     (out_total > 0 here) an empty extent would store the init sentinel
     (-inf / INT64_MIN). Reject before any kernel runs — the check lives here, in
     the shared driver, so no caller (funnel or raw) can leak the sentinel. */
  if (no_identity && reduced_len == 0) return NX_C_ERR_EMPTY_REDUCE;
  int64_t bytes = out_total * reduced_len * in_elem;

  /* Streaming decision: take it when a kept axis that carries iteration is
     strictly more contiguous than every reduced axis (its stride magnitude is
     the smallest overall), so the reduced run of the per-output path would
     gather with a large stride. The lane must have extent > 1: a size-1 lane
     gives the stream loop nothing to vectorize and more calls than the
     per-output path, and a size-1 axis carries no iteration anyway. The empty
     reduced extent (reduced_len == 0, a sum/prod identity fill) has no first row
     to seed the accumulators, so it stays on the per-output path. */
  if (stbl != NULL && stbl->stream[dt] != NULL && stbl->scatter[dt] != NULL &&
      e.nk >= 1 && e.nr >= 1 && reduced_len >= 1) {
    /* A broadcast (0-stride) kept axis can win the lane: every lane then folds
       the same reduced extent, and the equal outputs are the correct broadcast
       result. The per-lane sum reassociates differently from the per-output
       8-accumulator step, but that stays within the sum tolerance the
       conformance suite already grants (accepted review corner). */
    int lane = -1;
    for (int j = 0; j < e.nk; j++) {
      if (e.kshape[j] <= 1) continue;
      if (lane < 0 || llabs(e.k_in_stride[j]) < llabs(e.k_in_stride[lane]))
        lane = j;
    }
    if (lane >= 0) {
      int64_t run_stride = llabs(e.r_in_stride[0]);
      for (int d = 1; d < e.nr; d++) {
        int64_t s = llabs(e.r_in_stride[d]);
        if (s < run_stride) run_stride = s;
      }
      if (llabs(e.k_in_stride[lane]) < run_stride) {
        int64_t lane_len = e.kshape[lane];
        int64_t panels = out_total / lane_len; /* product of the panel shapes */
        int nth = nx_c_threads_for(cls, panels, lane_len * reduced_len, bytes);
        if (nth > panels) nth = (int)panels;
        if (nth < 1) nth = 1;
        /* Only stream when the accumulator scratch stays bounded; otherwise the
           per-output path (which allocates nothing) is both safe and, for a lane
           this wide, competitive. */
        int64_t scratch = (int64_t)nth * lane_len * (int64_t)sizeof(nx_c_acc);
        if (scratch <= NX_C_FOLD_STREAM_SCRATCH_CAP)
          return nx_c_fold_stream_run(stbl, dt, &e, lane, nth, panels, bytes);
      }
    }
  }

  /* Per-output path: stream the most contiguous reduced axis by moving the
     smallest-|stride| reduced axis to the run position (last). */
  if (e.nr > 1) {
    int best = 0;
    for (int r = 1; r < e.nr; r++)
      if (llabs(e.r_in_stride[r]) < llabs(e.r_in_stride[best])) best = r;
    int last = e.nr - 1;
    int64_t ts = e.rshape[best];
    e.rshape[best] = e.rshape[last];
    e.rshape[last] = ts;
    int64_t tt = e.r_in_stride[best];
    e.r_in_stride[best] = e.r_in_stride[last];
    e.r_in_stride[last] = tt;
  }

  int nth = nx_c_threads_for(cls, out_total, reduced_len, bytes);
  if (nth > out_total) nth = (int)out_total;
  nx_c_parallel_for(nth, out_total, bytes, nx_c_fold_body, &e, NULL);
  return NX_C_OK;
}

/* ── Argreduce driver ──────────────────────────────────────────────────────
   Argmax/argmin over one axis into an int32 output, parallelized over the
   non-axis nest. One run per output (the axis); the kernel carries the running
   extreme and its index. */

nx_c_status nx_c_argreduce_validate(int64_t axis_len) {
  if (axis_len == 0) return NX_C_ERR_EMPTY_REDUCE;
  if (axis_len > INT32_MAX) return NX_C_ERR_ARGREDUCE_CAP;
  return NX_C_OK;
}

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
                             nx_c_cost_class cls, void *ctx) {
  if (tbl->step[dt] == NULL)
    return nx_c_dtype_is_packed(dt) ? NX_C_ERR_PACKED : NX_C_ERR_UNSUPPORTED_DTYPE;

  if (axis < 0 || axis >= in->ndim) return NX_C_ERR_AXIS;
  if (out->ndim != in->ndim - 1) return NX_C_ERR_OUT_RANK;

  int64_t axis_len = in->shape[axis];
  nx_c_status vs = nx_c_argreduce_validate(axis_len);
  if (vs != NX_C_OK) return vs;

  nx_c_arg_exec e;
  e.step = tbl->step[dt];
  e.ctx = ctx;
  e.in_base = (char *)in->data + in->offset * in_elem;
  e.out_base = (char *)out->data + out->offset * (int64_t)sizeof(int32_t);
  e.axis_stride = in->strides[axis] * in_elem;
  e.axis_len = axis_len;

  e.nk = 0;
  for (int a = 0; a < in->ndim; a++) {
    if (a == axis) continue;
    e.kshape[e.nk] = in->shape[a];
    e.k_in_stride[e.nk] = in->strides[a] * in_elem;
    e.k_out_stride[e.nk] = out->strides[e.nk] * (int64_t)sizeof(int32_t);
    e.nk++;
  }

  int64_t out_total = 1;
  for (int d = 0; d < e.nk; d++) out_total *= e.kshape[d];
  if (out_total == 0) return NX_C_OK;
  int64_t bytes = out_total * axis_len * in_elem;

  int nth = nx_c_threads_for(cls, out_total, axis_len, bytes);
  if (nth > out_total) nth = (int)out_total;
  nx_c_parallel_for(nth, out_total, bytes, nx_c_arg_body, &e, NULL);
  return NX_C_OK;
}

/* ── Scan driver ───────────────────────────────────────────────────────────
   Inclusive scan over one axis. The non-axis nest indexes independent slices
   (parallelized); within a slice the kernel walks the axis sequentially. */

typedef struct {
  nx_c_scan_init *init;
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
} nx_c_scan_exec;

static void nx_c_scan_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  (void)worker;
  const nx_c_scan_exec *e = vctx;
  int64_t coord[NX_C_MAX_NDIM];
  char *ip;
  char *op;
  nx_c_seek2(e->nk, e->kshape, e->k_in_stride, e->k_out_stride, lo, coord,
            e->in_base, e->out_base, &ip, &op);
  for (int64_t it = lo; it < hi; it++) {
    nx_c_acc state;
    e->init(&state, e->ctx);
    e->step(op, e->axis_out_stride, ip, e->axis_in_stride, e->axis_len, &state,
            e->ctx);
    nx_c_next2(e->nk, e->kshape, e->k_in_stride, e->k_out_stride, coord, &ip,
              &op);
  }
}

nx_c_status nx_c_scan_run(const nx_c_scan_table *tbl, nx_c_dtype dt,
                        const nx_c_ndarray *in, int64_t in_elem,
                        const nx_c_ndarray *out, int64_t out_elem, int axis,
                        nx_c_cost_class cls, void *ctx) {
  if (tbl->init[dt] == NULL || tbl->step[dt] == NULL)
    return nx_c_dtype_is_packed(dt) ? NX_C_ERR_PACKED : NX_C_ERR_UNSUPPORTED_DTYPE;

  if (axis < 0 || axis >= in->ndim) return NX_C_ERR_AXIS;
  if (out->ndim != in->ndim) return NX_C_ERR_OUT_RANK;

  nx_c_scan_exec e;
  e.init = tbl->init[dt];
  e.step = tbl->step[dt];
  e.ctx = ctx;
  e.in_base = (char *)in->data + in->offset * in_elem;
  e.out_base = (char *)out->data + out->offset * out_elem;
  e.axis_in_stride = in->strides[axis] * in_elem;
  e.axis_out_stride = out->strides[axis] * out_elem;
  e.axis_len = in->shape[axis];

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

  int nth = nx_c_threads_for(cls, slices, e.axis_len, bytes);
  if (nth > slices) nth = (int)slices;
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
   (unsupported dtype, packed, bad kind, the argreduce cap, allocation) is a
   Failure. Runs only on the cold error path, so strcmp is free. */
NX_C_NORETURN void nx_c_raise_status(const char *op, nx_c_status s) {
  if (strcmp(s, NX_C_ERR_EMPTY_REDUCE) == 0 || strcmp(s, NX_C_ERR_AXES) == 0 ||
      strcmp(s, NX_C_ERR_AXIS) == 0 || strcmp(s, NX_C_ERR_OUT_RANK) == 0 ||
      strcmp(s, NX_C_ERR_OUT_ALIASED) == 0 || strcmp(s, NX_C_ERR_SHAPE) == 0)
    nx_c_raise_invalid(op, s);
  nx_c_raise(op, s);
}

/* dtype + element size of an operand, or a bad-kind status. */
static nx_c_status nx_c_operand_dtype(value v, nx_c_dtype *dt, int64_t *elem) {
  nx_c_dtype d = nx_c_dtype_of_value(v);
  if (d == NX_C_DTYPE_COUNT) return NX_C_ERR_BAD_KIND;
  *dt = d;
  *elem = nx_c_elem_size(d);
  return NX_C_OK;
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
    nx_c_dtype dk;
    s = nx_c_operand_dtype(vals[k], &dk, &elem[k]);
    if (s != NX_C_OK) nx_c_raise(op, s);
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
   (keepdims=true), inferred from its rank. Bounds/dup-checks the axes with an
   order-independent mask so a malformed axis is caught before use (the driver
   re-checks strict ordering). */
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
                     const nx_c_stream_table *stbl, nx_c_cost_class cls,
                     value vout, value vin, value vaxes, bool no_identity,
                     void *ctx) {
  nx_c_ndarray in, out;
  nx_c_status s = nx_c_ndarray_of_value(vin, &in);
  if (s != NX_C_OK) nx_c_raise(op, s);
  s = nx_c_ndarray_of_value(vout, &out);
  if (s != NX_C_OK) nx_c_raise(op, s);

  nx_c_dtype dt;
  int64_t in_elem;
  s = nx_c_operand_dtype(vin, &dt, &in_elem);
  if (s != NX_C_OK) nx_c_raise(op, s);
  nx_c_dtype odt;
  int64_t out_elem;
  s = nx_c_operand_dtype(vout, &odt, &out_elem);
  if (s != NX_C_OK) nx_c_raise(op, s);

  int n_reduce = (int)Wosize_val(vaxes);
  if (n_reduce > NX_C_MAX_NDIM) nx_c_raise(op, NX_C_ERR_NDIM);
  int axes[NX_C_MAX_NDIM];
  for (int i = 0; i < n_reduce; i++) axes[i] = (int)Long_val(Field(vaxes, i));

  nx_c_ndarray sq;
  s = nx_c_squeeze_out(&in, &out, axes, n_reduce, &sq);
  if (s != NX_C_OK) nx_c_raise_status(op, s);
  s = nx_c_fold_run(tbl, stbl, dt, &in, in_elem, &sq, out_elem, axes, n_reduce,
                   no_identity, cls, ctx);
  if (s != NX_C_OK) nx_c_raise_status(op, s);
}

void nx_c_argreduce_funnel(const char *op, const nx_c_arg_table *tbl,
                          nx_c_cost_class cls, value vout, value vin, int axis,
                          void *ctx) {
  nx_c_ndarray in, out;
  nx_c_status s = nx_c_ndarray_of_value(vin, &in);
  if (s != NX_C_OK) nx_c_raise(op, s);
  s = nx_c_ndarray_of_value(vout, &out);
  if (s != NX_C_OK) nx_c_raise(op, s);

  nx_c_dtype dt;
  int64_t in_elem;
  s = nx_c_operand_dtype(vin, &dt, &in_elem);
  if (s != NX_C_OK) nx_c_raise(op, s);

  nx_c_ndarray sq;
  s = nx_c_squeeze_out(&in, &out, &axis, 1, &sq);
  if (s != NX_C_OK) nx_c_raise_status(op, s);
  s = nx_c_argreduce_run(tbl, dt, &in, in_elem, &sq, axis, cls, ctx);
  if (s != NX_C_OK) nx_c_raise_status(op, s);
}

void nx_c_scan_funnel(const char *op, const nx_c_scan_table *tbl,
                     nx_c_cost_class cls, value vout, value vin, int axis,
                     void *ctx) {
  nx_c_ndarray in, out;
  nx_c_status s = nx_c_ndarray_of_value(vin, &in);
  if (s != NX_C_OK) nx_c_raise(op, s);
  s = nx_c_ndarray_of_value(vout, &out);
  if (s != NX_C_OK) nx_c_raise(op, s);

  nx_c_dtype dt;
  int64_t in_elem;
  s = nx_c_operand_dtype(vin, &dt, &in_elem);
  if (s != NX_C_OK) nx_c_raise(op, s);
  nx_c_dtype odt;
  int64_t out_elem;
  s = nx_c_operand_dtype(vout, &odt, &out_elem);
  if (s != NX_C_OK) nx_c_raise(op, s);

  s = nx_c_scan_run(tbl, dt, &in, in_elem, &out, out_elem, axis, cls, ctx);
  if (s != NX_C_OK) nx_c_raise_status(op, s);
}
