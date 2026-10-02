/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The host's thread pool: the one set of threads that computes on the host,
   for nx.cpu's kernels and for compiled programs (Nx_device.Program.call).

   A fixed set of persistent workers, created lazily on first parallel use and
   sized to the CPUs the process may use. Work is an integer range [0, total)
   cut into `nchunks` contiguous chunks; the participating threads (the caller
   as worker 0, then spawned workers) claim the next unclaimed chunk from a
   shared counter until none remain. A chunk's [lo, hi) is a pure function of
   its index, so one relaxed fetch-add is the whole scheduler, and a fast
   thread absorbs the tail a slow one (an E-core, a preempted core) would drag
   through the join. A thread's worker index is stable across the chunks it
   claims, so per-worker scratch stays exclusive.

   A job is published behind a generation counter, and its workers count down
   to the caller. Jobs come in bursts, such as the kernels of a compiled
   program, a few microseconds apart: a worker spins on the counter until
   [spin_ns] has passed since it last took part in a job, and the caller spins
   on the countdown for [spin_ns], before either parks on a condition variable.
   A job wakes the parked threads only if one of its workers parked. A job's
   publication then costs no system call, its workers start at once, and a
   worker that a burst's jobs leave out parks once its window ends.

   Workers run pure C and never touch the OCaml runtime, so they are not
   registered with it. The pool lives until process exit, and the OS reclaims
   its workers then. */

#if defined(__linux__)
#define _GNU_SOURCE /* sched_getaffinity, CPU_COUNT */
#include <sched.h>
#endif

#include <caml/alloc.h>
#include <caml/mlvalues.h>

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__) || defined(_SC_NPROCESSORS_ONLN)
#include <unistd.h>
#endif

#include "nx_device.h"

#define MAX_THREADS 64

/* How long a waiting thread spins before it parks. Waking a parked thread
   takes a system call and tens of microseconds before it runs, which a launch
   of a compiled program's kernel paid with each of its workers; spinning for
   longer than the gaps between such launches keeps the workers running for
   the next. A worker that took part in no job for that long parks, so an
   idle pool, and a worker that narrow jobs leave out, cost nothing. */
static const uint64_t spin_ns = 100000;
static const uint64_t busy_ns = 2000;

static inline void relax(void) {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield");
#endif
}

/* Spins while [*word] is [value] ([until] 0) or until it is ([until] 1), up
   to the clock's [deadline], and returns the last value read. The clock is
   read every few loads. Past [busy_ns] of spinning, the thread yields its core
   between reads, so that a spinning pool slows other threads little when the
   cores are all taken. */
static uint64_t spin(_Atomic uint64_t *word, uint64_t value, int until,
                     uint64_t deadline) {
  uint64_t v = atomic_load_explicit(word, memory_order_acquire);
  if ((v == value) == until) return v;
  uint64_t start = nx_device_now_ns();
  for (;;) {
    for (int i = 0; i < 64; i++) {
      relax();
      v = atomic_load_explicit(word, memory_order_acquire);
      if ((v == value) == until) return v;
    }
    uint64_t now = nx_device_now_ns();
    if (now >= deadline) return v;
    if (now - start >= busy_ns) sched_yield();
  }
}

typedef struct pool pool;

typedef struct {
  pool *pool;
  int id;
} worker_arg;

struct pool {
  pthread_t threads[MAX_THREADS]; /* [1, nworkers); slot 0 is the caller */
  worker_arg worker_args[MAX_THREADS];
  int nworkers;
  pthread_mutex_t drive; /* one published parallel region at a time */
  pthread_mutex_t mtx;   /* guards parking */
  pthread_cond_t wake;   /* parked workers wait here for a new generation */
  pthread_cond_t done;   /* a parked caller waits here for the job's end */
  _Atomic uint64_t generation; /* even; odd while a job is being written */
  _Atomic uint64_t pending;    /* participating workers not yet finished */
  _Atomic uint64_t parked;     /* bit [id] set while worker [id] is parked */
  _Atomic int waiting;         /* whether the caller parked on [done] */
  /* The job, written while the generation is odd and read while it is even:
     a worker that is no participant may read the next job's while the caller
     writes it, so each field is atomic, and a worker that finds the
     generation moved while it read them reads them again. */
  _Atomic int active; /* threads participating in the current job */
  _Atomic(nx_device_range_body) body;
  _Atomic(void *) body_ctx;
  _Atomic int64_t total;
  _Atomic int64_t nchunks;
  _Atomic int64_t next; /* next unclaimed chunk index; reset per job */
};

/* The pool is heap-owned so that a fork child can abandon the inherited one
   and lazily build its own: only the forking thread survives in the child, so
   the copied worker ids, mutexes and condition waiters are unusable, and
   reinitialising live pthread objects in place is undefined.

   init_mtx serialises lazy creation. The atfork prepare handler holds it and
   the current pool's drive and job locks, so fork sees no active region; the
   parent releases them, and the child clears the pointer and releases only the
   still-valid init_mtx. */
static _Atomic(pool *) g_pool;
static pthread_mutex_t init_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t atfork_once = PTHREAD_ONCE_INIT;
static int atfork_ok;

/* The CPUs the process may use, computed once. Apple: physical cores, which
   equal the logical ones on Apple Silicon. Linux: the affinity mask (taskset,
   a cgroup's cpuset, a container's set), bounded by the cgroup's quota
   (cpu.max), so that the pool has no more workers than CPUs it can occupy: 14
   workers taking turns on 6 CPUs ran a batched product 1.6 times slower.
   Elsewhere: the online logical CPUs. */
static int g_ncores;
static pthread_once_t ncores_once = PTHREAD_ONCE_INIT;

static int cpu_count(void) {
#if defined(__APPLE__)
  int n = 0;
  size_t sz = sizeof n;
  if (sysctlbyname("hw.physicalcpu", &n, &sz, NULL, 0) == 0 && n > 0) return n;
  return 1;
#elif defined(__linux__)
  cpu_set_t set;
  long n = sched_getaffinity(0, sizeof set, &set) == 0
               ? CPU_COUNT(&set)
               : sysconf(_SC_NPROCESSORS_ONLN);
  /* cpu.max is "max PERIOD" without a quota and "QUOTA PERIOD" with one. */
  FILE *f = fopen("/sys/fs/cgroup/cpu.max", "r");
  if (f) {
    long quota, period;
    if (fscanf(f, "%ld %ld", &quota, &period) == 2 && quota > 0 && period > 0 &&
        quota / period < n)
      n = quota / period > 0 ? quota / period : 1;
    fclose(f);
  }
  return (n > 0) ? (int)n : 1;
#elif defined(_SC_NPROCESSORS_ONLN)
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return (n > 0) ? (int)n : 1;
#else
  return 1;
#endif
}

static void ncores_init(void) {
  int n = cpu_count();
  if (n < 1) n = 1;
  if (n > MAX_THREADS) n = MAX_THREADS;
  g_ncores = n;
}

static int workers(void) {
  pthread_once(&ncores_once, ncores_init);
  return g_ncores;
}

/* The performance cores, for compute-bound work. Apple Silicon has P- and
   E-cores, and an E-core is a net loss for compute-bound work even under the
   claim dispatch: a chunk it claims runs 2-3x slower and drags the join, for
   little compute in return. f32 GEMM measured 418 GFLOP/s on the 8 P-cores
   against 390 on all 10. macOS reports the top performance level as
   hw.perflevel0.physicalcpu; a homogeneous machine, or a platform without
   the query, counts every core. */
static int g_pcores;
static pthread_once_t pcores_once = PTHREAD_ONCE_INIT;

static void pcores_init(void) {
  int p = 0;
#if defined(__APPLE__)
  size_t sz = sizeof p;
  if (sysctlbyname("hw.perflevel0.physicalcpu", &p, &sz, NULL, 0) != 0 || p <= 0)
    p = 0;
#endif
  if (p < 1 || p > workers()) p = workers();
  g_pcores = p;
}

static int compute_workers(void) {
  pthread_once(&pcores_once, pcores_init);
  return g_pcores;
}

/* Chunk [idx] of [parts] over [0, total): balanced to one unit and never
   empty when parts <= total. */
static void chunk(int64_t total, int64_t parts, int64_t idx, int64_t *lo,
                  int64_t *hi) {
  *lo = idx * total / parts;
  *hi = (idx + 1) * total / parts;
}

/* Parking: a thread sets its bit in [parked] (or sets [waiting]) and reads
   the word it waits on again before it waits, and the thread that changes the
   word reads the flag after the change, both sequentially consistent: one of
   them sees the other, so no wakeup is lost. The signal is sent under the
   mutex the parked thread waits with.

   A job wakes the parked workers only if one of its participants is among
   them, so a parked worker that takes no part in a job sleeps through it. A
   worker that a broadcast wakes for a job it takes no part in parks again at
   once, since its window has ended. */

static uint64_t park_worker(pool *p, int id, uint64_t seen) {
  uint64_t bit = (uint64_t)1 << id;
  pthread_mutex_lock(&p->mtx);
  atomic_fetch_or(&p->parked, bit);
  uint64_t g;
  while ((g = atomic_load(&p->generation)) == seen)
    pthread_cond_wait(&p->wake, &p->mtx);
  atomic_fetch_and(&p->parked, ~bit);
  pthread_mutex_unlock(&p->mtx);
  return g;
}

/* The bits of [parked] of the workers of a job of [nthreads] threads: 1 to
   nthreads - 1, the caller being worker 0. */
static uint64_t participants(int nthreads) {
  uint64_t all = nthreads >= 64 ? ~(uint64_t)0 : ((uint64_t)1 << nthreads) - 1;
  return all & ~(uint64_t)1;
}

/* The generation of the next job after [seen], once it is written, spinning
   for it until [deadline] and parked after. */
static uint64_t next_job(pool *p, int id, uint64_t seen, uint64_t deadline) {
  uint64_t g = seen;
  for (;;) {
    uint64_t v = spin(&p->generation, g, 0, deadline);
    if (v == g) v = park_worker(p, id, g);
    if (v % 2 == 0 && v != seen) return v;
    g = v;
  }
}

static void *worker(void *arg) {
  const worker_arg *a = arg;
  pool *p = a->pool;
  int id = a->id; /* 1 .. nworkers-1 */
  /* Workers start before the first job, at generation 0: seeding [seen] with 0
     makes a worker that first runs after job 1 was published process it. */
  uint64_t seen = 0;
  /* The spin window runs from the worker's last part in a job, and a job it
     takes no part in leaves the window as it was: a worker that narrow jobs
     leave out parks once the window ends, for the rest of their burst. */
  uint64_t deadline = nx_device_now_ns() + spin_ns;
  for (;;) {
    uint64_t g = next_job(p, id, seen, deadline);
    int active;
    nx_device_range_body body;
    void *ctx;
    int64_t total, nchunks;
    for (;;) {
      active = atomic_load_explicit(&p->active, memory_order_relaxed);
      body = atomic_load_explicit(&p->body, memory_order_relaxed);
      ctx = atomic_load_explicit(&p->body_ctx, memory_order_relaxed);
      total = atomic_load_explicit(&p->total, memory_order_relaxed);
      nchunks = atomic_load_explicit(&p->nchunks, memory_order_relaxed);
      atomic_thread_fence(memory_order_acquire);
      if (atomic_load_explicit(&p->generation, memory_order_relaxed) == g) break;
      g = next_job(p, id, seen, deadline);
    }
    seen = g;
    /* A participant keeps the next job from being written until it counts
       down, so the fields it read are its job's. */
    if (id >= active) continue;

    /* A relaxed fetch-add makes every claimed index unique; the ordering the
       job needs rides the generation and the countdown. */
    for (;;) {
      int64_t c = atomic_fetch_add_explicit(&p->next, 1, memory_order_relaxed);
      if (c >= nchunks) break;
      int64_t lo, hi;
      chunk(total, nchunks, c, &lo, &hi);
      body(lo, hi, id, ctx);
    }

    if (atomic_fetch_sub(&p->pending, 1) == 1 && atomic_load(&p->waiting)) {
      pthread_mutex_lock(&p->mtx);
      pthread_cond_signal(&p->done);
      pthread_mutex_unlock(&p->mtx);
    }
    deadline = nx_device_now_ns() + spin_ns;
  }
  return NULL;
}

static pool *create(void) {
  pool *p = calloc(1, sizeof(*p));
  if (!p) return NULL;
  p->nworkers = workers();
  atomic_init(&p->generation, 0);
  atomic_init(&p->parked, 0);
  atomic_init(&p->pending, 0);
  atomic_init(&p->waiting, 0);
  atomic_init(&p->active, 0);
  atomic_init(&p->body, NULL);
  atomic_init(&p->body_ctx, NULL);
  atomic_init(&p->total, 0);
  atomic_init(&p->nchunks, 0);
  atomic_init(&p->next, 0);
  if (pthread_mutex_init(&p->drive, NULL) != 0) goto fail_pool;
  if (pthread_mutex_init(&p->mtx, NULL) != 0) goto fail_drive;
  if (pthread_cond_init(&p->wake, NULL) != 0) goto fail_mtx;
  if (pthread_cond_init(&p->done, NULL) != 0) goto fail_wake;
  for (int i = 1; i < p->nworkers; i++) {
    p->worker_args[i].pool = p;
    p->worker_args[i].id = i;
    if (pthread_create(&p->threads[i], NULL, worker, &p->worker_args[i]) != 0) {
      /* Run with the workers we have, which is correct, only slower. */
      p->nworkers = i;
      break;
    }
  }
  return p;

fail_wake:
  pthread_cond_destroy(&p->wake);
fail_mtx:
  pthread_mutex_destroy(&p->mtx);
fail_drive:
  pthread_mutex_destroy(&p->drive);
fail_pool:
  free(p);
  return NULL;
}

#if defined(_WIN32)
static void register_atfork(void) { atfork_ok = 1; }
#else
static void atfork_prepare(void) {
  pthread_mutex_lock(&init_mtx);
  pool *p = atomic_load_explicit(&g_pool, memory_order_acquire);
  if (p) {
    pthread_mutex_lock(&p->drive);
    pthread_mutex_lock(&p->mtx);
  }
}

static void atfork_parent(void) {
  pool *p = atomic_load_explicit(&g_pool, memory_order_acquire);
  if (p) {
    pthread_mutex_unlock(&p->mtx);
    pthread_mutex_unlock(&p->drive);
  }
  pthread_mutex_unlock(&init_mtx);
}

static void atfork_child(void) {
  atomic_store_explicit(&g_pool, NULL, memory_order_release);
  pthread_mutex_unlock(&init_mtx);
}

static void register_atfork(void) {
  atfork_ok =
      pthread_atfork(atfork_prepare, atfork_parent, atfork_child) == 0;
}
#endif

/* The pool, made at first use; NULL if a later fork could not be made safe,
   which leaves the caller computing alone. */
static pool *get(void) {
  pthread_once(&atfork_once, register_atfork);
  if (!atfork_ok) return NULL;
  pool *p = atomic_load_explicit(&g_pool, memory_order_acquire);
  if (p) return p;
  pthread_mutex_lock(&init_mtx);
  p = atomic_load_explicit(&g_pool, memory_order_relaxed);
  if (!p) {
    p = create();
    atomic_store_explicit(&g_pool, p, memory_order_release);
  }
  pthread_mutex_unlock(&init_mtx);
  return p;
}

/* [total] split into [nchunks] chunks, run by at most [nthreads] threads with
   the caller as worker 0, one region at a time. The runtime lock must be
   released when nthreads > 1. The counter is reset under both mutexes and
   claimed only by the current generation's threads; the caller returns once
   they all counted down, so no thread touches it until the next publish. */
static void run(int nthreads, int64_t total, int64_t nchunks,
                nx_device_range_body body, void *ctx) {
  if (total <= 0) return;
  if (nchunks > total) nchunks = total;
  if (nchunks < 1) nchunks = 1;
  pool *p = nthreads > 1 && nchunks > 1 ? get() : NULL;
  if (p && nthreads > p->nworkers) nthreads = p->nworkers;
  if (!p || nthreads <= 1) {
    body(0, total, 0, ctx);
    return;
  }

  pthread_mutex_lock(&p->drive);
  /* A sequence lock: the odd generation marks the job being written. */
  uint64_t g = atomic_load_explicit(&p->generation, memory_order_relaxed);
  atomic_store_explicit(&p->generation, g + 1, memory_order_relaxed);
  atomic_thread_fence(memory_order_release);
  atomic_store_explicit(&p->active, nthreads, memory_order_relaxed);
  atomic_store_explicit(&p->body, body, memory_order_relaxed);
  atomic_store_explicit(&p->body_ctx, ctx, memory_order_relaxed);
  atomic_store_explicit(&p->total, total, memory_order_relaxed);
  atomic_store_explicit(&p->nchunks, nchunks, memory_order_relaxed);
  atomic_store_explicit(&p->next, 0, memory_order_relaxed);
  atomic_store_explicit(&p->pending, (uint64_t)(nthreads - 1),
                        memory_order_relaxed);
  atomic_store(&p->generation, g + 2);
  if (atomic_load(&p->parked) & participants(nthreads)) {
    pthread_mutex_lock(&p->mtx);
    pthread_cond_broadcast(&p->wake);
    pthread_mutex_unlock(&p->mtx);
  }

  for (;;) {
    int64_t c = atomic_fetch_add_explicit(&p->next, 1, memory_order_relaxed);
    if (c >= nchunks) break;
    int64_t lo, hi;
    chunk(total, nchunks, c, &lo, &hi);
    body(lo, hi, 0, ctx);
  }

  if (atomic_load(&p->pending) != 0 &&
      spin(&p->pending, 0, 1, nx_device_now_ns() + spin_ns) != 0) {
    pthread_mutex_lock(&p->mtx);
    atomic_store(&p->waiting, 1);
    while (atomic_load(&p->pending) != 0) pthread_cond_wait(&p->done, &p->mtx);
    atomic_store(&p->waiting, 0);
    pthread_mutex_unlock(&p->mtx);
  }
  pthread_mutex_unlock(&p->drive);
}

static const nx_device_pool the_pool = {workers, compute_workers, run};

/* The pool, for nx.device's own C: nx_device_stubs.c runs split calls on
   it. */
const nx_device_pool *nx_device_pool_get(void);

const nx_device_pool *nx_device_pool_get(void) { return &the_pool; }

value caml_nx_device_pool(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&the_pool);
}
