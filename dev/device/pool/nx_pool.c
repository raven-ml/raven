/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The host's worker threads (nx_pool.h).

   One pool per process: nx_pool_cores () - 1 persistent workers, made at
   the first job of more than one thread. A job is published behind a
   generation counter; its threads claim chunk indices from a shared counter
   with one relaxed fetch-add each. A worker enters the job before it
   claims, while the caller has not closed it. The caller closes the job
   once its own claims find no chunk left and waits only for the workers
   inside, so a worker late to see the job, or kept off its core by the
   system, delays nothing.

   Jobs come in bursts, such as the kernels of a compiled program a few
   microseconds apart. A waiting thread spins for spin_ns before it parks on
   a condition variable, and a job wakes the parked workers only if one of
   its own threads is among them, so a job's publication costs no system
   call. */

#define _GNU_SOURCE

#include "nx_pool.h"
#include "nx_pool_cgroup.h"

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <signal.h>
#include <time.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <errno.h>
#endif

/* How long a waiting thread spins before it parks. Waking a parked thread
   takes a system call and tens of microseconds before it runs; a spin
   longer than the gaps within a burst keeps the workers running for the
   next job. Past busy_ns of spinning, a thread yields its core between
   reads, so that a spinning pool slows other threads little when the cores
   are all taken. */
static const uint64_t spin_ns = 100000;
static const uint64_t busy_ns = 2000;

/* A worker's stack, the size of a main thread's on Linux and macOS, so that
   a body's safe depth does not depend on which thread claims its chunk. */
static const size_t stack_bytes = 8 << 20;

/* A cache line on Apple silicon, and the pair of lines x86 fetches
   together. */
enum { line_bytes = 128 };

/* Clock */

static uint64_t now_ns(void) {
#if defined(_WIN32)
  LARGE_INTEGER count, frequency;
  QueryPerformanceCounter(&count);
  QueryPerformanceFrequency(&frequency);
  uint64_t c = (uint64_t)count.QuadPart, f = (uint64_t)frequency.QuadPart;
  return c / f * 1000000000u + c % f * 1000000000u / f;
#elif defined(__APPLE__)
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
#endif
}

/* Cores */

#if defined(__linux__)

/* The CPUs of the affinity mask. The kernel refuses a mask smaller than
   its own, which can exceed the configured CPUs, so the mask doubles until
   it fits, up to 2^20 CPUs, far past any kernel's limit. */
static long affinity_cpus(void) {
  long conf = sysconf(_SC_NPROCESSORS_CONF);
  for (long n = conf > 0 ? conf : 1; n <= (1 << 20); n *= 2) {
    cpu_set_t *set = CPU_ALLOC(n);
    if (set == NULL) break;
    size_t size = CPU_ALLOC_SIZE(n);
    int ok = sched_getaffinity(0, size, set) == 0;
    long count = ok ? CPU_COUNT_S(size, set) : 0;
    CPU_FREE(set);
    if (ok) return count;
    if (errno != EINVAL) break;
  }
  return sysconf(_SC_NPROCESSORS_ONLN);
}

#endif

/* The cgroup quota is plain reading of files, compiled everywhere so that
   the suite runs it on every system. */

/* The file [name] under [root], opened for reading, or NULL. */
static FILE *open_under(const char *root, const char *name) {
  char file[4096];
  if (snprintf(file, sizeof file, "%s%s", root, name) >= (int)sizeof file)
    return NULL;
  return fopen(file, "r");
}

/* The process's cgroup v2 directory under [root], written to [dir]: the
   cgroup2 mount that holds the path of proc/self/cgroup's "0::" line. A
   mount's root is the cgroup it shows, so the directory is the mount point
   joined with the path below that root. Returns the length of [root] and
   the mount point, where the walk to the ancestors stops, or 0 if there is
   none. */
static size_t cgroup_dir(const char *root, char *dir, size_t size) {
  char line[4096], path[4096] = "", mount[4096], point[4096];
  FILE *f = open_under(root, "/proc/self/cgroup");
  if (f == NULL) return 0;
  while (fgets(line, sizeof line, f))
    if (strncmp(line, "0::", 3) == 0 && sscanf(line + 3, "%4095s", path) == 1)
      break;
  fclose(f);
  if (path[0] != '/') return 0;

  size_t stop = 0;
  f = open_under(root, "/proc/self/mountinfo");
  if (f == NULL) return 0;
  while (stop == 0 && fgets(line, sizeof line, f)) {
    if (strstr(line, " - cgroup2 ") == NULL) continue;
    if (sscanf(line, "%*s %*s %*s %4095s %4095s", mount, point) != 2)
      continue;
    size_t n = strcmp(mount, "/") == 0 ? 0 : strlen(mount);
    if (strncmp(path, mount, n) != 0 || (path[n] != '/' && path[n] != '\0'))
      continue;
    if (snprintf(dir, size, "%s%s%s", root, point, path + n) < (int)size)
      stop = strlen(root) + strlen(point);
  }
  fclose(f);
  return stop;
}

/* The smallest of ceil (quota / period) over the cgroup and its ancestors,
   since limits nest. Rounded up: n threads spend at most n CPUs a period,
   so rounding a quota of 1.5 down would leave a third of it idle. A file
   reads "max PERIOD" without a quota and "QUOTA PERIOD" with one. */
long nx_pool_cgroup_cpus(const char *root) {
  char dir[4096], file[4200];
  size_t stop = cgroup_dir(root, dir, sizeof dir);
  if (stop == 0) return -1;
  long least = -1;
  for (;;) {
    snprintf(file, sizeof file, "%s/cpu.max", dir);
    FILE *f = fopen(file, "r");
    long quota, period;
    if (f != NULL) {
      if (fscanf(f, "%ld %ld", &quota, &period) == 2 && quota > 0 &&
          period > 0) {
        long cpus = (quota + period - 1) / period;
        if (least < 0 || cpus < least) least = cpus;
      }
      fclose(f);
    }
    char *slash = strrchr(dir, '/');
    if (strlen(dir) <= stop || slash == NULL) return least;
    *slash = '\0';
  }
}

static int count_cores(void) {
  long n;
#if defined(__APPLE__)
  int c = 0;
  size_t size = sizeof c;
  n = sysctlbyname("hw.physicalcpu", &c, &size, NULL, 0) == 0 ? c : 1;
#elif defined(_WIN32)
  n = (long)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
#elif defined(__linux__)
  n = affinity_cpus();
  long quota = nx_pool_cgroup_cpus("");
  if (quota > 0 && quota < n) n = quota;
#else
  n = sysconf(_SC_NPROCESSORS_ONLN);
#endif
  return n < 1 ? 1 : (int)n;
}

static int count_performance_cores(int cores) {
#if defined(__APPLE__)
  int p = 0;
  size_t size = sizeof p;
  if (sysctlbyname("hw.perflevel0.physicalcpu", &p, &size, NULL, 0) == 0 &&
      p >= 1 && p <= cores)
    return p;
#endif
  return cores;
}

static int g_cores, g_performance_cores;
static pthread_once_t cores_once = PTHREAD_ONCE_INIT;

static void cores_init(void) {
  g_cores = count_cores();
  g_performance_cores = count_performance_cores(g_cores);
}

int nx_pool_cores(void) {
  pthread_once(&cores_once, cores_init);
  return g_cores;
}

int nx_pool_performance_cores(void) {
  pthread_once(&cores_once, cores_init);
  return g_performance_cores;
}

/* Jobs */

/* A job as its caller passes it. [wide] says i * total may overflow 64 bits
   for some chunk bound i <= chunks, so the bounds take 128. */
typedef struct {
  nx_pool_body body;
  void *ctx;
  int64_t total, chunks;
  int threads, wide;
} job;

/* floor (i * total / chunks), exactly. */
static inline int64_t bound(const job *j, int64_t i) {
  if (j->wide) return (int64_t)((__int128)i * j->total / j->chunks);
  return i * j->total / j->chunks;
}

typedef struct pool pool;

typedef struct {
  pool *pool;
  int id;
} worker;

struct pool {
  int threads;           /* the workers made, plus the caller */
  worker *workers;       /* [1, threads); slot 0 is the caller */
  pthread_mutex_t drive; /* one published job at a time */
  pthread_mutex_t mtx;   /* guards parking */
  pthread_cond_t wake;   /* parked workers wait here for a new generation */
  pthread_cond_t done;   /* a parked caller waits here for the job's end */
  /* The job's number in the high 32 bits and its threads in the low 32,
     so that one load tells a worker that a job began and whether it is one
     of its threads. */
  _Atomic uint64_t generation;
  /* The job, written while no worker is inside and read only inside. */
  job job;
  /* Each gap keeps what follows it at least a line away from what precedes
     it, whatever the pool's address. The claims' counter stays off the
     line of the job and the generation, which every claim would otherwise
     take from the threads that spin on it. The workers' entries stay off
     the claims' line, which they would otherwise take from the caller's
     claims as a job opens: an empty job on 16 x86 threads took 1.6 times
     as long. A job of two threads pays for it on Apple silicon, where it
     moves two lines between caller and worker instead of one: 100 ns
     became 130. */
  char gap[line_bytes];
  _Atomic int64_t next; /* next unclaimed chunk index */
  char gap2[line_bytes];
  _Atomic uint64_t inside; /* the workers inside the job, | closed */
  _Atomic int waiting;     /* whether the caller parked on [done] */
  /* Bit [id % 64] of word [id / 64] is set while worker [id] is parked. */
  _Atomic uint64_t parked[];
};

/* Whether the thread runs a body of a job of more than one thread: a job
   it begins then runs alone on it, since the pool runs one such job at a
   time and the outer job holds it. A job on one thread holds nothing. */
static _Thread_local int in_body;

static void relax(void) {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield");
#endif
}

/* What a spin waits for: [*word] to leave [value], or to reach it. */
typedef enum { LEAVE, REACH } spin_until;

/* Spins until [*word] leaves or reaches [value], up to the clock's
   [deadline], and returns the last value read. The clock is read every 64
   loads. */
static uint64_t spin(_Atomic uint64_t *word, uint64_t value, spin_until until,
                     uint64_t deadline) {
  int reach = until == REACH;
  uint64_t v = atomic_load_explicit(word, memory_order_acquire);
  if ((v == value) == reach) return v;
  uint64_t start = now_ns();
  for (;;) {
    for (int i = 0; i < 64; i++) {
      relax();
      v = atomic_load_explicit(word, memory_order_acquire);
      if ((v == value) == reach) return v;
    }
    uint64_t now = now_ns();
    if (now >= deadline) return v;
    if (now - start >= busy_ns) sched_yield();
  }
}

/* Claims chunks of [j] until none remains, one call a chunk, which
   nx_pool.h does not promise: the thread that frees first takes the next
   chunk, so a costly one holds only its own thread. A relaxed fetch-add
   makes every index unique; ordering rides the opening and closing of the
   job. */
static void claim(pool *p, const job *j, int id) {
  for (;;) {
    int64_t i = atomic_fetch_add_explicit(&p->next, 1, memory_order_relaxed);
    if (i >= j->chunks) return;
    j->body(bound(j, i), bound(j, i + 1), id, j->ctx);
  }
}

/* Entering. A worker claims only inside, so the caller never rewrites a job
   a worker reads, and a worker that arrives after the close turns back
   without a claim.

   A worker that sees the job closed turns back after one load. Else it
   enters with a fetch-add, which always lands: a compare-and-swap fails and
   retries while other workers enter, and with 15 at once on x86 an empty
   job took 1.4 times as long. An add that finds the job closed after all is
   undone at once, so the caller, which waits for the count to fall to zero,
   waits for that undo too. Such an add can even land after the caller's
   wait has ended; the next job therefore opens by clearing [closed] alone,
   and counts the add until its undo. */
static const uint64_t closed = (uint64_t)1 << 63;

static void leave(pool *p) {
  if (atomic_fetch_sub(&p->inside, 1) == (closed | 1) &&
      atomic_load(&p->waiting)) {
    pthread_mutex_lock(&p->mtx);
    pthread_cond_signal(&p->done);
    pthread_mutex_unlock(&p->mtx);
  }
}

static int enter(pool *p) {
  if (atomic_load_explicit(&p->inside, memory_order_relaxed) & closed)
    return 0;
  if (!(atomic_fetch_add_explicit(&p->inside, 1, memory_order_acquire) &
        closed))
    return 1;
  leave(p);
  return 0;
}

/* Parking: a thread sets its bit in [parked] (or sets [waiting]) and reads
   the word it waits on again before it waits, and the thread that changes the
   word reads the flag after the change, both sequentially consistent: one of
   them sees the other, so no wakeup is lost. The signal is sent under the
   mutex the parked thread waits with. A worker that a broadcast wakes for a
   job it takes no part in parks again at once, since its window has ended. */

static uint64_t park_worker(pool *p, int id, uint64_t seen) {
  _Atomic uint64_t *word = &p->parked[id / 64];
  uint64_t bit = (uint64_t)1 << (id % 64);
  pthread_mutex_lock(&p->mtx);
  atomic_fetch_or(word, bit);
  uint64_t g;
  while ((g = atomic_load(&p->generation)) == seen)
    pthread_cond_wait(&p->wake, &p->mtx);
  atomic_fetch_and(word, ~bit);
  pthread_mutex_unlock(&p->mtx);
  return g;
}

/* Whether one of a job's workers, 1 to [threads] - 1, is parked: one load
   for jobs of up to 64 threads. */
static int participant_parked(pool *p, int threads) {
  for (int w = 0; w * 64 < threads; w++) {
    int below = threads - w * 64;
    uint64_t mask = below >= 64 ? ~(uint64_t)0 : ((uint64_t)1 << below) - 1;
    if (w == 0) mask &= ~(uint64_t)1;
    if (atomic_load(&p->parked[w]) & mask) return 1;
  }
  return 0;
}

/* The generation of the next job after [g] that worker [id] is one of the
   threads of: spun for until [deadline], parked for after. A job without
   the worker is passed over, and after [deadline] slept through: jobs may
   follow each other faster than a spin reads the clock. */
static uint64_t next_job(pool *p, int id, uint64_t g, uint64_t deadline) {
  for (;;) {
    uint64_t v = spin(&p->generation, g, LEAVE, deadline);
    if (v == g || (id >= (int)(uint32_t)v && now_ns() >= deadline))
      v = park_worker(p, id, v);
    if (id < (int)(uint32_t)v) return v;
    g = v;
  }
}

static void *work(void *arg) {
  const worker *w = arg;
  pool *p = w->pool;
  in_body = 1;
  /* Workers start before the first job, at generation 0. */
  uint64_t seen = 0;
  /* The spin window runs from the worker's last part in a job, and a job it
     takes no part in leaves the window as it was: a worker that narrow jobs
     leave out parks once the window ends, for the rest of their burst. */
  uint64_t deadline = now_ns() + spin_ns;
  for (;;) {
    seen = next_job(p, w->id, seen, deadline);
    /* The job open by now may be a later one than [seen], whose threads
       the worker learns inside. */
    if (enter(p)) {
      if (w->id < p->job.threads) claim(p, &p->job, w->id);
      leave(p);
    }
    deadline = now_ns() + spin_ns;
  }
  return NULL;
}

/* Starts the workers of [p], with [stack_bytes] of stack and the signals
   sent to the process blocked, keeping those a body raises itself. A worker
   that cannot be made so ends the list: the job runs with those made. The
   workers are never joined. */
static void start_workers(pool *p, int wanted) {
  p->threads = 1;
  pthread_attr_t attr;
  if (pthread_attr_init(&attr) != 0) return;
  int sized = pthread_attr_setstacksize(&attr, stack_bytes) == 0;
#if !defined(_WIN32)
  sigset_t blocked, saved;
  sigfillset(&blocked);
  static const int faults[] = {SIGSEGV, SIGBUS,  SIGFPE, SIGILL,
                               SIGTRAP, SIGABRT, SIGSYS};
  for (size_t i = 0; i < sizeof faults / sizeof *faults; i++)
    sigdelset(&blocked, faults[i]);
  pthread_sigmask(SIG_SETMASK, &blocked, &saved);
#endif
  for (int id = 1; sized && id < wanted; id++) {
    worker *w = &p->workers[id];
    w->pool = p;
    w->id = id;
    pthread_t thread;
    if (pthread_create(&thread, &attr, work, w) != 0) break;
    p->threads = id + 1;
  }
#if !defined(_WIN32)
  pthread_sigmask(SIG_SETMASK, &saved, NULL);
#endif
  pthread_attr_destroy(&attr);
}

/* A new pool, or NULL if its memory or locks cannot be had. calloc's zeros
   are each atomic field's initial value. */
static pool *create(void) {
  int cores = nx_pool_cores();
  size_t words = ((size_t)cores + 63) / 64;
  pool *p = calloc(1, sizeof *p + words * sizeof(_Atomic uint64_t));
  if (p == NULL) return NULL;
  p->workers = calloc((size_t)cores, sizeof *p->workers);
  if (p->workers == NULL) goto fail_pool;
  if (pthread_mutex_init(&p->drive, NULL) != 0) goto fail_workers;
  if (pthread_mutex_init(&p->mtx, NULL) != 0) goto fail_drive;
  if (pthread_cond_init(&p->wake, NULL) != 0) goto fail_mtx;
  if (pthread_cond_init(&p->done, NULL) != 0) goto fail_wake;
  start_workers(p, cores);
  return p;

fail_wake:
  pthread_cond_destroy(&p->wake);
fail_mtx:
  pthread_mutex_destroy(&p->mtx);
fail_drive:
  pthread_mutex_destroy(&p->drive);
fail_workers:
  free(p->workers);
fail_pool:
  free(p);
  return NULL;
}

/* Fork

   The pool is heap-owned so that a fork child can abandon the inherited one
   and lazily build its own: only the forking thread survives in the child, so
   the copied workers, mutexes and condition waiters are unusable, and
   reinitialising live pthread objects in place is undefined.

   init_mtx serialises lazy creation. The prepare handler holds it and the
   current pool's drive mutex, so fork waits for a running job of more than
   one thread; the parent releases both, and the child clears the pointer
   and releases only the still-valid init_mtx. */
static _Atomic(pool *) g_pool;
static pthread_mutex_t init_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t atfork_once = PTHREAD_ONCE_INIT;
static int atfork_ok;

#if defined(_WIN32)
static void register_atfork(void) { atfork_ok = 1; }
#else
static void atfork_prepare(void) {
  pthread_mutex_lock(&init_mtx);
  pool *p = atomic_load_explicit(&g_pool, memory_order_acquire);
  if (p) pthread_mutex_lock(&p->drive);
}

static void atfork_parent(void) {
  pool *p = atomic_load_explicit(&g_pool, memory_order_acquire);
  if (p) pthread_mutex_unlock(&p->drive);
  pthread_mutex_unlock(&init_mtx);
}

static void atfork_child(void) {
  atomic_store_explicit(&g_pool, NULL, memory_order_release);
  pthread_mutex_unlock(&init_mtx);
}

static void register_atfork(void) {
  atfork_ok = pthread_atfork(atfork_prepare, atfork_parent, atfork_child) == 0;
}
#endif

/* The pool, made at first use; NULL if it cannot be made, or if a later
   fork could not be made safe, which leaves the caller computing alone. It
   exists only once the fork handlers do, so finding it skips the once. */
static pool *get(void) {
  pool *p = atomic_load_explicit(&g_pool, memory_order_acquire);
  if (p) return p;
  pthread_once(&atfork_once, register_atfork);
  if (!atfork_ok) return NULL;
  pthread_mutex_lock(&init_mtx);
  p = atomic_load_explicit(&g_pool, memory_order_relaxed);
  if (!p) {
    p = create();
    atomic_store_explicit(&g_pool, p, memory_order_release);
  }
  pthread_mutex_unlock(&init_mtx);
  return p;
}

static int64_t clamp(int64_t x, int64_t lo, int64_t hi) {
  return x < lo ? lo : x > hi ? hi : x;
}

/* Runs a job of [t] > 1 threads and [c] chunks: alone if begun from a body
   or without a pool, else on at most p->threads (<= nx_pool_cores ()). Once
   it returns, no thread touches the job or the claim counter until the next
   job opens. Out of line, so a serial nx_pool_run saves no registers. */
__attribute__((noinline)) static void run(int t, int64_t total, int64_t c,
                                          nx_pool_body body, void *ctx) {
  pool *p = in_body ? NULL : get();
  if (p && t > p->threads) t = p->threads;
  if (p == NULL || t == 1) {
    body(0, total, 0, ctx);
    return;
  }
  job j = {body, ctx, total, c, t, total > INT64_MAX / c};

  pthread_mutex_lock(&p->drive);
  p->job = j;
  atomic_store_explicit(&p->next, 0, memory_order_relaxed);
  atomic_fetch_and_explicit(&p->inside, ~closed, memory_order_release);
  uint64_t g = atomic_load_explicit(&p->generation, memory_order_relaxed);
  atomic_store(&p->generation, ((g >> 32) + 1) << 32 | (uint32_t)t);
  if (participant_parked(p, t)) {
    pthread_mutex_lock(&p->mtx);
    pthread_cond_broadcast(&p->wake);
    pthread_mutex_unlock(&p->mtx);
  }

  in_body = 1;
  claim(p, &j, 0);
  in_body = 0;

  if (atomic_fetch_or(&p->inside, closed) != 0 &&
      spin(&p->inside, closed, REACH, now_ns() + spin_ns) != closed) {
    pthread_mutex_lock(&p->mtx);
    atomic_store(&p->waiting, 1);
    while (atomic_load(&p->inside) != closed)
      pthread_cond_wait(&p->done, &p->mtx);
    atomic_store(&p->waiting, 0);
    pthread_mutex_unlock(&p->mtx);
  }
  pthread_mutex_unlock(&p->drive);
}

/* A serial job is one call, before any thread-local read or division. */
void nx_pool_run(int threads, int64_t total, int64_t chunks, nx_pool_body body,
                 void *ctx) {
  if (total <= 0) return;
  int64_t c = clamp(chunks, 1, total);
  int t = (int)clamp(threads, 1, c);
  if (t == 1)
    body(0, total, 0, ctx);
  else
    run(t, total, c, body, ctx);
}
