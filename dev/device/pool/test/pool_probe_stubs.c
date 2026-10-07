/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Probes of nx_pool.h for the pool's suite: jobs whose bodies record what the
   pool did with them, called from OCaml as a consumer's stubs call the pool.

   Some bodies wait for another chunk, which nx_pool.h forbids, to force a
   chunk onto a worker or to hold a job open. Each such wait gives up after
   [patience], so a pool that breaks a promise fails the test instead of
   hanging it. */

#define _GNU_SOURCE

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <dirent.h>
#include <sched.h>
#include <sys/syscall.h>
#endif

#include "nx_pool.h"

/* Time */

/* Far longer than any wakeup: only a pool that never runs the awaited chunk,
   or never ends the awaited job, reaches it. */
static const int64_t patience = INT64_C(10000000000);

static int64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

static void spin_ns(int64_t ns) {
  int64_t end = now_ns() + ns;
  while (now_ns() < end) {
  }
}

static void nothing(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)lo;
  (void)hi;
  (void)worker;
  (void)ctx;
}

/* Recorded jobs */

typedef struct {
  int64_t lo, hi;
  int worker;
  pthread_t thread;
} call;

typedef struct {
  call *calls; /* in the order the calls began */
  int64_t cap;
  _Atomic int64_t n;
  _Atomic int *busy; /* per worker index: a call is running */
  int slots;
  _Atomic int overlaps;
} record_job;

static void record(int64_t lo, int64_t hi, int worker, void *ctx) {
  record_job *j = ctx;
  int tracked = worker >= 0 && worker < j->slots;
  if (tracked && atomic_exchange(&j->busy[worker], 1))
    atomic_fetch_add(&j->overlaps, 1);
  int64_t k = atomic_fetch_add(&j->n, 1);
  if (k < j->cap)
    j->calls[k] = (call){lo, hi, worker, pthread_self()};
  /* Long enough that several threads take part and an overlap shows. */
  spin_ns(1000);
  if (tracked)
    atomic_store(&j->busy[worker], 0);
}

static const int64_t max_recorded = INT64_C(1) << 20;

/* [probe_record threads total chunks] is (calls, count, overlaps): the calls
   as (lo, hi, worker, thread) in the order they began, [thread] numbering the
   threads in order of their first call, the caller's 0; the number of calls;
   the calls that began while another of their worker ran. */
value probe_record(value v_threads, value v_total, value v_chunks) {
  CAMLparam3(v_threads, v_total, v_chunks);
  CAMLlocal4(result, calls, entry, bound);
  int threads = Int_val(v_threads);
  int64_t total = Int64_val(v_total), chunks = Int64_val(v_chunks);
  int64_t cap = 0;
  if (total > 0) {
    cap = chunks < 1 ? 1 : chunks;
    if (cap > total)
      cap = total;
  }
  if (cap > max_recorded)
    caml_invalid_argument("probe_record: too many chunks");
  int slots = nx_pool_cores();
  record_job j = {calloc((size_t)cap + 1, sizeof(call)),      cap,   0,
                  calloc((size_t)slots, sizeof(_Atomic int)), slots, 0};
  pthread_t *seen = malloc(((size_t)cap + 1) * sizeof(pthread_t));
  if (j.calls == NULL || j.busy == NULL || seen == NULL) {
    free(j.calls);
    free((void *)j.busy);
    free(seen);
    caml_raise_out_of_memory();
  }
  seen[0] = pthread_self();
  caml_enter_blocking_section();
  nx_pool_run(threads, total, chunks, record, &j);
  caml_leave_blocking_section();
  int64_t n = atomic_load(&j.n), kept = n < cap ? n : cap;
  int nseen = 1;
  calls = caml_alloc((mlsize_t)kept, 0);
  for (int64_t k = 0; k < kept; k++) {
    call c = j.calls[k];
    int id = 0;
    while (id < nseen && !pthread_equal(seen[id], c.thread))
      id++;
    if (id == nseen)
      seen[nseen++] = c.thread;
    entry = caml_alloc_tuple(4);
    bound = caml_copy_int64(c.lo);
    Store_field(entry, 0, bound);
    bound = caml_copy_int64(c.hi);
    Store_field(entry, 1, bound);
    Store_field(entry, 2, Val_int(c.worker));
    Store_field(entry, 3, Val_int(id));
    Store_field(calls, k, entry);
  }
  free(j.calls);
  free((void *)j.busy);
  free(seen);
  result = caml_alloc_tuple(3);
  Store_field(result, 0, calls);
  Store_field(result, 1, Val_long(n));
  Store_field(result, 2, Val_int(atomic_load(&j.overlaps)));
  CAMLreturn(result);
}

/* Visibility */

#define COPY_UNITS 4096
#define COPY_CHUNKS 64

typedef struct {
  const int64_t *in;
  int64_t *out;
  int64_t job;
  _Atomic int64_t *misses;
} copy_job;

/* Reads the caller's [in] and writes [out], both plain memory. */
static void copy(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  copy_job *j = ctx;
  int64_t misses = 0;
  for (int64_t i = lo; i < hi; i++) {
    if (j->in[i] != j->job + i)
      misses++;
    j->out[i] = j->job + i + 1;
  }
  if (misses > 0)
    atomic_fetch_add(j->misses, misses);
}

/* [probe_visibility jobs threads] runs [jobs] jobs, each over values the
   caller writes just before it, and is (the values bodies read stale, the
   values the caller read stale after a job). */
value probe_visibility(value v_jobs, value v_threads) {
  CAMLparam2(v_jobs, v_threads);
  CAMLlocal1(result);
  int64_t jobs = Long_val(v_jobs);
  int threads = Int_val(v_threads);
  int64_t *in = malloc(COPY_UNITS * sizeof(int64_t));
  int64_t *out = malloc(COPY_UNITS * sizeof(int64_t));
  if (in == NULL || out == NULL) {
    free(in);
    free(out);
    caml_raise_out_of_memory();
  }
  _Atomic int64_t body_misses = 0;
  int64_t caller_misses = 0;
  caml_enter_blocking_section();
  for (int64_t k = 0; k < jobs; k++) {
    for (int64_t i = 0; i < COPY_UNITS; i++)
      in[i] = k + i;
    copy_job j = {in, out, k, &body_misses};
    nx_pool_run(threads, COPY_UNITS, COPY_CHUNKS, copy, &j);
    for (int64_t i = 0; i < COPY_UNITS; i++)
      if (out[i] != k + i + 1)
        caller_misses++;
  }
  caml_leave_blocking_section();
  free(in);
  free(out);
  result = caml_alloc_tuple(2);
  Store_field(result, 0, Val_long(atomic_load(&body_misses)));
  Store_field(result, 1, Val_long(caller_misses));
  CAMLreturn(result);
}

/* Nested jobs */

#define MAX_INNER 64

typedef struct {
  pthread_t thread; /* the outer body's */
  _Atomic int *units;
  _Atomic int64_t *calls, *misplaced;
} inner_job;

static void inner(int64_t lo, int64_t hi, int worker, void *ctx) {
  inner_job *j = ctx;
  atomic_fetch_add(j->calls, 1);
  if (worker != 0 || !pthread_equal(pthread_self(), j->thread))
    atomic_fetch_add(j->misplaced, 1);
  for (int64_t i = lo; i < hi; i++)
    atomic_fetch_add(&j->units[i], 1);
}

typedef struct {
  int threads;
  int64_t inner_units;
  _Atomic int64_t outer_calls, inner_calls, misplaced, unit_errors;
} outer_job;

/* Begins a job of one unit per chunk, and counts its units not run once. */
static void outer(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)lo;
  (void)hi;
  (void)worker;
  outer_job *j = ctx;
  atomic_fetch_add(&j->outer_calls, 1);
  _Atomic int units[MAX_INNER] = {0};
  inner_job ij = {pthread_self(), units, &j->inner_calls, &j->misplaced};
  nx_pool_run(j->threads, j->inner_units, j->inner_units, inner, &ij);
  for (int64_t i = 0; i < j->inner_units; i++)
    if (atomic_load(&units[i]) != 1)
      atomic_fetch_add(&j->unit_errors, 1);
}

/* [probe_nested threads outer inner] runs a job of [outer] chunks of one unit
   on [threads] threads, whose every body begins a job of [inner] chunks of
   one unit on [threads] threads. It is (outer calls, inner calls, inner calls
   off their outer body's thread or not worker 0, inner units not run once). */
value probe_nested(value v_threads, value v_outer, value v_inner) {
  CAMLparam3(v_threads, v_outer, v_inner);
  CAMLlocal1(result);
  int64_t inner_units = Long_val(v_inner);
  if (inner_units < 0 || inner_units > MAX_INNER)
    caml_invalid_argument("probe_nested: inner");
  outer_job j = {Int_val(v_threads), inner_units, 0, 0, 0, 0};
  int64_t outer_units = Long_val(v_outer);
  caml_enter_blocking_section();
  nx_pool_run(j.threads, outer_units, outer_units, outer, &j);
  caml_leave_blocking_section();
  result = caml_alloc_tuple(4);
  Store_field(result, 0, Val_long(atomic_load(&j.outer_calls)));
  Store_field(result, 1, Val_long(atomic_load(&j.inner_calls)));
  Store_field(result, 2, Val_long(atomic_load(&j.misplaced)));
  Store_field(result, 3, Val_long(atomic_load(&j.unit_errors)));
  CAMLreturn(result);
}

/* Balance */

typedef struct {
  int64_t chunks;
  _Atomic int64_t done;
  int balanced;
} balance_job;

/* Chunk 0 runs until every other chunk has run: only a thread other than its
   own can run them. */
static void balance(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)hi;
  (void)worker;
  balance_job *j = ctx;
  if (lo != 0) {
    atomic_fetch_add(&j->done, 1);
    return;
  }
  int64_t deadline = now_ns() + patience;
  while (atomic_load(&j->done) < j->chunks - 1 && now_ns() < deadline) {
  }
  j->balanced = atomic_load(&j->done) == j->chunks - 1;
}

/* [probe_balance chunks] runs a job of [chunks] chunks of one unit on two
   threads, whose chunk 0 lasts until the others have run, and is whether
   they ran while it lasted. */
value probe_balance(value v_chunks) {
  balance_job j = {Long_val(v_chunks), 0, 0};
  caml_enter_blocking_section();
  nx_pool_run(2, j.chunks, j.chunks, balance, &j);
  caml_leave_blocking_section();
  return Val_bool(j.balanced);
}

/* Held, counted and burst jobs, observed from another domain */

static _Atomic int hold_arrived, hold_released, burst_stop;
static _Atomic int64_t counted_calls;

/* Every chunk waits for [probe_hold_release]; with [only_worker], the two
   chunks first meet, then the worker's alone waits. */
static void hold(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)lo;
  (void)hi;
  int only_worker = *(int *)ctx;
  atomic_fetch_add(&hold_arrived, 1);
  int64_t deadline = now_ns() + patience;
  if (only_worker) {
    while (atomic_load(&hold_arrived) < 2 && now_ns() < deadline) {
    }
    if (worker == 0)
      return;
  }
  while (!atomic_load(&hold_released) && now_ns() < deadline) {
  }
}

value probe_reset(value unit) {
  (void)unit;
  atomic_store(&hold_arrived, 0);
  atomic_store(&hold_released, 0);
  atomic_store(&burst_stop, 0);
  atomic_store(&counted_calls, 0);
  return Val_unit;
}

/* [probe_hold only_worker] runs a job of two chunks on two threads that
   [hold]s, and is whether it was released before patience ran out. */
value probe_hold(value v_only_worker) {
  int only_worker = Bool_val(v_only_worker);
  caml_enter_blocking_section();
  nx_pool_run(2, 2, 2, hold, &only_worker);
  caml_leave_blocking_section();
  return Val_bool(atomic_load(&hold_released));
}

value probe_hold_arrived(value unit) {
  (void)unit;
  return Val_int(atomic_load(&hold_arrived));
}

value probe_hold_release(value unit) {
  (void)unit;
  atomic_store(&hold_released, 1);
  return Val_unit;
}

static void counted(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)lo;
  (void)hi;
  (void)worker;
  (void)ctx;
  atomic_fetch_add(&counted_calls, 1);
}

value probe_counted(value v_threads, value v_total, value v_chunks) {
  int threads = Int_val(v_threads);
  int64_t total = Int64_val(v_total), chunks = Int64_val(v_chunks);
  caml_enter_blocking_section();
  nx_pool_run(threads, total, chunks, counted, NULL);
  caml_leave_blocking_section();
  return Val_unit;
}

value probe_counted_calls(value unit) {
  (void)unit;
  return Val_long(atomic_load(&counted_calls));
}

/* [probe_burst ()] runs a job on every core, then jobs of two chunks on two
   threads back to back until [probe_burst_stop]. */
value probe_burst(value unit) {
  (void)unit;
  int cores = nx_pool_cores();
  caml_enter_blocking_section();
  nx_pool_run(cores, cores, cores, nothing, NULL);
  while (!atomic_load(&burst_stop))
    nx_pool_run(2, 2, 2, nothing, NULL);
  caml_leave_blocking_section();
  return Val_unit;
}

value probe_burst_stop(value unit) {
  (void)unit;
  atomic_store(&burst_stop, 1);
  return Val_unit;
}

/* Bodies on a worker */

typedef struct {
  _Atomic int arrived;
  int worked;
  void (*on_worker)(void);
} meet_job;

/* The two chunks wait for each other, so a worker runs one of them; that
   call runs [on_worker]. */
static void meet(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)lo;
  (void)hi;
  meet_job *j = ctx;
  atomic_fetch_add(&j->arrived, 1);
  int64_t deadline = now_ns() + patience;
  while (atomic_load(&j->arrived) < 2 && now_ns() < deadline) {
  }
  if (worker == 0)
    return;
  j->worked = 1;
  if (j->on_worker != NULL)
    j->on_worker();
}

/* Whether a worker ran a chunk, and so [on_worker]. */
static int on_a_worker(void (*on_worker)(void)) {
  meet_job j = {0, 0, on_worker};
  nx_pool_run(2, 2, 2, meet, &j);
  return j.worked;
}

static const struct {
  const char *name;
  int number;
} signals[] = {
    {"SIGHUP", SIGHUP},     {"SIGINT", SIGINT},       {"SIGQUIT", SIGQUIT},
    {"SIGILL", SIGILL},     {"SIGTRAP", SIGTRAP},     {"SIGABRT", SIGABRT},
    {"SIGBUS", SIGBUS},     {"SIGFPE", SIGFPE},       {"SIGUSR1", SIGUSR1},
    {"SIGSEGV", SIGSEGV},   {"SIGUSR2", SIGUSR2},     {"SIGPIPE", SIGPIPE},
    {"SIGALRM", SIGALRM},   {"SIGTERM", SIGTERM},     {"SIGCHLD", SIGCHLD},
    {"SIGCONT", SIGCONT},   {"SIGTSTP", SIGTSTP},     {"SIGTTIN", SIGTTIN},
    {"SIGTTOU", SIGTTOU},   {"SIGURG", SIGURG},       {"SIGXCPU", SIGXCPU},
    {"SIGXFSZ", SIGXFSZ},   {"SIGVTALRM", SIGVTALRM}, {"SIGPROF", SIGPROF},
    {"SIGWINCH", SIGWINCH}, {"SIGSYS", SIGSYS},
};

#define SIGNALS (sizeof signals / sizeof signals[0])

static sigset_t worker_mask;

static void read_mask(void) { pthread_sigmask(SIG_BLOCK, NULL, &worker_mask); }

/* [probe_worker_mask ()] is (whether a worker ran, the signals of the table
   with whether a worker blocks them). */
value probe_worker_mask(value unit) {
  CAMLparam1(unit);
  CAMLlocal5(result, list, pair, name, cell);
  caml_enter_blocking_section();
  int worked = on_a_worker(read_mask);
  caml_leave_blocking_section();
  list = Val_emptylist;
  for (size_t k = SIGNALS; k > 0; k--) {
    name = caml_copy_string(signals[k - 1].name);
    pair = caml_alloc_tuple(2);
    Store_field(pair, 0, name);
    Store_field(
        pair, 1,
        Val_bool(worked && sigismember(&worker_mask, signals[k - 1].number)));
    cell = caml_alloc_small(2, Tag_cons);
    Field(cell, 0) = pair;
    Field(cell, 1) = list;
    list = cell;
  }
  result = caml_alloc_tuple(2);
  Store_field(result, 0, Val_bool(worked));
  Store_field(result, 1, list);
  CAMLreturn(result);
}

/* The signals a body may raise itself, in the order of [probe_faults]. */
static const int faults[] = {SIGSEGV, SIGBUS,  SIGFPE, SIGILL,
                             SIGTRAP, SIGABRT, SIGSYS};

#define FAULTS (sizeof faults / sizeof faults[0])

static volatile sig_atomic_t caught;
static int64_t delivered;

static void on_fault(int sig) {
  (void)sig;
  caught = 1;
}

/* raise sends a signal to the calling thread, and returns after its handler
   when the thread does not block it. */
static void raise_faults(void) {
  for (size_t k = 0; k < FAULTS; k++) {
    struct sigaction sa, old;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_fault;
    sigemptyset(&sa.sa_mask);
    if (sigaction(faults[k], &sa, &old) != 0)
      continue;
    caught = 0;
    raise(faults[k]);
    if (caught)
      delivered |= INT64_C(1) << k;
    sigaction(faults[k], &old, NULL);
  }
}

/* 7 MiB of the 8 MiB nx_pool.h promises, touched from the top down so an
   overflow meets the guard page first. */
#define DEEP_BYTES (7 << 20)

static __attribute__((noinline)) void deep_stack(void) {
  volatile char frame[DEEP_BYTES];
  for (size_t i = DEEP_BYTES; i > 0; i -= 4096)
    frame[i - 1] = (char)i;
}

/* Children made by fork */

#define CHILD_VALUES 4

#if defined(__APPLE__)
static int64_t thread_count(void) {
  task_t task = mach_task_self();
  thread_act_array_t threads;
  mach_msg_type_number_t n;
  if (task_threads(task, &threads, &n) != KERN_SUCCESS)
    return -1;
  for (mach_msg_type_number_t i = 0; i < n; i++)
    mach_port_deallocate(task, threads[i]);
  vm_deallocate(task, (vm_address_t)threads, n * sizeof *threads);
  return n;
}
#elif defined(__linux__)
static int64_t thread_count(void) {
  DIR *dir = opendir("/proc/self/task");
  if (dir == NULL)
    return -1;
  int64_t n = 0;
  struct dirent *e;
  while ((e = readdir(dir)) != NULL)
    if (e->d_name[0] != '.')
      n++;
  closedir(dir);
  return n;
}
#else
static int64_t thread_count(void) { return -1; }
#endif

/* The process's threads before any job, after a job of one thread, after the
   first job of two, and after jobs of every core. */
static void child_threads(int64_t *v) {
  int cores = nx_pool_cores();
  v[0] = thread_count();
  nx_pool_run(1, 1000, 10, nothing, NULL);
  v[1] = thread_count();
  nx_pool_run(2, 1000, 10, nothing, NULL);
  v[2] = thread_count();
  nx_pool_run(cores, 1000, 100, nothing, NULL);
  nx_pool_run(cores, 1000, 100, nothing, NULL);
  v[3] = thread_count();
}

#define CHILD_UNITS 1000

static _Atomic int child_units[CHILD_UNITS];

static void count_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  (void)ctx;
  for (int64_t i = lo; i < hi; i++)
    atomic_fetch_add(&child_units[i], 1);
}

/* Whether a worker ran a chunk, and the units of a job on every core not run
   once. */
static void child_job(int64_t *v) {
  v[0] = on_a_worker(NULL);
  nx_pool_run(nx_pool_cores(), CHILD_UNITS, 37, count_units, NULL);
  for (int i = 0; i < CHILD_UNITS; i++)
    if (atomic_load(&child_units[i]) != 1)
      v[1]++;
}

static void child_stack(int64_t *v) { v[0] = on_a_worker(deep_stack); }

static void child_faults(int64_t *v) {
  v[0] = on_a_worker(raise_faults);
  v[1] = delivered;
}

/* Runs [f] in a child made by fork right after a job of the parent's, and
   reads the values it writes. [status] says how the child ended: "exit 0"
   once it wrote them all. */
static void in_child(void (*f)(int64_t *), int64_t *values, char *status,
                     size_t len) {
  int fds[2];
  nx_pool_run(nx_pool_cores(), 1000, 100, nothing, NULL);
  if (pipe(fds) != 0) {
    snprintf(status, len, "pipe: %s", strerror(errno));
    return;
  }
  pid_t pid = fork();
  if (pid < 0) {
    snprintf(status, len, "fork: %s", strerror(errno));
    close(fds[0]);
    close(fds[1]);
    return;
  }
  if (pid == 0) {
    int64_t v[CHILD_VALUES] = {0};
    close(fds[0]);
    f(v);
    _exit(write(fds[1], v, sizeof v) == (ssize_t)sizeof v ? 0 : 2);
  }
  close(fds[1]);
  size_t want = CHILD_VALUES * sizeof(int64_t), got = 0;
  int64_t deadline = now_ns() + patience;
  while (got < want && now_ns() < deadline) {
    struct pollfd p = {fds[0], POLLIN, 0};
    int r = poll(&p, 1, (int)((deadline - now_ns()) / 1000000) + 1);
    if (r < 0 && errno == EINTR)
      continue;
    if (r <= 0)
      break;
    ssize_t n = read(fds[0], (char *)values + got, want - got);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      break;
    got += (size_t)n;
  }
  close(fds[0]);
  int late = got < want && now_ns() >= deadline;
  if (late)
    kill(pid, SIGKILL);
  int st;
  while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {
  }
  if (late)
    snprintf(status, len, "no answer within %llds",
             (long long)(patience / 1000000000));
  else if (WIFSIGNALED(st))
    snprintf(status, len, "killed by %s", strsignal(WTERMSIG(st)));
  else if (WEXITSTATUS(st) != 0)
    snprintf(status, len, "exit %d", WEXITSTATUS(st));
  else if (got < want)
    snprintf(status, len, "exit 0 after %zu of %zu bytes", got, want);
  else
    snprintf(status, len, "exit 0");
}

/* [probe_in_child scenario] is (status, values) of [in_child] for the
   scenario of that index: threads, job, stack, faults. */
value probe_in_child(value v_scenario) {
  CAMLparam1(v_scenario);
  CAMLlocal3(result, values, s);
  static void (*const scenarios[])(int64_t *) = {child_threads, child_job,
                                                 child_stack, child_faults};
  int64_t v[CHILD_VALUES] = {0};
  char status[128];
  caml_enter_blocking_section();
  in_child(scenarios[Int_val(v_scenario)], v, status, sizeof status);
  caml_leave_blocking_section();
  values = caml_alloc(CHILD_VALUES, 0);
  for (int k = 0; k < CHILD_VALUES; k++)
    Store_field(values, k, Val_long(v[k]));
  s = caml_copy_string(status);
  result = caml_alloc_tuple(2);
  Store_field(result, 0, s);
  Store_field(result, 1, values);
  CAMLreturn(result);
}

/* [probe_fork ()] forks a child that exits at once, and waits for it. */
value probe_fork(value unit) {
  (void)unit;
  caml_enter_blocking_section();
  pid_t pid = fork();
  if (pid == 0)
    _exit(0);
  int st;
  if (pid > 0)
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {
    }
  caml_leave_blocking_section();
  if (pid < 0)
    caml_failwith("probe_fork: fork failed");
  return Val_unit;
}

/* The host */

value probe_cores(value unit) {
  (void)unit;
  return Val_int(nx_pool_cores());
}

value probe_performance_cores(value unit) {
  (void)unit;
  return Val_int(nx_pool_performance_cores());
}

value probe_system(value unit) {
  (void)unit;
#if defined(__APPLE__)
  return caml_copy_string("macos");
#elif defined(__linux__)
  return caml_copy_string("linux");
#else
  return caml_copy_string("other");
#endif
}

/* [probe_sysctl name] is the integer [name] reads, or -1. */
value probe_sysctl(value v_name) {
#if defined(__APPLE__)
  int n;
  size_t len = sizeof n;
  if (sysctlbyname(String_val(v_name), &n, &len, NULL, 0) == 0)
    return Val_int(n);
#else
  (void)v_name;
#endif
  return Val_int(-1);
}

/* [probe_pinned_cores ()] pins the calling thread to one CPU of its affinity,
   reads nx_pool_cores, restores the affinity, reads it again: (first, second),
   or (-1, -1) where affinity is not Linux's. */
value probe_pinned_cores(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);
  int first = -1, second = -1;
#if defined(__linux__)
  cpu_set_t all, one;
  if (sched_getaffinity(0, sizeof all, &all) == 0) {
    CPU_ZERO(&one);
    for (int i = 0; i < CPU_SETSIZE; i++)
      if (CPU_ISSET(i, &all)) {
        CPU_SET(i, &one);
        break;
      }
    if (sched_setaffinity(0, sizeof one, &one) == 0) {
      first = nx_pool_cores();
      sched_setaffinity(0, sizeof all, &all);
      second = nx_pool_cores();
    }
  }
#endif
  result = caml_alloc_tuple(2);
  Store_field(result, 0, Val_int(first));
  Store_field(result, 1, Val_int(second));
  CAMLreturn(result);
}

/* The threads of the process, other than the calling one, that are running
   now, or -1 where the system does not say. */
value probe_running_threads(value unit) {
  (void)unit;
#if defined(__APPLE__)
  mach_port_t task = mach_task_self();
  thread_act_array_t threads;
  mach_msg_type_number_t n;
  if (task_threads(task, &threads, &n) != KERN_SUCCESS)
    return Val_int(-1);
  thread_t self = mach_thread_self();
  long running = 0;
  for (mach_msg_type_number_t i = 0; i < n; i++) {
    thread_basic_info_data_t info;
    mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
    if (threads[i] != self &&
        thread_info(threads[i], THREAD_BASIC_INFO, (thread_info_t)&info,
                    &count) == KERN_SUCCESS &&
        info.run_state == TH_STATE_RUNNING)
      running++;
    mach_port_deallocate(task, threads[i]);
  }
  mach_port_deallocate(task, self);
  vm_deallocate(task, (vm_address_t)threads, n * sizeof *threads);
  return Val_long(running);
#elif defined(__linux__)
  DIR *dir = opendir("/proc/self/task");
  if (dir == NULL)
    return Val_int(-1);
  long self = (long)syscall(SYS_gettid), running = 0;
  struct dirent *e;
  while ((e = readdir(dir)) != NULL) {
    if (e->d_name[0] == '.' || atol(e->d_name) == self)
      continue;
    char path[sizeof "/proc/self/task//stat" + sizeof e->d_name], line[512];
    snprintf(path, sizeof path, "/proc/self/task/%s/stat", e->d_name);
    FILE *f = fopen(path, "r");
    if (f == NULL)
      continue; /* the thread exited */
    size_t len = fread(line, 1, sizeof line - 1, f);
    fclose(f);
    line[len] = 0;
    /* "tid (name) S ...": the name may hold spaces and parentheses. */
    char *close = strrchr(line, ')');
    if (close != NULL && close[1] == ' ' && close[2] == 'R')
      running++;
  }
  closedir(dir);
  return Val_long(running);
#else
  return Val_int(-1);
#endif
}
