/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Interleavings of the pool's protocol, held. This file compiles nx_pool.c
   again with its hook points defined: a scenario arms hooks that make the
   caller and chosen workers wait for each other's events, so the timeline in
   the scenario's comment is the one that runs. Every wait gives up after
   [patience] (nx_pool_probe.h); a scenario then answers which event it
   waited for. The copy's public names carry nx_pool_hooked_, so it links
   beside the pool itself. POSIX only: scenarios fork, and workers are told
   apart by their pthread_t. */

#define _GNU_SOURCE

#include <pthread.h>

typedef enum {
  point_enter_open,
  point_enter_refused,
  point_left,
  point_entered,
  point_chunk,
  point_parking,
  point_park,
  point_opened,
  point_published,
  point_claimed,
  point_closed,
  point_waiting,
} point;

static void hook(point at);

#define NX_POOL_HOOK(at) hook(point_##at)
#define nx_pool_cores nx_pool_hooked_cores
#define nx_pool_performance_cores nx_pool_hooked_performance_cores
#define nx_pool_run nx_pool_hooked_run
#define nx_pool_cgroup_cpus nx_pool_hooked_cgroup_cpus
/* nx_pool.c's clock is static; renamed, it leaves now_ns to the probes'. */
#define now_ns nx_pool_now_ns
#include "../nx_pool.c"
#undef now_ns

#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

#include <sched.h>
#include <sys/wait.h>
#include <unistd.h>

#include "nx_pool_probe.h"

/* Workers by thread: a mapping job has each worker record its thread. A
   thread not among them is a caller, worker 0. */

enum { max_mapped = 64 };
static pthread_t worker_threads[max_mapped];
static _Atomic int mapped; /* worker_threads[1, mapped) are known */

static int who(void) {
  pthread_t self = pthread_self();
  int n = atomic_load(&mapped);
  for (int w = 1; w < n; w++)
    if (pthread_equal(worker_threads[w], self)) return w;
  return 0;
}

/* Events */

typedef enum {
  w_loaded,
  n_returned,
  w_added,
  n1_opened,
  n1_closed,
  w_undoing,
  n1_returned,
  w_entered,
  w_left,
  w_claimed,
  caller_waiting,
  w_subbed,
  w_decided,
  gen_stored,
  w_bit,
  forked,
  body_began,
  body_done,
  events
} event;

static const char *event_names[events] = {
    "the worker's load of an open job",
    "job N's return",
    "the worker's add to a closed job",
    "job N+1's opening",
    "job N+1's close",
    "the worker's undo",
    "job N+1's return",
    "the worker's entry",
    "the worker's leave",
    "the worker's claim",
    "the caller's wait",
    "the worker's leave while the caller held",
    "the worker's decision to park",
    "the job's publication",
    "the worker's parked bit",
    "the fork",
    "the job's body",
    "the body's end",
};

static _Atomic int signalled[events];
static _Atomic int order[events]; /* the order events were signalled in */
static _Atomic int order_next;
static _Atomic int timed_out = -1; /* the first event a wait gave up on */

static void sig(event e) {
  atomic_store(&order[e], atomic_fetch_add(&order_next, 1) + 1);
  atomic_store(&signalled[e], 1);
}

static int await_for(event e, int64_t ns) {
  int64_t deadline = now_ns() + ns;
  while (!atomic_load(&signalled[e])) {
    if (now_ns() > deadline) return 0;
    sched_yield();
  }
  return 1;
}

/* Waits for [e]; past patience, records it and lets the thread go on, so a
   scenario that cannot hold ends and answers instead of hanging. */
static void await(event e) {
  if (await_for(e, patience)) return;
  int none = -1;
  atomic_compare_exchange_strong(&timed_out, &none, (int)e);
}

/* Scenarios and their hooks */

typedef enum {
  none,
  stale_first,
  stale_next,
  late_first,
  late_next,
  caller_sleep,
  caller_held,
  wake_bit,
  wake_decided,
  wake_slept,
  narrow,
  fork_parked,
} phase;

static _Atomic int current = none;
static int target; /* the worker a scenario holds */
static _Atomic int go;      /* the caller's hooks act only once set */
static _Atomic int entered; /* the target entered a job it must not */
static uint64_t inside_at_close;

/* A worker's hook that holds it with the parking mutex acts only while
   armed, and disarms as it acts: a job run meanwhile would wait for that
   mutex forever. */
static _Atomic int armed;
static int take_arm(void) { return atomic_exchange(&armed, 0) == 1; }

static _Atomic int once_flags[events];
static int once(event e) { return atomic_exchange(&once_flags[e], 1) == 0; }

static void hook(point at) {
  int ph = atomic_load(&current);
  if (ph == none) return;
  int id = who();
  int caller = id == 0;
  if (caller && !atomic_load(&go)) return;
  int mine = id == target;
  switch (ph) {
  case stale_first:
    if (at == point_enter_open && mine && once(w_loaded)) {
      sig(w_loaded);
      await(n_returned);
    } else if (at == point_claimed && caller)
      await(w_loaded);
    else if (at == point_enter_refused && mine && once(w_added)) {
      sig(w_added);
      await(n1_closed);
      sig(w_undoing);
    } else if (at == point_entered && mine)
      atomic_store(&entered, 1);
    break;
  case stale_next:
    if (at == point_closed && caller) {
      inside_at_close = atomic_load(&g_pool->inside);
      sig(n1_closed);
    } else if (at == point_entered && mine)
      atomic_store(&entered, 1);
    break;
  case late_first:
    if (at == point_enter_open && mine && once(w_loaded)) {
      sig(w_loaded);
      await(n1_opened);
    } else if (at == point_claimed && caller)
      await(w_loaded);
    break;
  case late_next:
    if (at == point_opened && caller) {
      sig(n1_opened);
      await(w_left);
    } else if (at == point_entered && mine)
      sig(w_entered);
    else if (at == point_left && mine && once(w_left))
      sig(w_left);
    break;
  case caller_sleep:
  case caller_held:
    if (at == point_chunk && id == 1 && once(w_claimed))
      sig(w_claimed);
    else if (at == point_waiting && caller && once(caller_waiting)) {
      sig(caller_waiting);
      if (ph == caller_held) await(w_subbed);
    } else if (at == point_left && id == 1 && once(w_subbed))
      sig(w_subbed);
    break;
  case wake_bit:
    if (at == point_parking && mine && take_arm()) {
      sig(w_bit);
      await(gen_stored);
    } else if (at == point_published && caller)
      sig(gen_stored);
    break;
  case wake_decided:
    if (at == point_park && mine && take_arm()) {
      sig(w_decided);
      await(gen_stored);
    } else if (at == point_published && caller && once(gen_stored)) {
      sig(gen_stored);
      await(w_bit);
    } else if (at == point_parking && mine &&
               atomic_load(&signalled[w_decided]) && once(w_bit))
      sig(w_bit);
    break;
  case wake_slept:
    if (at == point_parking && mine && take_arm()) sig(w_bit);
    break;
  case narrow:
    if (at == point_parking && mine && once(w_bit)) sig(w_bit);
    break;
  case fork_parked:
    if (at == point_parking && mine && take_arm()) {
      sig(w_bit);
      await(forked);
    }
    break;
  }
}

static void begin(phase ph) {
  for (int e = 0; e < events; e++) {
    atomic_store(&signalled[e], 0);
    atomic_store(&order[e], 0);
    atomic_store(&once_flags[e], 0);
  }
  atomic_store(&order_next, 0);
  atomic_store(&timed_out, -1);
  atomic_store(&entered, 0);
  atomic_store(&go, 0);
  atomic_store(&armed, 0);
  atomic_store(&current, ph);
}

/* Tallies: jobs whose bodies count what ran where */

enum { max_chunks = 64 };

typedef struct {
  int64_t total, chunks;
  _Atomic int runs[max_chunks];   /* calls that ran chunk i */
  _Atomic int on[max_chunks];     /* the worker that ran chunk i */
  _Atomic int worked[max_mapped]; /* whether worker w ran a chunk */
  int sleeper;                    /* a worker whose chunks sleep 20 ms */
  int awaited;                    /* others' chunks wait until it worked */
  int signals;                    /* the first chunk signals body_began */
  _Atomic int signalled_body;
} tally;

static void body(int64_t lo, int64_t hi, int w, void *ctx) {
  tally *j = ctx;
  for (int64_t i = 0; i < j->chunks; i++)
    if (i * j->total / j->chunks >= lo &&
        (i + 1) * j->total / j->chunks <= hi) {
      atomic_fetch_add(&j->runs[i], 1);
      atomic_store(&j->on[i], w);
    }
  if (w < max_mapped) atomic_store(&j->worked[w], 1);
  if (w == j->sleeper) usleep(20000);
  if (j->awaited > 0 && w != j->awaited) {
    int64_t deadline = now_ns() + patience;
    while (!atomic_load(&j->worked[j->awaited]) && now_ns() < deadline)
      sched_yield();
  }
  if (j->signals && atomic_exchange(&j->signalled_body, 1) == 0) {
    sig(body_began);
    usleep(50000);
    sig(body_done);
  }
}

static void tally_init(tally *j, int64_t chunks) {
  *j = (tally){.total = chunks, .chunks = chunks, .sleeper = -1, .awaited = -1};
}

static void run_tally(int threads, tally *j) {
  nx_pool_run(threads, j->total, j->chunks, body, j);
}

/* The chunks of [j] that did not run once, and those that ran on a worker
   at or past [threads]. */
static int not_once(const tally *j) {
  int n = 0;
  for (int64_t i = 0; i < j->chunks; i++) n += j->runs[i] != 1;
  return n;
}

static int off(const tally *j, int threads) {
  int n = 0;
  for (int64_t i = 0; i < j->chunks; i++) n += j->on[i] >= threads;
  return n;
}

static int on_worker(const tally *j, int w) {
  int n = 0;
  for (int64_t i = 0; i < j->chunks; i++) n += j->on[i] == w;
  return n;
}

/* A job of one chunk per thread, each waiting until every thread holds
   one: each worker records its thread. Once, before the first scenario. */
static _Atomic int arrived;

static void map_body(int64_t lo, int64_t hi, int w, void *ctx) {
  (void)lo;
  (void)hi;
  int threads = *(int *)ctx;
  if (w < max_mapped) worker_threads[w] = pthread_self();
  atomic_fetch_add(&arrived, 1);
  int64_t deadline = now_ns() + patience;
  while (atomic_load(&arrived) < threads && now_ns() < deadline)
    sched_yield();
}

static void map_workers(void) {
  if (atomic_load(&mapped) > 0) return;
  pool *p = get();
  int threads = p ? p->threads : 1;
  nx_pool_run(threads, threads, threads, map_body, &threads);
  atomic_store(&mapped, threads < max_mapped ? threads : max_mapped);
}

/* Every thread runs a job, then idles: a worker parks within its window. */
static void kick(void) {
  int t = nx_pool_cores();
  tally j;
  tally_init(&j, max_chunks);
  run_tally(t, &j);
}

static void settle(void) {
  atomic_store(&current, none);
  for (int k = 0; k < 20; k++) kick();
}

/* Kicks the workers until the target's hook for [e] takes the arm on its way
   to park, then waits for [e]. */
static void await_park(event e) {
  for (int tries = 0; tries < 10; tries++) {
    kick();
    atomic_store(&armed, 1);
    if (await_for(e, patience / 20)) return;
    if (atomic_exchange(&armed, 0) == 0) break; /* the hook took it */
  }
  await(e);
}

/* Answers */

/* (timed out, values): [timed out] names the event a wait gave up on, ""
   when the timeline held. */
static value answer(const int *v, int n) {
  CAMLparam0();
  CAMLlocal3(r, why, values);
  int e = atomic_load(&timed_out);
  why = caml_copy_string(e < 0 ? "" : event_names[e]);
  values = caml_alloc(n, 0);
  for (int i = 0; i < n; i++) Store_field(values, i, Val_int(v[i]));
  r = caml_alloc_tuple(2);
  Store_field(r, 0, why);
  Store_field(r, 1, values);
  CAMLreturn(r);
}

/* Each scenario releases the runtime: it blocks on the pool's threads. */

/* An add that finds the job closed counts in the next job.
   1. The caller opens N (t = 2, two chunks) and claims both.
   2. W1 sees N and loads inside: open. Holds.
   3. The caller closes N; nothing was inside, so it returns.
   4. W1 adds: it reads closed, and the add counts. Holds.
   5. The caller opens N+1 by clearing closed alone: inside = 1. It claims
      every chunk, closes, reads 1 and waits.
   6. W1 undoes its add: inside = closed. The caller returns.
   Answers: N+1's count at its close, whether closed was set then, whether
   the undo began before N+1 returned, whether inside was closed after, the
   chunks of N and N+1 not run once, those of N+1 off worker 0, whether W1
   entered either job. */
value nx_pool_test_stale_add(value unit) {
  (void)unit;
  int v[8] = {0};
  caml_enter_blocking_section();
  map_workers();
  settle();
  begin(stale_first);
  target = 1;
  tally n, n1;
  tally_init(&n, 2);
  tally_init(&n1, 4);
  atomic_store(&go, 1);
  run_tally(2, &n);
  sig(n_returned);
  await(w_added);
  atomic_store(&current, stale_next);
  run_tally(2, &n1);
  sig(n1_returned);
  v[0] = (int)(inside_at_close & ~closed);
  v[1] = (inside_at_close & closed) != 0;
  v[2] = order[w_undoing] > 0 && order[w_undoing] < order[n1_returned];
  v[3] = atomic_load(&g_pool->inside) == closed;
  v[4] = not_once(&n);
  v[5] = not_once(&n1);
  v[6] = off(&n1, 1);
  v[7] = atomic_load(&entered);
  settle();
  caml_leave_blocking_section();
  return answer(v, 8);
}

/* A worker that saw N enters N+1 before N+1 is published.
   1. The caller opens N (t = 3, three chunks) and claims them.
   2. W sees N and loads inside: open. Holds.
   3. The caller closes N and returns, then opens N+1 (t = 2, 8 chunks) and
      holds before its generation is stored.
   4. W adds: N+1 is open, so W is inside N+1, whose threads it reads. W1
      claims every chunk; W2 claims none. W leaves.
   5. The caller publishes, claims what is left, closes and returns.
   Answers: whether W entered N+1, the chunks of N and N+1 not run once,
   those of N+1 off workers 0 and 1, those of N+1 on W. */
value nx_pool_test_late_entry(value v_worker) {
  int w = Int_val(v_worker);
  int v[5] = {0};
  caml_enter_blocking_section();
  map_workers();
  settle();
  begin(late_first);
  target = w;
  tally n, n1;
  tally_init(&n, 3);
  tally_init(&n1, 8);
  atomic_store(&go, 1);
  run_tally(3, &n);
  atomic_store(&current, late_next);
  run_tally(2, &n1);
  v[0] = atomic_load(&signalled[w_entered]);
  v[1] = not_once(&n);
  v[2] = not_once(&n1);
  v[3] = off(&n1, 2);
  v[4] = on_worker(&n1, w);
  settle();
  caml_leave_blocking_section();
  return answer(v, 5);
}

/* The caller sleeps on done; the last worker out wakes it. W1's chunk
   sleeps 20 ms, the caller's waits for W1's claim; the caller closes, spins
   out its window and sleeps; W1 leaves, sees waiting and signals. Held: the
   caller holds between setting waiting and reading inside until W1's leave
   landed, so it reads closed and never sleeps, and W1's signal finds no
   sleeper. Answers: whether the caller set waiting, whether W1 left while it
   held (held only), the chunks not run once. */
value nx_pool_test_caller_sleeps(value v_held) {
  int held = Bool_val(v_held);
  int v[3] = {0};
  caml_enter_blocking_section();
  map_workers();
  settle();
  begin(held ? caller_held : caller_sleep);
  tally j;
  tally_init(&j, 2);
  j.sleeper = 1;
  j.awaited = 1;
  atomic_store(&go, 1);
  run_tally(2, &j);
  int n = 0;
  v[n++] = atomic_load(&signalled[caller_waiting]);
  if (held) v[n++] = order[w_subbed] > order[caller_waiting];
  v[n++] = not_once(&j);
  settle();
  caml_leave_blocking_section();
  return answer(v, n);
}

/* A parked worker and a publication, in three orders, the constructors of
   nx_pool_hooked_probe.mli's wake_order.
   0: W1 sets its parked bit and holds, with the parking mutex, until the
      caller stored the generation; it reads the new one and does not sleep.
   1: W1 decides to park and holds; the caller stores the generation and
      holds until W1 set its bit; W1 reads the new generation.
   2: W1 sleeps; the caller sees its bit and broadcasts.
   Answers: whether W1 ran a chunk of the job (the other chunk waits for it
   up to patience), the chunks not run once. */
value nx_pool_test_wake(value v_order) {
  int which = Int_val(v_order);
  int v[2] = {0};
  caml_enter_blocking_section();
  map_workers();
  settle();
  begin(which == 0 ? wake_bit : which == 1 ? wake_decided : wake_slept);
  target = 1;
  await_park(which == 1 ? w_decided : w_bit);
  if (which == 2) usleep(2000);
  tally j;
  tally_init(&j, 2);
  j.awaited = 1;
  atomic_store(&go, 1);
  run_tally(2, &j);
  v[0] = atomic_load(&j.worked[1]);
  v[1] = not_once(&j);
  settle();
  caml_leave_blocking_section();
  return answer(v, 2);
}

/* Narrow jobs (t = 2) leave W2 out past its window: it parks while they
   run. A wide job (t = 3) then needs it. Answers: whether W2 parked during
   the burst, whether it ran a chunk of the wide job, the chunks of the
   narrow jobs and of the wide one not run once. */
value nx_pool_test_narrow_burst(value unit) {
  (void)unit;
  int v[4] = {0};
  caml_enter_blocking_section();
  map_workers();
  settle();
  begin(narrow);
  target = 2;
  int64_t deadline = now_ns() + patience;
  while (!atomic_load(&signalled[w_bit]) && now_ns() < deadline) {
    tally j;
    tally_init(&j, 2);
    run_tally(2, &j);
    v[2] += not_once(&j);
  }
  atomic_store(&current, none);
  tally wide;
  tally_init(&wide, 3);
  wide.awaited = 2;
  run_tally(3, &wide);
  v[0] = atomic_load(&signalled[w_bit]);
  v[1] = atomic_load(&wide.worked[2]);
  v[3] = not_once(&wide);
  settle();
  caml_leave_blocking_section();
  return answer(v, 4);
}

/* A child made by fork runs a job on every core; it exits with the chunks
   not run once, or SIGALRM ends it past 10 s. */
static int fork_child(void) {
  pid_t pid = fork();
  if (pid == 0) {
    atomic_store(&current, none);
    alarm(10);
    tally j;
    tally_init(&j, max_chunks);
    run_tally(nx_pool_cores(), &j);
    _exit(not_once(&j));
  }
  return pid;
}

static int child_status(pid_t pid) {
  int status;
  if (pid < 0 || waitpid(pid, &status, 0) != pid) return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : 256 + WTERMSIG(status);
}

/* Fork while W1 holds the parking mutex: the child builds its own pool.
   Answers: the child's exit status. */
value nx_pool_test_fork_parked(value unit) {
  (void)unit;
  int v[1] = {0};
  caml_enter_blocking_section();
  map_workers();
  settle();
  begin(fork_parked);
  target = 1;
  await_park(w_bit);
  pid_t pid = fork_child();
  sig(forked);
  v[0] = child_status(pid);
  settle();
  caml_leave_blocking_section();
  return answer(v, 1);
}

static void *runner(void *ctx) {
  tally *j = ctx;
  run_tally(2, j);
  return NULL;
}

/* Fork from a thread while another thread's job of two threads runs a
   body of 50 ms. Answers: whether fork returned after the body ended, the
   child's exit status. */
value nx_pool_test_fork_running(value unit) {
  (void)unit;
  int v[2] = {0};
  caml_enter_blocking_section();
  map_workers();
  settle();
  begin(none);
  tally j;
  tally_init(&j, 4);
  j.signals = 1;
  pthread_t thread;
  int made = pthread_create(&thread, NULL, runner, &j) == 0;
  if (made) await(body_began);
  pid_t pid = fork_child();
  v[0] = atomic_load(&signalled[body_done]);
  if (made) pthread_join(thread, NULL);
  v[1] = child_status(pid);
  settle();
  caml_leave_blocking_section();
  return answer(v, 2);
}

value nx_pool_test_hooked_cores(value unit) {
  (void)unit;
  return Val_int(nx_pool_cores());
}
