/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/


/* Probes of device_pool.h's threads for the pool's suite: what a body sees on a
   worker, fork children, and the process's threads. They fork and read
   signal masks and thread states, so they build where POSIX does.

   Some bodies wait for another call, which device_pool.h forbids, to force a
   chunk onto a worker. Each such wait gives up after [patience], so a pool
   that breaks a promise fails the test instead of hanging it. */

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
#elif defined(__linux__)
#include <dirent.h>
#include <sys/syscall.h>
#endif

#include "device_pool.h"

/* Time */

/* Far longer than any wakeup: only a pool that never runs the awaited chunk
   reaches it. */
static const int64_t patience = INT64_C(10000000000);

static int64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

static void nothing(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)lo;
  (void)hi;
  (void)worker;
  (void)ctx;
}


/* A burst of jobs */

static _Atomic int burst_stop;

/* [device_pool_test_burst ()] runs a job on every core, then jobs of two chunks
   on two threads back to back until [device_pool_test_burst_stop], whose
   request it then clears. */
value device_pool_test_burst(value unit) {
  (void)unit;
  int cores = device_pool_cores();
  caml_enter_blocking_section();
  device_pool_run(cores, cores, cores, nothing, NULL);
  while (!atomic_load(&burst_stop))
    device_pool_run(2, 2, 2, nothing, NULL);
  atomic_store(&burst_stop, 0);
  caml_leave_blocking_section();
  return Val_unit;
}

value device_pool_test_burst_stop(value unit) {
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
  device_pool_run(2, 2, 2, meet, &j);
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

/* [device_pool_test_worker_mask ()] is (whether a worker ran, the signals of
   the table with whether a worker blocks them). */
value device_pool_test_worker_mask(value unit) {
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

/* The signals a body may raise itself, in the order of [faults] in
   device_pool_thread_probe.ml. */
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

/* 7 MiB of the 8 MiB device_pool.h promises, touched from the top down so an
   overflow meets the guard page first. The deepest byte is read back, so
   the frame is used. */
#define DEEP_BYTES (7 << 20)

static volatile char deep_sink;

static __attribute__((noinline)) void deep_stack(void) {
  volatile char frame[DEEP_BYTES];
  for (size_t i = DEEP_BYTES; i > 0; i -= 4096)
    frame[i - 1] = (char)i;
  deep_sink = frame[4095];
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
  int cores = device_pool_cores();
  v[0] = thread_count();
  device_pool_run(1, 1000, 10, nothing, NULL);
  v[1] = thread_count();
  device_pool_run(2, 1000, 10, nothing, NULL);
  v[2] = thread_count();
  device_pool_run(cores, 1000, 100, nothing, NULL);
  device_pool_run(cores, 1000, 100, nothing, NULL);
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
  device_pool_run(device_pool_cores(), CHILD_UNITS, 37, count_units, NULL);
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
  device_pool_run(device_pool_cores(), 1000, 100, nothing, NULL);
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

/* [device_pool_test_in_child scenario] is (status, values) of [in_child] for
   the scenario of that index: threads, job, stack, faults. */
value device_pool_test_in_child(value v_scenario) {
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

/* [device_pool_test_fork ()] forks a child that exits at once, and waits for
   it. */
value device_pool_test_fork(value unit) {
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
    caml_failwith("device_pool_test_fork: fork failed");
  return Val_unit;
}

/* The threads of the process, other than the calling one, that are running
   now, or -1 where the system does not say. */
value device_pool_test_running_threads(value unit) {
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
