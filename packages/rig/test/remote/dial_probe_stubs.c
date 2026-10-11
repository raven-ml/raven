/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Whether a thread of this process waits for its socket, from the system's
   view of it: a thread blocked in the kernel, entered through the system call
   of select or poll. A suite holds a signal until the thread it interrupts
   waits there. */

#define _GNU_SOURCE

#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <time.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <pthread.h>
#include <sys/syscall.h>
#elif defined(__linux__)
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

/* What the system shows of a thread: [RUNS] is anything but a wait for a
   socket. */
enum seen { RUNS, WAITS, UNSEEN };

#if defined(__APPLE__)

typedef mach_port_t thread_id;

static thread_id self(void) { return pthread_mach_thread_np(pthread_self()); }

static int socket_call(long call) {
  return call == SYS_select || call == SYS_select_nocancel ||
         call == SYS_pselect || call == SYS_pselect_nocancel ||
         call == SYS_poll || call == SYS_poll_nocancel;
}

/* The system call a suspended thread entered: the number its stub put in x16
   (arm64) or rax (x86_64). */
static enum seen waits(thread_id t) {
#if defined(__arm64__) || defined(__x86_64__)
  if (thread_suspend(t) != KERN_SUCCESS) return UNSEEN;
  thread_basic_info_data_t info;
  mach_msg_type_number_t n = THREAD_BASIC_INFO_COUNT;
  int ok = thread_info(t, THREAD_BASIC_INFO, (thread_info_t)&info, &n) ==
           KERN_SUCCESS;
  int waiting = ok && info.run_state == TH_STATE_WAITING;
  long call = -1;
#if defined(__arm64__)
  arm_thread_state64_t s;
  n = ARM_THREAD_STATE64_COUNT;
  ok = ok && thread_get_state(t, ARM_THREAD_STATE64, (thread_state_t)&s,
                              &n) == KERN_SUCCESS;
  if (ok) call = (long)s.__x[16];
#else
  x86_thread_state64_t s;
  n = x86_THREAD_STATE64_COUNT;
  ok = ok && thread_get_state(t, x86_THREAD_STATE64, (thread_state_t)&s,
                              &n) == KERN_SUCCESS;
  if (ok) call = (long)(s.__rax & 0xffffff);
#endif
  thread_resume(t);
  if (!ok) return UNSEEN;
  return waiting && socket_call(call) ? WAITS : RUNS;
#else
  (void)t;
  return UNSEEN;
#endif
}

#elif defined(__linux__)

typedef pid_t thread_id;

static thread_id self(void) { return (pid_t)syscall(SYS_gettid); }

static int socket_call(long call) {
#ifdef SYS_select
  if (call == SYS_select) return 1;
#endif
#ifdef SYS_pselect6
  if (call == SYS_pselect6) return 1;
#endif
#ifdef SYS_pselect6_time64
  if (call == SYS_pselect6_time64) return 1;
#endif
#ifdef SYS_poll
  if (call == SYS_poll) return 1;
#endif
#ifdef SYS_ppoll
  if (call == SYS_ppoll) return 1;
#endif
#ifdef SYS_ppoll_time64
  if (call == SYS_ppoll_time64) return 1;
#endif
  return 0;
}

/* The first field of the thread's syscall file: the number of the call it is
   blocked in, or "running". */
static enum seen waits(thread_id t) {
  char path[64];
  snprintf(path, sizeof path, "/proc/self/task/%d/syscall", (int)t);
  FILE *f = fopen(path, "r");
  if (f == NULL) return UNSEEN;
  long call = -1;
  int got = fscanf(f, "%ld", &call);
  fclose(f);
  if (got == EOF) return UNSEEN;
  return got == 1 && socket_call(call) ? WAITS : RUNS;
}

#else

typedef int thread_id;

static thread_id self(void) { return 0; }

static enum seen waits(thread_id t) {
  (void)t;
  return UNSEEN;
}

#endif

value rig_dial_probe_self(value unit) {
  (void)unit;
  return Val_long((long)self());
}

/* Waits until the thread [t] waits for its socket, for at most [ms]
   milliseconds: [WAITS] once it does, [RUNS] if it did not, [UNSEEN] if the
   system does not show it. Releases the runtime. */
value rig_dial_probe_await_socket(value t, value ms) {
  thread_id id = (thread_id)Long_val(t);
  long left = Long_val(ms);
  struct timespec tick = {0, 1000000};
  caml_release_runtime_system();
  enum seen s = waits(id);
  for (; s == RUNS && left > 0; left--) {
    nanosleep(&tick, NULL);
    s = waits(id);
  }
  caml_acquire_runtime_system();
  return Val_int(s);
}
