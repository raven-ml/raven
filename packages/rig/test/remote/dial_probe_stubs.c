/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Whether a thread of this process waits in select, from the system's view of
   it: a thread blocked in the kernel, entered through select's system call.
   A suite holds a signal until the thread it interrupts waits there. */

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

#if defined(__APPLE__)

typedef mach_port_t thread_id;

static thread_id self(void) { return pthread_mach_thread_np(pthread_self()); }

/* The system call a suspended thread entered: the number its stub put in x16
   (arm64) or rax (x86_64). */
static int in_select(thread_id t) {
  if (thread_suspend(t) != KERN_SUCCESS) return 0;
  thread_basic_info_data_t info;
  mach_msg_type_number_t n = THREAD_BASIC_INFO_COUNT;
  int waiting = thread_info(t, THREAD_BASIC_INFO, (thread_info_t)&info, &n) ==
                    KERN_SUCCESS &&
                info.run_state == TH_STATE_WAITING;
  long call = -1;
#if defined(__arm64__)
  arm_thread_state64_t s;
  n = ARM_THREAD_STATE64_COUNT;
  if (thread_get_state(t, ARM_THREAD_STATE64, (thread_state_t)&s, &n) ==
      KERN_SUCCESS)
    call = (long)s.__x[16];
#elif defined(__x86_64__)
  x86_thread_state64_t s;
  n = x86_THREAD_STATE64_COUNT;
  if (thread_get_state(t, x86_THREAD_STATE64, (thread_state_t)&s, &n) ==
      KERN_SUCCESS)
    call = (long)(s.__rax & 0xffffff);
#endif
  thread_resume(t);
  return waiting && (call == SYS_select || call == SYS_select_nocancel);
}

#elif defined(__linux__)

typedef pid_t thread_id;

static thread_id self(void) { return (pid_t)syscall(SYS_gettid); }

/* The first field of the thread's syscall file: the number of the call it is
   blocked in, or "running". */
static int in_select(thread_id t) {
  char path[64];
  snprintf(path, sizeof path, "/proc/self/task/%d/syscall", (int)t);
  FILE *f = fopen(path, "r");
  if (f == NULL) return 0;
  long call = -1;
  int got = fscanf(f, "%ld", &call);
  fclose(f);
  if (got != 1) return 0;
#ifdef SYS_select
  if (call == SYS_select) return 1;
#endif
#ifdef SYS_pselect6
  if (call == SYS_pselect6) return 1;
#endif
#ifdef SYS_pselect6_time64
  if (call == SYS_pselect6_time64) return 1;
#endif
  return 0;
}

#else

typedef int thread_id;

static thread_id self(void) { return 0; }

static int in_select(thread_id t) {
  (void)t;
  return 0;
}

#endif

value rig_dial_probe_self(value unit) {
  (void)unit;
  return Val_long((long)self());
}

/* Waits until the thread [t] waits in select, for at most [ms]
   milliseconds: whether it does. Releases the runtime. */
value rig_dial_probe_await_select(value t, value ms) {
  thread_id id = (thread_id)Long_val(t);
  long left = Long_val(ms);
  struct timespec tick = {0, 1000000};
  caml_release_runtime_system();
  int found = in_select(id);
  for (; !found && left > 0; left--) {
    nanosleep(&tick, NULL);
    found = in_select(id);
  }
  caml_acquire_runtime_system();
  return Val_bool(found);
}
