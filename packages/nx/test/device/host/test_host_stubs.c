/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#if defined(__linux__)
#define _GNU_SOURCE /* syscall */
#endif

#include <caml/mlvalues.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#endif

/* Whether the page holding [addr] is mapped in the process. */
value test_host_mapped(value addr) {
#ifdef _WIN32
  MEMORY_BASIC_INFORMATION info;
  if (VirtualQuery((void *)Nativeint_val(addr), &info, sizeof info) == 0)
    return Val_false;
  return Val_bool(info.State != MEM_FREE);
#else
  uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE);
  uintptr_t a = (uintptr_t)Nativeint_val(addr) & ~(page - 1);
  return Val_bool(msync((void *)a, page, MS_ASYNC) == 0);
#endif
}

/* The number of the process's threads, other than the calling one, that run
   or wait for a core, as the scheduler reports them; -1 where this is not
   known. A thread parked on a condition variable, or sleeping, does not
   count; a thread that spins, yielding its core, does. */
value test_host_running_threads(value unit) {
  (void)unit;
#if defined(__APPLE__)
  mach_port_t task = mach_task_self();
  thread_act_array_t threads;
  mach_msg_type_number_t n;
  if (task_threads(task, &threads, &n) != KERN_SUCCESS) return Val_int(-1);
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
  if (dir == NULL) return Val_int(-1);
  long self = (long)syscall(SYS_gettid), running = 0;
  struct dirent *e;
  while ((e = readdir(dir)) != NULL) {
    if (e->d_name[0] == '.' || atol(e->d_name) == self) continue;
    char path[64], line[512];
    snprintf(path, sizeof path, "/proc/self/task/%s/stat", e->d_name);
    FILE *f = fopen(path, "r");
    if (f == NULL) continue; /* the thread exited */
    size_t len = fread(line, 1, sizeof line - 1, f);
    fclose(f);
    line[len] = 0;
    /* "tid (name) S ...": the name may hold spaces and parentheses. */
    char *close = strrchr(line, ')');
    if (close != NULL && close[1] == ' ' && close[2] == 'R') running++;
  }
  closedir(dir);
  return Val_long(running);
#else
  return Val_int(-1);
#endif
}
