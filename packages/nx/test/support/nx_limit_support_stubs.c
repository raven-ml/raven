/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A limit of the process's memory, for suites that run on Linux and Windows
   alone: their limits bound allocations, macOS's do not. */

#include <caml/mlvalues.h>

#if defined(__linux__)

#include <errno.h>
#include <stdio.h>
#include <sys/resource.h>
#include <unistd.h>

/* Lowers the process's soft limit of its address space to [room] bytes past
   what it maps (the first field of /proc/self/statm, in pages) where the
   limit is higher, so that it never raises a limit set outside and stays
   within the hard limit: 0, or the errno of a refusal. Holds the runtime:
   none of these calls blocks. RLIM_INFINITY is rlim_t's largest value. */
value nx_test_limit_memory(value v_room) {
  FILE *f = fopen("/proc/self/statm", "r");
  if (f == NULL) return Val_int(errno);
  unsigned long pages;
  int got = fscanf(f, "%lu", &pages);
  fclose(f);
  long page = sysconf(_SC_PAGESIZE);
  if (got != 1 || page <= 0) return Val_int(EINVAL);

  struct rlimit l;
  if (getrlimit(RLIMIT_AS, &l) != 0) return Val_int(errno);
  rlim_t n = (rlim_t)pages * (rlim_t)page + (rlim_t)Long_val(v_room);
  if (n < l.rlim_cur) l.rlim_cur = n;
  return Val_int(setrlimit(RLIMIT_AS, &l) == 0 ? 0 : errno);
}

#elif defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>

/* Puts the process in a job that limits its committed memory to [room]
   bytes past what it commits now: 0, or the GetLastError of a refusal. A
   job nested in the job of a limit set outside cannot raise it. The job
   keeps its limit on the process once its handle is closed.
   K32GetProcessMemoryInfo is kernel32's, so the stub links no other system
   library. */
value nx_test_limit_memory(value v_room) {
  PROCESS_MEMORY_COUNTERS c;
  if (!K32GetProcessMemoryInfo(GetCurrentProcess(), &c, sizeof c))
    return Val_int(GetLastError());

  JOBOBJECT_EXTENDED_LIMIT_INFORMATION l = {0};
  l.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
  l.ProcessMemoryLimit = c.PagefileUsage + (SIZE_T)Long_val(v_room);

  HANDLE job = CreateJobObjectW(NULL, NULL);
  if (job == NULL) return Val_int(GetLastError());
  DWORD e = 0;
  if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &l,
                               sizeof l) ||
      !AssignProcessToJobObject(job, GetCurrentProcess()))
    e = GetLastError();
  CloseHandle(job);
  return Val_int(e);
}

#else

#include <errno.h>

/* macOS's limits do not bound allocations; other systems are not
   supported. */
value nx_test_limit_memory(value v_room) {
  (void)v_room;
  return Val_int(ENOSYS);
}

#endif
