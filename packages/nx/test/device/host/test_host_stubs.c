/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <caml/mlvalues.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
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
