/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The floors of the host bench: what linking and calling cost in system
   calls alone. Each holds the runtime but the release floor. */

#define _GNU_SOURCE

#include <stdint.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/mlvalues.h>
#include <caml/threads.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/mman.h>
#endif

#if defined(__APPLE__) && defined(__aarch64__)
#include <pthread.h>
#endif

/* Maps [bytes]' length, copies them in, makes them executable, synchronizes
   the instruction caches with them and unmaps them, as a link does. */
value device_host_bench_map_install_unmap(value v_bytes) {
  size_t n = caml_string_length(v_bytes);
#if defined(_WIN32)
  char *p = VirtualAlloc(NULL, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  DWORD old;
  memcpy(p, Bytes_val(v_bytes), n);
  VirtualProtect(p, n, PAGE_EXECUTE_READ, &old);
  FlushInstructionCache(GetCurrentProcess(), p, n);
  VirtualFree(p, 0, MEM_RELEASE);
#else
  int prot = PROT_READ | PROT_WRITE, flags = MAP_PRIVATE | MAP_ANON;
#if defined(__APPLE__) && defined(__aarch64__)
  prot |= PROT_EXEC;
  flags |= MAP_JIT;
#endif
  char *p = mmap(NULL, n, prot, flags, -1, 0);
#if defined(__APPLE__) && defined(__aarch64__)
  pthread_jit_write_protect_np(0);
  memcpy(p, Bytes_val(v_bytes), n);
  pthread_jit_write_protect_np(1);
#else
  memcpy(p, Bytes_val(v_bytes), n);
  mprotect(p, n, PROT_READ | PROT_EXEC);
#endif
  __builtin___clear_cache(p, p + n);
  munmap(p, n);
#endif
  return Val_unit;
}

/* One lookup of a name in the process, as a link of linked.c makes. */
value device_host_bench_lookup(value unit) {
  (void)unit;
#if defined(_WIN32)
  return Val_long(
      (intnat)GetProcAddress(GetModuleHandleA("ucrtbase.dll"), "cbrt"));
#else
  return Val_long((intnat)dlsym(RTLD_DEFAULT, "cbrt"));
#endif
}

value device_host_bench_release_acquire(value unit) {
  (void)unit;
  caml_release_runtime_system();
  caml_acquire_runtime_system();
  return Val_unit;
}
