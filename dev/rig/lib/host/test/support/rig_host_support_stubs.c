/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Probes for the host suite and bench. The jobs release the runtime; the
   other probes hold it. */

#define _GNU_SOURCE

#include <stdint.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#else
#include <inttypes.h>
#include <stdio.h>
#endif

#include "rig_pool.h"

value rig_host_test_address(value v_b) {
  return Val_long((intnat)Caml_ba_data_val(v_b));
}

value rig_host_test_machine(value unit) {
  (void)unit;
#if defined(__x86_64__) || defined(_M_X64)
  return caml_copy_string("x86_64");
#elif defined(__aarch64__) || defined(_M_ARM64)
  return caml_copy_string("aarch64");
#else
  return caml_copy_string("unknown");
#endif
}

/* Whether the page holding [v_a] is mapped executable. A page the process
   unmapped may be mapped again at once by any other allocation, so only its
   protection tells code from what replaced it. */
value rig_host_test_executable(value v_a) {
#if defined(_WIN32)
  MEMORY_BASIC_INFORMATION info;
  if (VirtualQuery((void *)Long_val(v_a), &info, sizeof info) == 0)
    return Val_false;
  DWORD x = PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
            PAGE_EXECUTE_WRITECOPY;
  return Val_bool(info.State == MEM_COMMIT && (info.Protect & x) != 0);
#elif defined(__APPLE__)
  mach_vm_address_t at = (mach_vm_address_t)Long_val(v_a);
  mach_vm_size_t size;
  vm_region_basic_info_data_64_t info;
  mach_msg_type_number_t n = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t object;
  kern_return_t r =
      mach_vm_region(mach_task_self(), &at, &size, VM_REGION_BASIC_INFO_64,
                     (vm_region_info_t)&info, &n, &object);
  return Val_bool(r == KERN_SUCCESS &&
                  at <= (mach_vm_address_t)Long_val(v_a) &&
                  (info.protection & VM_PROT_EXECUTE) != 0);
#else
  uintptr_t a = (uintptr_t)Long_val(v_a), lo, hi;
  char perms[5];
  int x = 0;
  FILE *f = fopen("/proc/self/maps", "r");
  if (f == NULL) caml_failwith("rig_host_test_executable: no /proc/self/maps");
  while (fscanf(f, "%" SCNxPTR "-%" SCNxPTR " %4s%*[^\n]", &lo, &hi, perms) ==
         3)
    if (lo <= a && a < hi) {
      x = perms[2] == 'x';
      break;
    }
  fclose(f);
  return Val_bool(x);
#endif
}

static void count(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  int64_t *counts = ctx;
  for (int64_t i = lo; i < hi; i++)
    __atomic_fetch_add(&counts[i], 1, __ATOMIC_RELAXED);
}

static void nothing(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)lo;
  (void)hi;
  (void)worker;
  (void)ctx;
}

/* The counts lie outside the heap, in a bigarray the caller keeps alive. */
value rig_host_test_count_job(value v_threads, value v_total, value v_chunks,
                                 value v_counts) {
  CAMLparam4(v_threads, v_total, v_chunks, v_counts);
  int64_t *counts = Caml_ba_data_val(v_counts);
  int threads = Int_val(v_threads);
  int64_t total = Long_val(v_total), chunks = Long_val(v_chunks);
  caml_release_runtime_system();
  rig_pool_run(threads, total, chunks, count, counts);
  caml_acquire_runtime_system();
  CAMLreturn(Val_unit);
}

value rig_host_test_empty_job(value v_threads, value v_total,
                                 value v_chunks) {
  int threads = Int_val(v_threads);
  int64_t total = Long_val(v_total), chunks = Long_val(v_chunks);
  caml_release_runtime_system();
  rig_pool_run(threads, total, chunks, nothing, NULL);
  caml_acquire_runtime_system();
  return Val_unit;
}
