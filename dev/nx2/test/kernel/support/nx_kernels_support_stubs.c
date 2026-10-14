/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <stdatomic.h>
#include <string.h>

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"

/* Every table, the base first and each later one's instructions a superset
   of the one before. */
static const nx_cpu_target *const tables[] = {
    &nx_cpu_base,
#if defined(__x86_64__)
    &nx_cpu_v3,
#endif
#if defined(__APPLE__) && defined(__aarch64__)
    &nx_cpu_amx,
#endif
};

#define TABLES (int)(sizeof tables / sizeof tables[0])

/* The table the program started with, the best the host runs. */
static const nx_cpu_target *best(void) {
  static const nx_cpu_target *b = NULL;
  if (b == NULL) b = nx_cpu_table;
  return b;
}

value nx_kernels_support_targets(value unit) {
  CAMLparam1(unit);
  CAMLlocal2(l, cell);
  const nx_cpu_target *top = best();
  int last = 0;
  while (tables[last] != top) last++;
  l = Val_emptylist;
  for (int i = last; i >= 0; i--) {
    cell = caml_alloc_small(2, Tag_cons);
    Field(cell, 0) = Val_unit;
    Field(cell, 1) = l;
    l = cell;
    Store_field(cell, 0, caml_copy_string(tables[i]->name));
  }
  CAMLreturn(l);
}

value nx_kernels_support_current(value unit) {
  (void)unit;
  best();
  return caml_copy_string(nx_cpu_table->name);
}

/* A table past the best one may use instructions the host lacks. */
value nx_kernels_support_use(value name) {
  const nx_cpu_target *top = best();
  for (int i = 0; i < TABLES; i++) {
    if (strcmp(tables[i]->name, String_val(name)) == 0) {
      nx_cpu_table = tables[i];
      return Val_unit;
    }
    if (tables[i] == top) break;
  }
  caml_invalid_argument(
      "Nx_kernels_support.with_target: not a target the host runs");
}

/* Jobs begun from jobs' bodies: [n] outer units of [cost] bytes, each
   adding 0 to n − 1 by a job of [n] units of that cost begun in its
   body. */
typedef struct {
  int64_t n, cost;
  _Atomic int64_t sum;
} nested;

static void nested_inner(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  nested *q = ctx;
  for (int64_t i = lo; i < hi; i++) atomic_fetch_add(&q->sum, i);
}

static void nested_outer(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  nested *q = ctx;
  for (int64_t u = lo; u < hi; u++)
    nx_cpu_job(q->n, 0, q->cost, nested_inner, q);
}

value nx_kernels_support_nested(value n, value cost) {
  nested q = {.n = Long_val(n), .cost = Long_val(cost)};
  atomic_init(&q.sum, 0);
  nx_cpu_job(q.n, 0, q.cost, nested_outer, &q);
  return Val_long(atomic_load(&q.sum));
}

/* nx.cpu's matrix routines, their matrices on one thread. */
value nx_kernels_support_serial_cholesky(value t, value dst, value a) {
  return nx_cpu_cholesky_on(Int_val(t) == NX_TRIANGLE_UPPER, dst, a, 1);
}

value nx_kernels_support_serial_lu(value lu, value pivots, value perm,
                                   value a) {
  return nx_cpu_lu_on(lu, pivots, perm, a, 1);
}

value nx_kernels_support_serial_qr(value q, value r, value a) {
  return nx_cpu_qr_on(q, r, a, 1);
}

value nx_kernels_support_serial_solve_triangular(value t, value adjoint,
                                                 value unit, value dst,
                                                 value a, value b) {
  return nx_cpu_solve_triangular_on(Int_val(t) == NX_TRIANGLE_UPPER,
                                    Bool_val(adjoint), Bool_val(unit), dst, a,
                                    b, 1);
}

value nx_kernels_support_serial_solve_triangular_byte(value *argv, int argn) {
  (void)argn;
  return nx_kernels_support_serial_solve_triangular(
      argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

/* Guard pages */

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <caml/bigarray.h>

static intnat page_bytes(void) {
#if defined(_WIN32)
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return (intnat)si.dwPageSize;
#else
  return (intnat)sysconf(_SC_PAGESIZE);
#endif
}

/* [vbytes] bytes, a uint8 bigarray over memory of its own, whose bytes from
   [vfrom] on are readable up to an inaccessible page: their last byte just
   before it where [vafter], else their first byte, [vfrom], just after it.
   The bytes before [vfrom] lie in the page where [vafter] is false. With
   the mapping's start and length, to give to unguard. */
value nx_kernels_support_guarded(value vafter, value vfrom, value vbytes) {
  CAMLparam0();
  CAMLlocal2(ba, r);
  intnat page = page_bytes(), bytes = Long_val(vbytes);
  intnat from = Long_val(vfrom), readable = bytes - from;
  intnat pages = 1 + (readable + page - 1) / page;
  if (pages < 2) pages = 2;
  size_t total = (size_t)(pages * page);
#if defined(_WIN32)
  uint8_t *base = VirtualAlloc(NULL, total, MEM_RESERVE | MEM_COMMIT,
                               PAGE_READWRITE);
  if (base == NULL) caml_raise_out_of_memory();
#else
  uint8_t *base = mmap(NULL, total, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANON, -1, 0);
  if (base == MAP_FAILED) caml_raise_out_of_memory();
#endif
  uint8_t *guard, *data;
  if (Bool_val(vafter)) {
    guard = base + total - page;
    data = guard - bytes;
  } else {
    guard = base;
    data = base + page - from;
  }
#if defined(_WIN32)
  DWORD old;
  VirtualProtect(guard, (size_t)page, PAGE_NOACCESS, &old);
#else
  mprotect(guard, (size_t)page, PROT_NONE);
#endif
  ba = caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT | CAML_BA_EXTERNAL,
                          1, data, bytes);
  r = caml_alloc_tuple(3);
  Store_field(r, 0, ba);
  Store_field(r, 1, caml_copy_nativeint((intnat)base));
  Store_field(r, 2, Val_long((intnat)total));
  CAMLreturn(r);
}

value nx_kernels_support_unguard(value vbase, value vtotal) {
  void *base = (void *)Nativeint_val(vbase);
#if defined(_WIN32)
  (void)vtotal;
  VirtualFree(base, 0, MEM_RELEASE);
#else
  munmap(base, (size_t)Long_val(vtotal));
#endif
  return Val_unit;
}
