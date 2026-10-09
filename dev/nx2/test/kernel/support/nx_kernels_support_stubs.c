/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

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
