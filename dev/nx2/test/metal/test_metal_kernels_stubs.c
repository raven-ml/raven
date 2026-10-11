/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* kernels.h as the host's C compiler reads it: its kernels' names, its
   constants and its structs' fields. */

#include <stddef.h>

#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "kernels.h"
#include "nx_dtype.h"

/* Each X(name) of the list: the name. */
#define ROW(name) #name,
static const char *const kernel_rows[] = {NX_METAL_KERNELS(ROW) NULL};

CAMLprim value nx_metal_test_kernel_rows(value unit) {
  (void)unit;
  return caml_copy_string_array((const char **)kernel_rows);
}

/* A name and up to two numbers: a constant's value, or a field's offset and
   bytes, or a struct's bytes. */
typedef struct {
  const char *name;
  size_t x, y;
} fact;

static const fact constants[] = {
    {"threads", NX_METAL_THREADS, 0},     {"large", NX_METAL_LARGE, 0},
    {"small", NX_METAL_SMALL, 0},         {"wide_m", NX_METAL_WIDE_M, 0},
    {"wide_n", NX_METAL_WIDE_N, 0},       {"bk_half", NX_METAL_BK_HALF, 0},
    {"bk", NX_METAL_BK, 0},               {"bk_wide", NX_METAL_BK_WIDE, 0},
    {"skinny_t", NX_METAL_SKINNY_T, 0},   {"skinny_n", NX_METAL_SKINNY_N, 0},
    {"int_tile", NX_METAL_INT_TILE, 0},   {"int_threads", NX_METAL_INT_THREADS, 0},
    {"a_t", NX_METAL_A_T, 0},             {"b_t", NX_METAL_B_T, 0},
    {"no_init", NX_DTYPE_COUNT, 0}};

/* Every field of a struct must be listed: the test checks that the fields
   tile their struct. */
#define F(s, f) {#s "." #f, offsetof(s, f), sizeof(((s *)0)->f)}

static const fact fields[] = {
    F(nx_metal_contract, a),          F(nx_metal_contract, b),
    F(nx_metal_contract, init),       F(nx_metal_contract, out),
    F(nx_metal_contract, a_batch),    F(nx_metal_contract, b_batch),
    F(nx_metal_contract, init_batch), F(nx_metal_contract, a_m),
    F(nx_metal_contract, a_k),        F(nx_metal_contract, b_k),
    F(nx_metal_contract, b_n),        F(nx_metal_contract, init_m),
    F(nx_metal_contract, init_n),     F(nx_metal_contract, batch),
    F(nx_metal_contract, m),          F(nx_metal_contract, n),
    F(nx_metal_contract, k),          F(nx_metal_contract, init_dtype),
    F(nx_metal_contract, out_dtype),  F(nx_metal_contract, swizzle),
    F(nx_metal_contract, dtype),      F(nx_metal_contract, acc),
    F(nx_metal_contract, order),      F(nx_metal_combine, out),
    F(nx_metal_combine, parts),       F(nx_metal_combine, init),
    F(nx_metal_combine, init_batch),  F(nx_metal_combine, init_m),
    F(nx_metal_combine, init_n),      F(nx_metal_combine, batch),
    F(nx_metal_combine, m),           F(nx_metal_combine, n),
    F(nx_metal_combine, split),       F(nx_metal_combine, init_dtype),
    F(nx_metal_combine, out_dtype)};

#define S(s) {#s, sizeof(s), 0}

static const fact structs[] = {S(nx_metal_contract), S(nx_metal_combine)};

#define COUNT(xs) (sizeof xs / sizeof xs[0])

/* [n] facts as an array of triples (name, x, y). */
static value facts(const fact *fs, size_t n) {
  CAMLparam0();
  CAMLlocal3(r, t, s);
  r = caml_alloc_tuple(n);
  for (size_t i = 0; i < n; i++) {
    s = caml_copy_string(fs[i].name);
    t = caml_alloc_tuple(3);
    Store_field(t, 0, s);
    Store_field(t, 1, Val_long(fs[i].x));
    Store_field(t, 2, Val_long(fs[i].y));
    Store_field(r, i, t);
  }
  CAMLreturn(r);
}

CAMLprim value nx_metal_test_constants(value unit) {
  (void)unit;
  return facts(constants, COUNT(constants));
}

CAMLprim value nx_metal_test_fields(value unit) {
  (void)unit;
  return facts(fields, COUNT(fields));
}

CAMLprim value nx_metal_test_structs(value unit) {
  (void)unit;
  return facts(structs, COUNT(structs));
}
