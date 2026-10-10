/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* kernels.h as the host's C compiler reads it: its rows as written, its
   constants and its structs' fields. */

#include <stddef.h>

#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "kernels.h"

/* Each X(...) of a list: its arguments' text, as "pack, PACK". */
#define ROW(...) #__VA_ARGS__,
static const char *const kernel_rows[] = {NX_AMD_KERNELS(ROW) NULL};
static const char *const tile_rows[] = {NX_AMD_TILES(ROW) NULL};

CAMLprim value nx_amd_test_kernel_rows(value unit) {
  (void)unit;
  return caml_copy_string_array((const char **)kernel_rows);
}

CAMLprim value nx_amd_test_tile_rows(value unit) {
  (void)unit;
  return caml_copy_string_array((const char **)tile_rows);
}

/* A name and up to two numbers: a constant's value, or a field's offset and
   bytes, or a struct's bytes. */
typedef struct {
  const char *name;
  size_t x, y;
} fact;

static const fact constants[] = {
    {"a_vectors", NX_CONTRACT_A_VECTORS, 0},
    {"b_vectors", NX_CONTRACT_B_VECTORS, 0},
    {"y_whole", NX_CONTRACT_Y_WHOLE, 0},
    {"threads", NX_CONTRACT_THREADS, 0}};

/* Every field of a struct must be listed: the test checks that the fields
   tile their struct. */
#define F(s, f) {#s "." #f, offsetof(s, f), sizeof(((s *)0)->f)}

static const fact fields[] = {
    F(contract_params, a),          F(contract_params, b),
    F(contract_params, init),       F(contract_params, y),
    F(contract_params, partials),   F(contract_params, tickets),
    F(contract_params, sa),         F(contract_params, sb),
    F(contract_params, si),         F(contract_params, sy),
    F(contract_params, batch),      F(contract_params, m),
    F(contract_params, n),          F(contract_params, k),
    F(contract_params, splits),     F(contract_params, a_dtype),
    F(contract_params, b_dtype),    F(contract_params, init_dtype),
    F(contract_params, y_dtype),    F(contract_params, acc_dtype),
    F(contract_params, aligned),    F(contract_params, zero),
    F(pack_params, src),            F(pack_params, dst),
    F(pack_params, s),              F(pack_params, lead),
    F(pack_params, batch),          F(pack_params, rows),
    F(pack_params, k),              F(pack_params, dtype),
    F(pack_params, out),            F(pack_params, bytes),
    F(zero_params, p),              F(zero_params, n)};

#define S(s) {#s, sizeof(s), 0}

static const fact structs[] = {S(contract_params), S(pack_params),
                               S(zero_params)};

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

CAMLprim value nx_amd_test_constants(value unit) {
  (void)unit;
  return facts(constants, COUNT(constants));
}

CAMLprim value nx_amd_test_fields(value unit) {
  (void)unit;
  return facts(fields, COUNT(fields));
}

CAMLprim value nx_amd_test_structs(value unit) {
  (void)unit;
  return facts(structs, COUNT(structs));
}
