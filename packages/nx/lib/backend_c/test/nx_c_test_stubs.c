/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Test-only introspection stub for the binding's dtype table. Given an
   Nx_backend tensor, it returns what the engine knows of the dtype it reads
   from it: the byte extent of two elements, the class bits, and whether the
   row's category is a signed integer. test_backend_c compares these with
   Nx_dtype for every one of the 19 dtypes. Lives here, not in the library or
   an engine file: it is a binding pinning concern, wired only to the test's
   foreign_stubs. */

#include <caml/alloc.h>
#include <caml/memory.h>

#include "nx_c.h"

static int signed_int(nx_c_dtype dt) {
  static const int sint[NX_C_DTYPE_COUNT] = {
#define SINT_NX_C_CAT_SINT 1
#define SINT_NX_C_CAT_UINT 0
#define SINT_NX_C_CAT_FLOAT 0
#define SINT_NX_C_CAT_COMPLEX 0
#define SINT_NX_C_CAT_BOOL 0
#define SINT_ROW(sfx, storage, compute, ld, st, cat, sel)                      \
  [NX_C_DTYPE_##sfx] = SINT_##cat,
      NX_C_FOR_EACH_DTYPE(SINT_ROW)
#undef SINT_ROW
  };
  return sint[dt];
}

CAMLprim value caml_nx_c_dtype_facts(value v) {
  CAMLparam1(v);
  CAMLlocal1(facts);
  nx_c_dtype dt = nx_c_dtype_of_value(v);
  facts = caml_alloc_tuple(3);
  Store_field(facts, 0, Val_long(nx_c_dtype_bytes(dt, 2)));
  Store_field(facts, 1, Val_int(nx_c_dtype_class(dt)));
  Store_field(facts, 2, Val_bool(signed_int(dt)));
  CAMLreturn(facts);
}
