/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Test-only introspection stub for the binding's dtype->tag pin. Given an
   Nx_backend tensor, it returns the dtype enum the engine reads from it, so
   test_backend_c can assert that value equals Nx_dtype.t's constructor index
   for every one of the 19 dtypes. Lives here, not in the library or an engine
   file: it is a binding pinning concern, wired only to the test's
   foreign_stubs. nx_c.h supplies nx_c_dtype_of_value as a header inline, so
   this needs no extra link input. */

#include "nx_c.h"

CAMLprim value caml_nx_c_dtype_tag(value v) {
  return Val_int((int)nx_c_dtype_of_value(v));
}
