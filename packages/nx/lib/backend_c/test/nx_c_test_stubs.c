/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Test-only introspection stub for the binding's kind->tag pin. Given a bare
   bigarray (an Nx_backend buffer), it returns the dtype enum nx_c_dtype_of_kind
   maps its kind to, so test_nx_c can assert that value equals
   Dtype.Packed.tag for every one of the 19 dtypes. Independent of the t record
   layout — a pure kind->tag check. Lives here, not in the library or an engine
   file: it is a binding pinning concern, wired only to the test's foreign_stubs.
   nx_c.h supplies both pieces as header inlines (nx_c_dtype_of_kind,
   nx_buffer_get_kind), so this needs no extra link input. */

#include "nx_c.h"

CAMLprim value caml_nx_c_dtype_tag(value vba) {
  return Val_int(
      (int)nx_c_dtype_of_kind(nx_buffer_get_kind(Caml_ba_array_val(vba))));
}
