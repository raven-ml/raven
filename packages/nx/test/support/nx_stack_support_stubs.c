/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A C frame of a chosen depth under an OCaml callback. OCaml runs its
   calls to C from the C stack as the innermost callback left it, so the
   frame's depth moves every such call's stack address. */

#include <stddef.h>

#include <caml/callback.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#if defined(_WIN32)
#include <malloc.h>
#else
#include <alloca.h>
#endif

value nx_stack_support_deeper(value n, value f) {
  CAMLparam1(f);
  CAMLlocal1(r);
  /* Written before and after the callback, so that no compiler drops the
     allocation or frees it early. */
  volatile char *pad = alloca((size_t)Long_val(n) + 1);
  pad[0] = 0;
  r = caml_callback(f, Val_unit);
  pad[0] = 1;
  CAMLreturn(r);
}
