/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* BLAKE2b with a key and fed in pieces, from the runtime's implementation,
   which Stdlib's Digest.BLAKE256 exposes only whole and without a key. */

#define CAML_INTERNALS

#include <caml/alloc.h>
#include <caml/blake2.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

/* BLAKE2b */

/* A digest in progress lives in a bytes value the size of the runtime's
   context. Its fields are integers and bytes alone, so the GC may move it
   between calls. */

CAMLprim value caml_munin_blake2_size(value unit) {
  (void)unit;
  return Val_long(sizeof(struct BLAKE2_context));
}

CAMLprim value caml_munin_blake2_init(value ctx, value hashlen, value key) {
  caml_BLAKE2Init((struct BLAKE2_context *)Bytes_val(ctx), Long_val(hashlen),
                  caml_string_length(key), (const unsigned char *)String_val(key));
  return Val_unit;
}

CAMLprim value caml_munin_blake2_update(value ctx, value buf, value off,
                                        value len) {
  caml_BLAKE2Update((struct BLAKE2_context *)Bytes_val(ctx),
                    (const unsigned char *)Bytes_val(buf) + Long_val(off),
                    Long_val(len));
  return Val_unit;
}

CAMLprim value caml_munin_blake2_final(value ctx, value hashlen) {
  CAMLparam2(ctx, hashlen);
  CAMLlocal1(res);
  unsigned char out[64];
  caml_BLAKE2Final((struct BLAKE2_context *)Bytes_val(ctx), Long_val(hashlen),
                   out);
  res = caml_alloc_initialized_string(Long_val(hashlen), (const char *)out);
  CAMLreturn(res);
}
