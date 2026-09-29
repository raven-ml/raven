/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <caml/mlvalues.h>
#include <string.h>

/* Copies between host memory and a [bytes] value, whose bounds the caller
   checked. The runtime stays held: [bytes] lives in the OCaml heap. */

value caml_rune_host_to_bytes(value addr, value bytes, value pos, value len) {
  memcpy(Bytes_val(bytes) + Long_val(pos), (const void *)Nativeint_val(addr),
         (size_t)Long_val(len));
  return Val_unit;
}

value caml_rune_bytes_to_host(value bytes, value pos, value addr, value len) {
  memcpy((void *)Nativeint_val(addr), Bytes_val(bytes) + Long_val(pos),
         (size_t)Long_val(len));
  return Val_unit;
}
