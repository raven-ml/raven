/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <stdint.h>
#include <string.h>
#include <caml/mlvalues.h>

/* Fills [nbytes] bytes at [addr] with the bytes of [pattern] repeated: the
   first copy, then doubling runs of the filled prefix. [nbytes] is a whole
   number of copies. */
value caml_nx_array_fill(intnat addr, intnat nbytes, value pattern) {
  uint8_t *dst = (uint8_t *)addr;
  size_t width = caml_string_length(pattern);
  size_t n = (size_t)nbytes;
  if (n == 0) return Val_unit;
  if (width == 1) {
    memset(dst, Byte_u(pattern, 0), n);
    return Val_unit;
  }
  memcpy(dst, Bytes_val(pattern), width);
  size_t done = width;
  while (done < n) {
    size_t run = done < n - done ? done : n - done;
    memcpy(dst + done, dst, run);
    done += run;
  }
  return Val_unit;
}

value caml_nx_array_fill_byte(value addr, value nbytes, value pattern) {
  return caml_nx_array_fill((intnat)Nativeint_val(addr), Long_val(nbytes),
                            pattern);
}
