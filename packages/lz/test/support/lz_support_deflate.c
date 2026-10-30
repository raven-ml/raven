/* The deflate encoder's paths (lib/deflate_enc.h), which must all write
   the same bytes: their names, and path [i]'s whole-buffer raw deflate
   encode of [src] into [dst] at [level], the encoded length or -1 when it
   does not fit. */

#include <stdint.h>

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "deflate_enc.h"

CAMLprim value lz_support_deflate_enc_paths(value unit) {
  CAMLparam1(unit);
  CAMLlocal2(names, name);
  const char *const *p;
  size_t n = lz_deflate_enc_paths(&p);
  names = caml_alloc(n, 0);
  for (size_t i = 0; i < n; i++) {
    name = caml_copy_string(p[i]);
    Store_field(names, i, name);
  }
  CAMLreturn(names);
}

/* An empty bigarray may have no data; the kernel takes this byte. */
static uint8_t none;

CAMLprim value lz_support_deflate_encode(value i, value level, value src,
                                         value dst) {
  const char *const *names;
  if ((size_t)Long_val(i) >= lz_deflate_enc_paths(&names))
    caml_invalid_argument("Lz_support: no such path");
  size_t src_len = (size_t)Caml_ba_array_val(src)->dim[0];
  size_t dst_cap = (size_t)Caml_ba_array_val(dst)->dim[0];
  struct lz_io io = {
      .src = src_len ? (const uint8_t *)Caml_ba_data_val(src) : &none,
      .src_len = src_len,
      .dst = dst_cap ? (uint8_t *)Caml_ba_data_val(dst) : &none,
      .dst_cap = dst_cap};
  struct lz_deflate_enc st;
  struct lz_error err;
  lz_deflate_enc_init(&st, LZ_RAW, Int_val(level), false);
  lz_deflate_enc_path(&st, (size_t)Long_val(i));
  enum lz_status k = lz_deflate_encode(&st, &io, true, &err);
  lz_deflate_enc_free(&st);
  if (k == LZ_ERROR) caml_raise_out_of_memory();
  return Val_long(k == LZ_DONE ? (long)io.produced : -1);
}
