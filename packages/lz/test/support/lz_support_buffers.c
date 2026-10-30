/* Copies between strings and buffers for Lz_support's of_string,
   to_string and placed: one memcpy where a loop in OCaml moved a byte at
   a time. */

#include <string.h>

#include <caml/bigarray.h>
#include <caml/mlvalues.h>

/* [lz_support_blit s b] copies all of [s] to the start of [b], which holds
   at least as many bytes. */
CAMLprim value lz_support_blit(value s, value b) {
  size_t n = caml_string_length(s);
  if (n > 0) memcpy(Caml_ba_data_val(b), String_val(s), n);
  return Val_unit;
}

/* [lz_support_unblit b d] copies the first [length d] bytes of [b] to
   [d]. */
CAMLprim value lz_support_unblit(value b, value d) {
  size_t n = caml_string_length(d);
  if (n > 0) memcpy(Bytes_val(d), Caml_ba_data_val(b), n);
  return Val_unit;
}
