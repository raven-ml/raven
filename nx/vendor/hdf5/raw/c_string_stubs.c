#include <string.h>
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include "hdf5_caml.h"

value hdf5_c_string_of_string(value s_v)
{
  CAMLparam1(s_v);
  size_t len;
  char *s;

  len = caml_string_length(s_v);
  s = malloc(len + 1);
  if (s == NULL)
    caml_raise_out_of_memory();
  memcpy(s, String_val(s_v), len);
  s[len] = '\0';
  CAMLreturn((value) s);
}

value hdf5_c_string_to_string(char *s)
{
  CAMLparam0();
  CAMLreturn(caml_copy_string(s));
}

value hdf5_c_string_to_bigstring(char *v)
{
  CAMLparam0();
  size_t len;

  len = strnlen(v, MAX_BIGSTRING_LEN);
  if (len == MAX_BIGSTRING_LEN && v[len] != '\0')
    caml_failwith("The given C_string bigger than 1 Gb");
  CAMLreturn(
    caml_ba_alloc_dims(CAML_BA_CHAR | CAML_BA_C_LAYOUT | CAML_BA_MANAGED, 1, v, len));
}
