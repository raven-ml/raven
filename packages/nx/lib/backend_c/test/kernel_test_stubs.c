/* Test-only OCaml wrapper for internal C maintenance hooks. */

#include <stdint.h>

#include <caml/memory.h>
#include <caml/mlvalues.h>

int64_t nx_c_cast_convert_selfcheck(void);

CAMLprim value caml_nx_c_cast_convert_selfcheck_test(value unit) {
  CAMLparam1(unit);
  CAMLreturn(Val_long(nx_c_cast_convert_selfcheck()));
}
