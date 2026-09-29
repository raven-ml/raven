/* The benchmark's OCaml wrapper for the owned GEMM. */

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_c_matmul.h"

CAMLprim value caml_nx_c_owned_matmul(value vout, value va, value vb) {
  CAMLparam3(vout, va, vb);
  nx_c_matmul_owned(vout, va, vb);
  CAMLreturn(Val_unit);
}
