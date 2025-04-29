#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <stdint.h>

// Declaration of the CUDA launch function
extern void launch_add_float32(int64_t a_ptr, int64_t b_ptr, int64_t c_ptr,
                               int n);

CAMLprim value caml_cuda_add_float32(value a_val, value b_val, value c_val,
                                     value n_val) {
  CAMLparam4(a_val, b_val, c_val, n_val);

  int64_t a_ptr = Int64_val(a_val);
  int64_t b_ptr = Int64_val(b_val);
  int64_t c_ptr = Int64_val(c_val);
  int n = Int_val(n_val);

  launch_add_float32(a_ptr, b_ptr, c_ptr, n);

  CAMLreturn(Val_unit);
}