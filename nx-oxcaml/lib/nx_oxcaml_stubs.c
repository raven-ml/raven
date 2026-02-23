#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/fail.h>
#include <caml/bigarray.h>
#include <string.h>

CAMLprim value caml_make_unboxed_float64_vect(value len);
CAMLprim value caml_make_unboxed_float32_vect(value len);
CAMLprim value caml_make_unboxed_int64_vect(value len);
CAMLprim value caml_make_unboxed_int32_vect(value len);


CAMLprim value caml_ba_to_unboxed_float64_array(value v_ba)
{
  CAMLparam1(v_ba);

  struct caml_ba_array *ba = Caml_ba_array_val(v_ba);

  if (ba->num_dims != 1)
    caml_invalid_argument("Bigarray must be 1D");

  if ((ba->flags & CAML_BA_KIND_MASK) != CAML_BA_FLOAT64)
    caml_invalid_argument("Bigarray must be float64");

  mlsize_t len = ba->dim[0];
  void *data = ba->data;

  value arr = caml_make_unboxed_float64_vect(Val_long(len));

  memcpy((double *)arr, (double *)data, len * sizeof(double));

  CAMLreturn(arr);
}

CAMLprim value caml_ba_to_unboxed_float32_array(value v_ba)
{
  CAMLparam1(v_ba);

  struct caml_ba_array *ba = Caml_ba_array_val(v_ba);

  if (ba->num_dims != 1)
    caml_invalid_argument("Bigarray must be 1D");

  if ((ba->flags & CAML_BA_KIND_MASK) != CAML_BA_FLOAT32)
    caml_invalid_argument("Bigarray must be float32");

  mlsize_t len = ba->dim[0];
  void *data = ba->data;

  value arr = caml_make_unboxed_float32_vect(Val_long(len));

  memcpy((float *)arr, (float *)data, len * sizeof(float));

  CAMLreturn(arr);
}

CAMLprim value caml_ba_to_unboxed_int64_array(value v_ba)
{
  CAMLparam1(v_ba);

  struct caml_ba_array *ba = Caml_ba_array_val(v_ba);

  if (ba->num_dims != 1)
    caml_invalid_argument("Bigarray must be 1D");

  if ((ba->flags & CAML_BA_KIND_MASK) != CAML_BA_INT64)
    caml_invalid_argument("Bigarray must be int64");

  mlsize_t len = ba->dim[0];
  void *data = ba->data;

  value arr = caml_make_unboxed_int64_vect(Val_long(len));

  memcpy((int64_t *)arr, (int64_t *)data, len * sizeof(int64_t));

  CAMLreturn(arr);
}

CAMLprim value caml_ba_to_unboxed_int32_array(value v_ba)
{
  CAMLparam1(v_ba);

  struct caml_ba_array *ba = Caml_ba_array_val(v_ba);

  if (ba->num_dims != 1)
    caml_invalid_argument("Bigarray must be 1D");

  if ((ba->flags & CAML_BA_KIND_MASK) != CAML_BA_INT32)
    caml_invalid_argument("Bigarray must be int32");

  mlsize_t len = ba->dim[0];
  void *data = ba->data;

  value arr = caml_make_unboxed_int32_vect(Val_long(len));

  memcpy((int32_t *)arr, (int32_t *)data, len * sizeof(int32_t));

  CAMLreturn(arr);
}

/* ── Unboxed array → Bigarray (to_host direction) ── */

CAMLprim value caml_unboxed_float64_array_to_ba(value v_arr, value v_len)
{
  CAMLparam1(v_arr);
  mlsize_t len = Long_val(v_len);
  void *src = (void *)v_arr;

  intnat dims[1] = { (intnat)len };
  value ba = caml_ba_alloc(CAML_BA_FLOAT64 | CAML_BA_C_LAYOUT, 1, NULL, dims);

  memcpy(Caml_ba_data_val(ba), src, len * sizeof(double));

  CAMLreturn(ba);
}

CAMLprim value caml_unboxed_float32_array_to_ba(value v_arr, value v_len)
{
  CAMLparam1(v_arr);
  mlsize_t len = Long_val(v_len);
  void *src = (void *)v_arr;

  intnat dims[1] = { (intnat)len };
  value ba = caml_ba_alloc(CAML_BA_FLOAT32 | CAML_BA_C_LAYOUT, 1, NULL, dims);

  memcpy(Caml_ba_data_val(ba), src, len * sizeof(float));

  CAMLreturn(ba);
}

CAMLprim value caml_unboxed_int64_array_to_ba(value v_arr, value v_len)
{
  CAMLparam1(v_arr);
  mlsize_t len = Long_val(v_len);
  void *src = (void *)v_arr;

  intnat dims[1] = { (intnat)len };
  value ba = caml_ba_alloc(CAML_BA_INT64 | CAML_BA_C_LAYOUT, 1, NULL, dims);

  memcpy(Caml_ba_data_val(ba), src, len * sizeof(int64_t));

  CAMLreturn(ba);
}

CAMLprim value caml_unboxed_int32_array_to_ba(value v_arr, value v_len)
{
  CAMLparam1(v_arr);
  mlsize_t len = Long_val(v_len);
  void *src = (void *)v_arr;

  intnat dims[1] = { (intnat)len };
  value ba = caml_ba_alloc(CAML_BA_INT32 | CAML_BA_C_LAYOUT, 1, NULL, dims);

  memcpy(Caml_ba_data_val(ba), src, len * sizeof(int32_t));

  CAMLreturn(ba);
}
