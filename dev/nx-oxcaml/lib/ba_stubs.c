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

  mlsize_t len = ba->dim[0];

  value arr = caml_make_unboxed_float64_vect(Val_long(len));

  memcpy(
    (double *)arr,
    (double *)ba->data,
    len * sizeof(double)
  );

  CAMLreturn(arr);
}


CAMLprim value
caml_ba_to_unboxed_float32_array(value v_ba)
{
  CAMLparam1(v_ba);

  struct caml_ba_array *ba = Caml_ba_array_val(v_ba);

  /* Validate shape */
  if (ba->num_dims != 1)
    caml_invalid_argument("Bigarray must be 1D");

  /* Validate dtype */
  if ((ba->flags & CAML_BA_KIND_MASK) != CAML_BA_FLOAT32)
    caml_invalid_argument("Bigarray must be float32");

  mlsize_t len = ba->dim[0];

  /* Allocate GC-owned unboxed float32 array */
  value arr = caml_make_unboxed_float32_vect(Val_long(len));

  /* Copy payload */
  memcpy(
    (float *) arr,
    (float *) ba->data,
    len * sizeof(float)
  );

  CAMLreturn(arr);
}

CAMLprim value
caml_ba_to_unboxed_int64_array(value v_ba)
{
  CAMLparam1(v_ba);

  struct caml_ba_array *ba = Caml_ba_array_val(v_ba);

  if (ba->num_dims != 1)
    caml_invalid_argument("Bigarray must be 1D");

  if ((ba->flags & CAML_BA_KIND_MASK) != CAML_BA_INT64)
    caml_invalid_argument("Bigarray must be int64");

  mlsize_t len = ba->dim[0];

  value arr = caml_make_unboxed_int64_vect(Val_long(len));

  memcpy(
    (int64_t *) arr,
    (int64_t *) ba->data,
    len * sizeof(int64_t)
  );

  CAMLreturn(arr);
}

CAMLprim value
caml_ba_to_unboxed_int32_array(value v_ba)
{
  CAMLparam1(v_ba);

  struct caml_ba_array *ba = Caml_ba_array_val(v_ba);

  if (ba->num_dims != 1)
    caml_invalid_argument("Bigarray must be 1D");

  if ((ba->flags & CAML_BA_KIND_MASK) != CAML_BA_INT32)
    caml_invalid_argument("Bigarray must be int32");

  mlsize_t len = ba->dim[0];

  value arr = caml_make_unboxed_int32_vect(Val_long(len));

  memcpy(
    (int32_t *) arr,
    (int32_t *) ba->data,
    len * sizeof(int32_t)
  );

  CAMLreturn(arr);
}