/* Copyright (c) 2026 The Raven authors. ISC License. */
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <malloc.h>
#endif

CAMLprim value caml_tolk_host_alloc(value v_size) {
  CAMLparam1(v_size);
  CAMLlocal1(result);
  result = caml_copy_nativeint(0);
  size_t size = (size_t)Long_val(v_size);
#if defined(_WIN32)
  void *ptr = _aligned_malloc(size, 64);
  if (ptr == NULL) {
    caml_failwith("host allocation failed");
  }
  memset(ptr, 0, size);
#else
  void *ptr = NULL;
  if (posix_memalign(&ptr, 64, size == 0 ? 64 : size) != 0) {
    caml_failwith("host allocation failed");
  }
  memset(ptr, 0, size);
#endif
  Nativeint_val(result) = (intnat)ptr;
  CAMLreturn(result);
}

CAMLprim value caml_tolk_host_free(value v_ptr) {
  CAMLparam1(v_ptr);
  void *ptr = (void *)Nativeint_val(v_ptr);
#if defined(_WIN32)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_host_copyin(value v_ptr, value v_bytes) {
  CAMLparam2(v_ptr, v_bytes);
  void *ptr = (void *)Nativeint_val(v_ptr);
  size_t len = (size_t)caml_string_length(v_bytes);
  memcpy(ptr, Bytes_val(v_bytes), len);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_host_copyout(value v_bytes, value v_ptr) {
  CAMLparam2(v_bytes, v_ptr);
  void *ptr = (void *)Nativeint_val(v_ptr);
  size_t len = (size_t)caml_string_length(v_bytes);
  memcpy(Bytes_val(v_bytes), ptr, len);
  CAMLreturn(Val_unit);
}


CAMLprim value caml_tolk_host_view(value v_ptr, value v_len) {
  CAMLparam2(v_ptr, v_len);
  CAMLreturn(caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT, 1,
                              (void *)Nativeint_val(v_ptr), Long_val(v_len)));
}
