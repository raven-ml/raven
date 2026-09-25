/* Copyright (c) 2026 The Raven authors. ISC License. */
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <string.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

CAMLprim value caml_tolk_host_alloc(value v_size) {
  CAMLparam1(v_size);
  CAMLlocal1(result);
  result = caml_copy_nativeint(0);
  size_t size = (size_t)Long_val(v_size);
#if defined(_WIN32)
  void *ptr = VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (ptr == NULL) {
    caml_failwith("host allocation failed");
  }
#else
  void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON, -1, 0);
  if (ptr == MAP_FAILED) {
    caml_failwith("host allocation failed");
  }
#endif
  Nativeint_val(result) = (intnat)ptr;
  CAMLreturn(result);
}

CAMLprim value caml_tolk_host_free(value v_ptr, value v_size) {
  CAMLparam2(v_ptr, v_size);
  void *ptr = (void *)Nativeint_val(v_ptr);
#if defined(_WIN32)
  if (!VirtualFree(ptr, 0, MEM_RELEASE)) caml_failwith("host release failed");
#else
  if (munmap(ptr, (size_t)Long_val(v_size)) != 0)
    caml_failwith("host release failed");
#endif
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_host_copyin(value v_ptr, value v_bytes,
                                    value v_offset, value v_length) {
  CAMLparam4(v_ptr, v_bytes, v_offset, v_length);
  void *ptr = (void *)Nativeint_val(v_ptr);
  intnat offset = Long_val(v_offset), length = Long_val(v_length);
  size_t size = caml_string_length(v_bytes);
  if (offset < 0 || length < 0 || (uintnat)offset > size ||
      (uintnat)length > size - (uintnat)offset)
    caml_invalid_argument("host copy source range is outside bytes");
  if (length != 0) memcpy(ptr, Bytes_val(v_bytes) + offset, (size_t)length);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_host_copyout(value v_ptr, value v_bytes,
                                     value v_offset, value v_length) {
  CAMLparam4(v_ptr, v_bytes, v_offset, v_length);
  void *ptr = (void *)Nativeint_val(v_ptr);
  intnat offset = Long_val(v_offset), length = Long_val(v_length);
  size_t size = caml_string_length(v_bytes);
  if (offset < 0 || length < 0 || (uintnat)offset > size ||
      (uintnat)length > size - (uintnat)offset)
    caml_invalid_argument("host copy destination range is outside bytes");
  if (length != 0) memcpy(Bytes_val(v_bytes) + offset, ptr, (size_t)length);
  CAMLreturn(Val_unit);
}


CAMLprim value caml_tolk_host_view(value v_ptr, value v_len) {
  CAMLparam2(v_ptr, v_len);
  CAMLreturn(caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT, 1,
                              (void *)Nativeint_val(v_ptr), Long_val(v_len)));
}
