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
