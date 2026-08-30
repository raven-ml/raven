/*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*/

/* NVIDIA driver ioctls over caller-built parameter blobs. The OCaml side
   owns every structure layout and request code; these stubs only move the
   blob's address across the syscall boundary.

   The ioctl stub never releases the OCaml runtime: no blocking ioctl exists
   on this path (signal waits spin in userspace), so a release/acquire
   round-trip per call would buy nothing. */

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

CAMLprim value caml_tolk_nv_blob_addr(value v_blob) {
  CAMLparam1(v_blob);
  CAMLreturn(caml_copy_nativeint((intnat)Caml_ba_data_val(v_blob)));
}

#ifdef __linux__

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>

static void raise_errno(const char *what) {
  char buf[256];
  snprintf(buf, sizeof(buf), "%s: %s", what, strerror(errno));
  caml_failwith(buf);
}

CAMLprim value caml_tolk_nv_ioctl(value v_fd, value v_request, value v_blob) {
  CAMLparam3(v_fd, v_request, v_blob);
  int r;
  do
    r = ioctl(Int_val(v_fd), (unsigned long)(uintnat)Long_val(v_request),
              Caml_ba_data_val(v_blob));
  while (r < 0 && errno == EINTR);
  if (r < 0) raise_errno("nv ioctl");
  CAMLreturn(Val_int(r));
}

#else /* !__linux__ */

CAMLprim value caml_tolk_nv_ioctl(value v_fd, value v_request, value v_blob) {
  (void)v_fd;
  (void)v_request;
  (void)v_blob;
  caml_failwith("NV runtime requires Linux");
  return Val_unit; /* unreachable */
}

#endif /* __linux__ */
