/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#ifdef __linux__
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

static void fail_errno(const char *what, int e) {
  char msg[512];
  snprintf(msg, sizeof msg, "%s: %s", what, strerror(e));
  caml_failwith(msg);
}
#else
static void no_linux(void) { caml_failwith("NVIDIA GPUs need Linux"); }
#endif

value caml_nx_nv_linux(value unit) {
  (void)unit;
#ifdef __linux__
  return Val_true;
#else
  return Val_false;
#endif
}

value caml_nx_nv_address(value b) {
  return caml_copy_nativeint((intnat)Caml_ba_data_val(b));
}

/* Files */

/* Opens a device file; a GPU's first open initializes it, so the runtime is
   released. */
value caml_nx_nv_open(value path) {
  CAMLparam1(path);
#ifdef __linux__
  char *p = caml_stat_strdup(String_val(path));
  caml_release_runtime_system();
  int fd = open(p, O_RDWR | O_CLOEXEC);
  int e = errno;
  caml_acquire_runtime_system();
  if (fd < 0) {
    char msg[512];
    snprintf(msg, sizeof msg, "%s: %s", p, strerror(e));
    caml_stat_free(p);
    caml_failwith(msg);
  }
  caml_stat_free(p);
  CAMLreturn(Val_int(fd));
#else
  (void)path;
  no_linux();
  CAMLreturn(Val_unit);
#endif
}

value caml_nx_nv_close(value fd) {
#ifdef __linux__
  close(Int_val(fd));
#else
  (void)fd;
#endif
  return Val_unit;
}

/* Runs the ioctl [request] on [fd] with the argument [b]'s memory, which may
   point to other memory the caller keeps alive, retrying an interrupted call.
   Raises [Failure] naming [what] if it fails. */
value caml_nx_nv_ioctl(value fd, value request, value b, value what) {
  CAMLparam4(fd, request, b, what);
#ifdef __linux__
  int f = Int_val(fd);
  unsigned long req = (unsigned long)Long_val(request);
  void *arg = Caml_ba_data_val(b);
  int r;
  caml_release_runtime_system();
  do r = ioctl(f, req, arg);
  while (r < 0 && (errno == EINTR || errno == EAGAIN));
  int e = r < 0 ? errno : 0;
  caml_acquire_runtime_system();
  if (e) fail_errno(String_val(what), e);
  CAMLreturn(Val_unit);
#else
  (void)fd;
  (void)request;
  (void)b;
  (void)what;
  no_linux();
  CAMLreturn(Val_unit);
#endif
}

/* Mappings */

/* Maps [n] bytes at [at], readable and writable and shared: of the file [fd]
   from its start, or anonymous memory if [fd < 0]. */
value caml_nx_nv_map(value fd, value at, value n) {
#ifdef __linux__
  int f = Int_val(fd);
  void *want = (void *)Nativeint_val(at);
  size_t len = Long_val(n);
  caml_release_runtime_system();
  void *p = mmap(want, len, PROT_READ | PROT_WRITE,
                 MAP_SHARED | MAP_FIXED | (f < 0 ? MAP_ANONYMOUS : 0), f, 0);
  int e = errno;
  caml_acquire_runtime_system();
  if (p == MAP_FAILED) fail_errno("mapping memory for the GPU", e);
  return Val_unit;
#else
  (void)fd;
  (void)at;
  (void)n;
  no_linux();
  return Val_unit;
#endif
}

/* Returns [n] bytes at [at] to the inaccessible reservation they came from. */
value caml_nx_nv_release(value at, value n) {
#ifdef __linux__
  void *p = mmap((void *)Nativeint_val(at), Long_val(n), PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1,
                 0);
  if (p == MAP_FAILED) fail_errno("releasing GPU addresses", errno);
  return Val_unit;
#else
  (void)at;
  (void)n;
  no_linux();
  return Val_unit;
#endif
}
