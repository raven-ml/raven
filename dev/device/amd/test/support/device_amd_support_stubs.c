/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The GPU lock and host memory, for the AMD suites. */

#define _GNU_SOURCE

#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

/* Takes the lock file at [path] for the life of the process: [false] if
   another process holds it. */
value device_amd_test_lock(value v_path) {
#if defined(_WIN32)
  (void)v_path;
  return Val_false;
#else
  int fd = open(String_val(v_path), O_RDONLY | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0) return Val_false;
  if (flock(fd, LOCK_EX | LOCK_NB) == 0) return Val_true;
  close(fd);
  return Val_false;
#endif
}

value device_amd_test_read(value v_a, value v_n) {
  CAMLparam2(v_a, v_n);
  CAMLreturn(caml_alloc_initialized_string(Long_val(v_n),
                                           (const char *)Nativeint_val(v_a)));
}

value device_amd_test_write(value v_a, value v_s) {
  memcpy((void *)Nativeint_val(v_a), String_val(v_s), caml_string_length(v_s));
  return Val_unit;
}
