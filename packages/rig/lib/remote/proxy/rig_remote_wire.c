/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The handshake's system calls: sends that raise no SIGPIPE, and nonces from
   the system's random source. */

#define _GNU_SOURCE

#ifdef _WIN32
#define _CRT_RAND_S
#endif

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <caml/unixsupport.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "rig_remote_proxy_stubs.h"

#if !defined(_WIN32)
#include <sys/random.h>
#include <unistd.h>
#endif

#define NONCE 32

/* Sends all of [s] on the socket [fd]. Releases the runtime. Raises
   Unix.Unix_error if the stream fails. */
value caml_rig_remote_wire_send(value fd, value s) {
  CAMLparam2(fd, s);
  size_t n = caml_string_length(s);
  char *b = malloc(n ? n : 1);
  if (b == NULL) caml_raise_out_of_memory();
  memcpy(b, String_val(s), n);
  rig_remote_sock k = rig_remote_sock_val(fd);
  rig_remote_quiet(k);
  caml_release_runtime_system();
  int err = rig_remote_send_all(k, b, n);
  caml_acquire_runtime_system();
  free(b);
  if (err != 0) rig_remote_raise_error(err, "send");
  CAMLreturn(Val_unit);
}

/* [NONCE] bytes from the system's random source. Does not release the
   runtime: the call returns at once. */
value caml_rig_remote_wire_nonce(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(r);
  unsigned char b[NONCE];
#if defined(_WIN32)
  for (int i = 0; i < NONCE; i += 4) {
    unsigned int x;
    errno_t e = rand_s(&x);
    if (e != 0) {
      errno = e;
      caml_uerror("rand_s", Nothing);
    }
    memcpy(b + i, &x, 4);
  }
#else
  if (getentropy(b, NONCE) != 0) caml_uerror("getentropy", Nothing);
#endif
  r = caml_alloc_initialized_string(NONCE, (const char *)b);
  CAMLreturn(r);
}
