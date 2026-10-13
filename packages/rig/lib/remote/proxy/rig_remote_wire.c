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
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "rig_remote_proxy_stubs.h"

#if !defined(_WIN32)
#include <poll.h>
#include <sys/random.h>
#include <sys/time.h>
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

/* Waiting */

#ifdef _WIN32
/* An fd_set of [n] sockets: Winsock's fd_set is a count and an array, which
   FD_SETSIZE sizes only for its macros, so a set of any size is allocated
   to fit and filled by hand. */
static fd_set *socket_set(const SOCKET *s, mlsize_t n) {
  fd_set *set = malloc(sizeof(fd_set) + n * sizeof(SOCKET));
  if (set == NULL) return NULL;
  set->fd_count = (u_int)n;
  memcpy(set->fd_array, s, n * sizeof(SOCKET));
  return set;
}
#endif

/* The index of the first of the descriptors [fds] ready to read, or to
   write if [write], within [ms] milliseconds, -1 for no bound; -1 once [ms]
   passed. A descriptor in error, or whose peer hung up, is ready: the call
   that follows reports it. POSIX waits with poll, which takes descriptors
   of any number and count, where select takes none from FD_SETSIZE.
   Windows waits with select on sockets, whose sets it allocates to fit,
   and which reports a failed connect as an exception. Releases the
   runtime. Raises Unix.Unix_error if the wait fails, EINTR included. */
value caml_rig_remote_wire_wait(value fds, value write, value ms) {
  mlsize_t n = Wosize_val(fds);
  int w = Bool_val(write), t = Int_val(ms);
  if (n == 0) caml_invalid_argument("Rig_remote_proxy.Wire.ready: no socket");
#ifdef _WIN32
  SOCKET *s = malloc(n * sizeof(SOCKET));
  if (s == NULL) caml_raise_out_of_memory();
  for (mlsize_t i = 0; i < n; i++) s[i] = Socket_val(Field(fds, i));
  fd_set *ready = socket_set(s, n), *failed = socket_set(s, n);
  if (ready == NULL || failed == NULL) {
    free(s);
    free(ready);
    free(failed);
    caml_raise_out_of_memory();
  }
  struct timeval bound = {t / 1000, (t % 1000) * 1000};
  caml_release_runtime_system();
  int k = select(0, w ? NULL : ready, w ? ready : NULL, failed,
                 t < 0 ? NULL : &bound);
  int e = WSAGetLastError();
  caml_acquire_runtime_system();
  int r = -1;
  if (k != SOCKET_ERROR)
    for (mlsize_t i = 0; i < n && r < 0; i++)
      if (FD_ISSET(s[i], ready) || FD_ISSET(s[i], failed)) r = (int)i;
  free(s);
  free(ready);
  free(failed);
  if (k == SOCKET_ERROR) rig_remote_raise_error(e, "select");
  return Val_int(r);
#else
  struct pollfd *p = malloc(n * sizeof(struct pollfd));
  if (p == NULL) caml_raise_out_of_memory();
  for (mlsize_t i = 0; i < n; i++)
    p[i] = (struct pollfd){Int_val(Field(fds, i)), w ? POLLOUT : POLLIN, 0};
  caml_release_runtime_system();
  int k = poll(p, (nfds_t)n, t);
  int e = errno;
  caml_acquire_runtime_system();
  int r = -1;
  for (mlsize_t i = 0; k > 0 && i < n && r < 0; i++) {
    if (p[i].revents & POLLNVAL) {
      k = -1;
      e = EBADF;
    } else if (p[i].revents != 0)
      r = (int)i;
  }
  free(p);
  if (k < 0) rig_remote_raise_error(e, "poll");
  return Val_int(r);
#endif
}

/* Bounds each send on the socket [fd] to [s] seconds, [0.] for none. Winsock
   takes the bound as a DWORD of milliseconds where POSIX takes a timeval:
   Unix.setsockopt_float passes a timeval to both, which Windows reads as
   milliseconds from its seconds. */
value caml_rig_remote_wire_send_timeout(value fd, value s) {
  double t = Double_val(s);
#ifdef _WIN32
  DWORD bound = (DWORD)ceil(t * 1e3);
#else
  long long us = (long long)ceil(t * 1e6);
  struct timeval bound = {.tv_sec = (time_t)(us / 1000000),
                          .tv_usec = (suseconds_t)(us % 1000000)};
#endif
  if (setsockopt(rig_remote_sock_val(fd), SOL_SOCKET, SO_SNDTIMEO,
                 (const char *)&bound, sizeof bound) != 0)
    rig_remote_raise_error(rig_remote_sock_error(), "setsockopt");
  return Val_unit;
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
