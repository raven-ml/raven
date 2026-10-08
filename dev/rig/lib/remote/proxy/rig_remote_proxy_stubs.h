/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What the library's stub files share: sockets as C sees them. */

#ifndef RIG_REMOTE_PROXY_STUBS_H
#define RIG_REMOTE_PROXY_STUBS_H

#include <caml/mlvalues.h>
#include <caml/unixsupport.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#include <winsock2.h>
typedef SOCKET rig_remote_sock;
#define rig_remote_sock_val(v) Socket_val(v)
#define RIG_REMOTE_NOSIGNAL 0
#else
#include <sys/socket.h>
typedef int rig_remote_sock;
#define rig_remote_sock_val(v) Int_val(v)
#ifdef MSG_NOSIGNAL
#define RIG_REMOTE_NOSIGNAL MSG_NOSIGNAL
#else
#define RIG_REMOTE_NOSIGNAL 0
#endif
#endif

/* The last socket error of the calling thread. */
static inline int rig_remote_sock_error(void) {
#ifdef _WIN32
  return WSAGetLastError();
#else
  return errno;
#endif
}

/* Keeps sends on [s] from raising SIGPIPE where the system needs a socket
   option for that (macOS); elsewhere sends pass RIG_REMOTE_NOSIGNAL. */
static inline void rig_remote_quiet(rig_remote_sock s) {
#ifdef SO_NOSIGPIPE
  int one = 1;
  (void)setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#else
  (void)s;
#endif
}

/* Sends the [n] bytes at [p] on [s], all of them: 0, or the socket error. It
   blocks and calls nothing of the runtime. */
static inline int rig_remote_send_all(rig_remote_sock s, const void *p,
                                      size_t n) {
  const char *c = p;
  while (n > 0) {
    int chunk = n > (1u << 30) ? (1 << 30) : (int)n;
    long k = (long)send(s, c, chunk, RIG_REMOTE_NOSIGNAL);
    if (k < 0) {
      int e = rig_remote_sock_error();
#ifndef _WIN32
      if (e == EINTR) continue;
#endif
      return e;
    }
    c += k;
    n -= (size_t)k;
  }
  return 0;
}

/* Raises Unix.Unix_error for the socket error [e] of [op]. */
static inline void rig_remote_raise_error(int e, const char *op) {
#ifdef _WIN32
  caml_win32_maperr(e);
#else
  errno = e;
#endif
  caml_uerror(op, Nothing);
}

#endif
