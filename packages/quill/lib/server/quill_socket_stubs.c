/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Socket timeouts in the unit each system takes. */

#include <math.h>

#ifdef _WIN32
#include <winsock2.h>
#else
#include <sys/socket.h>
#include <sys/time.h>
#endif

#include <caml/mlvalues.h>
#include <caml/unixsupport.h>

/* Bounds each read ([opt] is SO_RCVTIMEO) or write (SO_SNDTIMEO) on a socket
   to [s] seconds. Winsock takes the bound as a DWORD of milliseconds where
   POSIX takes a timeval: OCaml's Unix.setsockopt_float passes a timeval to
   both, which Windows reads as milliseconds from its seconds, so 30 s
   meant 30 ms. */
value quill_socket_timeout(value fd, value opt, value s) {
  int o = Int_val(opt) == 0 ? SO_RCVTIMEO : SO_SNDTIMEO;
  double t = Double_val(s);
#ifdef _WIN32
  DWORD ms = (DWORD)ceil(t * 1e3);
  if (setsockopt(Socket_val(fd), SOL_SOCKET, o, (const char *)&ms,
                 sizeof ms) != 0) {
    caml_win32_maperr(WSAGetLastError());
    caml_uerror("setsockopt", Nothing);
  }
#else
  long long us = (long long)ceil(t * 1e6);
  struct timeval tv = {.tv_sec = (time_t)(us / 1000000),
                       .tv_usec = (suseconds_t)(us % 1000000)};
  if (setsockopt(Int_val(fd), SOL_SOCKET, o, &tv, sizeof tv) != 0)
    caml_uerror("setsockopt", Nothing);
#endif
  return Val_unit;
}
