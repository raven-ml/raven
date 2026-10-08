/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A descriptor's number, which the launchers pass to the processes they
   start in RIG_REMOTE_REPORT. It holds the runtime: it does not block. */

#define _GNU_SOURCE
#include <caml/mlvalues.h>

CAMLprim value caml_rig_bin_fd_number(value fd)
{
  return fd;
}
