/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  MD5 fed in pieces, for nx.io's C files. A context is NX_IO_MD5_CONTEXT
  bytes, which nx_io_md5.c checks against the runtime's own; its owner
  keeps it in memory of its own, and the functions take no lock.
  ---------------------------------------------------------------------------*/

#ifndef NX_IO_MD5_H
#define NX_IO_MD5_H

#include <stddef.h>

#define NX_IO_MD5_CONTEXT 88

/* [nx_io_md5_update ctx p n] adds the [n] bytes at [p] to the digest
   [ctx]. */
void nx_io_md5_update(unsigned char *ctx, const unsigned char *p, size_t n);

#endif
