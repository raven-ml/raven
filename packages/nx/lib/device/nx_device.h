/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* C access to nx.device buffers. */

#ifndef NX_DEVICE_H
#define NX_DEVICE_H

#include <caml/mlvalues.h>

/* The host address of the first byte of the Nx_device.Buffer.t [b], which
   the host addresses, as every host buffer's memory is: its base's memory,
   plus its offset. It reads fields and allocates nothing, so it needs the
   runtime but no rooting, and the address stays valid while [b] is reachable.
   The fields are those of nx_device.ml's [Buffer.t], [base] and [memory],
   whose [host] is [Some] address. */
static inline void *nx_device_buffer_host(value b) {
  value base = Field(b, 0);
  value host = Field(Field(base, 1), 0);
  return (char *)Nativeint_val(Field(host, 0)) + Long_val(Field(b, 1));
}

#endif
