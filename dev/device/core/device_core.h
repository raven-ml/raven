/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Reading buffers from C.

   A library that computes on host memory in C, such as nx's CPU kernels,
   reads a buffer (an OCaml value of type Device_core.Buffer.t) through these
   functions. Each reads the value without allocating and may be called with
   the domain lock held only. The address a buffer gives stays valid while the
   value is reachable. */

#ifndef DEVICE_CORE_H
#define DEVICE_CORE_H

#include <stddef.h>
#include <caml/mlvalues.h>

/* The host address of [b]'s first byte, or NULL if the host does not
   address [b]'s memory. */
void *device_core_buffer_host(value b);

/* The number of [b]'s bytes. */
size_t device_core_buffer_bytes(value b);

/* The reason the consumption of [b] gave if [b] is dead, as a C string that
   lives while [b] is reachable; NULL if [b] is live. */
const char *device_core_buffer_why(value b);

#endif
