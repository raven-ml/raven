/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What kernels read in C, beside nx_array.h: the code a kernel answers for
   a case it does not compute. Free of OCaml's headers, so device code
   includes it. */

#ifndef NX_SPEC_H
#define NX_SPEC_H

/* Nx_kernel.not_computed: answered before any write or queued work. The
   door's codes are non-negative, so it is none of them. */
#define NX_NOT_COMPUTED (-1)

#endif
