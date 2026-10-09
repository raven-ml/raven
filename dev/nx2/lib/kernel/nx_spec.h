/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Descriptors (Nx_kernel.Spec) as C kernels read them: a family's struct,
   filled by its encoder and read in place. Fields are int32 in the host's
   byte order, and a trailing array is as long as the counts before it say.
   Free of OCaml's headers, so device code includes it. */

#ifndef NX_SPEC_H
#define NX_SPEC_H

#include <stdint.h>

/* A descriptor's family, its first field. */
enum { NX_SPEC_CONTRACT = 1 };

/* Spec.contract: the dtypes the sum runs in and its result has (nx_dtype.h's
   codes), whether an init operand is given, and the pairs of an axis of a and
   an axis of b: [nbatch] batch pairs, then [ncontracting] contracting
   pairs. */
typedef struct {
  int32_t family; /* NX_SPEC_CONTRACT */
  int32_t acc, out;
  int32_t init;
  int32_t nbatch, ncontracting;
  int32_t pairs[][2];
} nx_spec_contract;

#endif
