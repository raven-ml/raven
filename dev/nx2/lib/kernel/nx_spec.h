/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Descriptors (Nx_kernel.Spec) as C kernels read them: a family's struct,
   filled by its encoder and read in place. Fields are int32 in the host's
   byte order; entries no attribute uses are zero. Free of OCaml's headers,
   so device code includes it. */

#ifndef NX_SPEC_H
#define NX_SPEC_H

#include <stdint.h>

/* Layout.max_rank. */
#define NX_SPEC_MAX_RANK 32

/* A descriptor's family, its first field. */
enum { NX_SPEC_CONTRACT = 1 };

/* Spec.contract: the pairs of an axis of a and an axis of b, batch then
   contracting, the dtypes the sum runs in and its result has (nx_dtype.h's
   codes), and whether an init operand is given. */
typedef struct {
  int32_t family; /* NX_SPEC_CONTRACT */
  int32_t acc, out;
  int32_t init;
  int32_t nbatch, ncontracting;
  int32_t batch[NX_SPEC_MAX_RANK][2];
  int32_t contracting[NX_SPEC_MAX_RANK][2];
} nx_spec_contract;

#endif
