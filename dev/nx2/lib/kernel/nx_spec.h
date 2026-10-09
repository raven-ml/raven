/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Descriptors (Nx_kernel.Spec) and contraction views as C kernels read
   them. A descriptor is its family's struct, int32 fields in the host's byte
   order, a trailing array as long as the counts before it say; a view is
   int64. Each is the start of an OCaml string or bytes, which the collector
   may move: a kernel copies what it reads into its own memory before
   anything that can run OCaml code, nx_read included. Free of OCaml's
   headers, so device code includes it. */

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

/* Spec.Contract_view: a contraction's operands and result grouped into four
   axes, as Contract_view.fill leaves them, its arrays indexed by the
   operands and axes below. [offset] is each operand's first element and
   [stride] its stride along each axis, in elements; an axis an operand
   lacks, and an absent Init, read 0. Kernels write
   v.stride[NX_VIEW_A][NX_VIEW_CONTRACTED]. */
enum { NX_VIEW_A, NX_VIEW_B, NX_VIEW_INIT, NX_VIEW_DST };
enum { NX_VIEW_BATCH, NX_VIEW_ROW, NX_VIEW_COLUMN, NX_VIEW_CONTRACTED };

typedef struct {
  int64_t extent[4];    /* by axis */
  int64_t offset[4];    /* by operand */
  int64_t stride[4][4]; /* by operand, then axis */
} nx_contract_view;

#endif
