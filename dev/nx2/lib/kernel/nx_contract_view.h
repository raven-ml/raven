/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Contraction views as C kernels read them: the start of OCaml bytes,
   which the collector may move, so a kernel copies the view before
   anything that can run OCaml code. Free of OCaml's headers, so device
   code includes it. */

#ifndef NX_CONTRACT_VIEW_H
#define NX_CONTRACT_VIEW_H

#include <stdint.h>

/* Nx_kernel.Contract_view: a contraction's operands and result grouped into four
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
