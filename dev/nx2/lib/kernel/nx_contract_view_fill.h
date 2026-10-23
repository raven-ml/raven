/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Contraction views filled on the host from a contraction's OCaml values,
   as Nx_kernel.Contract_view.v groups them, and the runs a group makes
   where its axes do not merge into one. */

#ifndef NX_CONTRACT_VIEW_FILL_H
#define NX_CONTRACT_VIEW_FILL_H

#include <caml/mlvalues.h>

#include "nx_array.h"
#include "nx_contract_view.h"

/* Groups the axes of the arrays [a] and [b], the init [i], an option, and
   the result [y], over the pairs [batch] and [contracting], into [w], and
   answers an nx_array.h code: NX_OK once grouped; NX_DECLINED if a group
   does not merge into one axis, [w] then unspecified; NX_SHAPE if the
   arrays do not fit the pairs, as Contract_view.v answers. It allocates
   nothing and raises nothing. */
int nx_contract_view_fill(nx_contract_view *w, value batch, value contracting,
                          value y, value a, value b, value i);

/* A contraction's groups as runs, indexed as nx_contract_view's arrays:
   each group's axes coalesced over its operands (nx_coalesce_dims) into
   [runs[g]] runs, at least one, in C order. Run [r] of group [g] has
   [extent[g][r]] elements, and operand [o] steps [stride[o][g][r]] along
   it, 0 in a group it lacks and for an absent init. A view is the runs
   where every group has one. */
typedef struct {
  int runs[4];
  int64_t extent[4][NX_MAX_RANK];
  int64_t stride[4][4][NX_MAX_RANK];
  int64_t offset[4];
} nx_contract_runs;

/* An array's layout as nx_array_layout reads it: its rank, its extents
   then its strides, and its first element. */
typedef struct {
  int rank;
  int64_t dim[2 * NX_MAX_RANK];
  int64_t offset;
} nx_layout;

/* Groups as nx_contract_view_fill does, reading each pair once: NX_OK
   with [w] filled, NX_DECLINED with [r] filled where a group makes
   several runs, or NX_SHAPE. It allocates nothing and raises nothing. */
int nx_contract_fill(nx_contract_view *w, nx_contract_runs *r, value batch,
                     value contracting, value y, value a, value b, value i);

/* Groups as nx_contract_fill does, the layouts read already: [y], [a],
   [b] and the init [i], NULL for none, over the [nb] batch pairs then the
   [nc] contracting pairs [pairs], each a's axis then b's. It reads no
   OCaml value. */
int nx_contract_fill_layouts(nx_contract_view *w, nx_contract_runs *r,
                             int nb, int nc, const int (*pairs)[2],
                             const nx_layout *y, const nx_layout *a,
                             const nx_layout *b, const nx_layout *i);

#endif
