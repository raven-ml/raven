/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Contraction views filled on the host from a contraction's OCaml values,
   as Nx_kernel.Contract_view.v groups them. */

#ifndef NX_CONTRACT_VIEW_FILL_H
#define NX_CONTRACT_VIEW_FILL_H

#include <caml/mlvalues.h>

#include "nx_contract_view.h"

/* Groups the axes of the arrays [a] and [b], the init [i], an option, and
   the result [y], over the pairs [batch] and [contracting], into [w], and
   answers an nx_array.h code: NX_OK once grouped; NX_DECLINED if a group
   does not merge into one axis, [w] then unspecified; NX_SHAPE if the
   arrays do not fit the pairs, as Contract_view.v answers. It allocates
   nothing and raises nothing. */
int nx_contract_view_fill(nx_contract_view *w, value batch, value contracting,
                          value y, value a, value b, value i);

#endif
