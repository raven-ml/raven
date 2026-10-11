/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Layouts as C reads them: a Layout.t is an OCaml record whose fields are, in
   this order, the extents and the strides (int arrays of one length, the
   rank), the offset, the flags (NX_CONTIGUOUS, NX_DISTINCT, NX_EMPTY), and
   the span's lo and hi, all ints. layout.ml declares the record and
   nx_array's stubs read it; the two change together. Kernels read
   descriptors (nx_array.h).

   The record lives in the OCaml heap, which a collection moves: a stub copies
   what it needs before it allocates or releases the domain lock. */

#ifndef NX_LAYOUT_H
#define NX_LAYOUT_H

enum {
  NX_LAYOUT_SHAPE,
  NX_LAYOUT_STRIDES,
  NX_LAYOUT_OFFSET,
  NX_LAYOUT_FLAGS,
  NX_LAYOUT_LO,
  NX_LAYOUT_HI
};

#endif /* NX_LAYOUT_H */
