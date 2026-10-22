/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Nx_kernel's constructors that C reads by tag, as their codes, in the
   order of the constructors. A constant constructor is its index among
   its type's constant constructors, read by Long_val; one with an
   argument is a block whose tag, read by Tag_val, is its index among
   those with one. Free of OCaml's headers, so device code includes it. */

#ifndef NX_KERNEL_H
#define NX_KERNEL_H

/* Nx_kernel.load: blocks. */
enum { NX_LOAD_PLAIN, NX_LOAD_PADDED };

/* Nx_kernel.monoid. */
enum { NX_MONOID_SUM, NX_MONOID_PROD, NX_MONOID_MAX, NX_MONOID_MIN };

/* Nx_kernel.reduction: Monoid and Arg are blocks, Logsumexp and Moments
   constants. */
enum { NX_REDUCTION_MONOID, NX_REDUCTION_ARG };
enum { NX_REDUCTION_LOGSUMEXP, NX_REDUCTION_MOMENTS };

/* Nx_kernel.combine. */
enum { NX_COMBINE_SET, NX_COMBINE_ADD, NX_COMBINE_MAX, NX_COMBINE_MIN };

/* Nx_kernel.triangle. */
enum { NX_TRIANGLE_LOWER, NX_TRIANGLE_UPPER };

#endif
