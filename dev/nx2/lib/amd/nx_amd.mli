(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels for AMD GPUs.

    [computes_on d] is [true] iff this library has a code object for [d]'s
    processor ({!Rig.arch}), which loads on [d]: [gfx1201]. The first call
    for [d], of [computes_on] or of a kernel, loads it, and a load that fails
    answers [false] from [computes_on] and [Declined] from a kernel, as every
    later call does. [computes_on d] raises {!Rig.Lost} if [d] is lost.
    Kernels claim through {!Nx_array.door} until their work is submitted, and
    answer [Done] once it is on [d]'s timeline; on a lost [d] they raise
    {!Rig.Lost}, as the door does. A kernel allocates a number of words its
    operands' extents do not change, and keeps none once it returns but the
    caches it fills for later calls on [d].

    [apply0] to [apply3], [map], [reduce], [scan], [gather], [scatter],
    [sort], [assemble], [fold], the transforms and the routines answer
    [Declined] for every case of the dtypes they take. [contract] computes
    into [float32], [float64] and integer accumulators, from operands, [init]
    and results of a byte or more that are not complex, whose layouts
    {!Nx_kernel.Contract_view} groups. It declines a float operand or [init]
    wider than a float accumulator, an operand or [init] of the other kind
    than the accumulator, integer or float, a batch above 65,535, a row,
    column or contracted extent above [2{^31} - 1], a product of more than
    [2{^31} - 1] output tiles in a batch element, and an operand it would
    repack whose elements, rows padded to 16 bytes, pass [2{^40}].

    A float contraction whose own order ({!Nx_kernel.section-orders}) is
    lane order adds in that order, and with no products an output is
    [init]'s bits, or [+0]. The bits of other contractions depend on the
    dtypes and the view's extents alone. A NaN result's payload is
    unspecified. *)

include Nx_kernel.S
