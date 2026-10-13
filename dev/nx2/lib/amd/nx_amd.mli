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

    [contract_order ~rows ~columns] is [Lane] where [rows] is 1 or [columns]
    are 4 or fewer, and [Chain] otherwise.

    A float contraction given lane order ({!Nx_kernel.section-orders}) adds
    in that order, and with no products an output is [init]'s bits, or [+0].
    In one given chain order, operands each bfloat16 or float8, or both
    float16, into [float32] sum on the matrix units, each output adding its
    products in steps of 16 in increasing k, one matrix product a step;
    other float sums add each output's products in increasing k, one fused
    multiply-add each. A chain sum covers one range of k from [0], or ranges
    whose count the plan chooses from the extents, their sums added in
    increasing k; [init] then enters the sum, and the result rounds once to
    [dst]'s dtype. The extents enter a chain sum's bits through that count
    alone, and a float sum's through the limits past which it declines.
    Integer sums wrap, so every order gives them the same bits. A NaN
    result's payload is unspecified. *)

include Nx_kernel.S
