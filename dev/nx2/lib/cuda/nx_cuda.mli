(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels for NVIDIA GPUs.

    [computes_on d] is [true] iff this library has a cubin for [d]'s
    architecture ({!Rig.arch}), which loads on [d]: [sm_89]. The first call
    for [d], of [computes_on] or of a kernel, loads it, and a load that fails
    answers [false] from [computes_on] and [Declined] from a kernel, as every
    later call does. [computes_on d] raises {!Rig.Lost} if [d] is lost.
    Kernels claim through {!Nx_array.door} until their work is submitted, and
    answer [Done] once it is on [d]'s timeline; on a lost [d] they raise
    {!Rig.Lost}, as the door does. A kernel allocates a number of words its
    operands' extents do not change, and keeps none once it returns but the
    caches it fills for later calls on [d].

    [apply0] to [apply3], [map], [gather], [scatter], [sort], [assemble],
    [fold], the transforms and the routines answer [Declined] for every case
    of the dtypes they take.

    [reduce] and [scan] compute one [Sum], [Prod], [Max] or [Min] of output
    [0] of a program that is its one operand, read plain, into that operand's
    dtype, at [float32], [float64] and the 8- to 64-bit integers, and [Max]
    and [Min] at [bool]; they answer [Declined] for the others. They refuse
    with [Wrong_dtype] an operand of another dtype than the program's, and
    with [Shape_mismatch] what {!Nx_kernel.Shape.reduce} and
    {!Nx_kernel.Shape.scan} answer [Error] for and destinations whose shapes
    are not the ones they give. They associate as nx.cpu states, so that the
    two give the same bits: in blocks of 1024 terms, 16 lanes a block, the
    lanes' and the blocks' trees, and a scan's chunks of 4096.

    [contract] computes into [float32], [float64] and integer accumulators,
    from operands, [init] and results of a byte or more that are not complex,
    whose layouts {!Nx_kernel.Contract_view} groups. It declines a float
    operand or [init] wider than a float accumulator, an operand or [init] of
    the other kind than the accumulator, integer or float, a batch above
    65,535, a row, column or contracted extent above [2{^30}], and a product
    of more than [2{^31} - 1] output tiles in a batch element.

    [contract_order ~rows ~columns] is [Lane] where [rows] or [columns] are 4
    or fewer, and [Chain] otherwise.

    Operands each bfloat16 or float8, or both float16, into [float32] sum on
    the matrix units in either order: each output adds its products in steps
    of 16 in increasing k, the unit summing a step's products and truncating
    to [float32]. Other float sums, in chain order, add each output's
    products in increasing k, one fused multiply-add each. In lane order,
    lane [l] of 32 adds the products [k0 + 128t + 4l + u], [u < 4], of a
    range of k from [k0] in increasing k, by fused multiply-adds, then lane
    [l] takes lane [l + 16] for [l] below 16, then lane [l + 8] for [l] below
    8, and so on to lane 0 taking lane 1. A call sums one range of k from
    [0], or ranges whose count the plan chooses from the extents, their sums
    added in increasing k. [init] then enters the sum, and the result rounds
    once to [dst]'s dtype. The extents enter a float sum's bits through that
    count alone, and through the limits past which it declines. Another
    library's sum in the same order may differ in its bits. Integer sums
    wrap, so every order gives them the same bits. *)

include Nx_kernel.S
