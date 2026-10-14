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
    caches it fills for later calls on [d]. The exception is a contraction
    past one launch's bounds: 65,535 batch elements, [2{^30}] rows or
    columns, or [2{^31} - 1] blocks of 4 rows by 32 columns, the skinny
    kernels' blocks, whichever kernel runs. It runs as one launch per window
    of its results, and its words grow with the number of windows.

    [apply0] to [apply3], [map], [gather], [scatter], [sort], [assemble],
    [fold], the transforms and the routines answer [Declined] for every case
    of the dtypes they take.

    [reduce] and [scan] compute one [Sum], [Prod], [Max] or [Min] of output
    [0] of a program that is its one operand, read plain, into that operand's
    dtype, at [float32], [float64] and the 8- to 64-bit integers, and [Max]
    and [Min] at [bool]; they answer [Declined] for the others. [reduce]
    first refuses with [Shape_mismatch] loads whose shape once loaded is not
    its [shape], whatever the case, as nx.cpu does. They refuse with
    [Wrong_dtype] an operand of another dtype than the program's, and with
    [Shape_mismatch] what {!Nx_kernel.Shape.reduce} and
    {!Nx_kernel.Shape.scan} answer [Error] for and destinations whose shapes
    are not the ones they give. They associate as nx.cpu states, so that the
    two give the same bits: in blocks of 1024 terms, 16 lanes a block, the
    lanes' and the blocks' trees, and a scan's chunks of 4096.

    [contract] computes into [float32], [float64] and integer accumulators,
    from operands, [init] and results of a byte or more that are not complex,
    whose layouts {!Nx_kernel.Contract_view} groups. It declines a float
    operand or [init] wider than a float accumulator, an operand or [init] of
    the other kind than the accumulator, integer or float, and a contracted
    extent above [2{^30}].

    [contract_order ~rows ~columns] is [Lane] where [rows] or [columns] are 4
    or fewer, and [Chain] otherwise.

    A float output's association depends on [k], the dtypes and the order
    alone. [k] is cut into [n] ranges of whole runs of 128 terms: of
    [r = ⌈k / 128⌉] runs, [n] is [max 1 (min 16 (r / 256))] in chain order
    and [max 1 (min 16 (r / 16))] in lane order, and range [s] takes the runs
    from [⌊s r / n⌋] to below [⌊(s + 1) r / n⌋]. Each range is summed from
    [+0], the ranges' sums are added in increasing k, [init] then enters the
    sum, and the result rounds once to [dst]'s dtype. Within a range,
    operands each bfloat16 or float8, or both float16, into [float32] sum on
    the matrix units in either order: products in steps of 16 in increasing
    k, the unit summing a step's products and truncating to [float32]. Other
    float sums, in chain order, add each output's products in increasing k,
    one fused multiply-add each. In lane order, lane [l] of 32 adds the
    products [k0 + 128t + 4l + u], [u < 4], of a range from [k0] in
    increasing k, by fused multiply-adds, then lane [l] takes lane [l + 16]
    for [l] below 16, then lane [l + 8] for [l] below 8, and so on to lane 0
    taking lane 1. The tiles, the packs and the launches a call is cut into
    follow its extents and layouts and change no bit: a window of a result,
    computed alone, has the whole's bits. Another library's sum in the same
    order may differ in its bits, and a NaN result's sign and payload are
    unspecified. Integer sums wrap, so every association gives them the same
    bits. *)

include Nx_kernel.S
