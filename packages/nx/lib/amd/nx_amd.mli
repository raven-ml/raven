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
    caches it fills for later calls on [d]. A contraction past one launch's
    extents (65,535 batch elements, [2{^30}] rows or columns, [2{^35}]
    outputs a batch element, or a batch times the larger of rows and columns
    times [k + 16] past [2{^40}]) runs as a launch a window of its results,
    each allocating alike.

    Several domains may call its kernels concurrently.

    [map], [reduce], [scan], [gather], [scatter], [sort], [assemble],
    [fold], the transforms and the routines answer [Declined] for every
    case of the dtypes they take, a scan given a carry included; [map]
    first refuses with [Shape_mismatch] destinations or loads other than its
    program's outputs and operands.

    [contract] computes into [float32], [float64] and integer accumulators,
    from operands, [init] and results of a byte or more that are not
    complex, unscaled and unsegmented, whose layouts group: each operand
    lays out the axes of each kind, batch, rows, columns and contracted, as
    one run of strides once their axes of one element drop out. It declines
    a scaled or segmented call once its shapes fit, a call whose axes do not
    group, a float operand or [init] wider than a float accumulator, an
    operand or [init] of the other kind than the accumulator, integer or
    float, and a contracted extent [k] above [2{^31} - 1].

    A float contraction's order of summation follows its rows, [a]'s free
    extents multiplied, and its columns, [b]'s: {e lane order} at 1 row or
    at 4 columns or fewer, {e chain order} otherwise. Products are numbered
    as {!Nx_kernel.section-contraction} numbers them.

    Lane order puts the products into blocks of 1024 consecutive numbers
    and, within a block, into 16 lanes by number modulo 16, each lane adding
    its products in increasing number from [+0]. Lane [i] takes lane
    [i + 8] for [i] below 8, then lane [i + 4] for [i] below 4, then
    [i + 2] for [i] below 2, then lane 0 takes lane 1, a lane holding no
    product taking no part. The blocks are combined by a binary tree whose
    left part holds the largest power of two of blocks below their count,
    each part combined the same way; then [init] is added, and the result
    rounds once to [dst]'s dtype. An output without products is [init]'s
    bits, or [+0].

    In chain order, an output's products fall into chunks of [c]
    consecutive ones, the last possibly shorter. Each chunk is summed from
    [+0] in increasing number, the chunks' sums are added to [+0] in
    increasing number, then [init] is added, and the result rounds once to
    [dst]'s dtype. Two float16 operands, or two of bfloat16 and the
    float8s, into a float32 accumulator sum on the matrix unit, which adds
    16 products a step as its instruction does; other operands convert
    exactly to the accumulator's dtype and sum by fused multiply-add, one
    product a step. [c = u ⌈⌈k / u⌉ / q⌉] for [q = max 1 (min 16 ⌊k / l⌋)]
    chunks of at least [l] products: [u = 64] and [l = 2048] on the matrix
    unit; [l = 128] otherwise, with [u = 16] for a float32 accumulator and
    [8] for float64. An integer sum wraps, so its bits are the exact sum's
    whatever the order. An output's bits thus depend on the operands'
    values, the dtypes, the order and [k] alone: never otherwise on the
    rows, columns and batch, nor on the layouts or the GPU's size. A box of a result computed alone
    in the whole's order has the whole's bits. *)

include Nx_kernel.S
