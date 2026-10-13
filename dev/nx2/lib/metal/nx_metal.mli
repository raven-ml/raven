(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels for Apple GPUs.

    [computes_on d] is [true] iff [d]'s {!Rig.arch} is [AppleN], an Apple GPU
    family, and this library's metallib loads on it. The first call
    for [d], of [computes_on] or of a kernel, loads it, and a load that fails
    answers [false] from [computes_on] and [Declined] from a kernel, as every
    later call does. [computes_on d] raises {!Rig.Lost} if [d] is lost.
    Kernels claim through {!Nx_array.door} until their work is submitted, and
    answer [Done] once it is on [d]'s timeline; on a lost [d] they raise
    {!Rig.Lost}, as the door does. A kernel allocates a number of words its
    operands' extents do not change, and keeps none once it returns but the
    caches it fills for later calls on [d].

    Every entry but [contract] answers [Declined] for every case of the
    dtypes it takes. [contract] computes in two families: floats, whose
    operands, [init] and result are float32, float16 or bfloat16, into a
    float32 accumulator; and integers, whose operands, [init], result and
    accumulator are 8- to 64-bit integers, which wrap. [a] and [b] are of one
    dtype, and the layouts are ones {!Nx_kernel.Contract_view} groups. It
    declines the other cases, an extent above [2{^32} - 1], an operand whose
    elements in a batch element lie [2{^32}] elements or more apart, and a
    float operand with neither axis of unit stride.

    [contract_order ~rows ~columns] is [Lane] at 1 row, and [Chain]
    otherwise.

    In chain order, a float sum runs on the matrix units, each output adding
    its products in increasing k, 8 at a time in the GPU's matrix product,
    over one range of k from [0], or ranges whose count the plan chooses
    from the extents, their sums added in increasing k. In lane order, k is
    cut into runs of 8, and strand [s] of 32 adds the products of the runs
    [s], [s + 32], [s + 64], ... in increasing k, by fused multiply-adds;
    the strands then add pairwise, [s] with [s + 1] for even [s], those sums
    2 apart, then 4, 8 and 16. [init] then enters the sum, and the result
    rounds once to [dst]'s dtype. The extents enter a float sum's bits
    through the count of ranges alone, and through the limits past which it
    declines. Another library's sum in the same order may differ in its
    bits. Integer sums wrap, so every order gives them the same bits.

    Float32 and bfloat16 subnormal operands read as a zero of their sign, and
    float32 subnormal products, sums and results store as a zero of their
    sign: Apple GPUs flush them. Float16 keeps its subnormals. The flushing
    widens the contraction's bound on an output by
    [2{^-126} (1 + Σ (1 + |a| + |b|))], the sum over its [K] products'
    operands [a] and [b]. *)

include Nx_kernel.S
