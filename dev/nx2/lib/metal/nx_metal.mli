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
    declines the other cases and a [k] above [2{^31} - 1]: over layouts it
    groups, whether it declines depends on the dtypes and [k] alone.

    [contract_order ~rows ~columns] is [Lane] at 1 row, and [Chain]
    otherwise.

    A float output's association depends on [k], the dtypes and the order
    alone, never on the batch, rows, columns or layouts, so a window of a
    result computed alone has the bits the whole result gives it. An
    output's terms are numbered by their index along k, from [0]:
    {ul
    {- In chain order the terms are added in increasing number, eight at a
       time in the GPU's 8 × 8 matrix product, from [+0]. A float16
       product's operands enter the matrix product as float16, the other
       dtypes' as float32.}
    {- In lane order, the terms are cut into runs of 8, and strand [s] of
       32 adds, from [+0], the terms of the runs [s], [s + 32], [s + 64],
       ... in increasing number, by fused multiply-adds; the strands then add
       pairwise, [s] with [s + 1] for even [s], those sums 2 apart, then 4,
       8 and 16.}}
    [init] then enters the sum, and the result rounds once to [dst]'s
    dtype. Another library's sum in the same order may differ in its bits.
    Integer sums wrap, so every order gives them the same bits.

    Float32 and bfloat16 subnormal operands read as a zero of their sign, and
    float32 subnormal products, sums and results store as a zero of their
    sign: Apple GPUs flush them. Float16 keeps its subnormals. The flushing
    widens the contraction's bound on an output by
    [2{^-126} (1 + Σ (1 + |a| + |b|))], the sum over its [K] products'
    operands [a] and [b]. *)

include Nx_kernel.S
