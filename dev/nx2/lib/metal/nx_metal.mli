(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels for Apple GPUs.

    [computes_on d] is [true] iff a queue of [d] runs launches ({!Rig.queues}),
    [d]'s {!Rig.arch} is [AppleN], an Apple GPU family, and this library's
    metallib loads on it. The first call for [d], of [computes_on] or of a
    kernel, loads it, and a load that fails answers [false] from [computes_on]
    and [Declined] from a kernel. That first call raises {!Rig.Lost} if [d] is
    lost; once it has answered, later calls of [computes_on d] answer the same.
    Kernels claim through {!Nx_array.door} until their work is submitted, and
    answer [Done] once it is on [d]'s timeline; on a lost [d] they raise
    {!Rig.Lost}, as the door does.

    Every entry but [contract] answers [Declined] for every case. [contract]
    computes plain loads in two families: floats, whose operands, [init] and
    result are float32, float16 or bfloat16, into a float32 accumulator; and
    integers, whose operands, [init], result and accumulator are 8- to 64-bit
    integers, which wrap. [a] and [b] are of one dtype, and the layouts are ones
    {!Nx_kernel.Spec.Contract_view} groups. It declines the other cases, an
    extent above [2{^32} - 1], an operand whose elements in a batch element lie
    [2{^32}] elements or more apart, and a float operand with neither axis of
    unit stride. Float32 and bfloat16 subnormal operands read as a zero of
    their sign, and float32 subnormal products, sums and results store as a
    zero of their sign: Apple GPUs flush them. Float16 keeps its subnormals.
    The flushing widens the contraction's bound on an output by
    [2{^-126} (1 + Σ (1 + |a| + |b|))], the sum over its [K] products'
    operands [a] and [b]. *)

include Nx_kernel.S
