(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels for NVIDIA GPUs.

    [computes_on d] is [true] iff a queue of [d] runs launches ({!Rig.queues})
    and this library has a cubin for [d]'s architecture ({!Rig.arch}), which
    loads on [d]: [sm_89]. The first call for [d], of [computes_on] or of a
    kernel, loads it, and a load that fails answers [false] from [computes_on]
    and [Declined] from a kernel. Raises {!Rig.Lost} if [d] is lost. Kernels
    claim through {!Nx_array.door} until their work is submitted, and answer
    [Done] once it is on [d]'s timeline.

    [apply0] to [apply3] and [map] answer [Declined] for every kind.

    [reduce] and [scan] compute one [Sum], [Prod], [Max] or [Min] of a
    program's one operand into its own dtype, read plain, at [float32],
    [float64] and the 8- to 64-bit integers, and [Max] and [Min] at [bool];
    they answer [Declined] for the others. They refuse with [Wrong_dtype] an
    operand or destination of another dtype than the program's, and with
    [Shape_mismatch] what {!Nx_kernel.Spec.shapes} answers [Error] for and
    destinations whose shapes are not the results'. They associate as
    nx.cpu states, so that the two give the same bits: in blocks of 1024
    terms, 16 lanes a block, the lanes' and the blocks' trees, and a scan's
    chunks of 4096.

    [contract]
    computes plain loads into [float32], [float64] and integer accumulators,
    from operands, [init] and results of a byte or more that are not complex,
    whose layouts {!Nx_kernel.Spec.Contract_view} groups. It declines a float
    operand or [init] wider than a float accumulator, an operand or [init] of
    the other kind than the accumulator, integer or float, a batch above 65,535,
    a row, column or contracted extent above [2{^31} - 1], and a product of
    more than [2{^31} - 1] output tiles in a batch element. *)

include Nx_kernel.S
