(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels for AMD GPUs.

    [computes_on d] is [true] iff a queue of [d] runs launches ({!Rig.queues})
    and this library has a code object for [d]'s processor ({!Rig.arch}), which
    loads on [d]: [gfx1201]. The first call for [d], of [computes_on] or of a
    kernel, loads it, and a load that fails answers [false] from [computes_on]
    and [Declined] from a kernel. Raises {!Rig.Lost} if [d] is lost. Kernels
    claim through {!Nx_array.door} until their work is submitted, and answer
    [Done] once it is on [d]'s timeline.

    [apply0] to [apply3], [map], [reduce], [scan], [gather], [scatter],
    [sort], [assemble], [fold], [fft] and [linalg] answer [Declined] for
    every case. [contract] computes plain loads into [float32], [float64]
    and integer accumulators, from operands, [init] and results of a byte or
    more that are not complex, whose layouts {!Nx_kernel.Spec.Contract_view} groups. It
    declines a float operand or [init] wider than a float accumulator, an
    operand or [init] of the other kind than the accumulator, integer or float,
    a batch above 65,535, a row, column or contracted extent above [2{^31} - 1],
    a product of more than [2{^31} - 1] output tiles in a batch element, and an
    operand it would repack whose elements, rows padded to 16 bytes, pass
    [2{^40}]. *)

include Nx_kernel.S
