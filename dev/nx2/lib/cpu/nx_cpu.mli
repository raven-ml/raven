(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels for host memory.

    [computes_on d] is {!Rig.shares_host_memory}[ d]. Each kernel computes from
    operands of [dst]'s shape and any layouts, on memory the host addresses. It
    claims every operand through the door of [nx_array.h] for the extent of its
    call. It answers [Done] once its work is done, or, before any write, a
    refusal: the door's, or [Shape_mismatch] if the operands' shapes differ.

    [apply0] computes [Fill] at every dtype of a byte or more and [Iota] at
    float32, float64 and the 8- to 64-bit integers, and declines the others.
    [apply1] to [apply3] compute every kind at every dtype of its domain. A
    [Copy] whose operand's dtype is not [dst]'s is refused with
    [Wrong_dtype]. [map] computes a program of plain loads whose outputs,
    loads and coordinate axes number at most [NX_MAX_OPERANDS] of
    [nx_array.h], and whose nodes, evaluated in the program's order, hold at
    most 256 values at once, a value held from the node that computes it
    through the last node that reads it, or to the end for an output; it
    answers [Declined] for the others and for a program that bitcasts a
    sub-byte dtype.

    [contract] computes contractions in [float32] or [float64] whose [out] is
    their [acc], whose [a], [b] and [init] each convert exactly into [acc],
    and whose layouts {!Nx_kernel.Spec.Contract_view} groups; it answers
    [Declined] for the others. [float32] holds [bool], [bit], the integers of
    at most 16 bits and the floats of at most 32; [float64] also [int32],
    [uint32] and [float64].

    A float contraction adds in its own order
    ({!Nx_kernel.Spec.section-orders}). With no products, an output is
    [init]'s bits, or [+0]. Otherwise a NaN result's payload is unspecified.

    [reduce] and [scan] compute one [Sum], [Prod], [Max] or [Min] of a
    program's one operand into its own dtype, read plain or padded, at
    [float32], [float64] and the 8- to 64-bit integers, and [Max] and [Min]
    at [bool]; they answer [Declined] for the others. A padded operand is
    first copied padded into host memory, which the call holds until it
    returns. They refuse with [Shape_mismatch] what
    {!Nx_kernel.Spec.shapes} answers [Error] for, and destinations whose
    shapes are not the results'.

    A float sum or product adds each output's terms in lane order
    ({!Nx_kernel.Spec.section-orders}). A scan cuts each slice along its
    axis into chunks of 4096 from its start. A chunk's total is its terms
    reduced in that order. The carry into the first chunk is [+0] ([1]), into
    the next the carry into this one combined with this one's total; each
    result of a chunk is its carry combined with the chunk's terms up to it,
    left to right.

    [gather] and [scatter] compute every case, at every dtype, and refuse with
    [Shape_mismatch] operands and destinations whose shapes do not fit. A
    scatter's [Add] adds a target's updates left to right.

    [sort] computes every dtype of a byte or more and declines the sub-byte
    ones. [fold] computes [float32], [float64] and the 8- to 64-bit integers
    and declines the others. [assemble] computes every dtype. The three
    refuse with [Shape_mismatch] operands and destinations whose shapes do
    not fit.

    [fft] and [linalg] answer [Declined] for the dtypes
    {!Nx_kernel.Spec.dtypes} gives, and [Wrong_dtype] for others. *)

include Nx_kernel.S
