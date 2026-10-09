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

    [apply1] computes [Copy], [Cast] and [Bitcast] at every dtype and the
    [Unary] kinds at float32, float64 and the 8- to 64-bit integers, and
    answers [Declined] for the others. A [Copy] whose operand's dtype is not
    [dst]'s is refused with [Wrong_dtype].

    [contract] computes contractions in [float32] or [float64] whose [out] is
    their [acc], whose [a], [b] and [init] each convert exactly into [acc],
    and whose layouts {!Nx_kernel.Spec.Contract_view} groups; it answers
    [Declined] for the others. [float32] holds [bool], [bit], the integers of
    at most 16 bits and the floats of at most 32; [float64] also [int32],
    [uint32] and [float64].

    With at least 64 outputs in each batch element, each output adds its
    products in increasing order of the contracting pairs' indices, taken in
    C order, each product fused into its addition, from [init] or [+0]. With
    fewer, each output's products fall into blocks of 1024 consecutive
    indices and, within a block, into 16 lanes by index modulo 16, each lane
    a fused chain from [+0]. Lane [i] takes lane [i + 8] for [i] below 8,
    then lane [i + 4] for [i] below 4, then [i + 2] for [i] below 2, then
    lane 0 takes lane 1. The blocks are summed by a binary tree whose left
    part holds the largest power of two of blocks below their count, each
    part summed the same way; then [init] is added. A NaN result's payload
    is unspecified. *)

include Nx_kernel.S
