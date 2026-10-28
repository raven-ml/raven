(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** nx's operations as tolk tensor graphs.

    {!op} answers an operation, given the node of each of its operands, with the
    node of each of its results, as {!Lower_arith} holds elements.

    {b Numerics.} A result computes what {!Nx.Prim.t} states of its operation,
    to the class {!Lower_arith} and {!Lower_reduce} state. *)

open Tolk

val op :
  device:Ops.device ->
  ('v 's 'd. ('v, 's, 'd) Nx.t -> Ops.t) ->
  'r Nx.Prim.t ->
  Ops.t array
(** [op ~device node o] is the node of each result of [o] in position order,
    its operand [x] read from [node x], computed on [device], the one device
    the graph computes on. A [Map] evaluates its program at once; a padded
    load reads its operand padded as {!Nx_kernel.load} states, the fill where
    it reads no element. A [Reduce] and a [Scan] reduce their program's
    outputs as {!Lower_reduce} does; a [Contract] sums its products in its
    accumulator as {!Lower_reduce.contract} does; a [Copy] is its operand, a
    [Move] its operand moved, a [Bitcast] its operand's bits as
    {!Lower_arith.bitcast} reads them, and a [Place] its operand, as its caller
    holds every value on [device]. A [Gather], a [Scatter], a [Fold], an
    [Assemble] and a [Sort] compute as {!Lower_index} states, a [Linalg] as {!Lower_linalg}
    states, and an [Fft] transforms as {!Lower_fft} does.

    Raises [Invalid_argument] naming the operation for one not lowered yet:
    [Check], a [Linalg] that {!Lower_linalg} refuses, a boolean [Add]
    [Scatter] or [Fold], a [Scatter] or an [Assemble] that would do more than
    16 times eager's work ({!Lower_index.scatter}, {!Lower_index.assemble}), a
    [Sort] along an axis of more than [2{^31}] elements, an [Fft] of a
    precision other than [float32] and [float64], or whose result's precision
    is not its operand's, a [Contract] scaled in an accumulator other than
    [float32] or [float64], a reduction or a scan of a complex [Prod], and a
    scan of a [complex128] [Max], [Min] or [Arg]. *)
