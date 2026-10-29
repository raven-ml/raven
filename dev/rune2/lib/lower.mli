(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** nx's operations as tolk tensor graphs.

    {!op} takes the node of each operand of an operation and gives the node of
    each result. A node holds its value's elements as {!Lower_arith} states. The
    graph computes on one device, and the caller must hold every value on it.

    Each operation lowers as follows:
    - A [Map] is its program's expression over its loads ({!Lower_arith.prog}).
      A padded load reads its operand padded as {!Nx_kernel.load} states, and
      gives the fill where it reads no element.
    - A [Reduce] and a [Scan] reduce their program's outputs as {!Lower_reduce}
      does, and a [Contract] sums its products in its accumulator as
      {!Lower_reduce.contract} does.
    - A [Copy] and a [Place] are their operand, a [Move] its operand moved, and
      a [Bitcast] its operand's bits as {!Lower_arith.bitcast} reads them.
    - A [Gather], a [Scatter], a [Fold], an [Assemble] and a [Sort] compute as
      {!Lower_index} states, a [Linalg] as {!Lower_linalg} states, and an [Fft]
      transforms as {!Lower_fft} does.

    {b Numerics.} A result is what {!Nx.Prim.t} states of its operation, within
    the bounds the modules above state. Those bounds implement
    {!Rune2.section-numerics}. *)

open Tolk

val op :
  ?dst:(Ops.t -> Ops.t) ->
  device:Ops.device ->
  host:bool ->
  ('v 's 'd. ('v, 's, 'd) Nx.t -> Ops.t) ->
  'r Nx.Prim.t ->
  Ops.t array
(** [op ~dst ~device ~host node o] is the node of each result of [o], in
    position order, computed on [device], whose kernels compile for the host's
    processors where [host] holds ({!Tolk_engine.target}). It reads each
    operand [x] from [node x]. A [Scatter] takes its storage from [dst] where
    {!Lower_index.scatter} does; another operation ignores it.

    Raises [Invalid_argument], naming the operation, for one it does not lower:
    [Check], which its caller lowers itself, a [Linalg] that {!Lower_linalg.op}
    refuses, a boolean [Add] [Scatter] or [Fold], an [Assemble] that would do
    more than 16 times the work of nx's kernel ({!Lower_index.assemble}), a
    [Sort] along an axis of more than [2{^31}] elements ({!Lower_index.sort}),
    an [Fft] of a precision other than [float32] and [float64] or whose
    result's precision is not its operand's, a [Contract] scaled in an
    accumulator other than [float32] or [float64], a reduction or a scan of a
    complex [Prod], and a scan of a [complex128] [Max], [Min] or [Arg]. *)
