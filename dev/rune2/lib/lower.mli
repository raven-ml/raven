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
  ('v 's 'd. ('v, 's, 'd) Nx.t -> Ops.t) -> 'r Nx.Prim.t -> Ops.t array
(** [op node o] is the node of each result of [o] in position order, its
    operand [x] read from [node x]. A [Map] evaluates its program at once; a
    padded load reads its operand padded as {!Nx_kernel.load} states, the fill
    where it reads no element. A [Reduce] and a [Scan] reduce their program's
    outputs as {!Lower_reduce} does; a [Contract] of no scale and no segments
    sums its products in its accumulator; a [Copy] is its operand, a [Move]
    its operand moved, a [Bitcast] of one width its operand's bits, a
    [Place] its operand, as a graph computes on one device, which its caller
    holds every value on. A
    [Gather], a [Scatter], a [Fold] and an [Assemble] compute as
    {!Lower_index} states.

    Raises [Invalid_argument] naming the operation for one not lowered yet:
    [Sort], [Fft], [Linalg] and [Check], a boolean [Add] [Scatter]
    or [Fold], a scaled or segmented [Contract], a [Bitcast]
    that changes width, a reduction or a scan of a complex [Prod], and a scan
    of a [complex128] [Max], [Min] or [Arg]. *)
