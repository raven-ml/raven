(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** nx's operations as tolk tensor graphs.

    {!op} answers an operation, given the node of each of its operands, with the
    node of each of its results, as {!Lower_arith} holds elements. An operation
    tolk has no direct form for lowers through its expansion
    ({!Nx.Prim.expand}): {!op} says so, and its caller, the interpretation that
    received the operation, expands it, so that each operation of the expansion
    reaches it and lowers in turn.

    {b Numerics.} A result computes what {!Nx.Prim.t} states of its operation,
    to the class {!Lower_arith} and {!Lower_reduce} state. *)

open Tolk

val op :
  ('v 's 'd. ('v, 's, 'd) Nx.t -> Ops.t) -> 'r Nx.Prim.t -> Ops.t array option
(** [op node o] is [Some ys], [ys] the node of each result of [o] in position
    order, its operand [x] read from [node x], or [None] where [o] lowers
    through its expansion: a reduction or a scan of [Logsumexp], [Moments] or
    [Arg]. A [Map] evaluates its program at once; a padded load reads its
    operand padded as {!Nx_kernel.load} states, the fill where it reads no
    element. A [Reduce] and a [Scan] fold their program's outputs; a [Contract]
    of no scale and no segments sums its products in its accumulator; a
    [Copy] is its operand, a [Move] its operand moved, and a [Bitcast] of one
    width its operand's bits.

    Raises [Invalid_argument] naming the operation for one not lowered yet:
    [Gather], [Scatter], [Sort], [Fold], [Assemble], [Fft], [Linalg], [Place]
    and [Check], a scaled or segmented [Contract], a [Bitcast] that changes
    width, a reduction or a scan of a complex [Prod], and a scan of a
    [complex128] [Max] or [Min]. *)
