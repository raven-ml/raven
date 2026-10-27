(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Gathers and scatters as tolk graphs.

    Each function takes nodes as {!Lower_arith} holds elements and computes
    what {!Nx_kernel.S} states of its operation, of static shapes.

    {b Numerics.} A gather and a scatter by [Set] move elements: they give
    eager's bits, a NaN's sign and payload included, and so does a scatter
    where no update lands. A scatter by [Max] or [Min] gives eager's bits but
    for a NaN, of no promised sign or payload, a complex one with two NaN
    parts. A scatter by [Add] wraps integers exactly. Its float sum, of [x] and
    the [n] updates [u1], …, [un] that land on it, adds in the carrier
    ({!Nx_kernel.carrier}) in an association tolk chooses and rounds once: it
    is within [γ(n) (|x| + Σ |ui|)] of the exact sum before that rounding, as
    {!Lower_reduce} states its sums, each part of a complex sum such a sum. It
    is [-0] where [x] and every [ui] are, and [+0] for other terms of zero
    sum. *)

open Tolk

val gather : Nx_array.Dtype.any -> axis:int -> Ops.t -> Ops.t -> Ops.t
(** [gather d ~axis idx x] is the node [x] of a value of [d] read at the
    [int64] positions [idx] along [axis], as {!Nx_kernel.S.gather} states: of
    [idx]'s shape, the element of zero bits where a position lies outside the
    axis. Each element is one indexed load. *)

val scatter :
  Nx_array.Dtype.any ->
  Nx_kernel.combine ->
  axis:int ->
  idx:Ops.t ->
  updates:Ops.t ->
  Ops.t ->
  Ops.t
(** [scatter d c ~axis ~idx ~updates into] is [into], a value of [d], with
    [updates] combined by [c] at the positions [idx] holds along [axis], as
    {!Nx_kernel.S.scatter} states. Each element of [into] meets each update
    along [axis]: [n k] comparisons for [n] positions and [k] updates along
    it. Under [Set], the last update in C order of [updates] wins; a promise of
    distinct targets changes nothing.

    Raises [Invalid_argument] for a boolean [Add]. *)
