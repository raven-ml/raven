(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Programs built node by node.

    A builder appends nodes and gives each its index, which later nodes refer
    to. Its operands are the inputs it pushes, in order: each states its dtype
    once, where the program reads it. *)

type t
(** The type for programs under construction: their operands' dtypes and
    their nodes, in order. *)

val make : unit -> t
(** [make ()] is a builder of no operand and no node. *)

val input : t -> Nx_array.Dtype.any -> int
(** [input b dt] pushes onto [b] the node [In k] of its next operand [k], of
    dtype [dt], and is its index. *)

val push : t -> Nx_kernel.Prog.node -> int
(** [push b n] appends [n] to [b]'s nodes and is its index.

    Raises [Invalid_argument] if [n] is an [In]: operands come from
    {!input}. *)

val const : t -> ('v, 's) Nx_array.Dtype.t -> 'v -> int
(** [const b dt x] pushes onto [b] the [Const] of [dt] whose bits are
    {!Nx_kernel.Prog.bits}[ dt x], and is its index.

    Raises [Invalid_argument] as {!Nx_kernel.Prog.bits} does. *)

val program : t -> int iarray -> Nx_kernel.Prog.t
(** [program b outs] is {!Nx_kernel.Prog.v}[ ~ins nodes ~outs], [ins] being
    [b]'s operands' dtypes and [nodes] its nodes.

    Raises [Invalid_argument] as {!Nx_kernel.Prog.v} does. *)
