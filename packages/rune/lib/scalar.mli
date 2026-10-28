(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tangents, parts and transposes of scalar programs.

    A {e derivative builder} holds a scalar program ([Nx_kernel.Prog.t]), its
    outputs' tangents ({!tangent}), and the nodes added to it since
    ({!section-nodes}). A tangent is a node over the program's operands and the
    tangents of those that have one, or nothing for zero. Only float and complex
    nodes have tangents, and an added node has none. Operand [i] of the program
    a builder builds is the program's operand [i]. The tangents' operands come
    next, in the order of the operands that have one, and the {e extra} operands
    last. Every node, output and operand index given to a builder must be one it
    has.

    Each kind's tangent is its derivative applied to its operands' tangents
    ({!Nx_kernel.Prog.type-unary} and its siblings). At its kinks, [Abs] at a
    zero of either sign passes the tangent. Tied operands of [Maximum] and
    [Minimum] take half of each tangent, and a NaN operand takes neither. A real
    [Sign], [Floor], [Ceil], [Round], [Trunc], the comparisons and [Bitcast]
    have no tangent. [Pow]'s tangent along its base is zero where its exponent
    is zero, and along its exponent where its base is zero. *)

module P := Nx_kernel.Prog
module D := Nx_array.Dtype

type t
(** The type for derivative builders. *)

val start : P.t -> tangents:bool array -> extra:D.any array -> t
(** [start p ~tangents ~extra] is the builder of [p]. Operand [i] of [p] has a
    tangent where [tangents.(i)] holds and its dtype is {!tangible}. [extra]
    gives the extra operands' dtypes. [tangents] must have one element per
    operand of [p]. *)

val ins : t -> D.any array
(** [ins b] is the dtypes of [b]'s operands. *)

val outputs : t -> int
(** [outputs b] is the number of outputs of [b]'s program. *)

val value : t -> int -> int
(** [value b k] is the node of [b]'s output [k]. *)

val tangent : t -> int -> int option
(** [tangent b k] is the node of the tangent of [b]'s output [k], [None] for
    zero. *)

val operand : t -> int -> int
(** [operand b j] adds a node to [b] that reads [b]'s operand [j], and is its
    index. *)

val dtype : t -> int -> D.any
(** [dtype b n] is node [n]'s dtype. *)

(** {1:nodes Nodes}

    Each of {!const} to {!where}, applied to [b], adds a node to [b] and is the
    node's index, as {!operand} does. The node has no tangent. *)

val const : t -> D.any -> float -> int
val cast : t -> D.any -> int -> int
val mul : t -> int -> int -> int
val sub : t -> int -> int -> int
val exp : t -> int -> int
val equal : t -> int -> int -> int
val where : t -> int -> int -> int -> int

val finish : t -> outs:int array -> P.t
(** [finish b ~outs] is [b]'s program with outputs [outs]. It holds the nodes
    these outputs reach, each distinct subexpression once. *)

val tangible : D.any -> bool
(** [tangible dt] is [true] iff [dt] is a float or complex dtype. *)

(** {1:parts Parts of programs} *)

val depends : P.t -> bool array -> bool array
(** [depends p reads] is, for each node of [p], whether it reads, through the
    nodes it is made of, an operand [j] with [reads.(j)]. *)

val reads : P.t -> outs:int array -> bool array
(** [reads p ~outs] is, for each operand of [p], whether one of the outputs
    [outs] reads it. *)

val prune : P.t -> keep:bool array -> outs:int array -> P.t
(** [prune p ~keep ~outs] is [p]'s outputs [outs], in order, over its operands
    [j] with [keep.(j)], in order.

    Raises [Invalid_argument] if one of [outs] reads another operand. *)

(** {1:transposes Transposes}

    A program is {e linear} in some of its operands, the others {e fixed}, if
    each output is a sum of products of a value computed from the fixed operands
    alone by one of the linear operands. Its transpose maps a cotangent [w_k] of
    each output [y_k] to a cotangent [u_j] of each such operand [x_j], such that
    [Σ_k <w_k, y_k> = Σ_j <u_j, x_j>] in the pairing [<a, b> = Re (a b)],
    unconjugated on complex numbers. A node of the fixed operands alone is
    computed again in the transpose. *)

exception Nonlinear of string
(** Raised with the kind, as {!Nx.Prim.kind} names it, of the node that applies
    a nonlinear function to a linear operand, as ["Exp"]. *)

val transpose :
  P.t -> linear:bool array -> cts:(int * D.any) array -> P.t option * bool array
(** [transpose p ~linear ~cts] is the transpose of [p] in its operands [j] with
    [linear.(j)], and for each operand of [p] whether it is linear and receives
    a contribution. The program's operands are [p]'s fixed operands, in order,
    then each cotangent [(k, dt)] of [cts], of output [k] in the dtype [dt], in
    order, cast into output [k]'s dtype. Its outputs are the cotangents of the
    linear operands that receive a contribution, in order. The program is [None]
    where none receives one. A node's cotangent is the sum of its contributions
    in the order they come, with no zero before the first; the branch of a
    [Where] its condition does not take receives [+0].

    Raises {!Nonlinear} if a node that a cotangent of [cts] reaches is not
    linear in those operands. *)
