(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tangents of scalar programs.

    A {e derivative builder} holds a scalar program ([Nx_kernel.Prog.t]) over
    its operands, each node's tangent over those operands and the tangents of
    some of them, and nodes added after. Operand [i] of the program it builds is
    the program's operand [i]; the tangents' operands come next, in the order of
    the operands that have one; {e extra} operands last. A tangent is a node or
    nothing, which stands for zero: only float and complex nodes have tangents.

    Each kind's tangent is its derivative applied to its operands' tangents
    ({!Nx_kernel.Prog.type-unary} and its siblings). At its kinks: [Abs] at a
    zero, of either sign, passes the tangent; tied operands of [Maximum] and
    [Minimum] take half of each tangent, and a NaN operand neither; a real
    [Sign], [Floor], [Ceil], [Round], [Trunc], the comparisons and [Bitcast]
    give nothing, as do integer and boolean nodes. [Pow]'s tangent along its
    base is zero where its exponent is zero, and along its exponent where its
    base is zero. *)

module P := Nx_kernel.Prog
module D := Nx_array.Dtype

type t
(** The type for derivative builders. *)

val start : P.t -> tangents:bool array -> extra:D.any array -> t
(** [start p ~tangents ~extra] is the builder of [p], operand [i] of [p] having
    a tangent where [tangents.(i)] holds and its dtype is a float or complex
    one, and operands of [extra]'s dtypes after. *)

val ins : t -> D.any array
(** [ins b] is the dtypes of [b]'s operands. *)

val outputs : t -> int
(** [outputs b] is the number of outputs of [b]'s program. *)

val value : t -> int -> int
(** [value b k] is the node of [b]'s output [k]. *)

val tangent : t -> int -> int option
(** [tangent b k] is the node of the tangent of [b]'s output [k]. *)

val operand : t -> int -> int
(** [operand b j] is the node of [b]'s operand [j]. *)

val dtype : t -> int -> D.any
(** [dtype b n] is node [n]'s dtype. *)

(** {1:nodes Nodes}

    Each adds a node to [b] and is its index. *)

val const : t -> D.any -> float -> int
val cast : t -> D.any -> int -> int
val mul : t -> int -> int -> int
val sub : t -> int -> int -> int
val exp : t -> int -> int
val equal : t -> int -> int -> int
val where : t -> int -> int -> int -> int

val finish : t -> outs:int array -> P.t
(** [finish b ~outs] is [b]'s program with outputs [outs], each node it reads
    once. *)

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

    A program is {e linear} in some of its operands, the others {e constants},
    if each output is a sum of products of a constant by one of those operands.
    Its transpose maps a cotangent of each output to one of each such operand:
    [Σ_k <w_k, y_k> = Σ_j <u_j, x_j>] for the pairing [<a, b> = Re (a b)],
    unconjugated on complex numbers. A node of the constants alone is computed
    again in the transpose. *)

exception Nonlinear of string
(** Raised with the kind, as {!Nx.Prim.kind} names it, of the node that applies
    a nonlinear function to a linear operand, as ["Exp"]. *)

val transpose :
  P.t -> linear:bool array -> cts:(int * D.any) array -> P.t option * bool array
(** [transpose p ~linear ~cts] is the transpose of [p] in its operands [j] with
    [linear.(j)]: a program over [p]'s constants, in order, then each cotangent
    [(k, dt)] of [cts], of output [k] in the dtype [dt], in order, cast into the
    output's dtype, whose outputs are the cotangents of the linear operands that
    receive a contribution, in order, and which those are. [None] where none
    receives one. A node's cotangent is the sum of its contributions in the
    order they come, with no zero before the first; the branch of a [Where] its
    condition does not take receives [+0].

    Raises {!Nonlinear} if [p] is not linear in those operands. *)
