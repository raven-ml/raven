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
