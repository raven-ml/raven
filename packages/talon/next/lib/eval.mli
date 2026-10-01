(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Expression evaluation.

    The evaluator computes bound expressions ({!Expr.bind_out},
    {!Expr.bind_predicate}, {!Expr.bind_value}) over a {e frame}: the rows of
    one batch. A row expression gives one value per row of the frame.

    {b Compiled once per step.} [outputs s os], [predicate s p] and [values s e]
    analyse their expressions once, when applied to them: the kernel of each
    node, the casts of its operands, and one slot per subexpression that the
    expressions share ({!Expr.same}), computed once per frame. The resulting
    function is applied to each frame. Every operation is eager nx on the
    frame's columns. A literal is a column of one row that nx broadcasts; an
    output of literals alone is broadcast to the frame's rows, last.

    {b Failures.} A failure is found at a row: a value that [Query.values]
    cannot decode, or an exception that a user function raises. The evaluator
    records the earliest, gives the failing value a null and goes on, so the
    rows before the failing one are computed as if it had not failed. It returns
    the failure with its result. A user function is not called again after it
    raises. *)

type frame
(** The type for frames. *)

val frame : Table.t -> frame
(** [frame b] is the rows of the one-batch table [b]. *)

(** The type for what fails at a row. *)
type cause =
  | Data of string  (** A value the data breaks, with the reason, a phrase. *)
  | Raised of exn * Printexc.raw_backtrace
      (** An exception a user function raised. *)

type failure = { row : int; cause : cause }
(** The type for the earliest failure of a call, at the frame's row [row]. *)

val outputs :
  Schema.t ->
  (string * Expr.packed) list ->
  frame ->
  Column.t list * failure option
(** [outputs s os] compiles the outputs [os], bound to the columns [s], each of
    a column typing. The result maps a frame over [s] to each output's column,
    of the frame's rows, and the failure, if any. *)

val predicate :
  Schema.t -> (bool, Expr.row) Expr.t -> frame -> Nx.bool_t * failure option
(** [predicate s p] compiles [p]; on a frame it is [true] at the rows where [p]
    is [true], and [false] where it is [false] or null. *)

val values :
  Schema.t -> ('a, Expr.row) Expr.t -> frame -> 'a option array * failure option
(** [values s e] compiles [e]; on a frame it is [e] on each row before the
    failure, if any, decoded to OCaml, [None] where it is null, an extension's
    values decoded with its declaration. A value outside the OCaml type is a
    failure at its row ({!Column.decoder}). *)

val not_lowered : string -> 'a
(** [not_lowered what] raises [Invalid_argument] saying that running [what], a
    step or an expression, is not implemented yet. It is the one place a plan
    the planning layer accepts is refused, and it goes with the last unit that
    lowers a step or an expression (RFC issue Core-1). *)
