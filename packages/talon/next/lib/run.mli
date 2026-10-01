(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Running plans.

    [Talon_next.Query.fold], [run] and [values] document running. This module is
    internal; [Talon_next] exports its functions in [Query].

    A run optimizes its plan ({!Optimize.query}), then compiles each step to a
    stream of one-batch tables that pulls the streams of its inputs, and each
    step's expressions once ({!Eval}). A step that streams transforms one batch
    at a time, in order. A step or an expression that no unit lowers yet is
    refused by {!Eval.not_lowered} before any data is read.

    {b Failures as one row at a time.} A run fails exactly where evaluating its
    optimized plan one row at a time would: at the first row, in that order,
    where a step fails. A streaming step whose evaluation of a batch fails at
    input row [r] emits its output for the rows before [r], then raises the
    failure at the next pull, so a step that needs no more rows never meets it.
    The failure becomes an [Error.t] naming the step ({!Query.pp_step}) and the
    row of its input, counted over the batches before.

    Every stream is closed exactly once: when no more of its rows are needed, or
    when the run ends, fails, or a user function raises. *)

val fold : Query.t -> init:'a -> ('a -> Table.t -> 'a) -> ('a, Error.t) result
val run : Query.t -> (Table.t, Error.t) result
val values : ('a, Expr.row) Expr.t -> Query.t -> ('a array, Error.t) result
