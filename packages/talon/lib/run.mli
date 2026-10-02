(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Running plans.

    [Talon.Query.fold], [run] and [values] document running. This module is
    internal; [Talon] exports its functions in [Query].

    A run optimizes its plan ({!Optimize.query}), then compiles each step to a
    stream of one-batch tables that pulls the streams of its inputs, and each
    step's expressions once ({!Eval}). A step that streams transforms one batch
    at a time, in order. A step that blocks (a [sort], an [aggregate], and a
    [select], [derive] or [filter] whose expressions read other rows than their
    own) pulls its input to its end, concatenates it and computes once; a slice
    from the start over a [sort] is one step, a top-k ({!Order_run.top_k}). A
    join blocks on both inputs, the left pulled first, except a join on
    [Join.all], which pulls its right to its end and then streams its left. A
    source's parts are opened in order, each when the one before ends, and its
    batches are checked against the request and the source's order.

    {b Failures as one row at a time.} A run fails exactly where evaluating its
    optimized plan one row at a time would: at the first row, in that order,
    where a step fails. A streaming step whose evaluation of a batch fails at
    input row [r] emits its output for the rows before [r], then raises the
    failure at the next pull, so a step that needs no more rows never meets it.
    A step that blocks emits nothing when it fails: its first row may depend on
    its last. A step that the optimized plan reaches more than once runs once,
    through a {!Tee}: it computes as far as its furthest reader has pulled, so a
    failure past every reader's need is never met. The failure becomes an
    [Error.t] naming the step ({!Query.pp_step}) and the row of its input,
    counted over the batches before.

    Every stream and every source reader is closed exactly once: when no more of
    its rows are needed, or when the run ends, fails, or a user function raises.
*)

val fold : Query.t -> init:'a -> ('a -> Table.t -> 'a) -> ('a, Error.t) result
val run : Query.t -> (Table.t, Error.t) result
val values : ('a, Expr.row) Expr.t -> Query.t -> ('a array, Error.t) result
