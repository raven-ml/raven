(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Reverse mode through a compiled call, split in two plain functions.

    A function's reverse mode ({!Construct.vjp}) is split at its {e residuals}:
    the values of its forward pass that the transpose of its derivative reads.
    They are fixed by tracing: the forward pass is traced keeping its tape, and
    the tape's transpose is traced against dense cotangents shaped like the real
    or complex results, each value of the forward trace it reads substituted. An
    argument stands for itself, a capture stays a capture, a constant folds, and
    a movement or a placement is made again over the substitution of its
    operand; any other value is a residual, in the order the transpose first
    reads them. Only computed values are results of the forward function.

    Traces are deterministic per dtype, shape and placement of the arguments, so
    a later run of the forward function computes the same values in the same
    order: the forward function returns, as its residuals, the values at the
    positions the residuals took in the traced run. The backward function
    transposes the traced tape under the same substitution, with each residual
    read as the backward function's argument in its place. It never runs the
    function, so each forward operation runs once per call, except where a
    construct's own reverse rule recomputes, as [remat] and [scan] do.

    Both are functions of explicit inputs, which any transformation derives
    from: under a map the residuals carry lanes, and the forward and backward
    derivatives of the backward function differentiate a function of explicit
    residuals. *)

val plan :
  'p Nx.Ptree.t ->
  'q Nx.Ptree.t ->
  ('p, 'q) Construct.vjp ->
  'p ->
  (Construct.axis * int) list * ('p, 'q) Construct.split
(** [plan p q vjp args] is the reverse mode [vjp] split at the residuals that
    tracing it at values like [args] fixes, where [vjp] makes a fresh tape on
    each run and tracks the same leaves, and the lane count of each map the
    traces asked for. The split holds while each map around the call has those
    lane counts.

    The traces answer every construct that no trace stages with its default,
    traced; a total receives nothing. A collective of a map around the call is
    answered at its lane count, so that the traces compute values of the shapes
    a run under that map does, and a gathering of the lanes or a lane's index is
    a value the forward function computes, a residual where the transpose reads
    it.

    Raises {!Lower.Jit_error} when the forward pass or the transpose cannot be
    traced, and as {!Lower.op} does. The forward function raises
    {!Lower.Jit_error} when a run computes values other than the traced run's,
    as a function that is not deterministic does. *)
