(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Constants: values of every set, computed where an operation reads them.

    A {e constant} is result [k] of a {e node} ([Value.Deferred]): an
    operation whose every operand is a constant, a creation included, with
    its results' layouts ({!Value.node}). A constant has no bytes and no
    placement, so it takes the brand of what reads it. {!defer} makes the
    constants of a node, and {!at} computes one at the placement an
    operation reads it at.

    Nothing keeps a constant's bytes. Each {!at} computes the nodes its
    constant reads, transitively, each once at each placement that call
    reads it at ({!Prim.evaluate}), and drops their bytes when it returns: a
    second call computes them again.

    A node wanted at [p] computes at its route ({!Route.at}): [p] with the
    axes its rule keeps whole made whole, over operands computed where that
    route reads them. It computes by {!Partial} where the route computes by
    partials and by {!Exec.windows} otherwise. Where the route's result is
    not [p], each device of [p] keeps its window of the results, copied into
    memory of its own, so that it holds no more.

    This module lies above {!Exec} and {!Partial}, which compute its nodes,
    and below {!Check} and {!Eval}, which compute the constants their
    operations read. *)

val defer : by:string -> 'r Value.prim -> 'r
(** [defer ~by op] is [op]'s results as the constants of a new node, after
    checking [op]'s operands against its rule ({!Prim.layouts}). Every
    operand of [op] must be a constant.

    Raises [Invalid_argument] naming [by] where [op]'s operands break its
    rule. *)

val at : 'd Devices.placement -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [at p c] is the constant [c] computed at [p]: each device of [p] holds
    its window of it. A movement or a bitcast of a constant at a placement
    that cuts no axis is a view of its operand computed there. A value that
    is not a constant is returned as it is.

    A kernel's refusal or decline raises here, as {!Exec} states, naming the
    function that made the node it computes. *)
