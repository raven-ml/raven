(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Composites: the arguments a composite operation's caller passed it, so
    that a refusal from inside names them.

    A composite computes by other operations, over values it derives from
    its arguments by movements, maps, folds and the rest. Where one of those
    operations refuses an operand ({!Route.route}), its remedy would place a
    value the caller never made. A composite declares its arguments by name
    for the extent of its body ({!call}). Each operation in that extent
    notes how its results derive from its operands that derive from an
    argument ({!note}), and a refusal of such an operand names, in its
    place, the argument and the grid that reads the operand where the route
    needs it: that grid moved back through each operation's rule
    ({!Route.backward}), to the first argument, in operand order, that it
    places elsewhere than it lies ({!named}). The arguments of the outermost
    composite that do so are named, so that the caller reads the names it
    passed. An operand that derives from no argument, or whose remedy moves
    none, keeps the route's message.

    The notes are the calling domain's. They hold no value alive, and live
    from the outermost composite's start to its end. A body that resumes on
    another domain notes nothing there, and its refusals there keep the
    route's message. This module lies above {!Route} and {!Value}; {!Prim}
    asks it for names and {!Eval} records. *)

(** The type for a value of any dtype and brand. *)
type value = Value : ('v, 's, 'd) Value.t -> value

val call :
  by:string ->
  (string * string * unit Devices.placement * value) list ->
  (unit -> 'a) ->
  'a
(** [call ~by args f] is [f ()], with each [(name, place, at, x)] of [args]
    the argument [name] of the composite [by], which the function [place]
    places, [x] lying at [at]. Several values may share a name: the parts
    of one structure, which its own function places. *)

val idle : unit -> bool
(** [idle ()] is [true] iff no composite's extent is open on the calling
    domain. It allocates nothing. *)

type node
(** The type for how a value derives from a composite's arguments. *)

val source : ('v, 's, 'd) Value.t -> node option
(** [source x] is how [x] derives from an argument of an open composite:
    the argument's node where [x] is one, else the node an operation noted
    for it ({!note}), else [None], as where no composite's extent is open.
    A miss allocates nothing. *)

val derives : ('v, 's, 'd) Value.t -> bool
(** [derives x] is [Option.is_some (source x)]. *)

val made : Route.rule -> (int iarray * node) option array -> node
(** [made r xs] is the node of a result an operation made by the rule [r]
    from its operands [xs], in operand order: [Some (s, n)] for an operand
    of shape [s] that derives as [n], [None] for one that does not. *)

val note : ('v, 's, 'd) Value.t -> node -> unit
(** [note y n] notes that [y] derives as [n], holding [y] weakly, until the
    outermost composite's extent ends. An operation notes each of its
    results of the first's shape, as its rule's windows are the first's. *)

val named : value -> Grid.t -> Route.named option
(** [named x q] is the argument of an open composite that [x] derives from
    and the grid that places it so that [x] lies at [q], as the preamble
    states, or [None]. *)
