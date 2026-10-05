(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Forward-mode differentiation: duals and the JVP table.

    An installation ({!t}) owns the duals it makes: traced values that pair a
    primal with a tangent, with the primal's metadata. A value with a zero
    tangent is no dual. The interpreter applies an operation's rule when an
    operand is one of its own duals and forwards every other operation
    unchanged: a rule forwards the operation on the primals and issues the
    tangent's operations outside the installation, so an enclosing installation
    sees them. Under reverse mode the tangents are a recorder's slots
    ({!Linear}), and those operations are the linear map reverse mode
    transposes. *)

type t
(** The type for installations of forward mode. *)

val create : ?slots:Linear.tape -> string -> t
(** [create ?slots entry] is a fresh installation, distinct from every other,
    for the entry point named [entry], which its errors name. Its tangents are
    values, or under reverse mode slots of [slots]. *)

val dual : t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [dual i x dx] is the dual of [i] whose primal is [x] and tangent [dx]. *)

val split : t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t * ('a, 'b) Nx.t option
(** [split i x] is [(p, Some dx)] if [x] is a dual of [i] of primal [p] and
    tangent [dx], and [(x, None)] otherwise. *)

val tangent : t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [tangent i x] is the tangent of [x] if it is a dual of [i], and otherwise
    zeros, or under reverse mode a slot nothing feeds. *)

val install : t -> (unit -> 'a) -> 'a
(** [install i f] is [f ()] under [i]'s interpreter ({!Construct.install}).

    A custom rule with one of [i]'s duals among its arguments runs at the
    arguments' primals, outside [i]: a [custom_jvp] rule's result is the answer
    of the differentiations around [i] to the rule there, and its tangent the
    rule's tangent map at [i]'s tangents, zeros or a slot nothing feeds for an
    argument [i] does not track; under reverse mode the map is not applied to a
    result that holds no tensor, and a loop in it unrolls. A [custom_vjp] rule's
    result is the rule's, with, under reverse mode, a linear call whose
    transpose is the rule's pullback between the conjugated cotangents and
    gradients.

    A root passes on with [i]'s values read as primals in its functions, so its
    solve is never differentiated; its tangent is the root, passed on, of the
    residual's derivative at the result plus the residual's tangent there. A
    linear function applied at another level ({!Construct.At_map}) to one of
    [i]'s duals is applied to its primal and to its tangent.

    Raises [Invalid_argument], at the operation, when an operation on one of
    [i]'s duals has no tangent rule; at a custom rule's operation on one of
    [i]'s duals; and at a [custom_vjp] call with a tensor in its result when
    [i]'s tangents are values. *)
