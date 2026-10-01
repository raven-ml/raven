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

val create : unit -> t
(** [create ()] is a fresh installation, distinct from every other. *)

val dual : t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [dual i x dx] is the dual of [i] whose primal is [x] and tangent [dx]. *)

val split : t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t * ('a, 'b) Nx.t option
(** [split i x] is [(p, Some dx)] if [x] is a dual of [i] of primal [p] and
    tangent [dx], and [(x, None)] otherwise. *)

val install : t -> (unit -> 'a) -> 'a
(** [install i f] is [f ()] under [i]'s interpreter ({!Construct.install}).

    Raises [Invalid_argument], at the operation, when an operation on one of
    [i]'s duals has no tangent rule. *)
