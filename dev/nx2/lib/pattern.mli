(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Axis patterns: the parser behind [Nx.Pattern], and the movements a
    one-operand pattern makes of a shape.

    A pattern is checked when parsed: its grammar, its names' places, its sums.
    Extents are checked where a pattern meets a shape. Messages start with the
    [by] the caller passes, then the pattern's text in quotes. *)

(** {1:patterns Patterns} *)

type axis =
  | Name of string
  | Group of string list  (** Merged axes, major first: at least one name. *)
  | Unit  (** [1]: an axis of extent [1]. *)
  | Ellipsis  (** [...]: the axes no name covers. *)

type layout = axis list

type plan
(** A one-operand pattern prepared for {!moves}: its names numbered, so that a
    call reads arrays and allocates only the movements. *)

(** The type for a pattern's operands. *)
type kind = private
  | One of { operand : layout; plan : plan }
  | Two of { a : layout; b : layout; summed : string list }
      (** [summed] is the names after [|], in order. With two operands, an
          [Ellipsis] leads all three layouts or none. *)

type t = private {
  text : string;  (** As written, for messages. *)
  result : layout;
  kind : kind;
}
(** A pattern whose structure holds the rules of [Nx.Pattern.v]. *)

val v : by:string -> string -> t
(** [v ~by s] is the pattern [s].

    Raises [Invalid_argument] as [Nx.Pattern.v] says, naming [by] and [s]. *)

val inverse : by:string -> t -> t
(** [inverse ~by p] is [p] with its operand and result exchanged.

    Raises [Invalid_argument] naming [by] if [p] has two operands. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a pattern as its text, quoted. *)

(** {1:moves Movements} *)

val moves :
  by:string ->
  sizes:(string * int) list ->
  t ->
  int array ->
  (Format.formatter -> unit) ->
  Nx_array.Move.t list
(** [moves ~by ~sizes p s what] is the movements, in order, that rearrange a
    value of shape [s] as the one-operand pattern [p] says: a [Reshape] that
    splits groups and drops units, a [Permute], and a [Reshape] that merges
    groups and adds units, each left out where it changes nothing. [what]
    formats the operand in messages.

    Raises [Invalid_argument] naming [by] as [Nx.rearrange] says. *)
