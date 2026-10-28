(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Axis patterns: the parser behind [Nx.Pattern], the extents a pattern gives
    its names where it meets shapes, and the movements a one-operand pattern
    makes of a shape.

    A pattern is checked when parsed: its grammar, its names' places, its sums.
    Extents are checked where a pattern meets shapes. Messages start with the
    [by] the caller passes, then the pattern's text in quotes.

    This module depends on nothing of the engine; {!Contraction} and [Nx]
    parse patterns by it. *)

(** {1:patterns Patterns} *)

(** An axis, its names written as ['n]: as parsed, or numbered. *)
type 'n axis =
  | Name of 'n
  | Group of 'n list  (** Merged axes, major first: at least one name. *)
  | Unit  (** [1]: an axis of extent [1]. *)
  | Ellipsis  (** [...]: the axes no name covers. *)

type t = private {
  text : string;  (** As written, for messages. *)
  names : string array;
      (** Each name once, in its first place among the operands. *)
  operands : int axis list list;
      (** Each operand's layout, a name numbered by its place in [names]. *)
  result : int axis list;  (** The result's layout, numbered alike. *)
}
(** A pattern whose structure holds the rules of [Nx.Pattern.v]. With two
    operands or more, an [Ellipsis] leads every layout or none. *)

val v : by:string -> string -> t
(** [v ~by s] is the pattern [s].

    Raises [Invalid_argument] as [Nx.Pattern.v] says, naming [by] and [s]. *)

val inverse : by:string -> t -> t
(** [inverse ~by p] is [p] with its operand and result exchanged.

    Raises [Invalid_argument] naming [by] if [p] has more than one operand. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a pattern as its text, quoted. *)

(** {1:extents Extents} *)

val resolve :
  by:string ->
  sizes:(string * int) list ->
  label:(int -> string) ->
  t ->
  (string * int array) list ->
  extents:int array
  * operands:int array list
  * result:int array
  * shape:int array
(** [resolve ~by ~sizes ~label p xs] meets [p] with the operands [xs], each a
    dtype's name and a shape, one for each of [p]'s operands. Of [p]'s [n]
    names and the [c] axes its [...] covers, numbered [n] to [n + c - 1]:

    - [extents] is each one's extent.
    - [operands] and [result] are each layout's in order, a group's names
      included, [...] written out and units left out.
    - [shape] is the result's shape.

    A name takes its extent from [sizes] or from an axis it stands on alone. A
    group whose known extents multiply to [0] has no element and leaves its
    other names unknown; otherwise its known extents divide its axis exactly
    and leave its one unknown name the quotient, which may give another group
    its own. [label k] names the [k]th operand in messages.

    Raises [Invalid_argument] naming [by] as [Nx.contract] says. *)

(** {1:moves Movements} *)

val moves :
  by:string ->
  sizes:(string * int) list ->
  t ->
  string * int array ->
  Nx_array.Move.t list
(** [moves ~by ~sizes p x] is the movements, in order, that rearrange the
    operand [x], a dtype's name and a shape, as the one-operand pattern [p]
    says: a [Reshape] that splits groups and drops units, a [Permute], and a
    [Reshape] that merges groups and adds units, each left out where it changes
    nothing.

    Raises [Invalid_argument] naming [by] as [Nx.rearrange] says. *)
