(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Channels: the data or constants a role is bound to. *)

module Scale := Hugin_next_kit.Scale
module Text := Hugin_next_text.Text

(** {1:kinds Kinds} *)

(** The kinds of data a channel holds. *)
type _ kind = Quantities : float kind | Categories : string kind

val equal_kind : 'a kind -> 'b kind -> ('a, 'b) Type.eq option
val pp_kind : Format.formatter -> 'd kind -> unit

val of_scale_kind : 'd Scale.kind -> 'd kind option
(** [of_scale_kind k] is the kind read by scales of kind [k], [None] for
    [Temporal]: no channel reads a time scale yet. *)

(** {1:lifts Lifts} *)

(** How a channel lifts its data into its scale's domain. [Scalar v] is the
    quantity [v] for every row: it contributes [v] to its scale's domain without
    taking part in broadcasting. *)
type _ lift =
  | Num : { x : ('a, 'b) Nx.t; valid : Nx.bool_t option } -> float lift
  | Index : int -> float lift
  | Scalar : float -> float lift
  | Cat : {
      codes : ('a, 'b) Nx.t;
      valid : Nx.bool_t option;
      labels : string array option;
    }
      -> string lift
  | Strings : string array -> string lift
  | Dim : {
      axis : int;
      valid : Nx.bool_t option;
      labels : string array option;
    }
      -> string lift

val kind : 'd lift -> 'd kind

val lift_shape : 'd lift -> int array option
(** [lift_shape l] is the shape [l] takes part in broadcasting with, if any. *)

(** {1:channels Channels} *)

type 'd data = {
  lift : 'd lift;
  spec : 'd Scale.t option;
  title : Text.t option;
}

type ('d, 'r) t =
  | Const : 'r -> ('d, 'r) t
  | Data : 'd data -> ('d, 'r) t
  | Map : ('r -> 'r) * ('d, 'r) t -> ('d, 'r) t

val equal : 'r Role.range -> ('d, 'r) t -> ('e, 'r) t -> bool

val data : ('d, 'r) t -> 'd data option
(** [data c] is the lift, specification and title of [c], if it holds data. *)

val constant : ('d, 'r) t -> 'r option
(** [constant c] is the value of [c] for every row, if it holds no data. *)

val mapping : ('d, 'r) t -> 'r -> 'r
(** [mapping c] is the composition of the {!map_range} functions of [c]. *)

(** {1:making Making channels} *)

val num :
  ?scale:float Scale.t ->
  ?valid:Nx.bool_t ->
  ?title:Text.t ->
  ('a, 'b) Nx.t ->
  (float, 'r) t

val cat :
  ?scale:string Scale.t ->
  ?valid:Nx.bool_t ->
  ?title:Text.t ->
  ?labels:string array ->
  ('a, 'b) Nx.t ->
  (string, 'r) t

val strings :
  ?scale:string Scale.t -> ?title:Text.t -> string array -> (string, 'r) t

val dim :
  ?scale:string Scale.t ->
  ?valid:Nx.bool_t ->
  ?title:Text.t ->
  ?labels:string array ->
  int ->
  (string, 'r) t

val index : ?scale:float Scale.t -> ?title:Text.t -> int -> (float, 'r) t
val const : 'r -> ('d, 'r) t
val map_range : ('r -> 'r) -> ('d, 'r) t -> ('d, 'r) t

(** {1:shapes Shapes} *)

val axis_of : int array -> int -> int option
(** [axis_of shape k] is the axis [k] of [shape], counting from the last for a
    negative [k]. *)
