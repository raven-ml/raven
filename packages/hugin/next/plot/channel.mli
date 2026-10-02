(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Channels: the data or constants a role is bound to. *)

module Scale := Hugin_next_kit.Scale
module Text := Hugin_next_text.Text

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

val lift_kind : 'd lift -> 'd Scale.kind

val lift_shape : 'd lift -> int array option
(** [lift_shape l] is the shape [l] takes part in broadcasting with, if any. *)

(** {1:channels Channels} *)

type ('d, 'r) t =
  | Const : 'r -> ('d, 'r) t
  | Data : {
      lift : 'd lift;
      scale : 'd Scale.t option;
      title : Text.t option;
    }
      -> ('d, 'r) t
  | Map : ('r -> 'r) * ('d, 'r) t -> ('d, 'r) t

val equal : 'r Role.range -> ('d, 'r) t -> ('e, 'r) t -> bool

type 'd data = {
  lift : 'd lift;
  spec : 'd Scale.t option;
  title : Text.t option;
}

val data : ('d, 'r) t -> 'd data option
(** [data c] is the lift, specification and title of [c], if it holds data. *)

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
