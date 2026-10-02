(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Roles and the ranges they map into. *)

module Color := Hugin_next_gg.Color
module Text := Hugin_next_text.Text
module Symbol := Hugin_next_kit.Symbol
module Dash := Hugin_next_kit.Dash
module Scale := Hugin_next_kit.Scale

(** {1:ranges Ranges} *)

type 'r param = { id : 'r Type.Id.t; equal : 'r -> 'r -> bool }
(** The range of a parameter role, minted by {!param}. *)

(** The ranges roles map into. *)
type _ range =
  | Floats : float range
  | Colors : Color.t range
  | Symbols : Symbol.t range
  | Dashes : Dash.t range
  | Texts : Text.t range
  | Panels : string range
  | Param : 'r param -> 'r range

val equal_range : 'r range -> 's range -> ('r, 's) Type.eq option
(** [equal_range r s] is a witness that [r] and [s] are one range. Two [Param]
    ranges are one iff they share their id. *)

val equal_in : 'r range -> 'r -> 'r -> bool
(** [equal_in r v v'] compares two values of [r], a parameter's with the
    equality it was made with. *)

(** {1:roles Roles} *)

type axis = X | Y

(** How an encoding maps a normalised value into its range. *)
type map =
  | Color  (** [fill], [stroke]: the scale's scheme. *)
  | Opacity  (** Clamped into \[[0];[1]\]. *)
  | Area  (** [size]: an area in pt², from the scale's areas. *)
  | Width  (** A line width, from the theme. *)
  | Shape  (** [symbol]: the scale's symbols, cycling. *)
  | Pattern  (** [dash]: the scale's dash patterns, cycling. *)

(** What a role is. *)
type use =
  | Position of { axis : axis; far : bool }  (** [x], [y]; far: [x2], [y2]. *)
  | Facet of axis  (** [fx], [fy]. *)
  | Encoding of { scale : string; map : map }
      (** [scale] is the role's default scale. *)
  | Value  (** [text], {!value} and {!param}: reads no scale. *)

type ('d, 'r) t = { name : string; range : 'r range; use : use }
(** Roles are identified by their names, and a parameter role by its name and
    its range's id. *)

val x : ('d, float) t
val x2 : ('d, float) t
val y : ('d, float) t
val y2 : ('d, float) t
val fill : ('d, Color.t) t
val stroke : ('d, Color.t) t
val opacity : ('d, float) t
val size : (float, float) t
val width : ('d, float) t
val symbol : (string, Symbol.t) t
val dash : (string, Dash.t) t
val text : ('d, Text.t) t
val fx : (string, string) t
val fy : (string, string) t
val value : name:string -> (float, float) t
val param : name:string -> equal:('r -> 'r -> bool) -> (unit, 'r) t

(** {1:meaning Meaning} *)

val scale : use -> string option
(** [scale u] is the name of the scale a role of use [u] reads by default: ["x"]
    for [x] and [x2], ["fx"] for [fx], an encoding's [scale], [None] for
    [Value]. *)

type shown = [ `Axis of axis | `Header of axis | `Legend ]

val shown_on : use -> shown option
(** [shown_on u] is the guide that shows the scale a role of use [u] reads: an
    axis for a position, a header for a facet, a legend for an encoding, [None]
    for a value. *)

val implied : use -> 'd Scale.kind -> 'd Scale.t option
(** [implied u k] is what a role of use [u] implies on a scale of kind [k]
    beyond what its mark does: [zero] for an area on quantities, [reverse] for a
    vertical position on categories. *)

val by_cell : string -> bool
(** [by_cell n] is [true] iff [n] is the default scale of a position or facet,
    which the innermost grid cell scopes. *)

val by_kind : string -> bool
(** [by_kind n] is [true] iff [n] is the default scale of an encoding, which a
    scope holds once per kind. *)
