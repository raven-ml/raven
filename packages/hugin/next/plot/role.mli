(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Roles and the ranges they map into. *)

module Color := Hugin_next_gg.Color
module Text := Hugin_next_text.Text
module Symbol := Hugin_next_kit.Symbol
module Curve := Hugin_next_kit.Curve

(** {1:ranges Ranges} *)

(** The ranges roles map into. [Curves] and [Pixels] are the ranges of roles the
    built-in marks keep their parameters in. *)
type _ range =
  | Floats : float range
  | Colors : Color.t range
  | Symbols : Symbol.t range
  | Texts : Text.t range
  | Panels : string range
  | Curves : Curve.t range
  | Pixels : Nx.packed range

val equal_range : 'r range -> 's range -> ('r, 's) Type.eq option
val equal_in : 'r range -> 'r -> 'r -> bool

(** {1:roles Roles} *)

type ('d, 'r) t = { name : string; range : 'r range; scale : string option }
(** [scale] is the name of the scale the role reads by default, [None] for a
    role that reads none. *)

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
val text : ('d, Text.t) t
val fx : (string, string) t
val fy : (string, string) t
val value : name:string -> (float, float) t

(** {1:parameters Parameters of built-in marks} *)

val curve : ('d, Curve.t) t
val dx : ('d, float) t
val dy : ('d, float) t
val pixels : ('d, Nx.packed) t
