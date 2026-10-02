(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Themes. *)

module Color := Hugin_next_gg.Color
module Font := Hugin_next_font.Font
module Scheme := Hugin_next_kit.Scheme
module Locale := Hugin_next_kit.Locale

type t

val v :
  ?ink:Color.t ->
  ?paper:Color.t ->
  ?accent:Color.t ->
  ?size:float ->
  ?fonts:Font.t list ->
  ?palette:Scheme.t ->
  ?scheme:Scheme.t ->
  ?locale:Locale.t ->
  unit ->
  t

val default : t
val dark : t
val talk : t
val poster : t
val ink : t -> Color.t
val paper : t -> Color.t
val accent : t -> Color.t
val size : t -> float
val fonts : t -> Font.t list
val palette : t -> Scheme.t
val scheme : t -> Scheme.t
val locale : t -> Locale.t
val equal : t -> t -> bool
val pp : Format.formatter -> t -> unit
