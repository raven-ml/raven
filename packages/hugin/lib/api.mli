(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The public names the built-in marks use, which {!Hugin} includes.

    {!Marks} opens this module and nothing else of the library, so a user
    can write each built-in mark against {!Hugin}. Every name here is
    documented in {!Hugin}; {!kind} is {!Channel.scale_kind}. *)

module P2 = Hugin_gg.P2
module Box2 = Hugin_gg.Box2
module Affine = Hugin_gg.Affine
module Path = Hugin_gg.Path
module Stroke = Hugin_gg.Stroke
module Color = Hugin_gg.Color
module Font = Hugin_font.Font
module Text = Hugin_text.Text
module Picture = Hugin_vg.Picture
module Renderable = Hugin_vg.Renderable
module Locale = Hugin_kit.Locale
module Number = Hugin_kit.Number
module Scale = Hugin_kit.Scale
module Scheme = Hugin_kit.Scheme
module Symbol = Hugin_kit.Symbol
module Dash = Hugin_kit.Dash
module Curve = Hugin_kit.Curve
module Stats = Hugin_kit.Stats

(* Figures *)

type t = Figure.t

(* Channels *)

type ('d, 'r) channel = ('d, 'r) Channel.t

val num :
  ?scale:float Scale.t ->
  ?valid:Nx.bool_t ->
  ?title:string ->
  ('a, 'b) Nx.t ->
  (float, 'r) channel

val cat :
  ?scale:string Scale.t ->
  ?valid:Nx.bool_t ->
  ?title:string ->
  ?labels:string array ->
  ('a, 'b) Nx.t ->
  (string, 'r) channel

val strings :
  ?scale:string Scale.t ->
  ?title:string ->
  string array ->
  (string, 'r) channel

val floats :
  ?scale:float Scale.t -> ?title:string -> float array -> (float, 'r) channel

val dim :
  ?scale:string Scale.t ->
  ?valid:Nx.bool_t ->
  ?title:string ->
  ?labels:string array ->
  int ->
  (string, 'r) channel

val index :
  ?scale:float Scale.t -> ?title:string -> int -> (float, 'r) channel

val const : 'r -> ('d, 'r) channel
val map_range : ('r -> 'r) -> ('d, 'r) channel -> ('d, 'r) channel
val kind : ('d, 'r) channel -> 'd Scale.kind option
val varies : int array -> ('d, 'r) channel -> int -> bool

(* Extending *)

module Coord = Coord
module Theme = Theme
module Role = Role
module Mark = Mark
