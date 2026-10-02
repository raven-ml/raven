(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The public names the built-in marks use, which [Hugin_next] includes. *)

module P2 = Hugin_next_gg.P2
module Box2 = Hugin_next_gg.Box2
module Affine = Hugin_next_gg.Affine
module Path = Hugin_next_gg.Path
module Stroke = Hugin_next_gg.Stroke
module Color = Hugin_next_gg.Color
module Font = Hugin_next_font.Font
module Text = Hugin_next_text.Text
module Picture = Hugin_next_vg.Picture
module Renderable = Hugin_next_vg.Renderable
module Locale = Hugin_next_kit.Locale
module Number = Hugin_next_kit.Number
module Scale = Hugin_next_kit.Scale
module Scheme = Hugin_next_kit.Scheme
module Symbol = Hugin_next_kit.Symbol
module Curve = Hugin_next_kit.Curve
module Stats = Hugin_next_kit.Stats

(* Figures *)

type t = Figure.t

(* Channels *)

type ('d, 'r) channel = ('d, 'r) Channel.t

let num = Channel.num
let cat = Channel.cat
let strings = Channel.strings
let floats = Channel.floats
let dim = Channel.dim
let index = Channel.index
let const = Channel.const
let map_range = Channel.map_range
let kind = Channel.scale_kind
let varies = Channel.varies

(* Extending *)

module Coord = Coord
module Theme = Theme
module Role = Role
module Mark = Mark
