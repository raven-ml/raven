(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The public names the built-in marks use, which [Hugin] includes. *)

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
