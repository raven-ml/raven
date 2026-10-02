(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

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
module Scale = Hugin_next_kit.Scale
module Scheme = Hugin_next_kit.Scheme
module Symbol = Hugin_next_kit.Symbol
module Curve = Hugin_next_kit.Curve

(* The stages after [layout] come with the drawing of figures. *)
let unimplemented fn = failwith ("Hugin_next." ^ fn ^ ": not implemented")

(* Figures *)

type t = Figure.t
type id = Common.id
type warning = Common.warning

let pp_warning = Common.pp_warning
let equal = Figure.equal

(* Channels *)

type ('d, 'r) channel = ('d, 'r) Channel.t

let num = Channel.num
let cat = Channel.cat
let strings = Channel.strings
let dim = Channel.dim
let index = Channel.index
let const = Channel.const
let map_range = Channel.map_range

(* Marks *)

let dot = Marks.dot
let line = Marks.line
let rect = Marks.rect
let rule = Marks.rule
let text = Marks.text
let image = Marks.image
let contour = Marks.contour

(* Coordinate systems and views *)

module Coord = Coord
module View = View

(* Composing *)

type sharing = Figure.sharing
type side = Figure.side

let layer = Figure.layer
let grid = Figure.grid
let span = Figure.span
let share = Figure.share
let title = Figure.title
let coord = Figure.coord
let name = Figure.name
let bind = Figure.bind
let axis = Figure.axis
let legend = Figure.legend

(* Sizes, themes and extending *)

module Size = Size
module Theme = Theme
module Role = Role
module Mark = Mark

(* Stages *)

module Resolved = Resolved
module Layout = Layout

module Drawing = struct
  type t = |

  let renderable (d : t) = match d with _ -> .
  let warnings (d : t) = match d with _ -> .
  let equal (d : t) _ = match d with _ -> .
  let pp _ (d : t) = match d with _ -> .
end

let resolve = Resolved.resolve
let layout = Layout.layout
let draw ?prev:_ ~density:_ _ = unimplemented "draw"

let render ?view ?theme ?(density = 2.) size f =
  draw ~density (layout ?theme size (resolve ?view f))

let save ?warn:_ ?view:_ ?theme:_ ?size:_ ?density:_ _ _ = unimplemented "save"
let pp _ _ = unimplemented "pp"
