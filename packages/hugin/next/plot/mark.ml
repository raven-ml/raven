(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Figure

type nonrec binding = binding

let bind ?imply ?guide role ch = B { role; ch; imply; guide }

type nonrec rows = rows

let id (r : rows) = r.Rows.id
let length = Rows.length
let shape (r : rows) = Array.copy r.Rows.shape
let index (r : rows) = Array.copy r.Rows.index
let get = Rows.get
let normalized = Rows.normalized
let range = Rows.range
let ticks = Rows.ticks
let scale = Rows.scale
let positions = Rows.positions
let points = Rows.points
let extent = Rows.extent
let projection (r : rows) = r.Rows.projection
let project = Rows.project
let series = Rows.series
let theme (r : rows) = r.Rows.theme
let text = Rows.text
let warn (r : rows) msg = r.Rows.warn msg

type nonrec reducer = reducer

let m4 = M4
let cells = Cells
let raster = Raster
let broadcast ?shape bindings = mark_shape "Mark.broadcast" ?shape bindings

let v ~name ?reduce ?coord ?shape ?swatch bindings draw =
  Mark (make_mark "Mark.v" ~name ?reduce ?coord ?shape ?swatch bindings draw)
