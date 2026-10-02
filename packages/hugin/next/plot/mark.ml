(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Figure

type nonrec binding = binding

let bind ?imply ?guide role ch = B { role; ch; imply; guide }

type nonrec rows = rows

let id (r : rows) = match r with _ -> .
let length (r : rows) = match r with _ -> .
let shape (r : rows) = match r with _ -> .
let index (r : rows) = match r with _ -> .
let get (r : rows) _ = match r with _ -> .
let normalized (r : rows) _ = match r with _ -> .
let range (r : rows) _ = match r with _ -> .
let ticks (r : rows) _ = match r with _ -> .
let points (r : rows) = match r with _ -> .
let extent (r : rows) _ = match r with _ -> .
let projection (r : rows) = match r with _ -> .
let project (r : rows) _ = match r with _ -> .
let series (r : rows) = match r with _ -> .
let theme (r : rows) = match r with _ -> .
let text ?halign:_ ?valign:_ (r : rows) _ _ _ = match r with _ -> .
let warn (r : rows) _ = match r with _ -> .

type nonrec reducer = reducer

let m4 = M4
let cells = Cells
let raster = Raster

let v ~name ?reduce ?coord ?swatch bindings draw =
  Mark (make_mark "Mark.v" ~name ?reduce ?coord ?swatch bindings draw)
