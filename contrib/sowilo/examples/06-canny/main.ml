(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin

let image_path = "sowilo/examples/chelsea.png"
let panel s px = image px |> title s

let () =
  let img = Sowilo.to_float (Nx_io.load_image image_path) in
  let gray = Sowilo.to_grayscale img in
  let edges = Sowilo.canny ~low:0.1 ~high:0.3 gray in
  grid [ [ panel "Grayscale" gray; panel "Canny Edges (0.1, 0.3)" edges ] ]
  |> save ~size:(Size.panels 240. 190.) "canny.png"
