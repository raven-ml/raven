(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin

let image_path = "sowilo/examples/lena.png"
let panel s px = image px |> title s

let () =
  let img = Sowilo.to_float (Nx_io.load_image image_path) in
  let gray = Sowilo.to_grayscale img in
  let median = Sowilo.median_blur ~ksize:5 gray in
  grid [ [ panel "Grayscale" gray; panel "Median Blur (k=5)" median ] ]
  |> save ~size:(Size.panels 240. 240.) "median_blur.png"
