(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin

let image_path = "sowilo/examples/lena.png"
let panel s px = image px |> title (Text.v s)

let () =
  let img = Sowilo.to_float (Nx_io.load_image image_path) in
  let gray = Sowilo.to_grayscale img in
  let blurred = Sowilo.gaussian_blur ~sigma:1.5 ~ksize:5 gray in
  grid
    [
      [ panel "Grayscale" gray; panel "Gaussian Blur (5x5, sigma=1.5)" blurred ];
    ]
  |> save ~size:(Size.panels 240. 240.) "gaussian_blur.png"
