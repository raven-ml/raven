(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin

let image_path = "sowilo/examples/lena.png"
let panel s px = image px |> title (Text.v s)

let () =
  let img_u8 = Nx_io.load_image image_path in
  let img = Sowilo.to_float img_u8 in
  let gray = Sowilo.to_grayscale img in
  grid [ [ panel "Original" img_u8; panel "Grayscale" gray ] ]
  |> save ~size:(Size.panels 240. 240.) "grayscale.png"
