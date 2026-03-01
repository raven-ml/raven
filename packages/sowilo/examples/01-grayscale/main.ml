(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let image_path = "sowilo/examples/lena.png"

let () =
  let img_u8 = Nx_io.load_image image_path in
  let img = Sowilo.to_float img_u8 in
  let gray = Sowilo.to_grayscale img in
  Hugin.hstack
    [
      Hugin.image img_u8 |> Hugin.title "Original";
      Hugin.imshow ~data:gray ~cmap:Hugin.Cmap.gray ()
      |> Hugin.title "Grayscale";
    ]
  |> Hugin.show
