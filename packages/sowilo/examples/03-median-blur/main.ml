(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let image_path = "sowilo/examples/lena.png"

let () =
  let img = Sowilo.to_float (Nx_io.load_image image_path) in
  let gray = Sowilo.to_grayscale img in
  let median = Sowilo.median_blur ~ksize:5 gray in
  Hugin.hstack
    [
      Hugin.imshow ~data:gray ~cmap:Hugin.Cmap.gray ()
      |> Hugin.title "Grayscale";
      Hugin.imshow ~data:median ~cmap:Hugin.Cmap.gray ()
      |> Hugin.title "Median Blur (k=5)";
    ]
  |> Hugin.show
