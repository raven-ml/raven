(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let image_path = "sowilo/examples/lena.png"

let () =
  let img = Sowilo.to_float (Nx_io.load_image image_path) in
  let gray = Sowilo.to_grayscale img in
  let blurred = Sowilo.gaussian_blur ~sigma:1.5 ~ksize:5 gray in
  Hugin.hstack
    [
      Hugin.imshow ~data:gray ~cmap:Hugin.Cmap.gray ()
      |> Hugin.title "Grayscale";
      Hugin.imshow ~data:blurred ~cmap:Hugin.Cmap.gray ()
      |> Hugin.title "Gaussian Blur (5x5, sigma=1.5)";
    ]
  |> Hugin.show
