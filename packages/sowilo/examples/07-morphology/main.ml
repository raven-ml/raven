(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let image_path = "sowilo/examples/lena.png"

let () =
  let img = Sowilo.to_float (Nx_io.load_image image_path) in
  let gray = Sowilo.to_grayscale img in
  let thresh = Sowilo.threshold 0.5 gray in
  let kernel = Sowilo.structuring_element Rect (5, 5) in
  let eroded = Sowilo.erode ~kernel thresh in
  let dilated = Sowilo.dilate ~kernel thresh in
  Hugin.hstack
    [
      Hugin.imshow ~data:thresh ~cmap:Hugin.Cmap.gray ()
      |> Hugin.title "Thresholded";
      Hugin.imshow ~data:eroded ~cmap:Hugin.Cmap.gray ()
      |> Hugin.title "Eroded (5x5)";
      Hugin.imshow ~data:dilated ~cmap:Hugin.Cmap.gray ()
      |> Hugin.title "Dilated (5x5)";
    ]
  |> Hugin.show
