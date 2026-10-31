(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin

let image_path = "sowilo/examples/chelsea.png"
let panel s px = image px |> title s

(* A boolean mask drawn as an image, [true] white. *)
let mask s m = panel s (Nx.cast Nx.float32 m)

let () =
  let img = Sowilo.to_float (Nx_io.load_image image_path) in
  let gray = Sowilo.to_grayscale img in
  let thresh = Sowilo.threshold 0.5 gray in
  let kernel = Sowilo.structuring_element Rect (5, 5) in
  let eroded = Sowilo.erode ~kernel thresh in
  let dilated = Sowilo.dilate ~kernel thresh in
  grid
    [
      [
        mask "Thresholded" thresh;
        mask "Eroded (5x5)" eroded;
        mask "Dilated (5x5)" dilated;
      ];
    ]
  |> save ~size:(Size.panels 240. 190.) "morphology.png"
