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
  grid [ [ panel "Grayscale" gray; mask "Binary Threshold (128)" thresh ] ]
  |> save ~size:(Size.panels 240. 190.) "threshold.png"
