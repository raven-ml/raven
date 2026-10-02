(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin

let image_path = "sowilo/examples/lena.png"
let panel s px = image px |> title s

let normalize_gradient img =
  let abs_img = Nx.abs img in
  let min_val = Nx.item [] (Nx.min ~keepdims:false abs_img) in
  let max_val = Nx.item [] (Nx.max ~keepdims:false abs_img) in
  let range = max_val -. min_val in
  if range <= 1e-6 then Nx.zeros_like img
  else
    Nx.div
      (Nx.sub abs_img (Nx.scalar Nx.float32 min_val))
      (Nx.scalar Nx.float32 range)

let () =
  let img = Sowilo.to_float (Nx_io.load_image image_path) in
  let gray = Sowilo.to_grayscale img in
  let gx, gy = Sowilo.sobel gray in
  grid
    [
      [
        panel "Grayscale" gray;
        panel "Sobel X" (normalize_gradient gx);
        panel "Sobel Y" (normalize_gradient gy);
      ];
    ]
  |> save ~size:(Size.panels 240. 240.) "sobel.png"
