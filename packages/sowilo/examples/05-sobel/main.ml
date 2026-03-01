(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let image_path = "sowilo/examples/lena.png"

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
  Hugin.hstack
    [
      Hugin.imshow ~data:gray ~cmap:Hugin.Cmap.gray ()
      |> Hugin.title "Grayscale";
      Hugin.imshow ~data:(normalize_gradient gx) ~cmap:Hugin.Cmap.gray ()
      |> Hugin.title "Sobel X";
      Hugin.imshow ~data:(normalize_gradient gy) ~cmap:Hugin.Cmap.gray ()
      |> Hugin.title "Sobel Y";
    ]
  |> Hugin.show
