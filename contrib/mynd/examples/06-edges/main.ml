(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Gradients of the grey photograph, and Canny's edges at two pairs of
   thresholds. *)

open Hugin

(* Canny's edges, from doc/edges.md (Canny's edges), with [`Sobel], which
   smooths across each difference as well: the maxima of the gradient's
   strength along the gradient that are above [low], kept where they join one
   above [high] through 8-connected pixels. *)
let canny ~radius ~sigma ~low ~high x =
  let edge = `Symmetric_whole in
  let b = Mynd.Filter.gaussian ~edge ~radius sigma x in
  let ((gx, gy) as g) = Mynd.Filter.gradient ~edge `Sobel b in
  let m = Nx.hypot gx gy in
  let candidates =
    Nx.logical_and (Mynd.Filter.gradient_maxima g) (Nx.greater_s m low)
  in
  (m, Mynd.Mask.reconstruct `Eight ~marker:(Nx.greater_s m high) candidates)

(* [pieces m] is the number of 8-connected components of a mask: a
   component's label is the least row-major index of its pixels, so exactly
   one of its pixels has its own index as its label. *)
let pieces m =
  let h = Nx.dim (-3) m and w = Nx.dim (-2) m in
  let index = Nx.reshape [| h; w; 1 |] (Nx.arange Nx.int64 0 (h * w) 1) in
  Nx.item [] (Nx.count (Nx.equal (Mynd.Mask.components `Eight m) index))

let () =
  let img = Mynd.convert Nx.float32 (Nx_io.load_image "../chelsea.png") in
  let grey = Mynd.Colour.luma img in
  let sigma = Nx.scalar Nx.float32 1.5 in

  (* Strengths are in values per pixel: a step from black to white between
     two columns, blurred by sigma 1.5, has strength (g(0) + g(1)) / 2 beside
     it, g the blur's weights, about 0.24. *)
  let edges (low, high) =
    let m, e = canny ~radius:5 ~sigma ~low ~high grey in
    Printf.printf "thresholds (%g, %g): %Ld edge pixels in %Ld pieces\n" low
      high
      (Nx.item [] (Nx.count e))
      (pieces e);
    (m, e)
  in
  let m, loose = edges (0.01, 0.03) in
  let _, strict = edges (0.02, 0.05) in
  Printf.printf "largest strength %.3f\n" (Nx.item [] (Nx.max m));

  let panel x name = image x |> title name in
  let mask e = Nx.cast Nx.float32 e in
  grid
    [
      [
        panel (Nx.div_s m (Nx.item [] (Nx.max m))) "strength";
        panel (mask loose) "edges (0.01, 0.03)";
        panel (mask strict) "edges (0.02, 0.05)";
      ];
    ]
  |> save ~size:(Size.figure 900. 240.) "edges.png"
