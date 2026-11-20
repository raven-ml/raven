(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Blurs against salt-and-pepper noise on the grey photograph, and what a
   filter reads past the border. *)

open Hugin

(* [noisy seed x] is the grey images [x], [[h; w; 1]], with one pixel in ten
   set to black or white, drawn from [seed]. *)
let noisy seed x =
  let st = Random.State.make [| seed |] in
  let v = Nx.to_array x in
  let noise =
    Array.init (Array.length v) (fun k ->
        let r = Random.State.int st 20 in
        if r = 0 then 0. else if r = 1 then 1. else v.(k))
  in
  Nx.create Nx.float32 (Nx.shape x) noise

(* A box blur of extent [k], from doc/filters.md (Box blur): the weights are
   a column of [1 / k] times a row of [1 / k], two calls. *)
let box edge k x =
  let w = Nx.full Nx.float32 [| k; 1 |] (1. /. Float.of_int k) in
  Mynd.Filter.correlate ~edge
    (Nx.reshape [| 1; k |] w)
    (Mynd.Filter.correlate ~edge w x)

(* [error a b] is the mean absolute difference of two images, in codes of
   255. *)
let error a b = 255. *. Nx.item [] (Nx.mean (Nx.abs (Nx.sub a b)))

let () =
  let img = Mynd.convert Nx.float32 (Nx_io.load_image "../chelsea.png") in
  let grey = Mynd.Colour.luma img in
  let noise = noisy 7 grey in

  (* Each filter reads past the border by the image mirrored about its
     border pixels. *)
  let edge = `Symmetric_whole in
  let sigma = Nx.scalar Nx.float32 1. in
  let blurred = Mynd.Filter.gaussian ~edge ~radius:3 sigma noise in
  let median = Mynd.Filter.median ~edge (5, 5) noise in
  Printf.printf "mean error to the photograph, in codes:\n";
  Printf.printf "  noisy     %.2f\n" (error noise grey);
  Printf.printf "  box 5     %.2f\n" (error (box edge 5 noise) grey);
  Printf.printf "  gaussian  %.2f\n" (error blurred grey);
  Printf.printf "  median 5  %.2f\n" (error median grey);

  (* Under `Zero a filter reads black past the border, so the border rows of
     a blur darken. *)
  let top x = 255. *. Nx.item [] (Nx.mean (Nx.slice [ Nx.I 0 ] x)) in
  let blur edge =
    Mynd.Filter.gaussian ~edge ~radius:6 (Nx.scalar Nx.float32 2.) grey
  in
  Printf.printf
    "top row's mean code: %.1f, blurred %.1f under `Zero, %.1f under \
     `Symmetric_whole\n"
    (top grey)
    (top (blur `Zero))
    (top (blur `Symmetric_whole));

  let panel x name = image x |> title name in
  grid
    [
      [
        panel noise "noisy";
        panel blurred "gaussian, σ 1";
        panel median "median 5 × 5";
      ];
    ]
  |> save ~size:(Size.figure 900. 240.) "filters.png"
