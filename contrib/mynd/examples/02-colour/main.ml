(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* An image holds encoded values, and every function computes on them as they
   are. Averaging light, or colours with alpha, is a choice the program
   makes by converting first. *)

open Hugin

(* [floats x] prints [x]'s elements to three decimals. A value that rounds
   to zero prints without a sign, whichever side of zero it lies on. *)
let floats x =
  let show v =
    match Printf.sprintf "%.3f" v with "-0.000" -> "0.000" | s -> s
  in
  String.concat " " (List.map show (Array.to_list (Nx.to_array x)))

(* [in_light f x] applies [f] to the light that the sRGB values [x] encode. *)
let in_light f x = Mynd.linear_to_srgb (f (Mynd.srgb_to_linear x))

let () =
  (* Black and white pixels in a checkerboard, halved by bilinear resizing:
     the mean of the encoded values is 0.5, while the mean of their light is
     0.5 in light, which sRGB encodes as about 0.735. *)
  let board = Nx.create Nx.float32 [| 2; 2; 1 |] [| 0.; 1.; 1.; 0. |] in
  let half = Mynd.Sample.resize ~edge:`Replicate `Triangle (1, 1) in
  Printf.printf "a checkerboard halved: %s in encoded values, %s in light\n"
    (floats (half board))
    (floats (in_light half board));

  (* The photograph quartered both ways. sRGB's encoding is concave, so the
     encoding of a mean of light is at least the mean of the encodings:
     averaging light gives lighter pixels where light and dark meet. *)
  let img = Mynd.convert Nx.float32 (Nx_io.load_image "../chelsea.png") in
  let quarter = Mynd.Sample.resize ~edge:`Replicate `Triangle (75, 113) in
  let encoded = quarter img and light = in_light quarter img in
  let codes x = Nx.mul_s x 255. in
  Printf.printf
    "chelsea quartered: mean code %.1f in encoded values, %.1f in light\n"
    (Nx.item [] (Nx.mean (codes encoded)))
    (Nx.item [] (Nx.mean (codes light)));
  Printf.printf "  mean difference %.2f codes, largest %.1f\n"
    (Nx.item [] (Nx.mean (Nx.abs (codes (Nx.sub light encoded)))))
    (Nx.item [] (Nx.max (Nx.abs (codes (Nx.sub light encoded)))));

  (* An opaque red pixel beside a transparent green one, averaged. Straight
     values give the transparent pixel's green half the weight; premultiplied
     ones give it none. *)
  let pair =
    Nx.create Nx.float32 [| 1; 2; 4 |] [| 1.; 0.; 0.; 1.; 0.; 1.; 0.; 0. |]
  in
  let mean = Mynd.Sample.resize ~edge:`Replicate `Box (1, 1) in
  Printf.printf "red and transparent green averaged (r g b a):\n";
  Printf.printf "  straight       %s\n" (floats (mean pair));
  Printf.printf "  premultiplied  %s\n"
    (floats (Mynd.unpremultiply (mean (Mynd.premultiply pair))));

  (* Luma weighs encoded values; luminance weighs light. *)
  let green = Nx.create Nx.float32 [| 3 |] [| 0.; 1.; 0. |] in
  Printf.printf "green: luma %s, luminance %s\n"
    (floats (Mynd.Colour.luma green))
    (floats (Mynd.Colour.luminance (Mynd.srgb_to_linear green)));

  grid
    [
      [
        image encoded |> title "encoded values averaged";
        image light |> title "light averaged";
      ];
    ]
  |> save ~size:(Size.figure 600. 240.) "colour.png"
