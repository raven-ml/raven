(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The photograph prepared as two models take their input, and a box a
   detector gives back mapped onto the photograph. *)

open Hugin

let shape x =
  let dims = Array.to_list (Array.map string_of_int (Nx.shape x)) in
  "[" ^ String.concat "; " dims ^ "]"

let floats fmt x =
  String.concat " "
    (List.map (Printf.sprintf fmt) (Array.to_list (Nx.to_array x)))

(* [channels x] is the per-channel mean and standard deviation of a batch of
   channels-first images [x], [[n; c; h; w]]. *)
let channels x =
  let axes = [ 0; 2; 3 ] in
  Printf.printf "  mean %s, std %s\n"
    (floats "%.3f" (Nx.mean ~axes x))
    (floats "%.2f" (Nx.std ~axes x))

(* [first x] is the images [x], [[…; h; w; c]], channels first,
   [[…; c; h; w]], as a model of 2-d convolutions reads them. *)
let first x = Nx.moveaxis (-1) (-3) x

let () =
  let img = Mynd.convert Nx.float32 (Nx_io.load_image "../chelsea.png") in
  let h = Nx.dim 0 img and w = Nx.dim 1 img in
  let batch = Nx.unsqueeze ~axes:[ 0 ] img in

  (* SAM 2 resizes to 1024 × 1024 by bilinear interpolation, without keeping
     the aspect ratio, then normalises by ImageNet's mean and deviation.
     Enlarging reads each new pixel's centre; past the border the border
     pixels repeat. *)
  let mean = Nx.create Nx.float32 [| 3 |] [| 0.485; 0.456; 0.406 |] in
  let std = Nx.create Nx.float32 [| 3 |] [| 0.229; 0.224; 0.225 |] in
  let sam = Mynd.Sample.resize ~edge:`Replicate `Triangle (1024, 1024) batch in
  let sam_input = first (Nx.div (Nx.sub sam mean) std) in
  Printf.printf "SAM 2 input %s\n" (shape sam_input);
  channels sam_input;

  (* RT-DETR resizes to 640 × 640 and keeps the fractions. *)
  let detr = Mynd.Sample.resize ~edge:`Replicate `Triangle (640, 640) batch in
  let detr_input = first detr in
  Printf.printf "RT-DETR input %s\n" (shape detr_input);
  channels detr_input;

  (* A detector gives boxes as centre and size, in fractions of its input's
     width and height. Scaling an image scales its coordinates, so a
     fraction of the input's width is the same fraction of the photograph's:
     multiplying by [w h w h] gives the box in the photograph's pixels, and
     by 1024 the box in SAM 2's input, where it can prompt for a mask. *)
  let cxcywh = Nx.create Nx.float32 [| 4 |] [| 0.5; 0.53125; 0.5; 0.6875 |] in
  let box = Mynd.Box.of_cxcywh cxcywh in
  let size =
    Nx.create Nx.float32 [| 4 |]
      Float.[| of_int w; of_int h; of_int w; of_int h |]
  in
  let in_photo = Nx.mul box size in
  Printf.printf "box [cx cy w h] %s of the input\n" (floats "%g" cxcywh);
  Printf.printf "  [x0 y0 x1 y1] %s in the photograph\n"
    (floats "%.2f" in_photo);
  Printf.printf "  [x0 y0 x1 y1] %s in SAM 2's input\n"
    (floats "%.2f" (Nx.mul_s box 1024.));

  let side i = num (Nx.slice [ Nx.R (i, i + 1) ] in_photo) in
  grid
    [
      [
        layer
          [
            image img;
            rect ~x:(side 0) ~y:(side 1) ~x2:(side 2) ~y2:(side 3)
              ~stroke:(const Color.white) ();
          ]
        |> title "the photograph and the box";
        image (Nx.squeeze ~axes:[ 0 ] detr) |> title "RT-DETR's input";
      ];
    ]
  |> save ~size:(Size.figure 640. 280.) "input.png"
