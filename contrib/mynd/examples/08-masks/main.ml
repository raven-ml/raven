(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* From a segmentation model's mask logits to the masks, boxes and COCO
   run-length encodings a user gets. A model such as SAM 2 predicts each mask
   as logits on a 256 × 256 grid that covers the whole image. The logits here
   are drawn by hand around the cat's eyes and nose. *)

open Hugin

let grid_size = 256

(* [logit (cx, cy) (rx, ry) x y] is positive inside the ellipse of centre
   [(cx, cy)] and radii [(rx, ry)], in image pixels. *)
let logit (cx, cy) (rx, ry) x y =
  let u = (x -. cx) /. rx and v = (y -. cy) /. ry in
  8. *. (1. -. ((u *. u) +. (v *. v)))

(* One function of the image position per mask. The left eye's has the two
   defects a cleaning step removes: a speck outside it and a hole inside. *)
let objects =
  [|
    (fun x y ->
      let eye = logit (172., 115.) (36., 31.) x y in
      let speck = logit (95., 40.) (5., 5.) x y in
      let hole = logit (160., 104.) (5., 5.) x y in
      Float.min (Float.max eye speck) (-.hole));
    logit (318., 137.) (29., 26.);
    logit (262., 243.) (30., 15.);
  |]

(* [logits h w] is the masks' logits on the model's grid, [[3; 256; 256; 1]],
   each cell read at the image position of its centre: the grid scales the
   image's [w] columns and [h] rows to 256 each. *)
let logits h w =
  let sx = Float.of_int w /. Float.of_int grid_size in
  let sy = Float.of_int h /. Float.of_int grid_size in
  Nx.init Nx.float32 [| 3; grid_size; grid_size; 1 |] (fun i ->
      let x = (Float.of_int i.(2) +. 0.5) *. sx in
      let y = (Float.of_int i.(1) +. 0.5) *. sy in
      objects.(i.(0)) x y)

(* Areas, from doc/morphology.md (Areas and small objects): [areas c m] is
   each pixel's component area, counted in each mask's own labels. *)
let areas c m =
  let s = Nx.shape m in
  let r = Array.length s in
  let lead = Array.sub s 0 (r - 3) in
  let flat x = Nx.reshape (Array.append lead [| s.(r - 3) * s.(r - 2) |]) x in
  let l = flat (Mynd.Mask.components c m) in
  let ones = Nx.ones_like l in
  let counts = Nx.scatter ~combine:Nx.Add ~axis:(-1) l ones (Nx.zeros_like l) in
  Nx.reshape s (Nx.take_along_axis ~axis:(-1) l counts)

(* SAM 2's cleaning of its logits [l], before they are upsampled: the
   8-connected pieces of background of at most [n] cells become foreground,
   their logits set to 10, and the pieces of foreground of at most [n] cells
   become background, their logits set to -10. Both are decided on [l], so a
   ring of at most [n] cells around a small hole becomes background and the
   hole foreground, where filling first would make one larger piece. *)
let clean n l =
  let small m =
    Nx.logical_and m (Nx.less_equal_s (areas `Eight m) (Int64.of_int n))
  in
  let holes = small (Nx.less_equal_s l 0.) in
  let specks = small (Nx.greater_s l 0.) in
  Nx.where specks (Nx.full_like l (-10.))
    (Nx.where holes (Nx.full_like l 10.) l)

(* [pieces c m] is the number of components of each mask. A component's label
   is the least row-major index of its pixels, so exactly one of its pixels
   has its own index as its label. *)
let pieces c m =
  let h = Nx.dim (-3) m and w = Nx.dim (-2) m in
  let index = Nx.reshape [| h; w; 1 |] (Nx.arange Nx.int64 0 (h * w) 1) in
  Nx.count ~axes:[ -3; -2; -1 ] (Nx.equal (Mynd.Mask.components c m) index)

let ints x = Array.map Int64.to_int (Nx.to_array x)

let shape x =
  let dims = Array.to_list (Array.map string_of_int (Nx.shape x)) in
  "[" ^ String.concat "; " dims ^ "]"

let names = [| "left eye"; "right eye"; "nose" |]

let () =
  let img = Nx_io.load_image "../chelsea.png" in
  let h = Nx.dim 0 img and w = Nx.dim 1 img in
  let l = logits h w in

  (* Bilinear upsampling to the image's size: each pixel reads the grid at its
     centre, and past the grid's border the border cells repeat. A pixel is in
     the mask where its logit is above 0. *)
  let masks l =
    Nx.greater_s (Mynd.Sample.resize ~edge:`Replicate `Triangle (h, w) l) 0.
  in
  let raw = masks l in
  let masks = masks (clean 50 l) in
  Printf.printf "logits %s, masks %s\n" (shape l) (shape masks);

  (* A mask's holes are its background's pieces under [`Four] but one, the
     background around it, for masks such as these that do not split the
     background at the image's border. *)
  let fg_before = ints (pieces `Eight raw) in
  let fg_after = ints (pieces `Eight masks) in
  let bg_before = ints (pieces `Four (Nx.logical_not raw)) in
  let bg_after = ints (pieces `Four (Nx.logical_not masks)) in
  let boxes = Mynd.Box.of_mask Nx.float32 masks in
  let iou = Mynd.Mask.iou Nx.float32 raw masks in
  Array.iteri
    (fun k name ->
      let mask = Nx.slice [ Nx.I k ] masks in
      let rle = Mynd.Rle.encode mask in
      let counts = Mynd.Rle.to_string rle in
      let b i = Nx.item [ k; i ] boxes in
      Printf.printf "%s: %d piece(s) and %d hole(s), cleaned to %d and %d\n"
        name fg_before.(k)
        (bg_before.(k) - 1)
        fg_after.(k)
        (bg_after.(k) - 1);
      Printf.printf "  IoU with the raw mask %.4f\n" (Nx.item [ k; k ] iou);
      Printf.printf "  bbox [%g, %g, %g, %g] (x, y, w, h), area %d\n" (b 0)
        (b 1)
        (b 2 -. b 0)
        (b 3 -. b 1)
        (Mynd.Rle.area rle);
      Printf.printf "  counts \"%s…\" (%d characters), decodes back: %b\n"
        (String.sub counts 0 16) (String.length counts)
        (Nx.to_array (Mynd.Rle.decode rle) = Nx.to_array mask))
    names;

  (* Each mask as a translucent colour over the photograph, and its box. An
     image covers [0, w] × [0, h] in the figure, row 0 at the top, which are
     mynd's coordinates, so each box draws where it lies. *)
  let colours =
    Nx.create Nx.float32 [| 3; 1; 1; 3 |]
      [| 1.; 0.8; 0.; 0.; 0.9; 1.; 1.; 0.3; 0.5 |]
  in
  let alpha = Nx.mul_s (Nx.cast Nx.float32 masks) 0.5 in
  let rgb = Nx.broadcast_to [| 3; h; w; 3 |] colours in
  let overlay = Nx.concatenate ~axis:(-1) [ rgb; alpha ] in
  let side i = num (Nx.slice [ Nx.A; Nx.I i ] boxes) in
  layer
    [
      image img;
      image overlay;
      rect ~x:(side 0) ~y:(side 1) ~x2:(side 2) ~y2:(side 3)
        ~stroke:(const Color.white) ();
    ]
  |> save ~size:(Size.figure 451. 300.) "masks.png"
