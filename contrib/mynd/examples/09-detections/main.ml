(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* From a detector's outputs to COCO's detections. A detector gives boxes as
   centre and size in fractions of its input, each with a class and a score.
   The twelve here are written by hand around the cat's eyes and nose, with
   the near-duplicates a detector gives. *)

open Hugin

(* Class, 0 for an eye and 1 for a nose, then [cx cy w h] and score. *)
let raw =
  [|
    (0, [| 0.3814; 0.3833; 0.1596; 0.2067 |], 0.92);
    (0, [| 0.3902; 0.3933; 0.1552; 0.2 |], 0.81);
    (0, [| 0.7051; 0.4567; 0.1286; 0.1733 |], 0.89);
    (0, [| 0.6918; 0.4433; 0.1375; 0.1867 |], 0.64);
    (1, [| 0.5809; 0.81; 0.133; 0.1 |], 0.87);
    (1, [| 0.5898; 0.82; 0.1242; 0.1 |], 0.58);
    (0, [| 0.4435; 0.4; 0.1596; 0.2067 |], 0.47);
    (0, [| 0.5809; 0.81; 0.133; 0.1 |], 0.35);
    (0, [| 0.2217; 0.2; 0.0887; 0.1333 |], 0.21);
    (1, [| 0.8426; 0.6667; 0.1109; 0.1667 |], 0.12);
    (0, [| 0.133; 0.8333; 0.0665; 0.1 |], 0.09);
    (1, [| 0.9313; 0.1333; 0.0443; 0.0667 |], 0.05);
  |]

let n = Array.length raw

let floats fmt x =
  String.concat ", "
    (List.map (Printf.sprintf fmt) (Array.to_list (Nx.to_array x)))

let () =
  let img = Nx_io.load_image "../chelsea.png" in
  let h = Float.of_int (Nx.dim 0 img) and w = Float.of_int (Nx.dim 1 img) in
  let cxcywh =
    Nx.create Nx.float32 [| n; 4 |]
      (Array.concat (Array.to_list (Array.map (fun (_, b, _) -> b) raw)))
  in
  let scores =
    Nx.create Nx.float32 [| n |] (Array.map (fun (_, _, s) -> s) raw)
  in
  let labels =
    Nx.create Nx.int64 [| n |] (Array.map (fun (c, _, _) -> Int64.of_int c) raw)
  in

  (* The eight best scores, best first; equal scores would keep the lower
     index first. *)
  let scores, top = Nx.top_k ~k:8 scores in
  let labels = Nx.take ~axis:0 top labels in
  Printf.printf "the 8 best of %d: %s\n" n (floats "%Ld" top);

  (* Corners in the photograph's pixels: a fraction of the input's width is
     the same fraction of the photograph's. *)
  let size = Nx.create Nx.float32 [| 4 |] [| w; h; w; h |] in
  let boxes = Nx.mul (Mynd.Box.of_cxcywh (Nx.take ~axis:0 top cxcywh)) size in

  (* Each eye box's intersection over union with each other eye box. *)
  let eyes = Nx.argwhere (Nx.equal_s labels 0L) in
  let eyes = Nx.reshape [| Nx.dim 0 eyes |] eyes in
  let eye_boxes = Nx.take ~axis:0 eyes boxes in
  let iou = Mynd.Box.iou eye_boxes eye_boxes in
  Printf.printf "IoU of the eye boxes %s:\n"
    (floats "%Ld" (Nx.take ~axis:0 eyes top));
  for i = 0 to Nx.dim 0 iou - 1 do
    Printf.printf "  %s\n" (floats "%.2f" (Nx.slice [ Nx.I i ] iou))
  done;

  (* Greedy suppression, by class: a box goes when a better box of its class
     that is kept overlaps it with an IoU above 0.5. *)
  let keep = Mynd.Box.nms ~classes:labels ~iou:0.5 boxes scores in
  let kept = Nx.reshape [| -1 |] (Nx.argwhere keep) in
  Printf.printf "kept: %s\n" (floats "%Ld" (Nx.take ~axis:0 kept top));

  (* COCO's detections: [x, y, w, h] in pixels from the top left corner, and
     categories numbered from 1, here 1 for an eye and 2 for a nose. *)
  let xywh = Mynd.Box.to_xywh boxes in
  Array.iter
    (fun k ->
      let k = Int64.to_int k in
      Printf.printf
        "{\"image_id\": 0, \"category_id\": %d, \"bbox\": [%s], \"score\": %.2f}\n"
        (Int64.to_int (Nx.item [ k ] labels) + 1)
        (floats "%.1f" (Nx.slice [ Nx.I k ] xywh))
        (Nx.item [ k ] scores))
    (Nx.to_array kept);

  let kept_boxes = Nx.take ~axis:0 kept boxes in
  let side i = num (Nx.slice [ Nx.A; Nx.I i ] kept_boxes) in
  layer
    [
      image img;
      rect ~x:(side 0) ~y:(side 1) ~x2:(side 2) ~y2:(side 3)
        ~stroke:(const Color.white) ();
    ]
  |> save ~size:(Size.figure 451. 300.) "detections.png"
