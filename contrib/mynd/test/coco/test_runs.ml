(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Mynd_coco
open Support

(* golden/polygons.npz: polygons and their masks' runs by rleFrPoly, each
   operation rounded on its own. The last [fused] polygons are those whose
   mask changes when a multiply-add is fused. *)
let polygons =
  lazy
    (let f = "polygons.npz" in
     let lengths = ints f "lengths" in
     let coords = split lengths (floats f "coords") in
     let polys = split (ints f "parts") coords in
     let counts = split (ints f "counts_lengths") (ints f "counts") in
     let h = ints f "height" and w = ints f "width" in
     ( Array.init (Array.length h) (fun i ->
           (i, h.(i), w.(i), Array.to_list polys.(i), counts.(i))),
       (ints f "fused").(0) ))

let check_case (_, height, width, parts, counts) =
  let r = runs ~height ~width (Polygons parts) in
  equal (array int) counts r.counts

let golden =
  let rows = Array.to_list (fst (Lazy.force polygons)) in
  cases "pycocotools' masks of drawn and named polygons"
    ~id:(fun (i, _, _, _, _) -> string_of_int i)
    rows check_case

let one_rounding =
  let rows, fused = Lazy.force polygons in
  let n = Array.length rows in
  cases "polygons whose mask a fused multiply-add changes"
    ~id:(fun (i, _, _, _, _) -> string_of_int i)
    (Array.to_list (Array.sub rows (n - fused) fused))
    check_case

let fixture_runs =
  test "the runs of val2017's annotations are pycocotools'" (fun () ->
      let d = Lazy.force fixture in
      let counts =
        split
          (ints "eval24.npz" "runs.lengths")
          (ints "eval24.npz" "runs.counts")
      in
      let size = Hashtbl.create 32 in
      Array.iter
        (fun (i : image) -> Hashtbl.add size i.id (i.height, i.width))
        d.images;
      Array.iteri
        (fun j (a : annotation) ->
          let height, width = Hashtbl.find size a.image in
          let r = runs ~height ~width a.segmentation in
          equal
            ~msg:(Printf.sprintf "annotation %d" a.id)
            (array int) counts.(j) r.counts)
        d.annotations)

(* Laws *)

let mask_of (r : Mynd.Rle.t) = Nx.to_array (Mynd.Rle.decode r)

let rectangles =
  prop "a rectangle of integer corners covers its cells in the image"
    Gen.(
      let* h, w = pair (int_range 0 9) (int_range 0 9) in
      let* inside = bool in
      (* Half the draws keep the corners in the image where it has pixels. *)
      let corners n =
        if inside && n > 0 then
          let* a = int_range 0 (n - 1) in
          let+ b = int_range (a + 1) n in
          (a, b)
        else
          let* a = int_range (-3) (n + 2) in
          let+ b = int_range (a + 1) (n + 3) in
          (a, b)
      in
      let+ x0, x1 = corners w and+ y0, y1 = corners h in
      (h, w, x0, y0, x1, y1))
    (fun (h, w, x0, y0, x1, y1) ->
      cover "partly outside" (x0 < 0 || y0 < 0 || x1 > w || y1 > h);
      cover "inside" (x0 >= 0 && y0 >= 0 && x1 <= w && y1 <= h && h * w > 0);
      let f = Float.of_int in
      let p = [| f x0; f y0; f x1; f y0; f x1; f y1; f x0; f y1 |] in
      let r = runs ~height:h ~width:w (Polygons [ p ]) in
      let want =
        Array.init (h * w) (fun k ->
            let i = k / w and j = k mod w in
            j >= x0 && j < x1 && i >= y0 && i < y1)
      in
      equal (array bool) want (mask_of r))

let few_vertices =
  prop "a polygon of fewer than three vertices covers no pixel"
    Gen.(
      let* k = int_range 0 2 in
      array ~size:(constant (2 * k)) (float_range (-20.) 40.))
    (fun p ->
      cover "two vertices" (Array.length p = 4);
      let r = runs ~height:23 ~width:31 (Polygons [ p ]) in
      equal int 0 (Mynd.Rle.area r))

let gen_parts =
  Gen.(
    let* h, w = pair (int_range 1 12) (int_range 1 12) in
    let poly =
      let* k = int_range 1 6 in
      array ~size:(constant (2 * k)) (float_range (-2.) 14.)
    in
    let+ ps = list ~size:(int_range 0 4) poly in
    (h, w, ps))

let union =
  prop "polygons cover the union of their masks, in any order" gen_parts
    (fun (height, width, ps) ->
      cover "several parts" (List.length ps >= 2);
      let r = runs ~height ~width (Polygons ps) in
      let parts = List.map (fun p -> runs ~height ~width (Polygons [ p ])) ps in
      let union =
        List.fold_left
          (fun acc p -> Nx.logical_or acc (Mynd.Rle.decode p))
          (Nx.zeros Nx.bool [| height; width; 1 |])
          parts
      in
      equal (array int) (Mynd.Rle.encode union).counts r.counts;
      let r' = runs ~height ~width (Polygons (List.rev ps)) in
      equal (array int) r.counts r'.counts)

let runs_pass =
  test "runs are their own mask" (fun () ->
      let r = Mynd.Rle.v ~height:2 ~width:3 [| 0; 2; 4 |] in
      equal (array int) r.counts (runs ~height:2 ~width:3 (Runs r)).counts)

let refusals =
  let raises f = raises_match (Exn.invalid_arg ~sub:"Mynd_coco.runs") f in
  let poly p () = ignore (runs ~height:10 ~width:10 (Polygons [ p ])) in
  group "misuse"
    [
      test "an odd number of coordinates" (fun () ->
          raises (poly [| 1.; 1.; 5.; 1.; 5.; 5.; 1. |]));
      test "a NaN coordinate" (fun () ->
          raises (poly [| 1.; 1.; Float.nan; 1.; 5.; 5. |]));
      test "an infinite coordinate" (fun () ->
          raises (poly [| 1.; 1.; 5.; Float.infinity; 5.; 5. |]));
      test "a coordinate whose grid point passes 2³¹" (fun () ->
          raises (poly [| 1.; 1.; 429496729.5; 1.; 5.; 5. |]);
          raises (poly [| 1.; -429496729.7; 5.; 1.; 5.; 5. |]));
      test "a negative size" (fun () ->
          raises (fun () -> ignore (runs ~height:(-1) ~width:2 (Polygons []))));
      test "runs of another size" (fun () ->
          let r = Mynd.Rle.v ~height:2 ~width:3 [| 6 |] in
          raises (fun () -> ignore (runs ~height:3 ~width:2 (Runs r))));
    ]

let () =
  exit
    (run "mynd.coco runs"
       [
         golden;
         one_rounding;
         fixture_runs;
         rectangles;
         few_vertices;
         union;
         runs_pass;
         refusals;
       ])
