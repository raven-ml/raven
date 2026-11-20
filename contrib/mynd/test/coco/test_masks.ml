(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Mynd_coco
open Support

let all v n = List.init n (fun _ -> v)
let below_one = Float.pred (Float.pred 1.)

let precisions ?(a = 0) e =
  List.concat_map
    (fun t -> List.init 101 (fun r -> precision e ~t ~r ~k:0 ~a ~m:2))
    (List.init 10 Fun.id)

let recalls ?(a = 0) e = List.init 10 (fun t -> recall e ~t ~k:0 ~a ~m:2)

(* [rect h w x0 y0 x1 y1] is the runs of the cells [x0, x1) × [y0, y1). *)
let rect h w x0 y0 x1 y1 =
  Mynd.Rle.encode
    (Nx.init Nx.bool [| h; w; 1 |] (fun i ->
         i.(1) >= x0 && i.(1) < x1 && i.(0) >= y0 && i.(0) < y1))

let empty h w = Mynd.Rle.v ~height:h ~width:w [| h * w |]

let ev ?(images = [ image ~height:20 ~width:30 1 ]) anns dets =
  evaluate_masks (dataset ~images anns) (Array.of_list dets)

let fixture_tables =
  let e = lazy (evaluate_masks (Lazy.force fixture) (Lazy.force mask_dets)) in
  let nk = Array.length (Lazy.force fixture).categories in
  cases "the tables are pycocotools' on val2017's images, bit for bit"
    ~id:string_of_int (List.init nk Fun.id) (fun k ->
      let p = of_coco (floats "eval24.npz" "segm.precision")
      and r = of_coco (floats "eval24.npz" "segm.recall") in
      let pick outer flat =
        List.init (outer * 12) (fun i ->
            flat.((((i / 12 * nk) + k) * 12) + (i mod 12)))
      in
      let ep, er = bits (Lazy.force e) in
      equal (list float_exact) (pick 1010 p) (pick 1010 ep);
      equal (list float_exact) (pick 10 r) (pick 10 er))

let iou_thresholds =
  [ 0.5; 0.55; 0.6; 0.65; 0.7; 0.75; 0.8; 0.85; 0x1.cccccccccccccp-1; 0.95 ]

let gen_mask h w =
  Gen.(
    let+ px =
      array
        ~size:(constant (h * w))
        (frequency [ (2, constant false); (3, constant true) ])
    in
    Mynd.Rle.encode (Nx.create Nx.bool [| h; w; 1 |] px))

let pixel_counts =
  prop "mask IoU is the quotient of exact pixel counts"
    Gen.(
      let* h, w = pair (int_range 1 6) (int_range 1 6) in
      let* d = gen_mask h w in
      let+ g =
        frequency
          [
            (2, map (fun r -> Runs r) (gen_mask h w));
            ( 1,
              map
                (fun p -> Polygons [ p ])
                (array ~size:(constant 8) (float_range (-1.) 7.)) );
          ]
      in
      (h, w, d, g))
    (fun (h, w, d, g) ->
      let gr = runs ~height:h ~width:w g in
      let iou =
        Nx.item [ 0; 0 ]
          (Mynd.Mask.iou Nx.float64
             (Nx.unsqueeze ~axes:[ 0 ] (Mynd.Rle.decode d))
             (Nx.unsqueeze ~axes:[ 0 ] (Mynd.Rle.decode gr)))
      in
      cover "a partial overlap" (iou > 0. && iou < 1.);
      cover "no overlap" (iou = 0.);
      let e =
        ev
          ~images:[ image ~height:h ~width:w 1 ]
          [
            ann ~area:1. ~segmentation:g ~image:1 ~category:1 (box 0. 0. 1. 1.);
          ]
          [ det ~image:1 ~category:1 0.5 d ]
      in
      equal (list float_exact)
        (List.map (fun t -> if iou >= t then 1. else 0.) iou_thresholds)
        (recalls e))

let crowd =
  test "against a crowd, IoU divides by the detection's pixels" (fun () ->
      (* A 4 by 4 detection of which 12 pixels lie in the crowd: 0.75. *)
      let crowd = rect 20 30 0 0 10 10 in
      let e =
        ev
          [
            ann ~crowd:true ~id:1 ~area:100. ~segmentation:(Runs crowd) ~image:1
              ~category:1 (box 0. 0. 10. 10.);
            ann ~id:2 ~area:16.
              ~segmentation:(Runs (rect 20 30 20 10 24 14))
              ~image:1 ~category:1 (box 20. 10. 4. 4.);
          ]
          [
            det ~image:1 ~category:1 0.9 (rect 20 30 7 0 11 4);
            det ~image:1 ~category:1 0.5 (rect 20 30 20 10 24 14);
          ]
      in
      (* Ignored where 0.75 reaches the threshold, a false positive above. *)
      equal (list float_exact)
        (List.map
           (fun t -> if 0.75 >= t then below_one else 0.5)
           iou_thresholds)
        (List.init 10 (fun t -> precision e ~t ~r:0 ~k:0 ~a:0 ~m:2)))

let areas =
  group "areas"
    [
      test "an annotation's range reads its area field" (fun () ->
          (* 30 pixels, an area of 2000: medium, never small. *)
          let m = rect 20 30 0 0 6 5 in
          let a =
            ann ~area:2000. ~segmentation:(Runs m) ~image:1 ~category:1
              (box 0. 0. 6. 5.)
          in
          let e = ev [ a ] [ det ~image:1 ~category:1 0.5 m ] in
          equal (list float_exact) (all 1. 10) (recalls ~a:2 e);
          equal (list float_exact) (all Float.nan 10) (recalls ~a:1 e));
      test "a mask detection's range reads its pixel count" (fun () ->
          (* An unmatched 36-pixel mask is a small false positive. *)
          let m = rect 20 30 0 0 4 4 in
          let a =
            ann ~area:16. ~segmentation:(Runs m) ~image:1 ~category:1
              (box 0. 0. 4. 4.)
          in
          let fp = det ~image:1 ~category:1 0.9 (rect 20 30 10 10 16 16) in
          let e = ev [ a ] [ fp; det ~image:1 ~category:1 0.5 m ] in
          equal (list float_exact) (all 0.5 1010) (precisions ~a:1 e));
      test "a results file's bbox does not range a mask" (fun () ->
          (* The 36-pixel false positive carries a 40 by 40 bbox. pycocotools
             would range it by the box's 1600, medium, and ignore it as
             small, giving 1 - 2⁻⁵²; its pixels make it small. *)
          let m = rect 20 30 0 0 4 4 in
          let a =
            ann ~area:16. ~segmentation:(Runs m) ~image:1 ~category:1
              (box 0. 0. 4. 4.)
          in
          let result score bbox r =
            Printf.sprintf
              "{\"image_id\":1,\"category_id\":1,\"score\":%g,\"bbox\":%s,\"segmentation\":{\"size\":[20,30],\"counts\":%S}}"
              score bbox (Mynd.Rle.to_string r)
          in
          let dets =
            decode mask_detections_jsont
              (Printf.sprintf "[%s,%s]"
                 (result 0.9 "[10,10,40,40]" (rect 20 30 10 10 16 16))
                 (result 0.5 "[0,0,4,4]" m))
          in
          let e =
            evaluate_masks
              (dataset ~images:[ image ~height:20 ~width:30 1 ] [ a ])
              dets
          in
          equal (list float_exact) (all 0.5 1010) (precisions ~a:1 e));
    ]

let empty_masks =
  group "an empty mask matches nothing"
    [
      test "against an empty annotation" (fun () ->
          let a =
            ann ~area:1.
              ~segmentation:(Runs (empty 20 30))
              ~image:1 ~category:1 (box 0. 0. 1. 1.)
          in
          let e = ev [ a ] [ det ~image:1 ~category:1 0.5 (empty 20 30) ] in
          equal (list float_exact) (all 0. 10) (recalls e));
      test "against an empty crowd" (fun () ->
          let crowd =
            ann ~crowd:true ~id:1 ~area:1.
              ~segmentation:(Runs (empty 20 30))
              ~image:1 ~category:1 (box 0. 0. 1. 1.)
          in
          let m = rect 20 30 0 0 4 4 in
          let a =
            ann ~id:2 ~area:16. ~segmentation:(Runs m) ~image:1 ~category:1
              (box 0. 0. 4. 4.)
          in
          let e =
            ev [ crowd; a ]
              [
                det ~image:1 ~category:1 0.9 (empty 20 30);
                det ~image:1 ~category:1 0.5 m;
              ]
          in
          equal (list float_exact) (all 0.5 1010) (precisions e));
    ]

let refusals =
  let raises f =
    raises_match (Exn.invalid_arg ~sub:"Mynd_coco.") (fun () -> ignore (f ()))
  in
  let one = ann ~area:16. ~image:1 ~category:1 (box 0. 0. 4. 4.) in
  group "misuse"
    [
      test "a mask of another size than its image" (fun () ->
          raises (fun () ->
              ev [ one ] [ det ~image:1 ~category:1 0.5 (empty 30 20) ]));
      test "an annotation's runs of another size" (fun () ->
          raises (fun () ->
              ev [ { one with segmentation = Runs (empty 2 2) } ] []));
      test "an annotation's polygon of an odd count" (fun () ->
          raises (fun () ->
              ev
                [ { one with segmentation = Polygons [ [| 1.; 1.; 2. |] ] } ]
                []));
      test "a NaN score" (fun () ->
          raises (fun () ->
              ev [ one ] [ det ~image:1 ~category:1 Float.nan (empty 20 30) ]));
    ]

let () =
  exit
    (run "mynd.coco masks"
       [ fixture_tables; pixel_counts; crowd; areas; empty_masks; refusals ])
