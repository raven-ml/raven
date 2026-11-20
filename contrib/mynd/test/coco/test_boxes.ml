(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Mynd_coco
open Support

let ev d dets = evaluate_boxes d (Array.of_list dets)

(* Every entry of a table at range [a] and limit 100, for category 0. *)
let precisions ?(a = 0) e =
  List.concat_map
    (fun t -> List.init 101 (fun r -> precision e ~t ~r ~k:0 ~a ~m:2))
    (List.init 10 Fun.id)

let recalls ?(a = 0) e = List.init 10 (fun t -> recall e ~t ~k:0 ~a ~m:2)
let all v n = List.init n (fun _ -> v)
let below_one = Float.pred (Float.pred 1.)

(* pycocotools on the fixture *)

let fixture_tables =
  let e = lazy (evaluate_boxes (Lazy.force fixture) (Lazy.force box_dets)) in
  let p = lazy (of_coco (floats "eval24.npz" "bbox.precision")) in
  let r = lazy (of_coco (floats "eval24.npz" "bbox.recall")) in
  let d = Lazy.force fixture in
  let nk = Array.length d.categories in
  cases "the tables are pycocotools' on val2017's images, bit for bit"
    ~id:string_of_int (List.init nk Fun.id) (fun k ->
      let e = Lazy.force e in
      let p = Lazy.force p and r = Lazy.force r in
      (* Category [k]'s entries of a table whose axes before the
         category's hold [outer] entries. *)
      let pick outer flat =
        List.init (outer * 12) (fun i ->
            flat.((((i / 12 * nk) + k) * 12) + (i mod 12)))
      in
      let ep, er = bits e in
      equal (list float_exact) (pick 1010 p) (pick 1010 ep);
      equal (list float_exact) (pick 10 r) (pick 10 er))

(* Thresholds *)

let iou_thresholds =
  [ 0.5; 0.55; 0.6; 0.65; 0.7; 0.75; 0.8; 0.85; 0x1.cccccccccccccp-1; 0.95 ]

let iou_steps =
  cases "a box matches at the IoU thresholds up to its IoU" ~id:string_of_int
    [ 50; 55; 60; 65; 70; 75; 80; 85; 90; 95; 100 ] (fun w ->
      (* IoU w / 100, rounded once, of a w by 100 box in a 100 by 100 one. *)
      let iou = Float.of_int w /. 100. in
      let d =
        dataset
          ~images:[ image ~height:200 ~width:200 1 ]
          [ ann ~image:1 ~category:1 (box 0. 0. 100. 100.) ]
      in
      let e =
        ev d [ det ~image:1 ~category:1 0.5 (box 0. 0. (Float.of_int w) 100.) ]
      in
      equal (list float_exact)
        (List.map (fun t -> if iou >= t then 1. else 0.) iou_thresholds)
        (recalls e))

let bumped = [ 35; 41; 47; 57; 69; 70; 82; 83; 94; 95 ]

let recall_steps =
  cases "a recall of exactly i / 100 reaches the ith recall threshold, but ten"
    ~id:string_of_int
    (List.init 100 (fun i -> i + 1))
    (fun i ->
      (* 100 images of one annotation each, the first i found. *)
      let d =
        dataset
          ~images:(List.init 100 (fun j -> image (j + 1)))
          (List.init 100 (fun j ->
               ann ~id:(j + 1) ~image:(j + 1) ~category:1 (box 0. 0. 10. 10.)))
      in
      let dets =
        List.init i (fun j ->
            det ~image:(j + 1) ~category:1 0.5 (box 0. 0. 10. 10.))
      in
      let e = ev d dets in
      let fi = Float.of_int i in
      let reached = fi /. (fi +. epsilon_float) in
      equal float_exact
        (if List.mem i bumped then 0. else reached)
        (precision e ~t:0 ~r:i ~k:0 ~a:0 ~m:2))

(* Matching *)

let later_of_equal =
  test "a detection takes the later of equal IoUs" (fun () ->
      (* The first detection has IoU 2/3 with both annotations and takes the
         second, which leaves the first to the second detection. *)
      let d =
        dataset
          [
            ann ~id:1 ~image:1 ~category:1 (box 0. 0. 10. 10.);
            ann ~id:2 ~image:1 ~category:1 (box 4. 0. 10. 10.);
          ]
      in
      let e =
        ev d
          [
            det ~image:1 ~category:1 0.9 (box 2. 0. 10. 10.);
            det ~image:1 ~category:1 0.8 (box 0. 0. 10. 10.);
          ]
      in
      equal (list float_exact)
        (List.map (fun t -> if t <= 2. /. 3. then 1. else 0.5) iou_thresholds)
        (recalls e))

let range_bounds =
  cases "an annotation's area range includes its bounds" ~id:fst
    [
      ("32²", (32., [ 1.; 1.; 1.; Float.nan ]));
      ("96²", (96., [ 1.; Float.nan; 1.; 1. ]));
    ]
    (fun (_, (side, want)) ->
      let b = box 0. 0. side side in
      let d = dataset [ ann ~area:(side *. side) ~image:1 ~category:1 b ] in
      let e = ev d [ det ~image:1 ~category:1 0.5 b ] in
      equal (list float_exact) want
        (List.init 4 (fun a -> recall e ~t:0 ~k:0 ~a ~m:2)))

(* The precision eps *)

let eps =
  group "precision adds 2⁻⁵² to its divisor"
    [
      test "one annotation found gives 1 - 2⁻⁵²" (fun () ->
          let d = dataset [ ann ~image:1 ~category:1 (box 5. 5. 10. 10.) ] in
          let e = ev d [ det ~image:1 ~category:1 0.9 (box 5. 5. 10. 10.) ] in
          equal (list float_exact) (all below_one 1010) (precisions e));
      test "two annotations found give 1" (fun () ->
          let d =
            dataset
              [
                ann ~id:1 ~image:1 ~category:1 (box 5. 5. 10. 10.);
                ann ~id:2 ~image:1 ~category:1 (box 50. 5. 10. 10.);
              ]
          in
          let e =
            ev d
              [
                det ~image:1 ~category:1 0.9 (box 5. 5. 10. 10.);
                det ~image:1 ~category:1 0.8 (box 50. 5. 10. 10.);
              ]
          in
          equal (list float_exact) (all 1. 1010) (precisions e));
    ]

(* Order *)

let ties =
  let d = dataset [ ann ~image:1 ~category:1 (box 0. 0. 10. 10.) ] in
  let tp = det ~image:1 ~category:1 0.5 (box 0. 0. 10. 10.) in
  let fp = det ~image:1 ~category:1 0.5 (box 50. 50. 10. 10.) in
  group "equal scores rank in array order"
    [
      test "a false positive first" (fun () ->
          equal (list float_exact) (all 0.5 1010) (precisions (ev d [ fp; tp ])));
      test "a true positive first" (fun () ->
          equal (list float_exact) (all below_one 1010)
            (precisions (ev d [ tp; fp ])));
    ]

let same e e' =
  let p, r = bits e and p', r' = bits e' in
  equal (array float_exact) p p';
  equal (array float_exact) r r'

let shuffle_dets =
  prop "detections of distinct scores evaluate alike in any order"
    Gen.(
      let* d, dets = gen_case ~ties:false in
      let+ order = permutation (List.init (Array.length dets) Fun.id) in
      (d, dets, Array.of_list (List.map (fun i -> dets.(i)) order)))
    (fun (d, dets, dets') ->
      cover "several detections" (Array.length dets >= 3);
      same (evaluate_boxes d dets) (evaluate_boxes d dets'))

(* Annotations keep their order within an image, which ties between equal
   IoUs read. *)
let interleave key (anns : annotation array) =
  let a =
    Array.mapi (fun i (x : annotation) -> (key.(x.image - 1), i, x)) anns
  in
  Array.stable_sort (fun (k, _, _) (k', _, _) -> Int.compare k k') a;
  Array.map (fun (_, _, x) -> x) a

let shuffle_dataset =
  prop "a dataset's images, categories and annotations evaluate in any order"
    Gen.(
      let* d, dets = gen_case ~ties:true in
      let* io = permutation (Array.to_list d.images) in
      let* co = permutation (Array.to_list d.categories) in
      let+ key =
        array ~size:(constant (Array.length d.images)) (int_range 0 3)
      in
      ( d,
        {
          images = Array.of_list io;
          categories = Array.of_list co;
          annotations = interleave key d.annotations;
        },
        dets ))
    (fun (d, d', dets) ->
      cover "several images" (Array.length d.images >= 2);
      same (evaluate_boxes d dets) (evaluate_boxes d' dets))

(* Laws *)

let found =
  prop "detections equal to the annotations find every one"
    Gen.(
      let+ d, _ = gen_case ~ties:false in
      let anns =
        Array.map
          (fun (a : annotation) -> { a with crowd = false })
          d.annotations
      in
      { d with annotations = anns })
    (fun d ->
      let dets =
        Array.mapi
          (fun j (a : annotation) ->
            det ~image:a.image ~category:a.category
              (Float.of_int (j + 1) /. 64.)
              a.box)
          d.annotations
      in
      let e = evaluate_boxes d dets in
      Array.iteri
        (fun k (c : category) ->
          let n =
            Array.fold_left
              (fun n (a : annotation) -> if a.category = c.id then n + 1 else n)
              0 d.annotations
          in
          cover "a category of two annotations or more" (n >= 2);
          if n >= 2 then begin
            equal (list float_exact) (all 1. 10)
              (List.init 10 (fun t -> recall e ~t ~k ~a:0 ~m:2));
            equal (list float_exact) (all 1. 1010)
              (List.concat_map
                 (fun t ->
                   List.init 101 (fun r -> precision e ~t ~r ~k ~a:0 ~m:2))
                 (List.init 10 Fun.id))
          end)
        e.categories)

let unrelated =
  prop "a detection of a category without annotations changes nothing"
    Gen.(
      let+ d, dets = gen_case ~ties:false
      and+ b = map (fun x -> box (Float.of_int x) 0. 5. 5.) (int_range 0 30) in
      (d, dets, b))
    (fun (d, dets, b) ->
      let d =
        { d with categories = Array.append d.categories [| category 9 |] }
      in
      let extra = det ~image:1 ~category:9 0.95 b in
      same (evaluate_boxes d dets)
        (evaluate_boxes d (Array.append dets [| extra |])))

(* Crowds, ranges and boundaries *)

let crowd =
  test "detections inside a crowd count neither way" (fun () ->
      let d =
        dataset
          [
            ann ~crowd:true ~id:1 ~image:1 ~category:1 (box 0. 0. 50. 50.);
            ann ~id:2 ~image:1 ~category:1 (box 60. 60. 10. 10.);
          ]
      in
      let inside j =
        det ~image:1 ~category:1 0.9 (box (Float.of_int (8 * j)) 5. 10. 10.)
      in
      let e =
        ev d
          (List.init 5 inside
          @ [ det ~image:1 ~category:1 0.5 (box 60. 60. 10. 10.) ])
      in
      equal (list float_exact) (all below_one 1010) (precisions e))

let boundaries =
  let found_with fp ~a ~ann_box =
    let d = dataset [ ann ~image:1 ~category:1 ann_box ] in
    let e = ev d [ fp; det ~image:1 ~category:1 0.5 ann_box ] in
    precisions ~a e
  in
  group "boundaries"
    [
      test "an unmatched 32 by 32 box counts as small and medium" (fun () ->
          let fp = det ~image:1 ~category:1 0.9 (box 60. 60. 32. 32.) in
          equal (list float_exact) (all 0.5 1010)
            (found_with fp ~a:1 ~ann_box:(box 0. 0. 10. 10.));
          equal (list float_exact) (all 0.5 1010)
            (found_with fp ~a:2 ~ann_box:(box 0. 0. 40. 40.)));
      test "an unmatched 31 by 31 box is ignored as medium" (fun () ->
          let fp = det ~image:1 ~category:1 0.9 (box 60. 60. 31. 31.) in
          equal (list float_exact) (all below_one 1010)
            (found_with fp ~a:2 ~ann_box:(box 0. 0. 40. 40.)));
      test "a detection in an image without annotations" (fun () ->
          let d =
            dataset
              ~images:[ image 1; image 2 ]
              [ ann ~image:1 ~category:1 (box 0. 0. 10. 10.) ]
          in
          let e =
            ev d
              [
                det ~image:2 ~category:1 0.9 (box 0. 0. 10. 10.);
                det ~image:1 ~category:1 0.5 (box 0. 0. 10. 10.);
              ]
          in
          equal (list float_exact) (all 0.5 1010) (precisions e));
      test "no detections" (fun () ->
          let e =
            ev (dataset [ ann ~image:1 ~category:1 (box 0. 0. 10. 10.) ]) []
          in
          equal (list float_exact) (all 0. 1010) (precisions e);
          equal (list float_exact) (all 0. 10) (recalls e));
      test "a category with nothing under a range is NaN there" (fun () ->
          let e =
            ev (dataset [ ann ~image:1 ~category:1 (box 0. 0. 10. 10.) ]) []
          in
          equal (list float_exact) (all Float.nan 1010) (precisions ~a:3 e);
          equal (list float_exact) (all Float.nan 10) (recalls ~a:3 e));
      test "a box of no area matches nothing, itself included" (fun () ->
          let b = box 5. 5. 0. 10. in
          let e =
            ev
              (dataset [ ann ~image:1 ~category:1 b ])
              [ det ~image:1 ~category:1 0.5 b ]
          in
          equal (list float_exact) (all 0. 10) (recalls e));
    ]

(* Departures from pycocotools *)

let positions =
  group "annotations are told apart by position"
    [
      test "annotations of id 0 are found" (fun () ->
          let d =
            dataset
              [
                ann ~id:0 ~image:1 ~category:1 (box 0. 0. 10. 10.);
                ann ~id:0 ~image:1 ~category:1 (box 50. 0. 10. 10.);
              ]
          in
          let e =
            ev d
              [
                det ~image:1 ~category:1 0.9 (box 0. 0. 10. 10.);
                det ~image:1 ~category:1 0.8 (box 50. 0. 10. 10.);
              ]
          in
          equal (list float_exact) (all 1. 1010) (precisions e));
      test "annotations of one id in two images are both found" (fun () ->
          let d =
            dataset
              ~images:[ image 1; image 2 ]
              [
                ann ~id:4 ~image:1 ~category:1 (box 0. 0. 10. 10.);
                ann ~id:4 ~image:2 ~category:1 (box 50. 0. 10. 10.);
              ]
          in
          let e =
            ev d
              [
                det ~image:1 ~category:1 0.9 (box 0. 0. 10. 10.);
                det ~image:2 ~category:1 0.8 (box 50. 0. 10. 10.);
              ]
          in
          equal (list float_exact) (all 1. 1010) (precisions e));
    ]

let refusals =
  let raises f =
    raises_match (Exn.invalid_arg ~sub:"Mynd_coco.evaluate_boxes") (fun () ->
        ignore (f ()))
  in
  let one = ann ~image:1 ~category:1 (box 0. 0. 10. 10.) in
  let good = det ~image:1 ~category:1 0.5 (box 0. 0. 10. 10.) in
  let with_ann a () = ev (dataset [ a ]) [ good ] in
  let with_det dt () = ev (dataset [ one ]) [ dt ] in
  group "misuse"
    [
      test "a detection of an unlisted category" (fun () ->
          raises (with_det { good with category = 2 }));
      test "a detection of an unlisted image" (fun () ->
          raises (with_det { good with image = 2 }));
      test "a NaN score" (fun () ->
          raises (with_det { good with score = Float.nan }));
      test "a detection box of negative or non-finite size" (fun () ->
          raises (with_det { good with region = box 0. 0. (-1.) 10. });
          raises (with_det { good with region = box 0. 0. 10. Float.nan });
          raises (with_det { good with region = box Float.infinity 0. 1. 1. }));
      test "a detection whose area lies past every range" (fun () ->
          raises (with_det { good with region = box 0. 0. 2e5 2e5 }));
      test "an annotation box of negative or non-finite size" (fun () ->
          raises (with_ann { one with box = box 0. 0. 10. (-0.5) });
          raises (with_ann { one with box = box Float.nan 0. 10. 10. }));
      test "an annotation area that is NaN, infinite or negative" (fun () ->
          raises (with_ann { one with area = Float.nan });
          raises (with_ann { one with area = Float.infinity });
          raises (with_ann { one with area = -1. }));
      test "an annotation of an unlisted image or category" (fun () ->
          raises (with_ann { one with image = 3 });
          raises (with_ann { one with category = 3 }));
      test "two images or two categories of one id" (fun () ->
          raises (fun () ->
              ev (dataset ~images:[ image 1; image 1 ] [ one ]) []);
          raises (fun () ->
              ev (dataset ~categories:[ category 1; category 1 ] [ one ]) []));
    ]

let () =
  exit
    (run "mynd.coco boxes"
       [
         fixture_tables;
         iou_steps;
         recall_steps;
         later_of_equal;
         range_bounds;
         eps;
         ties;
         shuffle_dets;
         shuffle_dataset;
         found;
         unrelated;
         crowd;
         boundaries;
         positions;
         refusals;
       ])
