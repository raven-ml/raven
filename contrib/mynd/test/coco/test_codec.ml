(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Mynd_coco
open Support

let json jsont =
  Testable.make
    ~pp:(fun ppf v -> Format.pp_print_string ppf (encode jsont v))
    ~equal:( = )

let fails jsont s =
  match Jsont_bytesrw.decode_string jsont s with
  | Ok _ -> failf "decoded %s" s
  | Error _ -> ()

(* The fixture *)

let fixture_decodes =
  test "val2017's file decodes" (fun () ->
      let d = Lazy.force fixture in
      let n = Array.length (ints "eval24.npz" "runs.lengths") in
      equal int 24 (Array.length d.images);
      equal int n (Array.length d.annotations);
      let crowds =
        Array.to_list d.annotations |> List.filter (fun a -> a.crowd)
      in
      equal int 2 (List.length crowds);
      List.iter
        (fun a ->
          match a.segmentation with
          | Runs _ -> ()
          | Polygons _ -> fail "a crowd's segmentation decodes as polygons")
        crowds;
      equal (json dataset_jsont) d
        (decode dataset_jsont (encode dataset_jsont d)))

let results_decode =
  test "a results file decodes in its order, for boxes and masks" (fun () ->
      let b = Lazy.force box_dets and m = Lazy.force mask_dets in
      equal int (Array.length b) (Array.length m);
      Array.iteri
        (fun i (d : box detection) ->
          let (e : Mynd.Rle.t detection) = m.(i) in
          equal int d.image e.image;
          equal int d.category e.category;
          equal float_exact d.score e.score)
        b)

(* Round trips *)

let gen_float = Gen.float_range (-1000.) 1000.

let gen_box =
  Gen.(
    let+ x = gen_float
    and+ y = gen_float
    and+ w = float_range 0. 500.
    and+ h = float_range 0. 500. in
    box x y w h)

let gen_runs =
  Gen.(
    let* h, w = pair (int_range 0 6) (int_range 0 6) in
    let+ px = array ~size:(constant (h * w)) bool in
    Mynd.Rle.encode (Nx.create Nx.bool [| h; w; 1 |] px))

let gen_segmentation =
  Gen.(
    frequency
      [
        ( 3,
          map
            (fun ps -> Polygons ps)
            (list ~size:(int_range 0 3)
               (array ~size:(map (fun n -> 2 * n) (int_range 0 6)) gen_float))
        );
        (1, map (fun r -> Runs r) gen_runs);
      ])

let gen_dataset =
  Gen.(
    let image =
      let+ id = int and+ h = int_range 0 1000 and+ w = int_range 0 1000 in
      image ~height:h ~width:w id
    in
    let category =
      let+ id = int
      and+ name = string_of ~size:(int_range 0 8) (char_range 'a' 'z') in
      ({ id; name } : category)
    in
    let annotation =
      let+ id = int
      and+ image = int
      and+ category = int
      and+ b = gen_box
      and+ area = float_range 0. 1e6
      and+ crowd = bool
      and+ segmentation = gen_segmentation in
      { id; image; category; box = b; area; crowd; segmentation }
    in
    let+ images = array ~size:(int_range 0 3) image
    and+ categories = array ~size:(int_range 0 3) category
    and+ annotations = array ~size:(int_range 0 4) annotation in
    { images; categories; annotations })

let gen_detections region =
  Gen.(
    array ~size:(int_range 0 5)
      (let+ image = int
       and+ category = int
       and+ score = float_range 0. 1.
       and+ region = region in
       { image; category; score; region }))

let round_trips =
  group "decoding what was encoded gives it back"
    [
      prop "datasets" gen_dataset (fun d ->
          cover "a crowd" (Array.exists (fun a -> a.crowd) d.annotations);
          equal (json dataset_jsont) d
            (decode dataset_jsont (encode dataset_jsont d)));
      prop "box detections" (gen_detections gen_box) (fun d ->
          equal
            (json box_detections_jsont)
            d
            (decode box_detections_jsont (encode box_detections_jsont d)));
      prop "mask detections" (gen_detections gen_runs) (fun d ->
          equal
            (json mask_detections_jsont)
            d
            (decode mask_detections_jsont (encode mask_detections_jsont d)));
    ]

(* Members *)

let ann_json ?(extra = "")
    ?(fields = "\"area\":4,\"bbox\":[1,2,2,2],\"iscrowd\":0") () =
  Printf.sprintf "{\"id\":7,\"image_id\":1,\"category_id\":2,%s%s}" fields extra

let file anns =
  Printf.sprintf
    "{\"images\":[{\"id\":1,\"file_name\":\"a.jpg\",\"height\":4,\"width\":3}],\"categories\":[{\"id\":2,\"name\":\"cat\",\"supercategory\":\"x\"}],\"annotations\":[%s],\"info\":{}}"
    anns

let members =
  group "members"
    [
      test "other members are skipped" (fun () ->
          let d =
            decode dataset_jsont (file (ann_json ~extra:",\"x\":[1]" ()))
          in
          equal int 1 (Array.length d.annotations);
          equal float_exact 4. d.annotations.(0).area);
      test "an absent segmentation is no polygon" (fun () ->
          let d = decode dataset_jsont (file (ann_json ())) in
          equal (json segmentation_jsont) (Polygons [])
            d.annotations.(0).segmentation);
      test "an image-information file has no annotations" (fun () ->
          let d =
            decode dataset_jsont
              "{\"images\":[{\"id\":1,\"file_name\":\"a.jpg\",\"height\":4,\"width\":3}],\"categories\":[]}"
          in
          equal int 0 (Array.length d.annotations));
      test "counts as a list and as a string decode alike" (fun () ->
          let a =
            decode segmentation_jsont "{\"size\":[4,3],\"counts\":[1,2,9]}"
          and r = Mynd.Rle.v ~height:4 ~width:3 [| 1; 2; 9 |] in
          let b =
            decode segmentation_jsont
              (Printf.sprintf "{\"size\":[4,3],\"counts\":%S}"
                 (Mynd.Rle.to_string r))
          in
          equal (json segmentation_jsont) (Runs r) a;
          equal (json segmentation_jsont) a b);
      test "runs encode as a string" (fun () ->
          let r = Mynd.Rle.v ~height:4 ~width:3 [| 1; 2; 9 |] in
          equal string
            (Printf.sprintf "{\"size\":[4,3],\"counts\":%S}"
               (Mynd.Rle.to_string r))
            (encode segmentation_jsont (Runs r)));
      test "a box is [x, y, w, h]" (fun () ->
          equal (json box_jsont) (box 1.5 2. 3. 4.25)
            (decode box_jsont "[1.5,2,3,4.25]"));
      test "a mask result's bbox is skipped" (fun () ->
          let r = Mynd.Rle.v ~height:2 ~width:2 [| 1; 2; 1 |] in
          let d =
            decode mask_detections_jsont
              (Printf.sprintf
                 "[{\"image_id\":1,\"category_id\":2,\"score\":0.5,\"bbox\":[0,0,9,9],\"segmentation\":{\"size\":[2,2],\"counts\":%S}}]"
                 (Mynd.Rle.to_string r))
          in
          equal (array int) r.counts d.(0).region.counts);
    ]

let refusals =
  group "refusals"
    [
      test "a missing area" (fun () ->
          fails dataset_jsont
            (file (ann_json ~fields:"\"bbox\":[1,2,2,2],\"iscrowd\":0" ())));
      test "a missing bbox" (fun () ->
          fails dataset_jsont
            (file (ann_json ~fields:"\"area\":4,\"iscrowd\":0" ())));
      test "a missing iscrowd" (fun () ->
          fails dataset_jsont
            (file (ann_json ~fields:"\"area\":4,\"bbox\":[1,2,2,2]" ())));
      test "an iscrowd of 2" (fun () ->
          fails dataset_jsont
            (file
               (ann_json ~fields:"\"area\":4,\"bbox\":[1,2,2,2],\"iscrowd\":2"
                  ())));
      test "an image without file_name" (fun () ->
          fails dataset_jsont
            "{\"images\":[{\"id\":1,\"height\":4,\"width\":3}],\"categories\":[]}");
      test "a bbox of three numbers" (fun () -> fails box_jsont "[1,2,3]");
      test "counts that do not cover the size" (fun () ->
          fails segmentation_jsont "{\"size\":[4,3],\"counts\":[1,2]}");
      test "a string outside the alphabet" (fun () ->
          fails segmentation_jsont "{\"size\":[1,1],\"counts\":\" \"}");
    ]

let () =
  exit
    (run "mynd.coco codecs"
       [ fixture_decodes; results_decode; round_trips; members; refusals ])
