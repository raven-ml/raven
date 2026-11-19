(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module V = Mynd

(* torchvision's box IoU and pycocotools' RLE, areas, boxes and mask IoU, from
   golden/annot.npz (contrib/mynd/test/gen/annot.py) *)
let golden = lazy (Nx_io.load_npz "golden/annot.npz")

let entry dt key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack dt p
  | None -> failf "golden/annot.npz has no %s" key

let floats x = Nx.to_array (Nx.cast Nx.float64 x)

let close tol want got =
  equal (array int) (Nx.shape want) (Nx.shape got);
  equal
    (array (if tol = 0. then float_exact else float tol))
    (floats want) (floats got)

let boxes a = Nx.create Nx.float32 [| Array.length a / 4; 4 |] a

(* Boxes *)

let conversions =
  test "conversions" (fun () ->
      let b = boxes [| 1.; 2.; 4.; 8.; -3.; 0.5; -1.; 0.75 |] in
      equal (array float_exact)
        [| 1.; 2.; 3.; 6.; -3.; 0.5; 2.; 0.25 |]
        (floats (V.Box.to_xywh b));
      equal (array float_exact)
        [| 2.5; 5.; 3.; 6.; -2.; 0.625; 2.; 0.25 |]
        (floats (V.Box.to_cxcywh b));
      equal (array float_exact) (floats b)
        (floats (V.Box.of_xywh (V.Box.to_xywh b)));
      equal (array float_exact) (floats b)
        (floats (V.Box.of_cxcywh (V.Box.to_cxcywh b))))

let gen_box =
  Gen.(
    let+ x = float_range (-50.) 50.
    and+ y = float_range (-50.) 50.
    and+ w = float_range (-5.) 40.
    and+ h = float_range (-5.) 40. in
    [| x; y; x +. w; y +. h |])

let areas =
  prop "area is the product of the sides, 0 when empty" gen_box (fun b ->
      let w = b.(2) -. b.(0) and h = b.(3) -. b.(1) in
      cover "empty" (w <= 0. || h <= 0.);
      let want = if w <= 0. || h <= 0. then 0. else w *. h in
      equal float_exact want
        (Nx.item [ 0 ] (V.Box.area (Nx.create Nx.float64 [| 1; 4 |] b))))

let iou_golden =
  test "box IoU is torchvision's" (fun () ->
      let a = entry Nx.float32 "box.a" and b = entry Nx.float32 "box.b" in
      close 0. (entry Nx.float32 "box.iou") (V.Box.iou a b))

(* IoU against its definition in float64, over boxes on a grid of eighths, so
   that every area is exact. *)
let iou_law =
  prop "IoU is the intersection over the union"
    Gen.(pair gen_box gen_box)
    (fun (a, b) ->
      let q a = Array.map (fun v -> Float.round (v *. 8.) /. 8.) a in
      let a = q a and b = q b in
      let area a =
        Float.max 0. (a.(2) -. a.(0)) *. Float.max 0. (a.(3) -. a.(1))
      in
      let area a = if a.(2) <= a.(0) || a.(3) <= a.(1) then 0. else area a in
      let i =
        Float.max 0. (Float.min a.(2) b.(2) -. Float.max a.(0) b.(0))
        *. Float.max 0. (Float.min a.(3) b.(3) -. Float.max a.(1) b.(1))
      in
      let u = area a +. area b -. i in
      cover "disjoint" (i = 0.);
      cover "overlapping" (i > 0.);
      let want = if u = 0. then 0. else i /. u in
      let got =
        V.Box.iou
          (Nx.create Nx.float64 [| 1; 4 |] a)
          (Nx.create Nx.float64 [| 1; 4 |] b)
      in
      equal (array int) [| 1; 1 |] (Nx.shape got);
      equal float_exact want (Nx.item [ 0; 0 ] got))

let iou_batches =
  test "IoU pairs every box with every box, batch axes broadcast" (fun () ->
      let a =
        Nx.create Nx.float64 [| 2; 3; 4 |]
          (Array.init 24 (fun i ->
               Float.of_int (i mod 7) *. if i mod 4 >= 2 then 2. else 1.))
      in
      let b =
        Nx.create Nx.float64 [| 1; 2; 4 |] [| 0.; 0.; 4.; 4.; 1.; 1.; 6.; 9. |]
      in
      let r = V.Box.iou a b in
      equal (array int) [| 2; 3; 2 |] (Nx.shape r);
      equal float_exact
        (Nx.item [ 1; 2; 1 ] r)
        (Nx.item [ 0; 0 ]
           (V.Box.iou
              (Nx.slice [ Nx.I 1; Nx.R (2, 3) ] a)
              (Nx.slice [ Nx.I 0; Nx.R (1, 2) ] b))))

let empty_iou =
  test "empty boxes have IoU 0" (fun () ->
      let z = boxes [| 1.; 1.; 1.; 5. |] in
      equal (array float_exact) [| 0. |] (floats (V.Box.iou z z)))

let of_mask =
  test "a mask's box" (fun () ->
      let m = Nx.zeros Nx.bool [| 3; 5; 6; 1 |] in
      let m =
        Nx.set [ Nx.I 0; Nx.R (1, 3); Nx.R (2, 5) ] (Nx.scalar Nx.bool true) m
      in
      let m = Nx.set [ Nx.I 1; Nx.I 4; Nx.I 0 ] (Nx.scalar Nx.bool true) m in
      equal (array float_exact)
        [| 2.; 1.; 5.; 3.; 0.; 4.; 1.; 5.; 0.; 0.; 0.; 0. |]
        (floats (V.Box.of_mask Nx.float32 m)))

let of_mask_golden =
  test "a mask's box is pycocotools'" (fun () ->
      let m = Nx.cast Nx.bool (entry Nx.uint8 "m") in
      let m = Nx.unsqueeze ~axes:[ -1 ] m in
      let want = entry Nx.float64 "bbox" in
      (* pycocotools gives [0 0 0 0] for an empty mask too. *)
      close 0. want (V.Box.to_xywh (V.Box.of_mask Nx.float64 m)))

let of_mask_empty =
  cases "a mask with no row or no column has the box 0 0 0 0"
    ~id:(fun s -> String.concat "x" (List.map string_of_int (Array.to_list s)))
    [ [| 3; 0; 1 |]; [| 0; 4; 1 |]; [| 0; 0; 1 |]; [| 2; 0; 3; 1 |] ]
    (fun shape ->
      let b = V.Box.of_mask Nx.float32 (Nx.zeros Nx.bool shape) in
      let lead = Array.sub shape 0 (Array.length shape - 3) in
      equal (array int) (Array.append lead [| 4 |]) (Nx.shape b);
      equal (array float_exact)
        (Array.make (Array.length (floats b)) 0.)
        (floats b))

(* Masks *)

let mask_iou =
  test "mask IoU is pycocotools'" (fun () ->
      let m =
        Nx.unsqueeze ~axes:[ -1 ] (Nx.cast Nx.bool (entry Nx.uint8 "m"))
      in
      close 1e-15 (entry Nx.float64 "iou") (V.Mask.iou Nx.float64 m m))

(* The quotient of the exact counts, rounded once into the dtype, where counts
   past float16's range or bfloat16's precision still give it: a full 300 × 300
   mask with itself, and with its first 100 rows, a third. *)
let mask_iou_narrow =
  test "mask IoU rounds the exact quotient once" (fun () ->
      let full = Nx.ones Nx.bool [| 1; 300; 300; 1 |] in
      let part =
        Nx.pad
          [| (0, 0); (0, 200); (0, 0); (0, 0) |]
          false
          (Nx.ones Nx.bool [| 1; 100; 300; 1 |])
      in
      let both = Nx.concatenate ~axis:0 [ full; part ] in
      let third = 30000. /. 90000. in
      equal (array float_exact)
        [| 1.; Nx.Dtype.of_float Nx.float16 third |]
        (floats (Nx.slice [ Nx.I 0 ] (V.Mask.iou Nx.float16 full both)));
      equal (array float_exact)
        [| 1.; Nx.Dtype.of_float Nx.bfloat16 third |]
        (floats (Nx.slice [ Nx.I 0 ] (V.Mask.iou Nx.bfloat16 full both))))

(* Run-length encoding *)

let rle_rows = [ 0; 1; 2; 3; 4; 5 ]

let mask i =
  Nx.unsqueeze ~axes:[ -1 ]
    (Nx.cast Nx.bool (Nx.slice [ Nx.I i ] (entry Nx.uint8 "m")))

let counts key = Array.map Int64.to_int (Nx.to_array (entry Nx.int64 key))

let text key =
  let a = Nx.to_array (entry Nx.uint8 key) in
  String.init (Array.length a) (fun i -> Char.chr a.(i))

let rle =
  cases "RLE is pycocotools'" ~id:string_of_int rle_rows (fun i ->
      let r = V.Rle.encode (mask i) in
      equal (array int) (counts (Printf.sprintf "rle.%d.counts" i)) r.counts;
      equal string (text (Printf.sprintf "rle.%d.s" i)) (V.Rle.to_string r);
      equal int
        (Nx.item [ i ] (entry Nx.int64 "area") |> Int64.to_int)
        (V.Rle.area r);
      equal (array bool) (Nx.to_array (mask i)) (Nx.to_array (V.Rle.decode r)))

let rle_big =
  test "long runs of both signs" (fun () ->
      let m =
        Nx.unsqueeze ~axes:[ -1 ] (Nx.cast Nx.bool (entry Nx.uint8 "big"))
      in
      let r = V.Rle.encode m in
      equal (array int) (counts "big.counts") r.counts;
      equal string (text "big.s") (V.Rle.to_string r);
      let back = V.Rle.of_string ~height:100 ~width:120 (text "big.s") in
      equal (array int) r.counts back.counts)

let gen_mask =
  Gen.(
    let* h, w = pair (int_range 0 7) (int_range 0 7) in
    let+ px = array ~size:(constant (h * w)) bool in
    (h, w, px))

let round_trips =
  prop "encode, decode and the string round trip" gen_mask (fun (h, w, px) ->
      let m = Nx.create Nx.bool [| h; w; 1 |] px in
      let r = V.Rle.encode m in
      cover "empty" (h * w = 0);
      equal (array bool) px (Nx.to_array (V.Rle.decode r));
      let r' = V.Rle.of_string ~height:h ~width:w (V.Rle.to_string r) in
      equal (array int) r.counts r'.counts;
      equal int
        (Array.fold_left (fun a p -> if p then a + 1 else a) 0 px)
        (V.Rle.area r))

let counts_round_trip =
  prop "any counts survive the string"
    Gen.(array ~size:(int_range 0 12) (int_range 0 100_000))
    (fun c ->
      let n = Array.fold_left ( + ) 0 c in
      let r = V.Rle.v ~height:1 ~width:n c in
      let r' = V.Rle.of_string ~height:1 ~width:n (V.Rle.to_string r) in
      equal (array int) c r'.counts)

let refusals =
  group "misuse"
    [
      test "counts that do not cover the mask" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Rle.v") (fun () ->
              V.Rle.v ~height:2 ~width:2 [| 1; 2 |]));
      test "a negative count" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Rle.v") (fun () ->
              V.Rle.v ~height:1 ~width:1 [| 2; -1 |]));
      test "a string that ends inside a count" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Rle.of_string") (fun () ->
              V.Rle.of_string ~height:1 ~width:1 "P"));
      test "a character outside the alphabet" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Rle.of_string") (fun () ->
              V.Rle.of_string ~height:1 ~width:1 " "));
      test "a string whose counts do not cover the mask" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Rle.of_string") (fun () ->
              V.Rle.of_string ~height:2 ~width:2 "1"));
      test "a mask without its channel axis" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Box.of_mask") (fun () ->
              V.Box.of_mask Nx.float32 (Nx.zeros Nx.bool [| 4; 4 |])));
      test "boxes end in 4" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Box.area") (fun () ->
              V.Box.area (Nx.zeros Nx.float32 [| 3 |])));
    ]

let () =
  exit
    (run "mynd.annot"
       [
         conversions;
         areas;
         iou_golden;
         iou_law;
         iou_batches;
         empty_iou;
         of_mask;
         of_mask_golden;
         of_mask_empty;
         mask_iou;
         mask_iou_narrow;
         rle;
         rle_big;
         round_trips;
         counts_round_trip;
         refusals;
       ])
