(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module V = Mynd

(* torchvision's NMS, from golden/select.npz *)
let golden = lazy (Nx_io.load_npz "golden/select.npz")

let entry dt key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack dt p
  | None -> failf "golden/select.npz has no %s" key

(* Non-maximum suppression *)

let floats x = Nx.to_array (Nx.cast Nx.float64 x)

(* [greedy ?classes iou t s] is greedy NMS from the IoUs [iou] (row-major n ×
   n): the boxes by descending score, equal scores lowest index first and NaN
   first, each kept unless a kept one before it has IoU above [t]. *)
let greedy ?classes iou t s =
  let n = Array.length s in
  let order =
    List.init n Fun.id
    |> List.stable_sort (fun i j ->
        match (Float.is_nan s.(i), Float.is_nan s.(j)) with
        | true, true -> 0
        | true, false -> -1
        | false, true -> 1
        | false, false -> compare s.(j) s.(i))
  in
  let keep = Array.make n false in
  let kept = ref [] in
  List.iter
    (fun i ->
      let same j =
        match classes with None -> true | Some c -> c.(i) = c.(j)
      in
      if not (List.exists (fun j -> same j && iou.((i * n) + j) > t) !kept) then (
        keep.(i) <- true;
        kept := i :: !kept))
    order;
  keep

let golden_nms =
  cases "nms is torchvision's"
    ~id:(fun (c, t) -> Printf.sprintf "%s%g" (if c then "classes, " else "") t)
    (List.concat_map
       (fun t -> [ (false, t); (true, t) ])
       [ 0.3; 0.5; 0.7; 0.85 ])
    (fun (by_class, t) ->
      let b = entry Nx.float32 "nms.boxes"
      and s = entry Nx.float32 "nms.scores" in
      let classes =
        if by_class then Some (entry Nx.int64 "nms.classes") else None
      in
      let key =
        Printf.sprintf "nms.%s%g" (if by_class then "classes." else "") t
      in
      equal (array bool)
        (Nx.to_array (entry Nx.bool key))
        (Nx.to_array (V.nms ?classes ~iou:t b s)))

let gen_boxes =
  Gen.(
    let* n = int_range 0 24 in
    let box =
      let+ x = float_range 0. 30.
      and+ y = float_range 0. 30.
      and+ w = float_range 1. 20.
      and+ h = float_range 1. 20. in
      [| x; y; x +. w; y +. h |]
    in
    let score =
      frequency
        [
          (8, map (fun k -> Float.of_int k /. 4.) (int_range 0 8));
          (1, constant Float.nan);
          (1, constant (-0.));
        ]
    in
    let+ boxes = array ~size:(constant n) box
    and+ scores = array ~size:(constant n) score
    and+ classes = array ~size:(constant n) (int_range 0 2)
    and+ t = float_range 0. 0.9 in
    (boxes, scores, classes, t))

let greedy_law =
  prop "nms is greedy suppression" ~count:300
    Gen.(pair bool gen_boxes)
    (fun (by_class, (bs, ss, cs, t)) ->
      let n = Array.length bs in
      let b =
        Nx.create Nx.float64 [| n; 4 |] (Array.concat (Array.to_list bs))
      in
      let s = Nx.create Nx.float64 [| n |] ss in
      let iou = floats (V.Box.iou b b) in
      let ties x =
        Array.fold_left (fun c y -> if y = x then c + 1 else c) 0 ss
      in
      cover "a tie" (Array.exists (fun x -> ties x > 1) ss);
      cover "a NaN score" (Array.exists Float.is_nan ss);
      cover "a -0 beside a 0"
        (Array.exists (fun x -> 1. /. x < 0.) ss
        && Array.exists (( = ) 0.) ss
        && Array.exists (fun x -> 1. /. x > 0. && x = 0.) ss);
      let classes, cl =
        if by_class then
          ( Some (Nx.create Nx.int64 [| n |] (Array.map Int64.of_int cs)),
            Some cs )
        else (None, None)
      in
      let keep, k = V.nms_rounds ?classes ~iou:t b s in
      equal (array bool) (greedy ?classes:cl iou t ss) (Nx.to_array keep);
      less int ~than:(n + 2) (Int32.to_int (Nx.item [] k)))

(* The second box's IoU with the first is about [t] in float32, and for 0.85
   exactly [t] rounded up into float32: greater than [t] as a real number,
   though equal to [t] compared in float32. *)
(* [-0.] and [0.] are equal scores: the lower position comes first. *)
let signed_zeros =
  test "-0 and 0 tie" (fun () ->
      let b =
        Nx.create Nx.float32 [| 2; 4 |] [| 0.; 0.; 2.; 2.; 0.; 0.; 2.; 2. |]
      in
      let s = Nx.create Nx.float32 [| 2 |] [| -0.; 0. |] in
      equal (array bool) [| true; false |] (Nx.to_array (V.nms ~iou:0.5 b s)))

let threshold =
  cases "the threshold is compared as a real number" ~id:string_of_float
    [ 0.85; 0.3; 0.7; 0.5 ] (fun t ->
      let t32 = Int32.float_of_bits (Int32.bits_of_float t) in
      let b =
        Nx.create Nx.float32 [| 2; 4 |] [| 0.; 0.; 1.; 1.; 0.; 0.; 1.; t32 |]
      in
      let s = Nx.create Nx.float32 [| 2 |] [| 1.; 0.5 |] in
      let iou = Nx.item [ 0; 1 ] (V.Box.iou b b) in
      if t = 0.85 then equal float_exact t32 iou;
      equal (array bool)
        [| true; not (iou > t) |]
        (Nx.to_array (V.nms ~iou:t b s)))

(* [steps n iou t s] is the parallel greedy's steps, each keeping every
   undecided box no earlier undecided box conflicts with and dropping those it
   suppresses: the dependence length. *)
let steps n iou t s =
  let rank = Array.make n 0 in
  List.iteri
    (fun r i -> rank.(i) <- r)
    (List.init n Fun.id |> List.stable_sort (fun i j -> compare s.(j) s.(i)));
  let conflict i j = i <> j && iou.((i * n) + j) > t in
  let undecided = Array.make n true in
  let count = ref 0 in
  while Array.exists Fun.id undecided do
    incr count;
    let roots =
      List.filter
        (fun i ->
          undecided.(i)
          && not
               (List.exists
                  (fun j ->
                    undecided.(j) && rank.(j) < rank.(i) && conflict i j)
                  (List.init n Fun.id)))
        (List.init n Fun.id)
    in
    List.iter
      (fun i ->
        undecided.(i) <- false;
        for j = 0 to n - 1 do
          if conflict i j then undecided.(j) <- false
        done)
      roots
  done;
  !count

(* A chain: each box overlaps the next above the threshold, by descending score,
   so each decision waits for the one before. *)
let chain n =
  ( Nx.create Nx.float64 [| n; 4 |]
      (Array.concat
         (List.init n (fun i ->
              let x = Float.of_int i in
              [| x; 0.; x +. 4.; 1. |]))),
    Nx.create Nx.float64 [| n |] (Array.init n (fun i -> Float.of_int (n - i)))
  )

let rounds_nms =
  cases "rounds stay within twice the dependence length and one" ~id:fst
    [
      ( "detections 0.5",
        ( entry Nx.float32 "nms.boxes" |> Nx.cast Nx.float64,
          entry Nx.float32 "nms.scores" |> Nx.cast Nx.float64,
          0.5 ) );
      ( "detections 0.7",
        ( entry Nx.float32 "nms.boxes" |> Nx.cast Nx.float64,
          entry Nx.float32 "nms.scores" |> Nx.cast Nx.float64,
          0.7 ) );
      ("chain of 40", (fst (chain 40), snd (chain 40), 0.5));
    ]
    (fun (_, (b, s, t)) ->
      let n = Nx.dim 0 b in
      let l = steps n (floats (V.Box.iou b b)) t (floats s) in
      let _, k = V.nms_rounds ~iou:t b s in
      less int ~than:((2 * l) + 2) (Int32.to_int (Nx.item [] k)))

let batches_nms =
  test "each set of leading axes selects alone" (fun () ->
      let b = entry Nx.float32 "nms.boxes"
      and s = entry Nx.float32 "nms.scores" in
      let b2 = Nx.stack [ b; Nx.flip ~axes:[ 0 ] b ]
      and s2 = Nx.stack [ s; Nx.flip ~axes:[ 0 ] s ] in
      let k = V.nms ~iou:0.5 b2 s2 in
      equal (array bool)
        (Nx.to_array (V.nms ~iou:0.5 b s))
        (Nx.to_array (Nx.slice [ Nx.I 0 ] k));
      equal (array bool)
        (Nx.to_array
           (V.nms ~iou:0.5 (Nx.flip ~axes:[ 0 ] b) (Nx.flip ~axes:[ 0 ] s)))
        (Nx.to_array (Nx.slice [ Nx.I 1 ] k)))

let compiled_nms =
  test "compiled is eager" (fun () ->
      let b = entry Nx.float32 "nms.boxes"
      and s = entry Nx.float32 "nms.scores" in
      equal (array bool)
        (Nx.to_array (V.nms ~iou:0.5 b s))
        (Nx.to_array (Rune.jit' (fun b -> V.nms ~iou:0.5 b s) b)))

let refusals =
  group "misuse"
    [
      test "scores that are not one a box" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.nms") (fun () ->
              V.nms ~iou:0.5
                (Nx.zeros Nx.float32 [| 3; 4 |])
                (Nx.zeros Nx.float32 [| 2 |])));
      test "boxes end in 4" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.nms") (fun () ->
              V.nms ~iou:0.5
                (Nx.zeros Nx.float32 [| 3; 5 |])
                (Nx.zeros Nx.float32 [| 3 |])));
    ]

let () =
  exit
    (run "mynd.nms"
       [
         golden_nms;
         greedy_law;
         signed_zeros;
         threshold;
         rounds_nms;
         batches_nms;
         compiled_nms;
         refusals;
       ])
