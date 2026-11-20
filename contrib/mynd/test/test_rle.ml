(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module V = Mynd

(* pycocotools' RLE and areas, from golden/annot.npz
   (contrib/mynd/test/gen/annot.py) *)
let golden = lazy (Nx_io.load_npz "golden/annot.npz")

let entry dt key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack dt p
  | None -> failf "golden/annot.npz has no %s" key

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
    ]

let () =
  exit
    (run "mynd.rle" [ rle; rle_big; round_trips; counts_round_trip; refusals ])
