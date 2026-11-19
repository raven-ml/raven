(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module V = Mynd

(* scipy's connected-component labels, from golden/select.npz
   (contrib/mynd/test/gen/select.py) *)
let golden = lazy (Nx_io.load_npz "golden/select.npz")

let entry dt key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack dt p
  | None -> failf "golden/select.npz has no %s" key

let longs x = Nx.to_array (Nx.cast Nx.int64 x)

(* Components *)

let golden_components =
  cases "components are scipy's labels" ~id:fst
    [ ("four", `Four); ("eight", `Eight) ]
    (fun (name, conn) ->
      let m = Nx.unsqueeze ~axes:[ -1 ] (entry Nx.bool "cc.m") in
      let want = entry Nx.int64 ("cc." ^ name) in
      equal (array int64) (longs want) (longs (V.components conn m)))

(* [reference conn h w px] labels by a sequential union-find, each component by
   its least row-major index. *)
let reference conn h w px =
  let parent = Array.init (h * w) Fun.id in
  let rec find i = if parent.(i) = i then i else find parent.(i) in
  let union a b =
    let a = find a and b = find b in
    if a < b then parent.(b) <- a else if b < a then parent.(a) <- b
  in
  let offsets =
    match conn with
    | `Four -> [ (0, 1); (1, 0) ]
    | `Eight -> [ (0, 1); (1, -1); (1, 0); (1, 1) ]
  in
  for i = 0 to h - 1 do
    for j = 0 to w - 1 do
      if px.((i * w) + j) then
        List.iter
          (fun (di, dj) ->
            let i' = i + di and j' = j + dj in
            if i' < h && j' >= 0 && j' < w && px.((i' * w) + j') then
              union ((i * w) + j) ((i' * w) + j'))
          offsets
    done
  done;
  Array.init (h * w) (fun k -> if px.(k) then Int64.of_int (find k) else -1L)

let gen_mask =
  Gen.(
    let* h, (w, d) =
      pair (int_range 0 14) (pair (int_range 0 14) (float_range 0.2 0.8))
    in
    let+ px =
      array ~size:(constant (h * w)) (map (fun f -> f < d) (float_range 0. 1.))
    in
    (h, w, px))

let union_find =
  prop "components equal a sequential union-find" ~count:200
    Gen.(pair bool gen_mask)
    (fun (eight, (h, w, px)) ->
      let conn = if eight then `Eight else `Four in
      cover "empty" (h * w = 0);
      let m = Nx.create Nx.bool [| h; w; 1 |] px in
      equal (array int64) (reference conn h w px) (longs (V.components conn m)))

(* A serpentine of one-pixel rows joined at alternate ends: one component whose
   least index is far along its path. *)
let serpentine n =
  Nx.init Nx.bool [| n; n; 1 |] (fun i ->
      let r = i.(0) and c = i.(1) in
      r mod 2 = 0 || c = if r / 2 mod 2 = 0 then n - 1 else 0)

let bound n =
  6 + Float.to_int (Float.floor (3. *. Float.log2 (1.5 *. Float.of_int n)))

let rounds =
  cases "rounds stay under Liu and Tarjan's bound" ~id:fst
    [
      ("serpentine 64", serpentine 64);
      ("serpentine 64, flipped", Nx.flip ~axes:[ 0; 1 ] (serpentine 64));
      ( "noise",
        Nx.unsqueeze ~axes:[ -1 ] (Nx.slice [ Nx.I 0 ] (entry Nx.bool "cc.m"))
      );
    ]
    (fun (_, m) ->
      let h = Nx.dim 0 m and w = Nx.dim 1 m in
      List.iter
        (fun conn ->
          let labels, k = V.components_rounds conn m in
          let px = Nx.to_array (Nx.reshape [| h * w |] m) in
          equal (array int64) (reference conn h w px) (longs labels);
          less int ~than:(bound (h * w) + 1) (Int32.to_int (Nx.item [] k)))
        [ `Four; `Eight ])

(* A batch padded with false keeps its components; a label, the least index,
   counts in the padded width. *)
let padded =
  prop "padding keeps components, relabelled by the padded width"
    Gen.(triple bool gen_mask (pair (int_range 0 3) (int_range 0 3)))
    (fun (eight, (h, w, px), (ph, pw)) ->
      let conn = if eight then `Eight else `Four in
      let m = Nx.create Nx.bool [| h; w; 1 |] px in
      let mp = Nx.pad [| (0, ph); (0, pw); (0, 0) |] false m in
      let w' = w + pw in
      let moved l =
        if l < 0L then l
        else Int64.(add (mul (div l (of_int w)) (of_int w')) (rem l (of_int w)))
      in
      let want = Array.map moved (longs (V.components conn m)) in
      let got =
        longs (Nx.slice [ Nx.R (0, h); Nx.R (0, w) ] (V.components conn mp))
      in
      equal (array int64) want got)

let batch_and_vmap =
  test "each mask of a batch is labelled alone, under vmap too" (fun () ->
      let m = Nx.unsqueeze ~axes:[ -1 ] (entry Nx.bool "cc.m") in
      let whole = longs (V.components `Eight m) in
      let one i = longs (V.components `Eight (Nx.slice [ Nx.I i ] m)) in
      equal (array int64) (Array.append (one 0) (one 1)) whole;
      equal (array int64) whole (longs (Rune.vmap' (V.components `Eight) m)))

let compiled =
  test "compiled is eager" (fun () ->
      let m = Nx.unsqueeze ~axes:[ -1 ] (entry Nx.bool "cc.m") in
      equal (array int64)
        (longs (V.components `Four m))
        (longs (Rune.jit' (V.components `Four) m)))

let areas =
  test "areas compose from the labels" (fun () ->
      let m =
        Nx.unsqueeze ~axes:[ -1 ] (Nx.slice [ Nx.I 1 ] (entry Nx.bool "cc.m"))
      in
      let labels = Nx.reshape [| 24 * 31 |] (V.components `Four m) in
      let ones = Nx.ones Nx.int64 [| 24 * 31 |] in
      let areas =
        Nx.take labels (Nx.reduce_segments Add ~segments:(24 * 31) labels ones)
      in
      let l = longs labels in
      let count x =
        Array.fold_left (fun a y -> if y = x then a + 1 else a) 0 l
      in
      equal (array int64)
        (Array.map (fun x -> if x < 0L then 0L else Int64.of_int (count x)) l)
        (longs areas))

let refusals =
  group "misuse"
    [
      test "a mask without its channel axis" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.components") (fun () ->
              V.components `Four (Nx.zeros Nx.bool [| 4; 4 |])));
    ]

let () =
  exit
    (run "mynd.components"
       [
         golden_components;
         union_find;
         rounds;
         padded;
         batch_and_vmap;
         compiled;
         areas;
         refusals;
       ])
