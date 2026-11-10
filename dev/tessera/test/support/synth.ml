(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

type spec = { s2 : int list; s1 : int list; seed : int }

(* A positive count of observations. *)
let count ~long =
  let near_bin =
    Gen.map
      (fun (k, d) -> (8 * k) + d)
      (Gen.pair (Gen.int_range 1 6) (Gen.int_range (-1) 1))
  in
  let base =
    [ (2, Gen.of_list [ 1; 2 ]); (3, near_bin); (2, Gen.int_range 3 20) ]
  in
  Gen.frequency
    (if long then (1, Gen.of_list [ 255; 256; 257; 295 ]) :: base else base)

(* A pixel's counts, its kind drawn first so that every case of a few pixels is
   likely to hold a pixel without observations and one of a single sensor: each
   kind has a branch of its own. *)
let pixel ~long =
  let open Gen in
  let c = count ~long in
  frequency
    [
      (1, constant (0, 0));
      (1, map (fun n -> (n, 0)) c);
      (1, map (fun n -> (0, n)) c);
      (4, pair c c);
    ]

let spec ?(pixels = 6) ?(long = false) () =
  let open Gen in
  let* n = int_range 1 pixels in
  let* counts = list ~size:(constant n) (pixel ~long) in
  let+ seed = int_range 0 1_000_000 in
  { s2 = List.map fst counts; s1 = List.map snd counts; seed }

let pp ppf { s2; s1; seed } =
  let ints l = String.concat "; " (List.map string_of_int l) in
  Format.fprintf ppf "{ s2 = [%s]; s1 = [%s]; seed = %d }" (ints s2) (ints s1)
    seed

let spec ?pixels ?long () = Gen.with_pp pp (spec ?pixels ?long ())

let pixels { s2; s1; seed } =
  let rng = Random.State.make [| seed |] in
  let draw lo hi = lo + Random.State.int rng (hi - lo + 1) in
  let rows dt counts cell =
    let total = List.fold_left ( + ) 0 counts in
    let c = Array.length (cell 0 0) in
    let values = Array.make (total * c) 0 in
    let at = ref 0 in
    List.iter
      (fun n ->
        for k = 0 to n - 1 do
          Array.iteri (fun i v -> values.((!at * c) + i) <- v) (cell n k);
          incr at
        done)
      counts;
    let lengths = Array.of_list (List.map Int64.of_int counts) in
    Nx_ragged.of_lengths
      (Nx.create Nx.int64 [| Array.length lengths |] lengths)
      (Nx.create dt [| total; c |] values)
  in
  let s2 =
    rows Nx.uint16 s2 (fun _ _ ->
        Array.init 11 (fun i -> if i < 10 then draw 1 12000 else draw 1 366))
  in
  (* Ascending passes come first, as a tile's rows hold them. *)
  let s1 =
    rows Nx.int16 s1 (fun n k ->
        [| draw 1 9000; draw 1 9000; draw 1 366; (if 2 * k < n then 0 else 1) |])
  in
  { Tessera.Pixels.s2; s1 }

let counts (c : Tessera.Batch.counts) =
  let ints x = Array.map Int32.to_int (Nx.to_array x) in
  (ints c.s2, ints c.s1)

let rebuild (px : Nx.host Tessera.Pixels.t) (c : Tessera.Batch.counts) f =
  let sensor (type s) (dt : (int, s) Nx.dtype) r counts =
    let offsets = Array.map Int64.to_int (Nx.to_array (Nx_ragged.offsets r)) in
    let values = Nx.to_array (Nx_ragged.values r) in
    let cells = Nx.dim 1 (Nx_ragged.values r) in
    let rows =
      Array.init (Nx_ragged.length r) (fun p ->
          let lo = offsets.(p) and hi = offsets.(p + 1) in
          let idx = f p (Array.sub counts lo (hi - lo)) in
          Array.map (fun i -> lo + i) idx)
    in
    let all = Array.concat (Array.to_list rows) in
    let v =
      Array.concat
        (Array.to_list
           (Array.map (fun j -> Array.sub values (j * cells) cells) all))
    in
    let lengths = Array.map (fun r -> Int64.of_int (Array.length r)) rows in
    ( Nx_ragged.of_lengths
        (Nx.create Nx.int64 [| Array.length lengths |] lengths)
        (Nx.create dt [| Array.length all; cells |] v),
      Nx.create Nx.int32
        [| Array.length all |]
        (Array.map (fun j -> Int32.of_int counts.(j)) all) )
  in
  let c2, c1 = counts c in
  let s2, n2 = sensor Nx.uint16 px.s2 c2 in
  let s1, n1 = sensor Nx.int16 px.s1 c1 in
  ({ Tessera.Pixels.s2; s1 }, { Tessera.Batch.s2 = n2; s1 = n1 })
