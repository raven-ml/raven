(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tessera

let rows dims l =
  Nx.create Nx.float32 [| List.length l / dims; dims |] (Array.of_list l)

(* Quantisation *)

(* A largest magnitude of 127 makes the scale 1, so each code is its value
   rounded, halves to even. *)
let ties =
  test "rounds halves to even" (fun () ->
      let codes, scales =
        Embedding.quantize
          (rows 8 [ 127.; 2.5; 3.5; -2.5; 0.5; -0.5; 1.5; -126.5 ])
      in
      equal (array int) [| 127; 2; 4; -2; 0; 0; 2; -126 |] (Nx.to_array codes);
      equal (array float_exact) [| 1. |] (Nx.to_array scales))

let no_embedding =
  test "codes a pixel without embedding as 0 of scale NaN" (fun () ->
      let codes, scales =
        Embedding.quantize (rows 2 [ Float.nan; Float.nan; 0.; 0. ])
      in
      equal (array int) [| 0; 0; 0; 0 |] (Nx.to_array codes);
      let s = Nx.to_array scales in
      equal ~msg:"no embedding" bool true (Float.is_nan s.(0));
      (* A zero vector takes the least scale, 1e-12 / 127 in float32. *)
      equal ~msg:"zeros" (float_rel ~rel:1e-6 ~abs:0.) (1e-12 /. 127.) s.(1))

let round_trip =
  prop "codes times the scale are within half a scale of the embedding"
    Gen.(list ~size:(int_range 1 40) (float_range (-6.) 6.))
    (fun l ->
      let e = rows (List.length l) l in
      let codes, scales = Embedding.quantize e in
      let s = Nx.item [ 0 ] scales in
      List.iteri
        (fun i x ->
          let c = Nx.item [ 0; i ] codes in
          at_most int ~than:127 (abs c);
          (* [e / s] rounds once in float32: a relative 2⁻²⁴ of 127. *)
          less float_exact
            ~than:(s /. 2. *. (1. +. 1e-5))
            (Float.abs ((Float.of_int c *. s) -. x)))
        l)

(* Files *)

let with_dir f =
  let dir = Filename.temp_dir "tessera" "out" in
  Fun.protect
    ~finally:(fun () ->
      Array.iter (fun n -> Sys.remove (Filename.concat dir n)) (Sys.readdir dir);
      Sys.rmdir dir)
    (fun () -> f dir)

let shape_of file = Nx.shape (Nx.unpack Nx.float32 (Nx_io.load_npy file))

let files =
  group "files"
    [
      test "save writes float32 [h; w; dims]" (fun () ->
          with_dir (fun dir ->
              let name = Filename.concat dir "tile" in
              Embedding.save name ~grid:(2, 3) (Nx.ones Nx.float32 [| 6; 4 |]);
              equal (array int) [| 2; 3; 4 |] (shape_of (name ^ ".npy"))));
      test "save_int8 writes GeoTessera's codes and scales" (fun () ->
          with_dir (fun dir ->
              let name = Filename.concat dir "tile" in
              Embedding.save_int8 name ~grid:(2, 3)
                (Nx.ones Nx.float32 [| 6; 4 |]);
              let codes = Nx.unpack Nx.int8 (Nx_io.load_npy (name ^ ".npy")) in
              equal (array int) [| 2; 3; 4 |] (Nx.shape codes);
              equal (array int) [| 2; 3 |] (shape_of (name ^ "_scales.npy"))));
      test "save refuses rows that are not the grid's" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"rows") (fun () ->
              Embedding.save "unused" ~grid:(2, 2)
                (Nx.ones Nx.float32 [| 6; 4 |])));
    ]

(* Encoding *)

let model =
  Model.init ~key:(Nx.Rng.key 3)
    {
      Model.pre_ln = false;
      width = 16;
      heads = 2;
      layers = 1;
      ffn = 24;
      dims = 16;
    }

let shape =
  { Batch.pixels = 3; s2_rows = 2; s1_rows = 2; slots = 2; width = 256 }

let undefined =
  prop "encode gives NaN exactly to pixels without observations"
    (Synth.spec ~pixels:8 ()) (fun spec ->
      let px = Synth.pixels spec in
      let e = Embedding.encode ~shape Nx.Host.on (Model.apply model) px in
      equal (array int) [| Pixels.length px; 16 |] (Nx.shape e);
      List.iteri
        (fun p (n2, n1) ->
          let row = Array.init 16 (fun c -> Nx.item [ p; c ] e) in
          let empty = n2 = 0 && n1 = 0 in
          cover "a pixel without observations" empty;
          cover "a pixel of one sensor" (n2 = 0 <> (n1 = 0));
          equal
            ~msg:(Printf.sprintf "pixel %d" p)
            bool empty
            (Array.for_all Float.is_nan row);
          if not empty then
            equal ~msg:"finite" bool true (Array.for_all Float.is_finite row))
        (List.combine spec.s2 spec.s1))

let () =
  exit
    (run "tessera.embedding"
       [
         group "quantize" [ ties; no_embedding; round_trip ];
         files;
         group "encode" [ undefined ];
       ])
