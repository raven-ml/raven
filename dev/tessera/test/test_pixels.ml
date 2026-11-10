(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tessera

(* A tile of one row of two pixels, written as the preprocessing writes it, and
   its expected rows. Sentinel-2: three frames; pixel 0 is clear in frames 0 and
   2, pixel 1 never. Sentinel-1: pixel 0 has VV only in its first ascending
   frame and both bands in the descending one; pixel 1 has VH only, then both,
   ascending. *)
let s2_frames = 3
let doys = [| 10; 20; 30 |]
let mask = [| 1; 0; 0; 0; 1; 0 |]
let band t p c = (100 * t) + (10 * p) + c + 1
let asc = [| 5; 0; 0; 7; 0; 0; 3; 3 |] (* [t; p; band] *)
let asc_doys = [| 40; 50 |]
let desc = [| 1; 1; 0; 0 |]
let desc_doys = [| 60 |]

let write ?(skip = []) dir =
  let save name dt shape v =
    if not (List.mem name skip) then
      Nx_io.save_npy (Filename.concat dir name) (Nx.create dt shape v)
  in
  save "bands.npy" Nx.uint16 [| s2_frames; 1; 2; 10 |]
    (Array.init
       (s2_frames * 2 * 10)
       (fun i -> band (i / 20) (i / 10 mod 2) (i mod 10)));
  save "masks.npy" Nx.uint8 [| s2_frames; 1; 2 |] mask;
  save "doys.npy" Nx.uint16 [| s2_frames |] doys;
  save "sar_ascending.npy" Nx.int16 [| 2; 1; 2; 2 |] asc;
  save "sar_ascending_doy.npy" Nx.int16 [| 2 |] asc_doys;
  save "sar_descending.npy" Nx.int16 [| 1; 1; 2; 2 |] desc;
  save "sar_descending_doy.npy" Nx.int16 [| 1 |] desc_doys

let with_tile ?skip f =
  let dir = Filename.temp_dir "tessera" "tile" in
  Fun.protect
    ~finally:(fun () ->
      Array.iter (fun n -> Sys.remove (Filename.concat dir n)) (Sys.readdir dir);
      Sys.rmdir dir)
    (fun () ->
      write ?skip dir;
      f dir)

let rows r =
  let o = Array.map Int64.to_int (Nx.to_array (Nx_ragged.offsets r)) in
  let v = Nx.to_array (Nx_ragged.values r) in
  let c = Nx.dim 1 (Nx_ragged.values r) in
  List.init (Nx_ragged.length r) (fun p ->
      List.init
        (o.(p + 1) - o.(p))
        (fun k -> Array.to_list (Array.sub v ((o.(p) + k) * c) c)))

let s2_cell t p = List.init 10 (band t p) @ [ doys.(t) ]
let expected_s2 = [ [ s2_cell 0 0; s2_cell 2 0 ]; [] ]

let tiles =
  group "of_tile"
    [
      test "reads the valid observations in order" (fun () ->
          with_tile (fun dir ->
              let px, grid = Pixels.of_tile dir in
              equal (pair int int) (1, 2) grid;
              equal (list (list (list int))) expected_s2 (rows px.s2);
              equal
                (list (list (list int)))
                [
                  [ [ 5; 0; 40; 0 ]; [ 1; 1; 60; 1 ] ];
                  [ [ 0; 7; 40; 0 ]; [ 3; 3; 50; 0 ] ];
                ]
                (rows px.s1)));
      test "reads a missing orbit as no observation" (fun () ->
          with_tile ~skip:[ "sar_descending.npy"; "sar_descending_doy.npy" ]
            (fun dir ->
              let px, _ = Pixels.of_tile dir in
              equal
                (list (list (list int)))
                [ [ [ 5; 0; 40; 0 ] ]; [ [ 0; 7; 40; 0 ]; [ 3; 3; 50; 0 ] ] ]
                (rows px.s1)));
      test "reads a missing mask as no observation" (fun () ->
          with_tile ~skip:[ "masks.npy" ] (fun dir ->
              let px, _ = Pixels.of_tile dir in
              equal (list (list (list int))) [ []; [] ] (rows px.s2)));
      test "refuses a tile without bands" (fun () ->
          with_tile ~skip:[ "bands.npy" ] (fun dir ->
              raises_match (Exn.invalid_arg ~sub:"bands.npy") (fun () ->
                  Pixels.of_tile dir)));
      test "refuses a mask of another shape" (fun () ->
          with_tile (fun dir ->
              Nx_io.save_npy
                (Filename.concat dir "masks.npy")
                (Nx.zeros Nx.uint8 [| 2; 1; 2 |]);
              raises_match (Exn.invalid_arg ~sub:"masks.npy") (fun () ->
                  Pixels.of_tile dir)));
    ]

let () = exit (run "tessera.pixels" [ tiles ])
