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

(* Corpora *)

let rm_rf dir =
  let rec go d =
    Array.iter
      (fun n ->
        let f = Filename.concat d n in
        if Sys.is_directory f then go f else Sys.remove f)
      (Sys.readdir d);
    Sys.rmdir d
  in
  if Sys.file_exists dir then go dir

let with_dir f =
  let dir = Filename.temp_dir "tessera" "corpus" in
  Fun.protect ~finally:(fun () -> rm_rf dir) (fun () -> f dir)

let pixel_rows px = List.combine (rows px.Pixels.s2) (rows px.Pixels.s1)

let round_trip =
  prop "a saved corpus loads as its pixels" ~count:20 (Synth.spec ~long:true ())
    (fun spec ->
      let px = Synth.pixels spec in
      let n = Pixels.length px in
      let pos = n / 3 in
      let part = Pixels.sub px ~pos ~len:(n - pos) in
      cover "a part that starts past its values' first row" (pos > 0);
      with_dir (fun dir ->
          let d = Filename.concat dir "c" in
          Pixels.save d part;
          let back = Pixels.load d in
          equal
            (list (pair (list (list int)) (list (list int))))
            (pixel_rows part) (pixel_rows back)))

let shuffled =
  prop "a shuffle puts pixel j at position permute j" ~count:40
    Gen.(
      triple (Synth.spec ()) (Synth.spec ())
        (frequency
           [
             (1, int_range 1 21); (1, int_range 22 100); (1, int_range 101 200);
           ]))
    (fun (a, b, bytes) ->
      let pa = Synth.pixels a and pb = Synth.pixels b in
      with_dir (fun dir ->
          let part name px =
            let d = Filename.concat dir name in
            Pixels.save d px;
            d
          in
          let parts = [ part "a" pa; part "b" pb ] in
          let out = Filename.concat dir "out" in
          let key = Nx.Rng.key 3 in
          let n = Pixels.shuffle ~key ~bytes out parts in
          equal int (Pixels.length pa + Pixels.length pb) n;
          let all = Array.of_list (pixel_rows pa @ pixel_rows pb) in
          let dest =
            Nx.to_array (Nx.Rng.permute ~key ~n (Nx.arange Nx.int64 0 n 1))
          in
          let expected = Array.make n ([], []) in
          Array.iteri (fun j q -> expected.(Int64.to_int q) <- all.(j)) dest;
          cover "runs of several pixels" (bytes > 100);
          cover "a pixel larger than a run" (bytes < 22);
          equal
            (list (pair (list (list int)) (list (list int))))
            (Array.to_list expected)
            (pixel_rows (Pixels.load out))))

let corpora =
  group "corpora"
    [
      round_trip;
      shuffled;
      test "refuses a run of no byte" (fun () ->
          raises
            (Invalid_argument "Tessera.Pixels.shuffle: bytes is not positive")
            (fun () -> Pixels.shuffle ~key:(Nx.Rng.key 0) ~bytes:0 "x" []));
      test "refuses a corpus of other cells" (fun () ->
          with_dir (fun dir ->
              Nx_io.save_npy
                (Filename.concat dir "s2.npy")
                (Nx.zeros Nx.uint16 [| 2; 3 |]);
              Nx_io.save_npy
                (Filename.concat dir "s2_offsets.npy")
                (Nx.zeros Nx.int64 [| 1 |]);
              raises_match (Exn.invalid_arg ~sub:"s2.npy") (fun () ->
                  Pixels.load dir)));
    ]

let () = exit (run "tessera.pixels" [ tiles; corpora ])
