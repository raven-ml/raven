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
              Embedding.save name (Nx.ones Nx.float32 [| 2; 3; 4 |]);
              equal (array int) [| 2; 3; 4 |] (shape_of (name ^ ".npy"))));
      test "save_int8 writes GeoTessera's codes and scales" (fun () ->
          with_dir (fun dir ->
              let name = Filename.concat dir "tile" in
              Embedding.save_int8 name (Nx.ones Nx.float32 [| 2; 3; 4 |]);
              let codes = Nx.unpack Nx.int8 (Nx_io.load_npy (name ^ ".npy")) in
              equal (array int) [| 2; 3; 4 |] (Nx.shape codes);
              equal (array int) [| 2; 3 |] (shape_of (name ^ "_scales.npy"))));
      test "save refuses an embedding that is not a tile's" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"rank") (fun () ->
              Embedding.save "unused" (Nx.ones Nx.float32 [| 6; 4 |])));
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

(* GeoTIFF *)

let grid_t =
  Testable.make
    ~pp:(fun ppf (g : Jord.Grid.t) ->
      let (a, b), (c, d), (e, f) = (g.origin, g.row_step, g.col_step) in
      Format.fprintf ppf
        "%a %dx%d origin (%.17g, %.17g) row (%g, %g) col (%g, %g)" Jord.Crs.pp
        g.crs g.rows g.cols a b c d e f)
    ~equal:Jord.Grid.equal

let utm31 = Jord.Crs.utm ~zone:31 ~north:true

(* 10 m cells, north up, as GeoTessera's landmasks place a tile. *)
let grid ~rows ~cols =
  Jord.Grid.v utm31 ~rows ~cols ~origin:(299_960., 5_800_040.)
    ~row_step:(0., -10.) ~col_step:(10., 0.)

let ok = function Ok x -> x | Error e -> failf "%s" e

let read_tiff file =
  let ifd = List.hd (Nx_io.load_tiff file) in
  (Nx_io.Tiff.tags ifd, Nx_io.Tiff.samples ifd)

(* What the file states: its grid, nodata and band descriptions, and its
   bands. *)
let read_geotiff file =
  let tags, samples = read_tiff file in
  let descriptions =
    match List.assoc_opt 42112 tags with
    | Some (Nx_io.Tiff.Ascii s) -> s
    | _ -> failf "%s: no GDAL_METADATA text" file
  in
  ( ok (Jord.Geotiff.grid tags),
    ok (Jord.Geotiff.nodata tags),
    descriptions,
    Nx.unpack Nx.float32 samples )

(* The prefix of [dims] values of a tile's embedding of [dims + 2], a strided
   view, whose values are distinct: the first pixel without embedding, and -0.
   and infinity among the others. *)
let distinct ~rows ~cols ~dims =
  let n = rows * cols * (dims + 2) in
  let v =
    Array.init n (fun i ->
        if i < dims + 2 then Float.nan
        else if i = dims + 2 then -0.
        else if i = n - 3 then Float.infinity
        else (Float.of_int i *. 0.25) -. 7.)
  in
  let e = Nx.create Nx.float32 [| rows; cols; dims + 2 |] v in
  Nx.slice [ Nx.A; Nx.A; Nx.R (0, dims) ] e

let on_grid =
  prop "save_geotiff writes the embedding on its grid, read back exactly"
    Gen.(triple (int_range 1 300) (int_range 1 300) (int_range 1 3))
    (fun (rows, cols, dims) ->
      cover "more than a tile down" (rows > 256);
      cover "more than a tile across" (cols > 256);
      with_dir (fun dir ->
          let name = Filename.concat dir "tile" in
          let g = grid ~rows ~cols in
          let e = distinct ~rows ~cols ~dims in
          Embedding.save_geotiff name ~grid:g e;
          let g', nodata, _, bands = read_geotiff (name ^ ".tiff") in
          equal grid_t g g';
          equal (option float_exact) (Some Float.nan) nodata;
          equal (array int) [| rows; cols; dims |] (Nx.shape bands);
          equal (array float_exact) (Nx.to_array e) (Nx.to_array bands)))

(* GDAL 3.12's GDAL_METADATA text for bands described as GeoTessera's
   _write_embedding_tiff describes them, as rasterio 1.5.2 writes it. *)
let gdal_descriptions =
  "<GDALMetadata>\n\
  \  <Item name=\"DESCRIPTION\" sample=\"0\" \
   role=\"description\">Tessera_Band_0</Item>\n\
  \  <Item name=\"DESCRIPTION\" sample=\"1\" \
   role=\"description\">Tessera_Band_1</Item>\n\
  \  <Item name=\"DESCRIPTION\" sample=\"2\" \
   role=\"description\">Tessera_Band_2</Item>\n\
   </GDALMetadata>\n"

(* A tile of [n] pixels as [rows × cols] cells. *)
let tile_shape n = if n mod 2 = 0 then (2, n / 2) else (1, n)

let geotiff =
  group "geotiff"
    [
      on_grid;
      test "describes each band as GeoTessera does" (fun () ->
          with_dir (fun dir ->
              let name = Filename.concat dir "tile" in
              Embedding.save_geotiff name ~grid:(grid ~rows:2 ~cols:2)
                (Nx.ones Nx.float32 [| 2; 2; 3 |]);
              let _, _, descriptions, _ = read_geotiff (name ^ ".tiff") in
              equal string gdal_descriptions descriptions));
      prop "encode's embedding of a tile lands on its landmask's grid"
        (Synth.spec ~pixels:8 ()) (fun spec ->
          with_dir (fun dir ->
              let px = Synth.pixels spec in
              let rows, cols = tile_shape (Pixels.length px) in
              let landmask = Filename.concat dir "landmask.tiff" in
              Nx_io.save_tiff
                ~tags:(Jord.Geotiff.tags (grid ~rows ~cols))
                landmask
                (Nx.ones Nx.uint8 [| rows; cols; 1 |]);
              let g = ok (Jord.Geotiff.grid (fst (read_tiff landmask))) in
              let e = Embedding.encode ~shape Nx.Host.on (Model.apply model) px in
              let e = Nx.reshape [| rows; cols; 16 |] e in
              let name = Filename.concat dir "tile" in
              Embedding.save_geotiff name ~grid:g e;
              let g', nodata, _, bands = read_geotiff (name ^ ".tiff") in
              equal grid_t (grid ~rows ~cols) g';
              equal (option float_exact) (Some Float.nan) nodata;
              List.iteri
                (fun p (n2, n1) ->
                  let empty = n2 = 0 && n1 = 0 in
                  cover "a pixel without observations" empty;
                  let at a c = Nx.item [ p / cols; p mod cols; c ] a in
                  let row = Array.init 16 (at bands) in
                  equal
                    ~msg:(Printf.sprintf "pixel %d" p)
                    (array float_exact)
                    (Array.init 16 (at e))
                    row;
                  equal ~msg:"no data" bool empty (Array.exists Float.is_nan row))
                (List.combine spec.s2 spec.s1)));
      test "save_geotiff refuses a tile of another shape than the grid's"
        (fun () ->
          raises_match (Exn.invalid_arg ~sub:"a grid of 2 × 2") (fun () ->
              Embedding.save_geotiff "unused" ~grid:(grid ~rows:2 ~cols:2)
                (Nx.ones Nx.float32 [| 1; 4; 3 |])));
    ]

let () =
  exit
    (run "tessera.embedding"
       [
         group "quantize" [ ties; no_embedding; round_trip ];
         files;
         group "encode" [ undefined ];
         geotiff;
       ])
