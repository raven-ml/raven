(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tessera

let rows dims l =
  Nx.create Nx.float32 [| List.length l / dims; dims |] (Array.of_list l)

(* Quantisation *)

(* numpy's rule, written out: the nearest integer, a half to the even one. *)
let np_round x =
  let f = Float.floor x in
  let d = x -. f in
  if d < 0.5 then f
  else if d > 0.5 then f +. 1.
  else if Float.rem f 2. = 0. then f
  else f +. 1.

(* A code's quotient is within [−127, 127], so its ties are k + 1/2 for k in
   −127..126, each in a row of scale 1. *)
let every_tie =
  test "rounds every tie as numpy does" (fun () ->
      let ties = List.init 254 (fun i -> Float.of_int (i - 127) +. 0.5) in
      let e = rows 2 (List.concat_map (fun x -> [ 127.; x ]) ties) in
      let want x = [ 127; Float.to_int (np_round x) ] in
      equal (array int)
        (Array.of_list (List.concat_map want ties))
        (Nx.to_array (fst (Embedding.quantize e))))

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
      test "load reads save's file back" (fun () ->
          with_dir (fun dir ->
              let name = Filename.concat dir "tile" in
              let e =
                Nx.reshape [| 1; 2; 3 |] (rows 3 [ 1.; -2.; 0.5; 3.; 0.; -1. ])
              in
              Embedding.save name e;
              equal (array float_exact) (Nx.to_array e)
                (Nx.to_array (Embedding.load name))));
      test "load reads codes times scale, NaN where the scale is not finite"
        (fun () ->
          with_dir (fun dir ->
              let name = Filename.concat dir "tile" in
              Nx_io.save_npy (name ^ ".npy")
                (Nx.create Nx.int8 [| 1; 3; 2 |] [| 3; -127; 0; 0; 5; 1 |]);
              Nx_io.save_npy (name ^ "_scales.npy")
                (Nx.create Nx.float32 [| 1; 3 |]
                   [| 0.5; Float.nan; Float.infinity |]);
              let e = Nx.to_array (Embedding.load name) in
              equal (array float_exact) [| 1.5; -63.5 |] (Array.sub e 0 2);
              equal ~msg:"no data" (array bool) [| true; true; true; true |]
                (Array.map Float.is_nan (Array.sub e 2 4))));
      test "load of save_int8's files is the int8 form's values" (fun () ->
          with_dir (fun dir ->
              let name = Filename.concat dir "tile" in
              let e =
                Nx.reshape [| 2; 1; 3 |]
                  (rows 3 [ 1.; -2.; 0.5; Float.nan; Float.nan; Float.nan ])
              in
              Embedding.save_int8 name e;
              let codes, scales = Embedding.quantize e in
              let want =
                Nx.mul (Nx.cast Nx.float32 codes)
                  (Nx.unsqueeze ~axes:[ -1 ] scales)
              in
              equal (array float_exact) (Nx.to_array want)
                (Nx.to_array (Embedding.load name))));
      prop "load of save_int8's files is within (1/2 + 2⁻¹⁵) of a scale"
        Gen.(list ~size:(int_range 1 40) (float_range (-6.) 6.))
        (fun l ->
          with_dir (fun dir ->
              let name = Filename.concat dir "tile" in
              let n = List.length l in
              let e = Nx.reshape [| 1; 1; n |] (rows n l) in
              Embedding.save_int8 name e;
              let s = Nx.item [ 0; 0 ] (snd (Embedding.quantize e)) in
              let got = Embedding.load name in
              for i = 0 to n - 1 do
                at_most float_exact
                  ~than:(s *. (0.5 +. ldexp 1. (-15)))
                  (Float.abs (Nx.item [ 0; 0; i ] got -. Nx.item [ 0; 0; i ] e))
              done));
      test "load refuses scales of another shape than the codes'" (fun () ->
          with_dir (fun dir ->
              let name = Filename.concat dir "tile" in
              Nx_io.save_npy (name ^ ".npy") (Nx.zeros Nx.int8 [| 2; 3; 4 |]);
              Nx_io.save_npy (name ^ "_scales.npy")
                (Nx.ones Nx.float32 [| 3; 2 |]);
              raises_match (Exn.failure ~sub:"_scales.npy") (fun () ->
                  Embedding.load name)));
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

(* What the file states: its grid, nodata and band descriptions, and its
   bands. *)
let read_geotiff file =
  let ifd = List.hd (Nx_io.load_tiff file) in
  let tags = Nx_io.Tiff.tags ifd in
  let descriptions =
    match List.assoc_opt 42112 tags with
    | Some (Nx_io.Tiff.Ascii s) -> s
    | _ -> failf "%s: no GDAL_METADATA text" file
  in
  ( ok (Jord_geotiff.grid tags),
    ok (Jord_geotiff.nodata tags),
    descriptions,
    Nx.unpack Nx.float32 (Nx_io.Tiff.samples ifd) )

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

let geotiff =
  prop "save_geotiff writes the embedding on its grid, as GeoTessera does"
    Gen.(pair (int_range 1 300) (int_range 1 300))
    (fun (rows, cols) ->
      cover "more than a tile down" (rows > 256);
      cover "more than a tile across" (cols > 256);
      with_dir (fun dir ->
          let name = Filename.concat dir "tile" in
          let g = grid ~rows ~cols in
          let e = distinct ~rows ~cols ~dims:3 in
          Embedding.save_geotiff name ~grid:g e;
          let g', nodata, descriptions, bands = read_geotiff (name ^ ".tiff") in
          equal grid_t g g';
          equal (option float_exact) (Some Float.nan) nodata;
          equal string gdal_descriptions descriptions;
          equal (array int) [| rows; cols; 3 |] (Nx.shape bands);
          equal (array float_exact) (Nx.to_array e) (Nx.to_array bands)))

let () =
  exit
    (run "tessera.embedding"
       [
         group "quantize" [ every_tie; no_embedding; round_trip ];
         files;
         group "encode" [ undefined ];
         group "geotiff" [ geotiff ];
       ])
