(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Encoding *)

(* Writes row [i] of [e] to row [pixel.{i}] of [out], for each [i] holding a
   pixel. *)
let scatter (out : (float, Bigarray.float32_elt) Flat.t) ~dims e
    (b : Nx.host Batch.t) =
  Flat.read b.pixel @@ fun pixel ->
  Flat.read e @@ fun (e : (float, Bigarray.float32_elt) Flat.t) ->
  for i = 0 to Bigarray.Array1.dim pixel - 1 do
    let p = Int64.to_int pixel.{i} in
    if p >= 0 then
      for c = 0 to dims - 1 do
        out.{(p * dims) + c} <- e.{(i * dims) + c}
      done
  done

(* Every batch, the last included, goes through [f], so that a tile whose pixels
   all lack an embedding still learns the embedding's length. *)
let encode ~shape at f px =
  let n = Pixels.length px in
  let counts = Batch.v2_counts px in
  let pack from = Batch.pack shape px counts ~from in
  let out = ref None in
  let rec go (b, next) =
    let e = f (Batch.place at b) in
    let following = if next < n then Some (pack next) else None in
    let e = Nx.place Nx.Host.on e in
    let dims = Nx.dim 1 e in
    let o =
      match !out with
      | Some (o, _) -> o
      | None ->
          let o = Flat.create Bigarray.float32 (n * dims) Float.nan in
          out := Some (o, dims);
          o
    in
    scatter o ~dims e b;
    Option.iter go following
  in
  go (pack 0);
  let o, dims = Option.get !out in
  Flat.wrap Nx.float32 [| n; dims |] o

(* Quantisation *)

(* GeoTessera's codes (tessera_infer_v2 student/quantize.py:37-50), whose
   [np.round] takes halves to the even neighbour. [Nx.round] takes them away
   from zero, so a half is rounded as twice the rounding of its half, which is
   exact in binary. *)
let round_even q =
  let tie = Nx.equal_s (Nx.abs (Nx.sub q (Nx.trunc q))) 0.5 in
  Nx.where tie (Nx.mul_s (Nx.round (Nx.div_s q 2.)) 2.) (Nx.round q)

let qmax = 127.
let absmax_floor = 1e-12

let quantize e =
  let absmax = Nx.maximum_s (Nx.max ~axes:[ -1 ] (Nx.abs e)) absmax_floor in
  let scale = Nx.div_s absmax qmax in
  let q = Nx.div e (Nx.unsqueeze ~axes:[ -1 ] scale) in
  let codes = Nx.clamp ~min:(-.qmax) ~max:qmax (round_even q) in
  (Nx.cast Nx.int8 codes, scale)

(* Files *)

let check_tile name e =
  if Nx.ndim e <> 3 then
    invalid_arg
      (Printf.sprintf "Tessera.Embedding.%s: rank %d, [h; w; dims] expected"
         name (Nx.ndim e))

let save name e =
  check_tile "save" e;
  Nx_io.save_npy (name ^ ".npy") e

let save_int8 name e =
  check_tile "save_int8" e;
  let codes, scales = quantize e in
  Nx_io.save_npy (name ^ ".npy") codes;
  Nx_io.save_npy (name ^ "_scales.npy") scales

(* GeoTessera's GeoTIFF (geotessera core.py:1313-1361, at 2814c5b): a band per
   value, tiled 256 × 256, nodata NaN, described in GDAL_METADATA in the text
   GDAL writes. GeoTessera compresses by LZW, which nx.io does not write. *)
let gdal_metadata = 42112
let geotiff_tile = (256, 256)

let descriptions dims =
  let b = Buffer.create (80 * (dims + 1)) in
  Buffer.add_string b "<GDALMetadata>\n";
  for k = 0 to dims - 1 do
    Printf.bprintf b
      "  <Item name=\"DESCRIPTION\" sample=\"%d\" \
       role=\"description\">Tessera_Band_%d</Item>\n"
      k k
  done;
  Buffer.add_string b "</GDALMetadata>\n";
  Buffer.contents b

let save_geotiff name ~(grid : Jord.Grid.t) e =
  check_tile "save_geotiff" e;
  if Nx.dim 0 e <> grid.rows || Nx.dim 1 e <> grid.cols then
    invalid_arg
      (Printf.sprintf
         "Tessera.Embedding.save_geotiff: a tile of %d × %d, a grid of %d × %d"
         (Nx.dim 0 e) (Nx.dim 1 e) grid.rows grid.cols);
  let tags =
    (gdal_metadata, Nx_io.Tiff.Ascii (descriptions (Nx.dim 2 e)))
    :: Jord_geotiff.tags ~nodata:Float.nan grid
  in
  Nx_io.save_tiff ~tile:geotiff_tile ~compression:Nx_io.Tiff.Deflate ~tags
    (name ^ ".tiff") e
