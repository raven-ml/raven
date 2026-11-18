(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Jord

let utm33 = Crs.utm ~zone:33 ~north:true
let read file = In_channel.with_open_bin file In_channel.input_all

let labels dir =
  let text = read (Filename.concat dir "austria_crops.geojson") in
  match
    Result.bind
      (Jsont_bytesrw.decode_string Jsont.json text)
      Jord_geojson.polygons
  with
  | Ok (p, _) -> p
  | Error e -> failwith e

let ints a = Nx.create Nx.int64 [| Array.length a |] (Array.map Int64.of_int a)

let project dir =
  let p = Polygons.to_crs utm33 (labels dir) in
  let save name t = Nx_io.save_npy (Filename.concat dir name) t in
  save "jord_points.npy" p.points;
  save "jord_rings.npy" (ints p.rings);
  save "jord_features.npy" (ints p.features);
  Printf.printf "%d features, %d points\n" (Polygons.count p)
    (Nx.dim 0 p.points)

(* A landmask's grid and nodata, from its first IFD's tags. *)
let landmask file =
  let ifd = List.hd (Nx_io.load_tiff file) in
  let tags = Nx_io.Tiff.tags ifd in
  match (Jord_geotiff.grid tags, Jord_geotiff.nodata tags) with
  | Ok g, Ok nodata -> (ifd, g, nodata)
  | Error e, _ | _, Error e -> failwith (file ^ ": " ^ e)

let describe file =
  let ifd, (g : Grid.t), nodata = landmask file in
  let (ox, oy), (_, ry), (cx, _) = (g.origin, g.row_step, g.col_step) in
  let mask = Nx.unpack Nx.uint8 (Nx_io.Tiff.samples ifd) in
  let ones =
    Array.fold_left (fun n v -> if v = 1 then n + 1 else n) 0 (Nx.to_array mask)
  in
  Format.printf
    "%s: %a, %d × %d cells, origin (%.17g, %.17g), steps %g, %g, nodata %s, %d \
     land cells@."
    (Filename.basename file) Crs.pp g.crs g.rows g.cols ox oy cx ry
    (Option.fold ~none:"none" ~some:string_of_float nodata)
    ones

let owner (g : Grid.t) (feature, cell) =
  let none = Nx.full Nx.int64 [| g.rows * g.cols |] (-1L) in
  Nx.to_array (Nx.scatter ~axis:0 cell feature none)

let compare dir masks =
  let p = labels dir in
  let tiles =
    List.sort Stdlib.compare
      (List.filter_map
         (fun f ->
           if Filename.check_suffix f ".tiff" then
             Some (Filename.chop_suffix f ".tiff")
           else None)
         (Array.to_list (Sys.readdir masks)))
  in
  List.iter
    (fun tile ->
      let path ext = Filename.concat dir (tile ^ ext) in
      let _, g, _ = landmask (Filename.concat masks (tile ^ ".tiff")) in
      let (ox, oy), (_, ry), (cx, _) = (g.origin, g.row_step, g.col_step) in
      Printf.printf "%s: %d × %d cells, origin (%.17g, %.17g), steps %g, %g\n"
        tile g.rows g.cols ox oy cx ry;
      let q = Polygons.to_crs g.crs p in
      List.iter
        (fun (rule, f) ->
          let mine = owner g (f g q) in
          List.iter
            (fun source ->
              let theirs =
                Nx.to_array
                  (Nx.unpack Nx.int64
                     (Nx_io.load_npy
                        (path (Printf.sprintf ".%s.%s.npy" source rule))))
              in
              let differ = ref 0 and labelled = ref 0 in
              Array.iteri
                (fun k o ->
                  if o <> theirs.(k) then incr differ;
                  if theirs.(k) >= 0L then incr labelled)
                mine;
              Printf.printf
                "  %-8s vertices by %-4s: %6d labelled cells, %d differ\n" rule
                source !labelled !differ)
            [ "jord"; "proj" ])
        [ ("centres", Rasterize.centres); ("touched", Rasterize.touched) ])
    tiles

let () =
  match Sys.argv with
  | [| _; "project"; dir |] -> project dir
  | [| _; "compare"; dir; masks |] -> compare dir masks
  | a when Array.length a >= 3 && a.(1) = "landmask" ->
      Array.iter describe (Array.sub a 2 (Array.length a - 2))
  | _ ->
      prerr_endline
        "usage: parity.exe (project DIR | compare DIR MASKS | landmask FILE...)";
      exit 2
