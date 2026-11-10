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
    Result.bind (Jsont_bytesrw.decode_string Jsont.json text) Geojson.polygons
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

(* A tile's tags, one per line: the tag, the TIFF type and the values. *)
let tags file =
  List.filter_map
    (fun line ->
      match String.split_on_char ' ' (String.trim line) with
      | tag :: ty :: values when line <> "" -> (
          let tag = int_of_string tag in
          let ints () = Array.of_list (List.map int_of_string values) in
          let floats () = Array.of_list (List.map float_of_string values) in
          match int_of_string ty with
          | 2 -> Some (tag, Geotiff.Ascii (String.concat " " values))
          | 3 -> Some (tag, Geotiff.Short (ints ()))
          | 4 -> Some (tag, Geotiff.Long (ints ()))
          | 12 -> Some (tag, Geotiff.Double (floats ()))
          | _ -> None)
      | _ -> None)
    (String.split_on_char '\n' (read file))

let owner (g : Grid.t) (feature, cell) =
  let none = Nx.full Nx.int64 [| g.rows * g.cols |] (-1L) in
  Nx.to_array (Nx.scatter ~axis:0 cell feature none)

let compare dir =
  let p = labels dir in
  let tiles =
    List.sort Stdlib.compare
      (List.filter_map
         (fun f ->
           if Filename.check_suffix f ".tags" then
             Some (Filename.chop_suffix f ".tags")
           else None)
         (Array.to_list (Sys.readdir dir)))
  in
  List.iter
    (fun tile ->
      let path ext = Filename.concat dir (tile ^ ext) in
      let g =
        match Geotiff.grid (tags (path ".tags")) with
        | Ok g -> g
        | Error e -> failwith e
      in
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
  | [| _; "compare"; dir |] -> compare dir
  | _ ->
      prerr_endline "usage: parity.exe (project | compare) DIR";
      exit 2
