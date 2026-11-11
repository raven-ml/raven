(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* make_labelled.exe DIR: a tile of 6 × 6 pixels on a 10 m grid of WGS 84 /
   UTM 33N in DIR, with its landmask.tiff and fields.geojson. Three fields in
   the file's order: b0, rows 0-2 and columns from 2.3 to 6; b1, rows 3-5 of
   the same columns; a, columns 0 to 2.3 of every row. Column 2's centres lie
   in b0 and b1, and field a's edge crosses column 2's cells. The pixel at
   (5, 5) has no observation. *)

let rows = 6
let cols = 6
let res = 10.
let origin = (480000., 5320000.)
let s2_frames = 8
let s1_frames = 5

let () =
  let dir = Sys.argv.(1) in
  if not (Sys.file_exists dir) then Sys.mkdir dir 0o755;
  let file f = Filename.concat dir f in
  let n = rows * cols in
  (* Class a's bands rise over the year, class b's fall. *)
  let a_of p = p mod cols < 2 in
  let empty p = p = n - 1 in
  let bands =
    Array.init (s2_frames * n * 10) (fun i ->
        let t = i / (n * 10) and p = i / 10 mod n and c = i mod 10 in
        let season = if a_of p then t else s2_frames - t in
        1000 + (300 * season) + (37 * c) + (11 * (p mod 7)))
  in
  Nx_io.save_npy (file "bands.npy")
    (Nx.create Nx.uint16 [| s2_frames; rows; cols; 10 |] bands);
  Nx_io.save_npy (file "masks.npy")
    (Nx.create Nx.uint8 [| s2_frames; rows; cols |]
       (Array.init (s2_frames * n) (fun i -> if empty (i mod n) then 0 else 1)));
  Nx_io.save_npy (file "doys.npy")
    (Nx.create Nx.uint16 [| s2_frames |]
       (Array.init s2_frames (fun t -> 20 + (40 * t))));
  let sar =
    Array.init (s1_frames * n * 2) (fun i ->
        let p = i / 2 mod n and c = i mod 2 in
        if empty p then 0 else if a_of p then 6000 + (200 * c) else 4000 + c)
  in
  Nx_io.save_npy (file "sar_ascending.npy")
    (Nx.create Nx.int16 [| s1_frames; rows; cols; 2 |] sar);
  Nx_io.save_npy (file "sar_ascending_doy.npy")
    (Nx.create Nx.int16 [| s1_frames |]
       (Array.init s1_frames (fun t -> 30 + (60 * t))));
  let crs = Jord.Crs.utm ~zone:33 ~north:true in
  let grid =
    Jord.Grid.v crs ~rows ~cols ~origin ~row_step:(0., -.res)
      ~col_step:(res, 0.)
  in
  Nx_io.save_tiff
    ~tags:(Jord.Geotiff.tags grid)
    (file "landmask.tiff")
    (Nx.ones Nx.uint8 [| rows; cols; 1 |]);
  (* A field's ring of pixel points (row, col), in CRS84. *)
  let ring (r0, c0) (r1, c1) =
    let corners = [ (r0, c0); (r0, c1); (r1, c1); (r1, c0); (r0, c0) ] in
    let px =
      Nx.create Nx.float64 [| 5; 2 |]
        (Array.of_list (List.concat_map (fun (r, c) -> [ r; c ]) corners))
    in
    let ll =
      Jord.Crs.transform ~src:crs ~dst:Jord.Crs.lonlat (Jord.Grid.world grid px)
    in
    String.concat ", "
      (List.init 5 (fun k ->
           Printf.sprintf "[%.17g, %.17g]" (Nx.item [ k; 0 ] ll)
             (Nx.item [ k; 1 ] ll)))
  in
  let feature cls corners =
    Printf.sprintf
      {|{"type": "Feature", "properties": {"class": "%s"}, "geometry": {"type": "Polygon", "coordinates": [[%s]]}}|}
      cls (ring (fst corners) (snd corners))
  in
  Out_channel.with_open_bin (file "fields.geojson") (fun oc ->
      Printf.fprintf oc {|{"type": "FeatureCollection", "features": [%s]}|}
        (String.concat ", "
           [
             feature "b" ((0., 2.3), (3., 6.));
             feature "b" ((3., 2.3), (6., 6.));
             feature "a" ((0., 0.), (6., 2.3));
           ]))
