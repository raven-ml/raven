(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tessera

let rec rm_rf path =
  if Sys.file_exists path then
    if Sys.is_directory path then begin
      Array.iter (fun n -> rm_rf (Filename.concat path n)) (Sys.readdir path);
      Sys.rmdir path
    end
    else Sys.remove path

let with_dir f =
  let dir = Filename.temp_dir "tessera" "tile" in
  Fun.protect ~finally:(fun () -> rm_rf dir) (fun () -> f dir)

let read file = In_channel.with_open_bin file In_channel.input_all
let ints x = Nx.to_array (Nx.cast Nx.int32 x)

(* A tile of 2 × 3 cells: two Sentinel-2 frames, negatives among their bands,
   one ascending frame and no descending one. *)
let grid =
  Jord.Grid.v
    (Jord.Crs.utm ~zone:31 ~north:true)
    ~rows:2 ~cols:3 ~origin:(448590., 5416580.) ~row_step:(0., -10.)
    ~col_step:(10., 0.)

let place =
  {
    Tile.grid;
    roi = Nx.create Nx.uint8 [| 2; 3 |] [| 1; 1; 0; 1; 0; 1 |];
    source = "mpc";
  }

let tile () =
  let t = Tile.empty ~place (2, 3) in
  {
    t with
    bands =
      Nx.create Nx.int16 [| 2; 2; 3; 10 |]
        (Array.init 120 (fun i -> (i * 37) - 999));
    masks =
      Nx.create Nx.uint8 [| 2; 2; 3 |] [| 1; 1; 0; 1; 0; 1; 0; 1; 0; 0; 0; 1 |];
    doys = Nx.create Nx.uint16 [| 2 |] [| 60; 366 |];
    ascending =
      Nx.create Nx.int16 [| 1; 2; 3; 2 |] (Array.init 12 (fun i -> i * 1000));
    ascending_doys = Nx.create Nx.int16 [| 1 |] [| 61 |];
  }

let crs = Testable.make ~pp:Jord.Crs.pp ~equal:Jord.Crs.equal

let grid_t =
  let pp ppf (g : Jord.Grid.t) =
    Format.fprintf ppf "%a %d × %d at (%g, %g)" Jord.Crs.pp g.crs g.rows g.cols
      (fst g.origin) (snd g.origin)
  in
  Testable.make ~pp ~equal:Jord.Grid.equal

let same_tile (a : Tile.t) (b : Tile.t) =
  equal (array int32) (ints a.bands) (ints b.bands);
  equal (array int32) (ints a.masks) (ints b.masks);
  equal (array int32) (ints a.doys) (ints b.doys);
  equal (array int32) (ints a.ascending) (ints b.ascending);
  equal (array int32) (ints a.ascending_doys) (ints b.ascending_doys);
  equal (array int) (Nx.shape a.descending) (Nx.shape b.descending);
  equal (array int32) (ints a.descending_doys) (ints b.descending_doys)

let files =
  group "files"
    [
      test "save then load gives the tile back, with its place" (fun () ->
          with_dir (fun dir ->
              let t = tile () in
              Tile.save dir t;
              let t' = Tile.load dir in
              same_tile t t';
              match t'.place with
              | None -> fail "no place"
              | Some p ->
                  equal grid_t grid p.grid;
                  equal string "mpc" p.source;
                  equal (array int32) (ints place.roi) (ints p.roi)));
      (* A saturated DN at frame 0's cell (0, 0) and frame 1's (1, 2), both
         observed, and at frame 0's (0, 2), unobserved. *)
      test "reads a uint16 value above 32767 as an invalid observation"
        (fun () ->
          with_dir (fun dir ->
              Tile.save dir (tile ());
              let at t c b = (((t * 6) + c) * 10) + b in
              let saturated = [ at 0 0 3; at 1 5 9; at 0 2 0 ] in
              let value i =
                if i = at 1 5 9 then 64535
                else if List.mem i saturated then 65535
                else 1000 + i
              in
              Nx_io.save_npy
                (Filename.concat dir "bands.npy")
                (Nx.create Nx.uint16 [| 2; 2; 3; 10 |] (Array.init 120 value));
              let t = Tile.load dir in
              let gone i = List.exists (fun s -> s / 10 = i / 10) saturated in
              equal (array int32)
                (Array.init 120 (fun i ->
                     Int32.of_int (if gone i then 0 else 1000 + i)))
                (ints t.bands);
              equal (array int32)
                (Array.map Int32.of_int
                   [| 0; 1; 0; 1; 0; 1; 0; 1; 0; 0; 0; 0 |])
                (ints t.masks)));
      test "reads missing masks, days, orbits and place as none" (fun () ->
          with_dir (fun dir ->
              Tile.save dir (tile ());
              List.iter
                (fun f -> Sys.remove (Filename.concat dir f))
                [
                  "masks.npy";
                  "doys.npy";
                  "sar_ascending.npy";
                  "sar_ascending_doy.npy";
                  "roi.tiff";
                ];
              let t = Tile.load dir in
              equal (array int32) (Array.make 12 0l) (ints t.masks);
              equal (array int32) [| 0l; 0l |] (ints t.doys);
              equal (array int) [| 0; 2; 3; 2 |] (Nx.shape t.ascending);
              is_none t.place));
      test "refuses a tile without bands" (fun () ->
          with_dir (fun dir ->
              Tile.save dir (tile ());
              Sys.remove (Filename.concat dir "bands.npy");
              raises_match (Exn.failure ~sub:"bands.npy") (fun () ->
                  ignore (Tile.load dir))));
      test "concat joins the frames of tiles in order" (fun () ->
          with_dir (fun dir ->
              let a = tile () in
              let b =
                {
                  (Tile.empty ~place (2, 3)) with
                  descending = Nx.ones Nx.int16 [| 1; 2; 3; 2 |];
                  descending_doys = Nx.create Nx.int16 [| 1 |] [| 200 |];
                }
              in
              Tile.save (Filename.concat dir "a") a;
              Tile.save (Filename.concat dir "b") b;
              let shards =
                List.map
                  (fun n -> Tile.load (Filename.concat dir n))
                  [ "a"; "b"; "a" ]
              in
              let out = Filename.concat dir "out" in
              Tile.concat ~place:(Some place) out shards;
              let t = Tile.load (Filename.concat dir "out") in
              equal (array int32)
                (Array.append (ints a.bands) (ints a.bands))
                (ints t.bands);
              equal (array int32) [| 60l; 366l; 60l; 366l |] (ints t.doys);
              equal (array int32) [| 61l; 61l |] (ints t.ascending_doys);
              equal (array int32) [| 200l |] (ints t.descending_doys);
              is_some t.place));
      cases "day_of_year counts from the first of January"
        ~id:(fun ((y, m, d), _) -> Printf.sprintf "%d-%d-%d" y m d)
        [
          ((2025, 1, 1), 1);
          ((2025, 7, 1), 182);
          ((2024, 2, 29), 60);
          ((2024, 12, 31), 366);
          ((2025, 12, 31), 365);
        ]
        (fun (date, n) -> equal int n (Tile.day_of_year date));
    ]

(* A square ring of [side] degrees with its corner at (lon, lat). *)
let square ~lon ~lat side =
  let pts =
    [| lon; lat; lon +. side; lat; lon +. side; lat +. side; lon; lat +. side |]
  in
  Jord.Polygons.v Jord.Crs.lonlat
    (Nx.create Nx.float64 [| 4; 2 |] pts)
    ~rings:[| 0; 4 |] ~features:[| 0; 1 |]

let polygons =
  group "of_polygons"
    [
      cases "takes TESSERA's UTM zone, with Norway's and Svalbard's"
        ~id:(fun (lon, lat, _) -> Printf.sprintf "%g, %g" lon lat)
        [
          (2.3, 48.8, Jord.Crs.utm ~zone:31 ~north:true);
          (5., 60., Jord.Crs.utm ~zone:32 ~north:true);
          (10., 78., Jord.Crs.utm ~zone:33 ~north:true);
          (-58.4, -34.6, Jord.Crs.utm ~zone:21 ~north:false);
        ]
        (fun (lon, lat, zone) ->
          let p = Tile.of_polygons ~source:"mpc" (square ~lon ~lat 0.01) in
          equal crs zone p.grid.crs);
      (* A hull from easting 500007 to 500012: TESSERA's tile keeps
         ⌈5 / 10⌉ = 1 column from 500000, and leaves 500010-500012 out. *)
      test "covers the polygons' bounds" (fun () ->
          let zone = Jord.Crs.utm ~zone:31 ~north:true in
          let pts =
            [| 500007.; 5000003.; 500012.; 5000003.; 500012.; 5000006. |]
          in
          let p =
            Tile.of_polygons ~source:"mpc"
              (Jord.Polygons.v zone
                 (Nx.create Nx.float64 [| 3; 2 |] pts)
                 ~rings:[| 0; 3 |] ~features:[| 0; 1 |])
          in
          equal crs zone p.grid.crs;
          equal (pair float_exact float_exact) (500000., 5000010.)
            p.grid.origin;
          equal int 2 p.grid.cols;
          equal int 1 p.grid.rows);
      test "lies on the 10 m lattice and covers the region's centres" (fun () ->
          let poly = square ~lon:2.3 ~lat:48.8 0.01 in
          let p = Tile.of_polygons ~source:"mpc" poly in
          let x, y = p.grid.origin in
          equal float_exact 0. (Float.rem x 10.);
          equal float_exact 0. (Float.rem y 10.);
          let poly = Jord.Polygons.to_crs p.grid.crs poly in
          let _, cell = Jord.Rasterize.centres p.grid poly in
          let roi = Nx.to_array (Nx.reshape [| -1 |] p.roi) in
          equal int (Nx.numel cell) (Array.fold_left (fun n v -> n + v) 0 roi);
          Array.iter
            (fun c -> equal int 1 roi.(Int64.to_int c))
            (Nx.to_array cell));
    ]

(* Places of landmasks *)

let utm31 = Jord.Crs.utm ~zone:31 ~north:true

let landmask ?(res = 10.) ~rows ~cols origin =
  Jord.Grid.v utm31 ~rows ~cols ~origin ~row_step:(0., -.res)
    ~col_step:(res, 0.)

let ones rows cols = Nx.ones Nx.uint8 [| rows; cols |]

let snapped ~rows ~cols (x, y) =
  (Tile.of_landmask ~source:"mpc" (landmask ~rows ~cols (x, y))
     (ones rows cols))
    .grid

let roi_list (p : Tile.place) = Array.to_list (ints p.roi)

(* Origins of GeoTessera's v2 landmasks and the windows the M7 and M4 checks
   read, by rasterio 1.5.2 (_plans/tessera-port/m7-revisit/lm.py), and the
   corner stackstac snaps each to. The central-meridian tiles lie a
   micrometre or less below easting 500000, and one just above. *)
let corners =
  [
    ("Paris", (448595.715, 5416575.514), (448590., 5416580.));
    ("zone edge", (287595.763, 5544659.906), (287590., 5544660.));
    ("Vienna F3", (596597.83, 5339477.33), (596590., 5339480.));
    ("Vienna window", (600957.831, 5331897.331), (600950., 5331900.));
    ("grid_3.05_45.05", (499999.99999918084, 4994064.093802556),
      (499990., 4994070.));
    ("grid_-20.95_81.55", (499999.9999999998, 9060179.843305765),
      (499990., 9060180.));
    ("grid_123.05_-15.15", (500000.00000000274, 8330613.514185729),
      (500000., 8330620.));
    ("grid_-52.35_0.15", (344199.67887027323, 22112.65541606301),
      (344190., 22120.));
  ]

(* A landmask GeoTIFF of [values] on a 10 m grid of 2 × 3 cells at [dt], with
   a second band of zeros if [two]. *)
let save_landmask ?(two = false) dt values file =
  let g = landmask ~rows:2 ~cols:3 (448595.715, 5416575.514) in
  let band = Nx.create dt [| 2; 3; 1 |] values in
  let samples =
    if two then Nx.concatenate ~axis:2 [ band; Nx.zeros_like band ] else band
  in
  Nx_io.save_tiff ~tags:(Jord_geotiff.tags g) file samples;
  (g, Nx.reshape [| 2; 3 |] band)

let landmask_files =
  [
    test "a landmask file is of_landmask of its grid and first band" (fun () ->
        with_dir (fun dir ->
            let file = Filename.concat dir "landmask.tiff" in
            let check ?two dt values =
              let g, mask = save_landmask ?two dt values file in
              let p = Tile.load_landmask ~source:"mpc" file in
              let want = Tile.of_landmask ~source:"mpc" g mask in
              equal grid_t want.grid p.grid;
              equal (list int32) (roi_list want) (roi_list p)
            in
            check Nx.uint8 [| 0; 1; 7; 1; 0; 255 |];
            check Nx.int16 [| -3; 1; 0; 2; 0; 9 |];
            check Nx.float32 [| -1.; 0.5; 0.; 0.5; Float.nan; 0.5 |];
            check ~two:true Nx.uint8 [| 1; 0; 1; 1; 0; 1 |]));
    test "a landmask file without a grid is refused, naming it" (fun () ->
        with_dir (fun dir ->
            let file = Filename.concat dir "plain.tiff" in
            Nx_io.save_tiff file (Nx.ones Nx.uint8 [| 2; 3; 1 |]);
            raises_match
              (Exn.failure ~sub:("Tessera.Tile.load_landmask: " ^ file))
              (fun () -> Tile.load_landmask ~source:"mpc" file)));
  ]

let landmasks =
  group "of_landmask"
    (landmask_files @ [
      cases "moves the corner west and north to the 10 m lattice"
        ~id:(fun (name, _, _) -> name)
        corners
        (fun (_, corner, want) ->
          let g = snapped ~rows:1111 ~cols:788 corner in
          equal (pair float_exact float_exact) want g.origin;
          equal int 1111 g.rows;
          equal int 788 g.cols;
          equal (pair float_exact float_exact) (0., -10.) g.row_step;
          equal (pair float_exact float_exact) (10., 0.) g.col_step);
      test "snaps a float either side of a lattice line as stackstac" (fun () ->
          let x = 500000. and y = 5000000. in
          let at c = (snapped ~rows:1 ~cols:1 c).origin in
          let o = pair float_exact float_exact in
          equal o (x, y) (at (x, y));
          equal o (x, y) (at (Float.succ x, Float.pred y));
          equal o (x -. 10., y +. 10.) (at (Float.pred x, Float.succ y)));
      test "keeps a 10 m mask cell for cell, and is idempotent" (fun () ->
          let mask = Nx.create Nx.uint8 [| 2; 3 |] [| 0; 1; 7; 1; 0; 255 |] in
          let g = landmask ~rows:2 ~cols:3 (448595.715, 5416575.514) in
          let p = Tile.of_landmask ~source:"mpc" g mask in
          equal (list int32) [ 0l; 1l; 1l; 1l; 0l; 1l ] (roi_list p);
          equal string "mpc" p.source;
          let p' = Tile.of_landmask ~source:"mpc" p.grid p.roi in
          equal grid_t p.grid p'.grid;
          equal (list int32) (roi_list p) (roi_list p'));
      test "gives a landmask and its snapped grid one place" (fun () ->
          let mask = Nx.create Nx.uint8 [| 2; 2 |] [| 1; 0; 0; 1 |] in
          let at c =
            Tile.of_landmask ~source:"mpc" (landmask ~rows:2 ~cols:2 c) mask
          in
          let l = at (287595.763, 5544659.906)
          and l' = at (287590., 5544660.) in
          equal grid_t l.grid l'.grid;
          equal (list int32) (roi_list l) (roi_list l'));
      (* rasterio 1.5.2's reproject by nearest, as load_roi calls it (GDAL
         3.12.2), of these masks from (448595.715, 5416575.514) onto its 10 m
         template, cut on 2026-11-16 by
           python -I _plans/tessera-port/m7-revisit/near.py
         in M7's upstream environment, outside the repository. *)
      cases "reads a mask of other cells by GDAL's nearest"
        ~id:(fun (res, _, _, _) -> Printf.sprintf "%g m" res)
        [
          ( 20., (3, 4), [ 0; 0; 0; 1; 1; 0; 1; 0; 0; 1; 1; 1 ],
            [ 0; 0; 0; 0; 0; 0; 1; 1; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 0; 0;
              1; 1; 0; 0; 1; 1; 0; 0; 1; 1; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1;
              0; 0; 1; 1; 1; 1; 1; 1 ] );
          ( 15., (4, 3), [ 1; 1; 0; 0; 0; 0; 0; 0; 1; 1; 0; 1 ],
            [ 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0;
              1; 0; 0; 1; 0; 1; 0; 0; 1; 0 ] );
          ( 30., (2, 3), [ 1; 0; 1; 0; 0; 0 ],
            [ 1; 1; 1; 0; 0; 0; 1; 1; 1; 1; 1; 1; 0; 0; 0; 1; 1; 1; 1; 1;
              1; 0; 0; 0; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0;
              0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0 ] );
          ( 7.5, (5, 6),
            [ 1; 1; 1; 1; 0; 1; 1; 1; 0; 1; 1; 0; 0; 0; 0; 0; 1; 0; 0; 0;
              1; 0; 1; 1; 1; 1; 0; 1; 0; 0 ],
            [ 1; 1; 1; 0; 0; 0; 0; 0; 1; 0; 0; 1; 0; 1; 0; 1; 0; 1; 0; 0 ] );
        ]
        (fun (res, (rows, cols), mask, want) ->
          let g = landmask ~res ~rows ~cols (448595.715, 5416575.514) in
          let mask =
            Nx.create Nx.uint8 [| rows; cols |] (Array.of_list mask)
          in
          let p = Tile.of_landmask ~source:"mpc" g mask in
          equal (pair float_exact float_exact) (448590., 5416580.)
            p.grid.origin;
          equal int (List.length want) (p.grid.rows * p.grid.cols);
          equal (list int32) (List.map Int32.of_int want) (roi_list p));
      (* At 5 m every centre of the template lies on a border of the mask's
         cells. GDAL's arithmetic lands just below some borders and above
         others: from this corner it takes source row 2i and column 2j + 1. *)
      test "takes the cell after a border, at 5 m" (fun () ->
          let g = landmask ~res:5. ~rows:4 ~cols:4 (448595.715, 5416575.514) in
          let one (r, c) =
            let m = Array.make 16 0 in
            m.((r * 4) + c) <- 1;
            roi_list (Tile.of_landmask ~source:"mpc" g
                        (Nx.create Nx.uint8 [| 4; 4 |] m))
          in
          equal (list int32) [ 1l; 0l; 0l; 0l ] (one (1, 1));
          equal (list int32) [ 0l; 0l; 0l; 1l ] (one (3, 3));
          List.iter
            (fun rc -> equal (list int32) [ 0l; 0l; 0l; 0l ] (one rc))
            [ (0, 0); (0, 1); (1, 0); (2, 3) ]);
      test "takes the cells above 0 of a mask of any dtype" (fun () ->
          let g = landmask ~rows:1 ~cols:5 (448590., 5416580.) in
          let roi x = roi_list (Tile.of_landmask ~source:"mpc" g x) in
          let want = [ 0l; 0l; 1l; 0l; 1l ] in
          equal (list int32) want
            (roi
               (Nx.create Nx.float32 [| 1; 5 |]
                  [| -1.; 0.; 0.5; Float.nan; 2. |]));
          equal (list int32) want
            (roi (Nx.create Nx.int16 [| 1; 5 |] [| -5; 0; 3; -1; 7 |]));
          equal (list int32) want
            (roi (Nx.create Nx.uint16 [| 1; 5 |] [| 0; 0; 65535; 0; 1 |])));
    ])

(* The stackers' files (upstream/l2/stacked) against Tile.concat of the same
   days' frames (upstream/l2/days.npz; its dune file names the command that
   wrote them): equal arrays, in equal bytes. *)

let npy_header s = 10 + Char.code s.[8] + (256 * Char.code s.[9])

let payload file =
  let s = read file in
  let n = npy_header s in
  String.sub s n (String.length s - n)

let shape_text file =
  let s = read file in
  let h = String.sub s 10 (npy_header s - 10) in
  let i = String.index h '(' and j = String.index h ')' in
  String.sub h i (j - i + 1)

let stacking =
  test "concat writes the bytes TESSERA's stackers write" (fun () ->
      let days = Nx_io.load_npz "upstream/l2/days.npz" in
      let get dt name =
        match Nx_io.Archive.find name days with
        | Some x -> Nx.unpack dt x
        | None -> failwith name
      in
      let h = 12 and w = 16 in
      let all = Nx.ones Nx.bool [| h; w |] in
      let s2 day =
        let scl = get Nx.uint8 ("s2_" ^ day ^ "_scl") in
        let choice = Sentinel2.first_valid ~roi:all [ scl ] in
        ( Nx.reshape [| h; w; 10 |] (get Nx.int16 ("s2_" ^ day ^ "_bands")),
          Nx.cast Nx.uint8 (Nx.greater_equal_s choice 0l) )
      in
      let s1 day orbit =
        let p q = get Nx.int16 (Printf.sprintf "s1_%s_%s_%s" day orbit q) in
        Nx.stack ~axis:2 [ p "vv"; p "vh" ]
      in
      let date d = Scanf.sscanf d "%d-%d-%d" (fun y m d -> (y, m, d)) in
      let shard ?s2 ?ascending ?descending d =
        Tile.shard (h, w) (date d) ~s2 ~ascending ~descending
      in
      let shard = function
        | `S2 d -> shard ~s2:(s2 d) d
        | `Asc d -> shard ~ascending:(s1 d "ascending") d
        | `Desc d -> shard ~descending:(s1 d "descending") d
      in
      let days =
        [
          `S2 "2024-01-01";
          `Asc "2024-01-10";
          `S2 "2024-02-29";
          `Asc "2024-03-05";
          `Desc "2024-06-30";
          `S2 "2024-07-04";
          `S2 "2024-12-31";
        ]
      in
      with_dir (fun dir ->
          Tile.concat ~place:None dir (List.map shard days);
          List.iter
            (fun f ->
              let mine = Filename.concat dir f
              and theirs = Filename.concat "upstream/l2/stacked" f in
              equal ~msg:f string (shape_text theirs) (shape_text mine);
              equal ~msg:f string (payload theirs) (payload mine))
            [
              "bands.npy";
              "masks.npy";
              "doys.npy";
              "sar_ascending.npy";
              "sar_ascending_doy.npy";
              "sar_descending.npy";
              "sar_descending_doy.npy";
            ]))

let shard_place =
  test "a day's shard lies at its place, its frames of that day" (fun () ->
      let codes = Nx.ones Nx.int16 [| 2; 3; 2 |] in
      let t =
        Tile.shard ~place (2, 3) (2024, 3, 1) ~s2:None ~ascending:None
          ~descending:(Some codes)
      in
      equal (option grid_t) (Some place.grid)
        (Option.map (fun (p : Tile.place) -> p.grid) t.place);
      equal (array int) [| 0; 2; 3; 10 |] (Nx.shape t.bands);
      equal (array int) [| 0; 2; 3; 2 |] (Nx.shape t.ascending);
      equal (array int) [| 1; 2; 3; 2 |] (Nx.shape t.descending);
      equal (array int32) [| 61l |] (ints t.descending_doys))

let () =
  exit
    (run "tessera.tile"
       [
         files;
         landmasks;
         polygons;
         group "stackers" [ stacking; shard_place ];
       ])
