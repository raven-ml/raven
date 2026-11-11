(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord

let decode s =
  match Jsont_bytesrw.decode_string Jsont.json s with
  | Ok j -> j
  | Error e -> failwith e

let page file =
  let json = decode (In_channel.with_open_bin file In_channel.input_all) in
  match Jord_stac.items json with
  | Ok [ item ] -> item
  | Ok l -> failf "%s: %d items" file (List.length l)
  | Error e -> failf "%s: %s" file e

let time = pair (triple int int int) float_exact
let instant (t : Jord_stac.time) = (t.date, t.seconds)

let grid =
  Testable.make
    ~pp:(fun ppf (g : Grid.t) ->
      let (a, b), (c, d), (e, f) = (g.origin, g.row_step, g.col_step) in
      Format.fprintf ppf
        "%a %dx%d origin (%.17g, %.17g) row (%g, %g) col (%g, %g)" Crs.pp g.crs
        g.rows g.cols a b c d e f)
    ~equal:Grid.equal

let north_up epsg ~rows ~cols ~x ~y d =
  Grid.v
    (Result.get_ok (Crs.of_epsg epsg))
    ~rows ~cols ~origin:(x, y) ~row_step:(0., -.d) ~col_step:(d, 0.)

let band =
  Testable.make
    ~pp:(fun ppf (b : Band.t) ->
      let str = Option.value ~default:"-" in
      Format.fprintf ppf "{%s %s %s %s scale %g offset %g nodata %s}"
        (str b.name) (str b.common_name) (str b.unit)
        (match b.kind with Measurement -> "measurement" | Bitfield -> "bits")
        b.scale b.offset
        (Option.fold ~none:"-" ~some:string_of_float b.nodata))
    ~equal:(fun a b -> compare a b = 0)

let measured ?name ?common_name ?unit ?(scale = 1.) ?(offset = 0.) ?nodata
    () =
  {
    Band.name;
    common_name;
    unit;
    kind = Measurement;
    scale;
    offset;
    nodata;
    saturated = None;
  }

let asset (item : Jord_stac.item) key =
  match List.assoc_opt key item.assets with
  | Some a -> a
  | None -> failf "no asset %s" key

(* An asset's or a band's grid, failing on an [Error]. *)
let ok = function
  | None -> None
  | Some (Ok g) -> Some g
  | Some (Error e) -> failf "grid: %s" e

let bands (a : Jord_stac.asset) = List.map fst a.bands

(* Microsoft's Planetary Computer *)

let sentinel2 =
  test "MPC's Sentinel-2 L2A: grids per asset, eo:bands alone" (fun () ->
      let item = page "mpc_sentinel2_l2a.json" in
      equal string "S2A_MSIL2A_20240629T102021_R065_T33UUU_20240629T161811"
        item.id;
      equal (option string) (Some "sentinel-2-l2a") item.collection;
      equal (option time)
        (Some ((2024, 6, 29), 37221.024))
        (Option.map instant item.datetime);
      equal bool true (item.interval = None);
      let b04 = asset item "B04" and b01 = asset item "B01" in
      equal (option grid)
        (Some
           (north_up 32633 ~rows:10980 ~cols:10980 ~x:300000. ~y:5900040. 10.))
        (ok b04.grid);
      equal (option grid)
        (Some (north_up 32633 ~rows:1830 ~cols:1830 ~x:300000. ~y:5900040. 60.))
        (ok b01.grid);
      equal (list string) [ "data" ] b04.roles;
      equal (list band)
        [ measured ~name:"B04" ~common_name:"red" () ]
        (bands b04);
      (* The properties state a system and no shape or transform. *)
      equal (option grid) None (ok (asset item "preview").grid);
      equal (list band) [] (bands (asset item "SCL"));
      equal (list (option string))
        [ Some "B04"; Some "B03"; Some "B02" ]
        (List.map (fun (b : Band.t) -> b.name) (bands (asset item "visual"))))

let sentinel1 =
  test "MPC's Sentinel-1 RTC: the properties' grid, a spaced interval"
    (fun () ->
      let item = page "mpc_sentinel1_rtc.json" in
      equal (option time)
        (Some ((2024, 6, 29), 61257.420551))
        (Option.map instant item.datetime);
      equal (option (pair time time))
        (Some (((2024, 6, 29), 61244.921157), ((2024, 6, 29), 61269.919946)))
        (Option.map (fun (a, b) -> (instant a, instant b)) item.interval);
      let vv = asset item "vv" in
      (* proj:transform has 9 numbers. *)
      equal (option grid)
        (Some
           (north_up 32632 ~rows:22115 ~cols:29551 ~x:547450. ~y:5894760. 10.))
        (ok vv.grid);
      equal (list band) [ measured ~nodata:(-32768.) () ] (bands vv))

let landsat =
  test "MPC's Landsat C2 L2: raster and eo bands, bitfields" (fun () ->
      let item = page "mpc_landsat_c2_l2.json" in
      let red = asset item "red" in
      equal (option grid)
        (Some
           (north_up 32633 ~rows:8141 ~cols:8061 ~x:228885. ~y:5850915. 30.))
        (ok red.grid);
      equal (list band)
        [
          measured ~name:"OLI_B4" ~common_name:"red" ~scale:2.75e-05
            ~offset:(-0.2) ~nodata:0. ();
        ]
        (bands red);
      equal (list band)
        [
          measured ~name:"TIRS_B10" ~common_name:"lwir11" ~unit:"kelvin"
            ~scale:0.00341802 ~offset:149. ~nodata:0. ();
        ]
        (bands (asset item "lwir11"));
      equal (list band)
        [
          {
            (measured ~unit:"bit index" ~nodata:1. ()) with
            kind = Bitfield;
          };
        ]
        (bands (asset item "qa_pixel")))

(* Times *)

let times =
  cases ~id:fst "RFC 3339's date-times, in UTC"
    [
      ("2024-06-29T10:20:21.024000Z", ((2024, 6, 29), 37221.024));
      ("2024-06-29 17:00:44.921157+00:00", ((2024, 6, 29), 61244.921157));
      ("2024-06-29t00:00:00z", ((2024, 6, 29), 0.));
      ("2024-06-29T23:30:00-01:00", ((2024, 6, 30), 1800.));
      ("2024-01-01T00:30:00+01:00", ((2023, 12, 31), 84600.));
      ("2024-03-01T00:00:00+00:01", ((2024, 2, 29), 86340.));
      ("1998-12-31T23:59:60.5Z", ((1998, 12, 31), 86400.5));
      ("1999-01-01T00:59:60+01:00", ((1998, 12, 31), 86400.));
    ]
    (fun (s, want) ->
      match Jord_stac.time s with
      | Ok t -> equal time want (instant t)
      | Error e -> fail e)

let bad_times =
  cases ~id:Fun.id "texts that are not date-times"
    [
      "2024-06-29";
      "2024-06-29T10:20:21";
      "2024-06-29T10:20:21.Z";
      "2024-06-29X10:20:21Z";
      "2024-06-29T10:20:21Zx";
      "2024-13-01T00:00:00Z";
      "2023-02-29T00:00:00Z";
      "2024-06-29T24:00:00Z";
      "2024-06-29T10:20:61Z";
      "2024-06-29T10:20:21+0100";
      "2024-06-29T10:20:21+24:00";
      "+024-06-29T10:20:21Z";
    ]
    (fun s ->
      match Jord_stac.time s with
      | Ok _ -> failf "%S read" s
      | Error e -> contains ~sub:"RFC 3339" e)

(* Items of both STAC versions *)

let item ?(props = {|"datetime": "2024-06-29T00:00:00Z"|}) assets =
  decode
    (Printf.sprintf
       {|{"type": "Feature", "stac_version": "1.1.0", "id": "x",
          "geometry": null, "properties": {%s}, "assets": {%s}}|}
       props assets)

let read json =
  match Jord_stac.item json with Ok i -> i | Error e -> fail e

let stac11 =
  test "STAC 1.1's bands take their asset's fields" (fun () ->
      let i =
        read
          (item
             {|"a": {"href": "a.tif", "raster:scale": 0.5, "nodata": "nan",
                     "bands": [{"name": "b1", "eo:common_name": "red"},
                               {"name": "b2", "raster:scale": 2, "nodata": 0,
                                "raster:offset": -1}]},
               "one": {"href": "one.tif", "nodata": "-inf", "unit": "m",
                       "classification:bitfields": []},
               "none": {"href": "none.tif"}|})
      in
      equal (list band)
        [
          measured ~name:"b1" ~common_name:"red" ~scale:0.5
            ~nodata:Float.nan ();
          measured ~name:"b2" ~scale:2. ~offset:(-1.) ~nodata:0. ();
        ]
        (bands (asset i "a"));
      equal (list band)
        [
          {
            (measured ~unit:"m" ~nodata:Float.neg_infinity ()) with
            kind = Bitfield;
          };
        ]
        (bands (asset i "one"));
      equal (list band) [] (bands (asset i "none")))

(* An item whose properties state the data's grid in UTM 33N, 10 m cells,
   by [proj:epsg]. *)
let utm_item assets =
  item
    ~props:
      {|"datetime": "2024-06-29T00:00:00Z", "proj:epsg": 32633,
        "proj:shape": [2, 3],
        "proj:transform": [10, 0, 500000, 0, -10, 5300000, 0, 0, 1]|}
    assets

let data_grid = north_up 32633 ~rows:2 ~cols:3 ~x:500000. ~y:5300000. 10.

let projection =
  test "the system is one field, stated most specifically" (fun () ->
      let i =
        read
          (utm_item
             {|"data": {"href": "d.tif"},
               "thumb": {"href": "t.png", "proj:code": null,
                         "proj:shape": [64, 64]},
               "lonlat": {"href": "l.tif", "proj:code": "EPSG:4326",
                          "proj:transform": [0, 1, 10, 1, 0, 20]}|})
      in
      equal (option grid) (Some data_grid) (ok (asset i "data").grid);
      (* proj:code null states no system, over the properties' proj:epsg. *)
      equal (option grid) None (ok (asset i "thumb").grid);
      (* proj:code over the properties' proj:epsg, not against it. *)
      equal (option grid)
        (Some
           (Grid.v Crs.lonlat ~rows:2 ~cols:3 ~origin:(10., 20.)
              ~row_step:(1., 0.) ~col_step:(0., 1.)))
        (ok (asset i "lonlat").grid))

let band_grids =
  test "a band's projection fields override its asset's" (fun () ->
      let i =
        read
          (utm_item
             {|"multi": {"href": "m.tif",
                         "bands": [{"name": "b10"},
                                   {"name": "b60", "proj:shape": [1, 1],
                                    "proj:transform":
                                      [60, 0, 500000, 0, -60, 5300000]}]}|})
      in
      let multi = asset i "multi" in
      equal (option grid) (Some data_grid) (ok multi.grid);
      equal (list (option grid))
        [
          Some data_grid;
          Some (north_up 32633 ~rows:1 ~cols:1 ~x:500000. ~y:5300000. 60.);
        ]
        (List.map (fun (_, g) -> ok g) multi.bands))

(* An asset [a] of shape [shape] with the projection fields [fields]. *)
let stated ?(shape = "[1, 1]") fields =
  utm_item
    (Printf.sprintf
       {|"data": {"href": "d.tif"},
         "a": {"href": "a", "proj:shape": %s, %s}|}
       shape fields)

let bad_grids =
  cases ~id:snd "a grid that cannot be read is its asset's Error"
    [
      ( stated
          {|"proj:code": "EPSG:3857",
            "proj:transform": [1, 0, 0, 0, -1, 0]|},
        "3857" );
      ( stated {|"proj:transform": [1, 0, 0, 0, -1, 0, 0, 1, 1]|},
        "last row" );
      (stated {|"proj:transform": [1, 0, 0, 0, -1, 0, 0]|}, "6 nor 9");
      (stated {|"proj:transform": [1, 2, 0, 2, 4, 0]|}, "not invertible");
      ( stated
          {|"proj:code": "EPSG:32633", "proj:epsg": 32632,
            "proj:transform": [1, 0, 0, 0, -1, 0]|},
        "proj:epsg" );
      ( stated
          {|"proj:code": "IAU_2015:30100",
            "proj:transform": [1, 0, 0, 0, -1, 0]|},
        "not an EPSG code" );
      ( stated ~shape:"[1]" {|"proj:transform": [1, 0, 0, 0, -1, 0]|},
        "two counts" );
    ]
    (fun (json, sub) ->
      let i = read json in
      equal (option grid) (Some data_grid) (ok (asset i "data").grid);
      match (asset i "a").grid with
      | Some (Error e) -> contains ~sub e
      | Some (Ok _) | None -> fail "read")

let refusals =
  cases ~id:snd "items that break STAC"
    [
      (item ~props:{|"datetime": null|} "", "no datetime");
      ( item
          ~props:{|"datetime": null, "start_datetime": "2024-06-29T00:00:00Z"|}
          "",
        "no end_datetime" );
      (item ~props:{|"datetime": "2024-06-29"|} "", "RFC 3339");
      (item {|"a": {"type": "image/tiff"}|}, "no href");
      (item {|"a": {"href": "a", "nodata": "none"}|}, "nodata");
      ( item
          {|"a": {"href": "a", "eo:bands": [{}, {}], "raster:bands": [{}]}|},
        "differ in length" );
      (decode {|{"type": "FeatureCollection", "features": []}|}, "Feature");
    ]
    (fun (json, sub) ->
      match Jord_stac.item json with
      | Ok _ -> fail "read"
      | Error e -> contains ~sub e)

let () =
  exit
    (run "jord.stac"
       [
         group "mpc" [ sentinel2; sentinel1; landsat ];
         group "time" [ times; bad_times ];
         group "items"
           [ stac11; projection; band_grids; bad_grids; refusals ];
       ])
