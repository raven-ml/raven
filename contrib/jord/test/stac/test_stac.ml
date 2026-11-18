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

(* Instants to 0.5 µs, a float64's spacing in 2024 being 0.24 µs; [june29]
   is 2024-06-29T00:00:00Z. *)
let time = float 5e-7
let june29 = 1719619200.

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
      let num = Option.fold ~none:"-" ~some:string_of_float in
      Format.fprintf ppf "{%s %s %s %s %s %s scale %g offset %g nodata %s}"
        (str b.name) (str b.common_name) (num b.center_wavelength)
        (num b.full_width_half_max) (str b.unit)
        (match b.kind with
        | Measurement -> "measurement"
        | Bitfield -> "bits"
        | Classes l -> Printf.sprintf "%d classes" (List.length l))
        b.scale b.offset (num b.nodata))
    ~equal:(fun a b -> compare a b = 0)

let measured ?name ?common_name ?centre ?width ?unit ?(scale = 1.)
    ?(offset = 0.) ?nodata () =
  {
    Band.name;
    common_name;
    center_wavelength = centre;
    full_width_half_max = width;
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

let bands (a : Jord_stac.asset) =
  List.map (fun (b : Jord_stac.band) -> b.band) a.bands

(* Microsoft's Planetary Computer *)

let sentinel2 =
  test "MPC's Sentinel-2 L2A: grids per asset, eo:bands alone" (fun () ->
      let item = page "mpc_sentinel2_l2a.json" in
      equal string "S2A_MSIL2A_20240629T102021_R065_T33UUU_20240629T161811"
        item.id;
      equal (option string) (Some "sentinel-2-l2a") item.collection;
      equal (option time) (Some (june29 +. 37221.024)) item.datetime;
      equal bool true (item.interval = None);
      let b04 = asset item "B04" and b01 = asset item "B01" in
      equal (option grid)
        (Some
           (north_up 32633 ~rows:10980 ~cols:10980 ~x:300000. ~y:5900040. 10.))
        (ok b04.grid);
      equal (option grid)
        (Some (north_up 32633 ~rows:1830 ~cols:1830 ~x:300000. ~y:5900040. 60.))
        (ok b01.grid);
      equal (option int) (Some 65) item.relative_orbit;
      equal (option float_exact) None item.center_frequency;
      equal (list string) [ "data" ] b04.roles;
      equal (list band)
        [
          measured ~name:"B04" ~common_name:"red" ~centre:0.665 ~width:0.038
            ();
        ]
        (bands b04);
      (* The properties state a system and no shape or transform. *)
      equal (option grid) None (ok (asset item "preview").grid);
      equal (list band) [] (bands (asset item "SCL"));
      equal (list (option string))
        [ Some "B04"; Some "B03"; Some "B02" ]
        (List.map (fun (b : Band.t) -> b.name) (bands (asset item "visual"))))

let classes =
  let pp ppf (c : Band.class_) =
    let str = Option.value ~default:"-" in
    Format.fprintf ppf "%d %s %s%s" c.value (str c.name) (str c.description)
      (if c.nodata then " nodata" else "")
  in
  list (Testable.make ~pp ~equal:( = ))

let class_ ?name ?title ?description ?(nodata = false) value =
  { Band.value; name; title; description; nodata }

let classes_of (b : Band.t) =
  match b.kind with Classes l -> l | _ -> fail "not a classified band"

(* ESA WorldCover's map states classification 1.0's classes on the asset,
   with descriptions and colours and no names, beside raster:bands. *)
let worldcover =
  test "MPC's WorldCover: a classified band of classification 1.0" (fun () ->
      let item = page "mpc_esa_worldcover.json" in
      let map =
        match bands (asset item "map") with
        | [ b ] -> b
        | l -> failf "map: %d bands" (List.length l)
      in
      equal (option float_exact) (Some 0.) map.nodata;
      equal classes
        (List.map
           (fun (v, d) -> class_ ~description:d v)
           [
             (10, "Tree cover");
             (20, "Shrubland");
             (30, "Grassland");
             (40, "Cropland");
             (50, "Built-up");
             (60, "Bare / sparse vegetation");
             (70, "Snow and ice");
             (80, "Permanent water bodies");
             (90, "Herbaceous wetland");
             (95, "Mangroves");
             (100, "Moss and lichen");
           ])
        (classes_of map);
      raises_match (Exn.invalid_arg ~sub:"classes") (fun () ->
          Band.decode map Nx.float32 (Nx.zeros Nx.uint8 [| 1 |])))

(* MPC's MODIS states its system by proj:wkt2 alone: GDAL's WKT 1 for
   MODLAND's sinusoid. *)
let modis =
  test "MPC's MODIS: proj:wkt2 states the sinusoid" (fun () ->
      let item = page "mpc_modis_13q1.json" in
      equal (option grid)
        (Some
           (Grid.v Crs.modis ~rows:4800 ~cols:4800 ~origin:(0., 5559752.598333)
              ~row_step:(0., -231.65635826375006)
              ~col_step:(231.65635826395834, 0.)))
        (ok (asset item "250m_16_days_NDVI").grid);
      match bands (asset item "250m_16_days_pixel_reliability") with
      | [ b ] ->
          equal (list int) [ 0; 1; 2; 3 ]
            (List.map (fun (c : Band.class_) -> c.value) (classes_of b))
      | l -> failf "pixel reliability: %d bands" (List.length l))

let orbit =
  Testable.make
    ~pp:(fun ppf o ->
      Format.pp_print_string ppf
        (match o with
        | Jord_stac.Ascending -> "ascending"
        | Descending -> "descending"
        | Geostationary -> "geostationary"
        | Crossing -> "crossing"))
    ~equal:( = )

let sentinel1 =
  test "MPC's Sentinel-1 RTC: the properties' grid, a spaced interval"
    (fun () ->
      let item = page "mpc_sentinel1_rtc.json" in
      equal (option time) (Some (june29 +. 61257.420551)) item.datetime;
      equal (option (pair time time))
        (Some (june29 +. 61244.921157, june29 +. 61269.919946))
        item.interval;
      let vv = asset item "vv" in
      (* proj:transform has 9 numbers. *)
      equal (option grid)
        (Some
           (north_up 32632 ~rows:22115 ~cols:29551 ~x:547450. ~y:5894760. 10.))
        (ok vv.grid);
      equal (list band) [ measured ~nodata:(-32768.) () ] (bands vv);
      equal (option orbit) (Some Jord_stac.Ascending) item.orbit_state;
      equal (option int) (Some 44) item.relative_orbit;
      equal (option float_exact) (Some 5.405) item.center_frequency)

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
          measured ~name:"OLI_B4" ~common_name:"red" ~centre:0.65 ~width:0.04
            ~scale:2.75e-05 ~offset:(-0.2) ~nodata:0. ();
        ]
        (bands red);
      equal (list band)
        [
          measured ~name:"TIRS_B10" ~common_name:"lwir11" ~centre:10.9
            ~width:0.59 ~unit:"kelvin" ~scale:0.00341802 ~offset:149.
            ~nodata:0. ();
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

(* A catalogue's wavelengths against its producer's *)

let wavelengths (b : Band.t) = (b.center_wavelength, b.full_width_half_max)
let stated = pair (option float_exact) (option float_exact)

(* The HLS guide's S30 bands are "Adjusted to OLI-like (except red edge,
   water vapor and cirrus bands)" (Table 2), while MPC's S30 items state one
   set near Sentinel-2A MSI's (SentiWiki, "S2 Mission", Table 3). B11's
   centre is S2A's 1613.7 nm, 4.9 nm from OLI's band 6; B12's is S2A's
   2202.4 nm with a digit repeated. The widths are the extents of S2A's
   tabulated responses, MAX - MIN in its MTD_MSIL2A.xml: 1682 - 1539 nm for
   B11 and 2320 - 2078 nm for B12, whose width SentiWiki states as 180 nm.
   So [Hls] states none. *)
let hls_s30 =
  test "MPC's HLS S30 states near-MSI wavelengths, which Hls does not adopt"
    (fun () ->
      let item = page "mpc_hls2_s30.json" in
      List.iter
        (fun b ->
          let name = Hls.name S30 b in
          equal ~msg:name stated (None, None) (wavelengths (Hls.band S30 b));
          match bands (asset item name) with
          | [ { center_wavelength = Some _; full_width_half_max = Some _; _ } ]
            ->
              ()
          | _ -> failf "%s: not one band with wavelengths" name)
        Hls.
          [
            Coastal; Blue; Green; Red; Red_edge1; Red_edge2; Red_edge3;
            Nir_broad; Nir_narrow; Water_vapour; Cirrus; Swir16; Swir22;
          ];
      let stated_by k = wavelengths (List.hd (bands (asset item k))) in
      equal stated (Some 1.6137, Some 0.143) (stated_by "B11");
      equal stated (Some 2.22024, Some 0.242) (stated_by "B12");
      equal (option float_exact) (Some 1.6088)
        (Landsat.band Landsat_8 Swir16).center_wavelength)

(* MPC's Landsat 9 items state band 10 as Landsat 8's TIRS, to 0.01 µm, not
   as TIRS-2, whose values [Landsat] keeps. *)
let landsat9_tirs =
  test "MPC's Landsat 9 band 10 is Landsat 8's TIRS, which Landsat does not \
        adopt" (fun () ->
      let item = page "mpc_landsat_c2_l2.json" in
      let cents = Option.map (fun x -> Float.round (x *. 100.) /. 100.) in
      let tirs = Landsat.band Landsat_8 Lwir in
      equal stated (Some 10.9, Some 0.59)
        (wavelengths (List.hd (bands (asset item "lwir11"))));
      equal stated (Some 10.9, Some 0.59)
        (cents tirs.center_wavelength, cents tirs.full_width_half_max);
      equal stated (Some 10.825, Some 0.75)
        (wavelengths (Landsat.band Landsat_9 Lwir)))

(* MPC's MODIS items state each tile's grid from its product's corners, which
   the product's metadata gives to the micrometre: MODLAND's grid, rounded.
   [Modis.tile] keeps the grid. *)
let modis_grids =
  cases "MPC's MODIS grids are the tile's corners to the micrometre" ~id:fst
    [
      ("mpc_modis_13q1.json", (Modis.Mod13q1, "250m_16_days_NDVI"));
      ("mpc_modis_09a1.json", (Modis.Mod09a1, "sur_refl_b01"));
    ]
    (fun (file, (p, key)) ->
      let item = page file in
      let mpc =
        match ok (asset item key).grid with
        | Some g -> g
        | None -> failf "%s: no grid" key
      in
      let ours = Modis.tile p ~h:18 ~v:4 in
      is_true ~msg:"equal" (not (Grid.equal mpc ours));
      let corner g (row, col) =
        let p = Nx.create Nx.float64 [| 2 |] [| row; col |] in
        let x = Nx.to_array (Grid.world g p) in
        (x.(0), x.(1))
      in
      let um x = Float.round (x *. 1e6) /. 1e6 in
      let n = Float.of_int ours.rows in
      List.iter
        (fun c ->
          let x, y = corner ours c in
          equal (pair (float 1e-8) (float 1e-8)) (um x, um y) (corner mpc c))
        [ (0., 0.); (0., n); (n, 0.); (n, n) ])

(* MPC's MODIS assets are named by the products' layers and state their
   scales, but no fill: a catalogue band decodes MOD09A1's fill, -28672, as a
   reflectance of -2.8672. [Modis] states the guides' fills. *)
let modis_layers =
  cases "MPC's MODIS assets are the guides' layers, without their fills"
    ~id:fst
    [ ("mpc_modis_13q1.json", Modis.Mod13q1); ("mpc_modis_09a1.json", Mod09a1) ]
    (fun (file, p) ->
      let item = page file in
      List.iter
        (fun b ->
          let ours = Modis.band p b in
          let key =
            String.map (function ' ' -> '_' | c -> c) (Modis.name p b)
          in
          match bands (asset item key) with
          | [ m ] ->
              equal ~msg:key float_exact ours.scale m.scale;
              equal ~msg:key (option float_exact) None m.nodata;
              is_true ~msg:key (ours.nodata <> None)
          | l -> failf "%s: %d bands" key (List.length l))
        (Modis.bands p);
      if p = Mod09a1 then begin
        let dn = Nx.create Nx.int16 [| 1 |] [| -28672 |] in
        let b01 = List.hd (bands (asset item "sur_refl_b01")) in
        let v, valid = Band.decode b01 Nx.float64 dn in
        equal (array bool) [| true |] (Nx.to_array valid);
        equal (array (float 1e-15)) [| -2.8672 |] (Nx.to_array v);
        let _, valid = Band.decode (Modis.band p Red) Nx.float64 dn in
        equal (array bool) [| false |] (Nx.to_array valid)
      end)

(* MOD09 Table 6 names each band by its range. MPC's MOD09A1 items state the
   ranges' midpoints as centres and ten times the ranges as widths, band 2 as
   nir08, which STAC's own table lists as MODIS's nir, and band 5, at 1.24
   µm, as lwir12, 11.5 to 12.5 µm. [Modis] states no wavelengths. *)
let modis_wavelengths =
  test "MPC's MOD09A1 widths are ten times the guide's ranges, which Modis \
        does not adopt" (fun () ->
      let item = page "mpc_modis_09a1.json" in
      List.iter
        (fun (b, (lo, hi)) ->
          let key = Modis.name Mod09a1 b in
          equal ~msg:key stated (None, None)
            (wavelengths (Modis.band Mod09a1 b));
          match bands (asset item key) with
          | [ { center_wavelength = Some c; full_width_half_max = Some w; _ } ]
            ->
              equal ~msg:key (float 1e-12) ((lo +. hi) /. 2000.) c;
              equal ~msg:key (float 1e-12) (10. *. (hi -. lo) /. 1000.) w
          | _ -> failf "%s: not one band with wavelengths" key)
        Modis.
          [
            (Red, (620., 670.)); (Nir, (841., 876.)); (Blue, (459., 479.));
            (Green, (545., 565.)); (Swir12, (1230., 1250.));
            (Swir16, (1628., 1652.)); (Swir22, (2105., 2155.));
          ];
      let common k = (List.hd (bands (asset item k))).common_name in
      equal (option string) (Some "nir08") (common "sur_refl_b02");
      equal (option string) (Some "lwir12") (common "sur_refl_b05");
      equal (option string) (Some "nir") (Modis.band Mod09a1 Nir).common_name;
      equal (option string) None (Modis.band Mod09a1 Swir12).common_name)

(* A catalogue's bit fields *)

let fields (a : Jord_stac.asset) =
  List.concat_map
    (fun (b : Jord_stac.band) ->
      List.map
        (fun (f : Jord_stac.bitfield) ->
          (Option.value f.name ~default:"-", (f.offset, f.length)))
        b.bitfields)
    a.bands

let bitfields =
  test "MPC's Landsat states QA_PIXEL's bit fields beside its band" (fun () ->
      let item = page "mpc_landsat_c2_l2.json" in
      let qa = asset item "qa_pixel" in
      equal
        (list (pair string (pair int int)))
        [
          ("fill", (0, 1));
          ("dilated_cloud", (1, 1));
          ("cirrus", (2, 1));
          ("cloud", (3, 1));
          ("cloud_shadow", (4, 1));
          ("snow", (5, 1));
          ("clear", (6, 1));
          ("water", (7, 1));
          ("cloud_confidence", (8, 2));
          ("cloud_shadow_confidence", (10, 2));
          ("snow_confidence", (12, 2));
          ("cirrus_confidence", (14, 2));
        ]
        (fields qa);
      (* The catalogue's text for bit 6, which the guides' values
         contradict (Landsat.Qa_pixel.Clear). *)
      let clear =
        List.find
          (fun (f : Jord_stac.bitfield) -> f.name = Some "clear")
          (List.hd qa.bands).bitfields
      in
      equal (option string) (Some "Cloud and dilated cloud bits are not set")
        (List.find (fun (c : Band.class_) -> c.value = 1) clear.classes)
          .description;
      equal (list (pair string (pair int int))) [] (fields (asset item "red")))

(* View angles *)

let view (v : Jord_stac.view) =
  [
    v.off_nadir; v.incidence_angle; v.azimuth; v.sun_azimuth; v.sun_elevation;
  ]

let views =
  test "MPC's Landsat and HLS state the scene's view angles" (fun () ->
      let angles = list (option float_exact) in
      equal angles
        [ Some 0.; None; None; Some 148.0085495; Some 58.45652183 ]
        (view (page "mpc_landsat_c2_l2.json").view);
      equal angles
        [ None; None; Some 226.65225969; Some 158.89867018; None ]
        (view (page "mpc_hls2_s30.json").view);
      equal angles [ None; None; None; None; None ]
        (view (page "mpc_sentinel2_l2a.json").view))

(* CDSE's Sentinel-2 L2A items (view 1.1) state each band's incidence angle
   and azimuth on its asset, and the scene's on the item. *)
let asset_views =
  test "an asset's view angles are its own over the item's" (fun () ->
      let angles = list (option float_exact) in
      let item = page "cdse_sentinel2_l2a.json" in
      equal angles
        [
          None;
          Some 10.019876372485358;
          Some 283.89173796727266;
          Some 160.32246592849;
          Some 47.7393365938335;
        ]
        (view item.view);
      equal angles
        [
          None;
          Some 9.92498944231438;
          Some 283.58495842916;
          Some 160.32246592849;
          Some 47.7393365938335;
        ]
        (view (asset item "B02_10m").view);
      equal angles (view item.view) (view (asset item "AOT_10m").view))

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
                     "bands": [{"name": "b1", "eo:common_name": "red",
                                "eo:center_wavelength": 0.665,
                                "eo:full_width_half_max": 0.038},
                               {"name": "b2", "raster:scale": 2, "nodata": 0,
                                "raster:offset": -1}]},
               "one": {"href": "one.tif", "nodata": "-inf", "unit": "m",
                       "classification:bitfields":
                         [{"offset": 0, "length": 1,
                           "classes": [{"value": 1, "name": "set"}]}]},
               "centre": {"href": "c.tif", "eo:center_wavelength": 0.665},
               "none": {"href": "none.tif"}|})
      in
      equal (list band)
        [
          measured ~name:"b1" ~common_name:"red" ~centre:0.665 ~width:0.038
            ~scale:0.5 ~nodata:Float.nan ();
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
      equal (list band)
        [ measured ~centre:0.665 () ]
        (bands (asset i "centre"));
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
        (List.map (fun (b : Jord_stac.band) -> ok b.grid) multi.bands))

(* An asset [a] of shape [shape] with the projection fields [fields]. *)
let stated ?(shape = "[1, 1]") fields =
  utm_item
    (Printf.sprintf
       {|"data": {"href": "d.tif"},
         "a": {"href": "a", "proj:shape": %s, %s}|}
       shape fields)

(* A Pseudo-Mercator system in WKT2, PROJ's, as JSON text. *)
let wkt2_3857 =
  {|"PROJCRS[\"WGS 84 / Pseudo-Mercator\",BASEGEOGCRS[\"WGS 84\",ENSEMBLE[\"World Geodetic System 1984 ensemble\",MEMBER[\"World Geodetic System 1984 (G2296)\"],ELLIPSOID[\"WGS 84\",6378137,298.257223563,LENGTHUNIT[\"metre\",1]],ENSEMBLEACCURACY[2.0]],PRIMEM[\"Greenwich\",0,ANGLEUNIT[\"degree\",0.0174532925199433]],ID[\"EPSG\",4326]],CONVERSION[\"Popular Visualisation Pseudo-Mercator\",METHOD[\"Popular Visualisation Pseudo Mercator\",ID[\"EPSG\",1024]],PARAMETER[\"Latitude of natural origin\",0,ANGLEUNIT[\"degree\",0.0174532925199433],ID[\"EPSG\",8801]],PARAMETER[\"Longitude of natural origin\",0,ANGLEUNIT[\"degree\",0.0174532925199433],ID[\"EPSG\",8802]],PARAMETER[\"False easting\",0,LENGTHUNIT[\"metre\",1],ID[\"EPSG\",8806]],PARAMETER[\"False northing\",0,LENGTHUNIT[\"metre\",1],ID[\"EPSG\",8807]]],CS[Cartesian,2],AXIS[\"easting (X)\",east,ORDER[1],LENGTHUNIT[\"metre\",1]],AXIS[\"northing (Y)\",north,ORDER[2],LENGTHUNIT[\"metre\",1]]]"|}

let wkt2 =
  test "proj:wkt2 states the system only where no code is" (fun () ->
      let mercator =
        north_up 3857 ~rows:1 ~cols:1 ~x:500000. ~y:5300000. 10.
      in
      let one fields =
        let transform = {|"proj:transform": [10, 0, 500000, 0, -10, 5300000], |} in
        (asset (read (stated (transform ^ fields))) "a").grid
      in
      (* The item states 32633; the asset's null code gives way to its WKT2. *)
      equal (option grid) (Some mercator)
        (ok (one ("\"proj:code\": null, \"proj:wkt2\": " ^ wkt2_3857)));
      (* A stated code wins, and a WKT2 beside it is not read. *)
      equal (option grid)
        (Some (north_up 32633 ~rows:1 ~cols:1 ~x:500000. ~y:5300000. 10.))
        (ok (one {|"proj:code": "EPSG:32633", "proj:wkt2": "not WKT"|}));
      (* The most specific object decides: the asset's WKT2 over the item's
         code. *)
      equal (option grid) (Some mercator)
        (ok (one ("\"proj:wkt2\": " ^ wkt2_3857)));
      match one "\"proj:code\": null, \"proj:wkt2\": \"GEOGCS[\\\"x\\\"]\"" with
      | Some (Error e) -> contains ~sub:"proj:wkt2" e
      | _ -> fail "read")

let bad_grids =
  cases ~id:snd "a grid that cannot be read is its asset's Error"
    [
      ( stated
          {|"proj:code": "EPSG:3395",
            "proj:transform": [1, 0, 0, 0, -1, 0]|},
        "3395" );
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

let orbit_states =
  test "an unknown sat:orbit_state is no state, and leaves the page readable"
    (fun () ->
      let feature id state =
        Printf.sprintf
          {|{"type": "Feature", "id": %S, "geometry": null,
             "properties": {"datetime": "2024-06-29T00:00:00Z"%s},
             "assets": {}}|}
          id
          (match state with
          | Some s -> Printf.sprintf {|, "sat:orbit_state": %s|} s
          | None -> "")
      in
      let page =
        decode
          (Printf.sprintf {|{"type": "FeatureCollection", "features": [%s]}|}
             (String.concat ", "
                [
                  feature "a" (Some {|"ascending"|});
                  feature "b" (Some {|"polar"|});
                  feature "c" (Some "7");
                  feature "d" None;
                  feature "e" (Some {|"descending"|});
                  feature "f" (Some {|"crossing"|});
                ]))
      in
      match Jord_stac.items page with
      | Error e -> fail e
      | Ok items ->
          equal
            (list (option orbit))
            [
              Some Jord_stac.Ascending;
              None;
              None;
              None;
              Some Descending;
              Some Crossing;
            ]
            (List.map (fun (i : Jord_stac.item) -> i.orbit_state) items))

let classification_2 =
  test "classification 2.0's classes have names and may mark no data"
    (fun () ->
      let i =
        read
          (item
             {|"a": {"href": "a.tif",
                     "bands": [{"name": "b1", "classification:classes":
                                 [{"value": 0, "name": "no_data",
                                   "nodata": true},
                                  {"value": 1, "name": "water",
                                   "title": "Water",
                                   "description": "Open water"}]}]}|})
      in
      equal classes
        [
          class_ ~name:"no_data" ~nodata:true 0;
          class_ ~name:"water" ~title:"Water" ~description:"Open water" 1;
        ]
        (classes_of (List.hd (bands (asset i "a")))))

let view_null =
  test "an asset's null view angle states that there is none" (fun () ->
      let i =
        read
          (item
             ~props:
               {|"datetime": "2024-06-29T00:00:00Z", "view:off_nadir": 5,
                 "view:sun_azimuth": 120|}
             {|"a": {"href": "a", "view:off_nadir": null}|})
      in
      equal
        (list (option float_exact))
        [ None; None; None; Some 120.; None ]
        (view (asset i "a").view))

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
      ( item
          ~props:
            {|"datetime": "2024-06-29T00:00:00Z", "view:sun_elevation": -91|}
          "",
        "view:sun_elevation -91 is outside -90 to 90" );
      ( item
          ~props:{|"datetime": "2024-06-29T00:00:00Z", "view:azimuth": "N"|}
          "",
        "view:azimuth is not a finite number" );
      ( item
          ~props:{|"datetime": "2024-06-29T00:00:00Z", "sat:relative_orbit": 0|}
          "",
        "sat:relative_orbit 0 is below 1" );
      ( item
          ~props:
            {|"datetime": "2024-06-29T00:00:00Z", "sat:relative_orbit": 4.5|}
          "",
        "sat:relative_orbit holds a value that is not an integer" );
      ( item
          {|"a": {"href": "a", "classification:classes": [{"value": 1}]}|},
        "neither name nor description" );
      ( item
          {|"a": {"href": "a", "classification:classes": [{"name": "x"}]}|},
        "no value" );
      ( item {|"a": {"href": "a", "classification:classes": []}|},
        "classification:classes is empty" );
      ( item
          {|"a": {"href": "a", "classification:classes":
                  [{"value": 1e20, "name": "x"}]}|},
        "outside an int's range" );
      ( item {|"a": {"href": "a", "classification:bitfields": []}|},
        "classification:bitfields is empty" );
      ( item
          {|"a": {"href": "a", "classification:bitfields":
                  [{"offset": 0, "classes": [{"value": 1, "name": "x"}]}]}|},
        "no length" );
      ( item
          {|"a": {"href": "a", "classification:bitfields":
                  [{"offset": 0, "length": 0,
                    "classes": [{"value": 0, "name": "x"}]}]}|},
        "length 0" );
      ( item
          {|"a": {"href": "a", "classification:bitfields":
                  [{"offset": 0, "length": 2,
                    "classes": [{"value": 4, "name": "x"}]}]}|},
        "outside its 2-bit field" );
      (decode {|{"type": "FeatureCollection", "features": []}|}, "Feature");
    ]
    (fun (json, sub) ->
      match Jord_stac.item json with
      | Ok _ -> fail "read"
      | Error e -> contains ~sub e)

(* STAC allows these; jord refuses them because a band has one kind and a
   code stands for one class. *)
let jord_rules =
  cases ~id:snd "items that break jord's rules for classes"
    [
      ( item
          {|"a": {"href": "a", "classification:classes":
                  [{"value": 1, "name": "x"}, {"value": 1, "name": "y"}]}|},
        "value 1 twice" );
      ( item
          {|"a": {"href": "a",
                  "classification:bitfields":
                    [{"offset": 0, "length": 1,
                      "classes": [{"value": 1, "name": "set"}]}],
                  "bands": [{"classification:classes":
                             [{"value": 1, "name": "x"}]}]}|},
        "classification:classes and classification:bitfields" );
    ]
    (fun (json, sub) ->
      match Jord_stac.item json with
      | Ok _ -> fail "read"
      | Error e -> contains ~sub e)

(* Landsat *)

(* The bands of the assets MPC names by these keys. *)
let landsat_assets =
  Landsat.
    [
      ("coastal", Coastal);
      ("blue", Blue);
      ("green", Green);
      ("red", Red);
      ("nir08", Nir);
      ("swir16", Swir16);
      ("swir22", Swir22);
      ("lwir11", Lwir);
      ("lwir", Lwir);
      ("qa", St_qa);
      ("trad", St_trad);
      ("urad", St_urad);
      ("drad", St_drad);
      ("atran", St_atran);
      ("emis", St_emis);
      ("emsd", St_emsd);
      ("cdist", St_cdist);
      ("atmos_opacity", Sr_atmos_opacity);
      ("qa_pixel", Qa_pixel);
      ("qa_radsat", Qa_radsat);
      ("qa_aerosol", Sr_qa_aerosol);
      ("cloud_qa", Sr_cloud_qa);
    ]

let landsat_items =
  [
    ("mpc_landsat_c2_l2.json", Landsat.Landsat_9);
    ("mpc_landsat_c2_l2_le07.json", Landsat.Landsat_7);
  ]

(* MPC's names of a band that Landsat names [SR_Bn] or [ST_Bn]. *)
let eo_name (s : Landsat.satellite) b n =
  match (Landsat.sensor s, b) with
  | Landsat.Oli_tirs, Landsat.Lwir -> "TIRS_B" ^ n
  | Oli_tirs, _ -> "OLI_B" ^ n
  | Etm_plus, _ -> "ETM_B" ^ n
  | Tm, _ -> "TM_B" ^ n

let landsat_bands =
  cases "MPC's Landsat raster bands are the guides' bands" ~id:fst
    landsat_items
    (fun (file, s) ->
      let item = page file in
      is_true (Landsat.satellite item.id = Ok s);
      let stated =
        List.filter_map
          (fun (k, (a : Jord_stac.asset)) ->
            match bands a with
            | [] -> None
            | [ b ] -> Some (k, b)
            | l -> failf "%s: %d bands" k (List.length l))
          item.assets
      in
      let ours = List.map (fun (k, _) -> List.assoc k landsat_assets) stated in
      equal (list string)
        (List.map (Landsat.name s) (Landsat.bands s))
        (List.map (Landsat.name s)
           (List.filter (fun b -> List.mem b ours) (Landsat.bands s)));
      equal int (List.length (Landsat.bands s)) (List.length ours);
      List.iter
        (fun (k, (theirs : Band.t)) ->
          let b = List.assoc k landsat_assets in
          let ours = Landsat.band s b in
          let msg = k in
          equal ~msg float_exact ours.scale theirs.scale;
          equal ~msg float_exact ours.offset theirs.offset;
          is_true ~msg (ours.kind = theirs.kind);
          (* MPC states no fill value for SR_CLOUD_QA, which LSDS-1618
             Table 5-1 states as 0. *)
          let nodata = if k = "cloud_qa" then None else ours.nodata in
          equal ~msg (option float_exact) nodata theirs.nodata;
          (* MPC gives bitfields and emissivities a unit; Landsat gives a
             unit only to a decoded value that has one. *)
          let unit =
            match (ours.kind, k) with
            | Band.Bitfield, _ -> Some "bit index"
            | _, ("emis" | "emsd") -> Some "emissivity coefficient"
            | _ -> ours.unit
          in
          equal ~msg (option string) unit theirs.unit;
          (* MPC names the reflective and thermal bands by sensor, as
             "OLI_B4" for the file the USGS names SR_B4, and TM and ETM+'s
             band 4 "nir08", which STAC's table names "nir". *)
          let eo =
            match (ours.common_name, ours.name) with
            | Some _, Some n ->
                let i = String.index n 'B' + 1 in
                Some (eo_name s b (String.sub n i (String.length n - i)))
            | _ -> None
          in
          equal ~msg (option string) eo theirs.name;
          let common =
            if Landsat.sensor s <> Oli_tirs && k = "nir08" then Some "nir08"
            else ours.common_name
          in
          equal ~msg (option string) common theirs.common_name)
        stated)

(* Every class of every bitfield MPC states, read back through Landsat. *)
module Landsat_bits = struct
  open Landsat

  let dn x = Nx.create Nx.uint16 [| 1 |] [| x |]
  let at m = (Nx.to_array m).(0)

  (* QA_RADSAT's bands as MPC names them, by sensor. *)
  let radsat_names = function
    | Oli_tirs ->
        [
          ("band1", Coastal);
          ("band2", Blue);
          ("band3", Green);
          ("band4", Red);
          ("band5", Nir);
          ("band6", Swir16);
          ("band7", Swir22);
        ]
    | Tm | Etm_plus ->
        [
          ("band1", Blue);
          ("band2", Green);
          ("band3", Red);
          ("band4", Nir);
          ("band5", Swir16);
          ("band6", Lwir);
          ("band7", Swir22);
        ]

  (* Whether this module reads the value [v] from [x], a DN that holds [v] in
     the field MPC names [field] of asset [asset] and nothing else; [None] for
     a field it does not read. *)
  let reads s asset field v x =
    let flag m = Some (at m = (v = 1)) in
    match (asset, field) with
    | "qa_pixel", _ when String.ends_with ~suffix:"_confidence" field ->
        let c =
          List.assoc field
            Qa_pixel.
              [
                ("cloud_confidence", Cloud_confidence);
                ("cloud_shadow_confidence", Cloud_shadow_confidence);
                ("snow_confidence", Snow_ice_confidence);
                ("cirrus_confidence", Cirrus_confidence);
              ]
        in
        let l =
          List.assoc v Qa_pixel.[ (0, Unset); (1, Low); (2, Medium); (3, High) ]
        in
        Some (at (Qa_pixel.confidence c [ l ] (dn x)))
    | "qa_pixel", _ ->
        let f =
          List.assoc field
            Qa_pixel.
              [
                ("fill", Fill);
                ("dilated_cloud", Dilated_cloud);
                ("cirrus", Cirrus);
                ("cloud", Cloud);
                ("cloud_shadow", Cloud_shadow);
                ("snow", Snow);
                ("clear", Clear);
                ("water", Water);
              ]
        in
        flag (Qa_pixel.any [ f ] (dn x))
    | "qa_aerosol", "level" ->
        let l =
          List.assoc v
            Sr_qa_aerosol.[ (0, Climatology); (1, Low); (2, Medium); (3, High) ]
        in
        Some (at (Sr_qa_aerosol.level [ l ] (dn x)))
    | "qa_aerosol", _ ->
        let f =
          List.assoc field
            Sr_qa_aerosol.
              [
                ("fill", Fill);
                ("retrieval", Valid_retrieval);
                ("water", Water);
                ("interpolated", Interpolated);
              ]
        in
        flag (Sr_qa_aerosol.any [ f ] (dn x))
    | "cloud_qa", _ ->
        let f =
          List.assoc field
            Sr_cloud_qa.
              [
                ("ddv", Ddv);
                ("cloud", Cloud);
                ("cloud_shadow", Cloud_shadow);
                ("cloud_adjacent", Adjacent_cloud);
                ("snow", Snow);
                ("water", Water);
              ]
        in
        flag (Sr_cloud_qa.any [ f ] (dn x))
    | "qa_radsat", "occlusion" -> flag (Qa_radsat.occluded (dn x))
    | "qa_radsat", "dropped" -> flag (Qa_radsat.dropped (dn x))
    (* ETM+'s temperature is saturated where both gains are. *)
    | "qa_radsat", "band6L" ->
        flag (Qa_radsat.saturated s Lwir (dn (x lor 256)))
    | "qa_radsat", "band6H" ->
        flag (Qa_radsat.saturated s Lwir (dn (x lor 32)))
    | "qa_radsat", _ -> (
        match List.assoc_opt field (radsat_names (Landsat.sensor s)) with
        | Some b -> flag (Qa_radsat.saturated s b (dn x))
        | None -> None)
    | _ -> failf "%s: unexpected bitfields" asset

  (* The fields of [asset] that this module does not read, after checking
     that it reads every other field's every class as MPC states it. *)
  let unread s (asset, (a : Jord_stac.asset)) =
    List.concat_map
      (fun (b : Jord_stac.band) ->
        List.concat_map
          (fun (f : Jord_stac.bitfield) ->
            let field = Option.get f.name in
            List.filter_map
              (fun (c : Band.class_) ->
                match reads s asset field c.value (c.value lsl f.offset) with
                | Some ok ->
                    is_true
                      ~msg:(Printf.sprintf "%s %s %d" asset field c.value)
                      ok;
                    None
                | None -> Some (asset ^ " " ^ field))
              f.classes)
          b.bitfields)
      a.bands

  let test =
    cases "MPC's Landsat bitfields read as the guides' bits" ~id:fst
      landsat_items
      (fun (file, s) ->
        (* OLI's band 9, cirrus, has no surface reflectance. *)
        let want =
          if Landsat.sensor s = Oli_tirs then
            [ "qa_radsat band9"; "qa_radsat band9" ]
          else []
        in
        equal (list string) want
          (List.concat_map (unread s) (page file).assets))
end

let () =
  exit
    (run "jord.stac"
       [
         group "proj:wkt2" [ modis; wkt2 ];
         group "classification" [ worldcover; classification_2; bitfields ];
         group "mpc"
           [
             sentinel2;
             sentinel1;
             landsat;
             landsat_bands;
             Landsat_bits.test;
             views;
             asset_views;
             hls_s30;
             landsat9_tirs;
             modis_grids;
             modis_layers;
             modis_wavelengths;
           ];
         group "items"
           [
             stac11;
             projection;
             band_grids;
             bad_grids;
             orbit_states;
             view_null;
             refusals;
             jord_rules;
           ];
       ])
