(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord
open Nx_io.Tiff
open Jord_geotiff

let grid_t =
  Testable.make
    ~pp:(fun ppf (g : Grid.t) ->
      let (a, b), (c, d), (e, f) = (g.origin, g.row_step, g.col_step) in
      Format.fprintf ppf
        "%a %dx%d origin (%.17g, %.17g) row (%g, %g) col (%g, %g)" Crs.pp g.crs
        g.rows g.cols a b c d e f)
    ~equal:Grid.equal

(* The tags of GeoTessera's landmask grid_0.15_52.25.tiff, its geographic tags
   as the file holds them. *)
let landmask_keys code =
  [|
    1;
    1;
    0;
    7;
    1024;
    0;
    1;
    1;
    1025;
    0;
    1;
    1;
    1026;
    34737;
    22;
    0;
    2049;
    34737;
    7;
    22;
    2054;
    0;
    1;
    9102;
    3072;
    0;
    1;
    code;
    3076;
    0;
    1;
    9001;
  |]

let landmask ~cols ~rows ~x ~y code =
  [
    (256, Short [| cols |]);
    (257, Short [| rows |]);
    (33550, Double [| 10.; 10.; 0. |]);
    (33922, Double [| 0.; 0.; 0.; x; y; 0. |]);
    (34735, Short (landmask_keys code));
    (34737, Ascii "WGS 84 / UTM zone 31N|WGS 84|");
    (42113, Ascii "0");
  ]

let north_up crs ~rows ~cols ~x ~y =
  Grid.v crs ~rows ~cols ~origin:(x, y) ~row_step:(0., -10.) ~col_step:(10., 0.)

let landmasks =
  cases "GeoTessera's landmasks state their grids"
    ~id:(fun (n, _, _) -> n)
    [
      ( "0.15, 52.25",
        landmask ~cols:726 ~rows:1139 ~x:301822.86256412434 ~y:5798366.72759726
          32631,
        north_up
          (Crs.utm ~zone:31 ~north:true)
          ~rows:1139 ~cols:726 ~x:301822.86256412434 ~y:5798366.72759726 );
      ( "15.05, 78.25",
        landmask ~cols:228 ~rows:1116 ~x:500000.00000000035 ~y:8691852.036646556
          32633,
        north_up
          (Crs.utm ~zone:33 ~north:true)
          ~rows:1116 ~cols:228 ~x:500000.00000000035 ~y:8691852.036646556 );
      ( "-58.45, -34.55",
        landmask ~cols:934 ~rows:1122 ~x:362289.4465982177 ~y:6181513.229923251
          32721,
        north_up
          (Crs.utm ~zone:21 ~north:false)
          ~rows:1122 ~cols:934 ~x:362289.4465982177 ~y:6181513.229923251 );
    ]
    (fun (_, tags, g) ->
      equal (result grid_t string) (Ok g) (grid tags);
      equal (result (option float_exact) string) (Ok (Some 0.)) (nodata tags))

(* The tags of Luke's MS-NFI 2023 site-type COG (kasvupaikka_vmi1x_1923.tif,
   read from its first IFD): the landmasks' GeoKeys with ProjectedCRSGeoKey
   3067, and no GeodeticCRSGeoKey. *)
let ms_nfi =
  test "MS-NFI's tags state a TM35FIN grid" (fun () ->
      let tags =
        [
          (256, Short [| 42240 |]);
          (257, Long [| 73472 |]);
          (33550, Double [| 16.; 16.; 0. |]);
          (33922, Double [| 0.; 0.; 0.; 57632.; 7778304.; 0. |]);
          (34735, Short (landmask_keys 3067));
          (34737, Ascii "ETRS89 / TM35FIN(E,N)|ETRS89|");
          (42113, Ascii "32767");
        ]
      in
      let tm35fin = Result.get_ok (Crs.of_epsg 3067) in
      let g =
        Grid.v tm35fin ~rows:73472 ~cols:42240 ~origin:(57632., 7778304.)
          ~row_step:(0., -16.) ~col_step:(16., 0.)
      in
      equal (result grid_t string) (Ok g) (grid tags);
      equal
        (result (option float_exact) string)
        (Ok (Some 32767.))
        (nodata tags))

(* The tags of MPC's MOD13Q1.A2023241.h18v04 NDVI COG (250m_16_days_NDVI.tif
   of its STAC item, read from its first IFD): user-defined keys for
   MODLAND's sinusoid, its doubles in GeoDoubleParams (34736). *)
let modis_keys =
  [|
    1; 1; 0; 18;
    1024; 0; 1; 1;
    1025; 0; 1; 1;
    1026; 34737; 8; 0;
    2048; 0; 1; 32767;
    2049; 34737; 153; 8;
    2050; 0; 1; 32767;
    2054; 0; 1; 9102;
    2056; 0; 1; 32767;
    2057; 34736; 1; 3;
    2058; 34736; 1; 4;
    2061; 34736; 1; 5;
    3072; 0; 1; 32767;
    3074; 0; 1; 32767;
    3075; 0; 1; 24;
    3076; 0; 1; 9001;
    3082; 34736; 1; 1;
    3083; 34736; 1; 2;
    3088; 34736; 1; 0;
  |]

let modis_tags ?(keys = modis_keys)
    ?(doubles = [| 0.; 0.; 0.; 6371007.181; 6371007.181; 0. |]) () =
  [
    (256, Short [| 4800 |]);
    (257, Short [| 4800 |]);
    (33550, Double [| 231.65635826395834; 231.65635826375006; 0. |]);
    (33922, Double [| 0.; 0.; 0.; 0.; 5559752.598333; 0. |]);
    (34735, Short keys);
    (34736, Double doubles);
    ( 34737,
      Ascii
        "unnamed|GCS Name = Unknown datum based upon the custom \
         spheroid|Datum = Not specified (based on custom spheroid)|Ellipsoid \
         = Custom spheroid|Primem = Greenwich||" );
    (42113, Ascii "-3000");
  ]

let with_value k v keys =
  let keys = Array.copy keys in
  for i = 1 to keys.(3) do
    if keys.(4 * i) = k then keys.((4 * i) + 3) <- v
  done;
  keys

let modis =
  group "MODIS"
    [
      test "MODIS's user-defined keys state its sinusoid" (fun () ->
          let g =
            Grid.v Crs.modis ~rows:4800 ~cols:4800 ~origin:(0., 5559752.598333)
              ~row_step:(0., -231.65635826375006)
              ~col_step:(231.65635826395834, 0.)
          in
          equal (result grid_t string) (Ok g) (grid (modis_tags ())));
      cases "other user-defined systems are refused" ~id:fst
        (let doubles a b m = [| 0.; 0.; 0.; a; b; m |] in
         let keys k v = with_value k v modis_keys in
         [
           ( "another sphere",
             ( modis_tags ~doubles:(doubles 6370997. 6370997. 0.) (),
               "MODIS's sphere" ) );
           ( "an ellipsoid",
             ( modis_tags ~doubles:(doubles 6371007.181 6356752. 0.) (),
               "MODIS's sphere" ) );
           ("another method", (modis_tags ~keys:(keys 3075 1) (), "method 1"));
           ("a coded datum", (modis_tags ~keys:(keys 2050 6326) (), "2050"));
           ( "another prime meridian",
             ( modis_tags ~doubles:(doubles 6371007.181 6371007.181 10.) (),
               "Greenwich" ) );
         ])
        (fun (_, (tags, sub)) ->
          match grid tags with
          | Ok g -> failf "read as %a" Crs.pp g.crs
          | Error e -> contains ~sub e);
      test "tags refuses MODIS's system, which has no code" (fun () ->
          let g = Result.get_ok (grid (modis_tags ())) in
          raises_match (Exn.invalid_arg ~sub:"no EPSG code") (fun () ->
              tags g));
    ]

let replace k v tags =
  List.map (fun (t, x) -> if t = k then (t, v) else (t, x)) tags

let base = landmask ~cols:4 ~rows:3 ~x:500000. ~y:5000000. 32633
let utm33 = Crs.utm ~zone:33 ~north:true

let raster_types =
  test "PixelIsPoint ties the first cell's centre" (fun () ->
      let keys = landmask_keys 32633 in
      keys.(11) <- 2;
      let g = Result.get_ok (grid (replace 34735 (Short keys) base)) in
      equal (pair float_exact float_exact) (499995., 5000005.) g.origin)

let transformation =
  test "ModelTransformation may rotate the grid" (fun () ->
      let m =
        [|
          6.; 8.; 0.; 1000.; 8.; -6.; 0.; 2000.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 1.;
        |]
      in
      let tags =
        List.filter (fun (t, _) -> t <> 33550 && t <> 33922) base
        @ [ (34264, Double m) ]
      in
      let g = Result.get_ok (grid tags) in
      equal (pair float_exact float_exact) (1000., 2000.) g.origin;
      equal (pair float_exact float_exact) (8., -6.) g.row_step;
      equal (pair float_exact float_exact) (6., 8.) g.col_step)

let geographic =
  test "a geographic model is CRS84" (fun () ->
      let keys =
        [| 1; 1; 1; 3; 1024; 0; 1; 2; 1025; 0; 1; 1; 2048; 0; 1; 4326 |]
      in
      let g = Result.get_ok (grid (replace 34735 (Short keys) base)) in
      equal bool true (Crs.equal Crs.lonlat g.crs))

let base_system =
  cases "a projected system's GeodeticCRSGeoKey is not read"
    ~id:(fun (p, _) -> string_of_int p)
    [
      (3067, "ETRS89-FIN [EUREF-FIN] / TM35FIN(E,N)");
      (32633, "WGS 84 / UTM zone 33N");
    ]
    (fun (projected, name) ->
      let keys =
        [| 1; 1; 1; 3; 1024; 0; 1; 1; 2048; 0; 1; 4258; 3072; 0; 1; projected |]
      in
      match grid (replace 34735 (Short keys) base) with
      | Ok g -> equal string name (Format.asprintf "%a" Crs.pp g.crs)
      | Error e -> failf "refused: %s" e)

let integer_types =
  test "the size may be of any unsigned integer type" (fun () ->
      let tags = replace 256 (Long [| 4 |]) (replace 257 (Byte [| 3 |]) base) in
      equal (result grid_t string) (grid base) (grid tags))

let with_key k v =
  let keys = Array.to_list (landmask_keys 32633) in
  let rec go = function
    | key :: loc :: n :: _ :: rest when key = k -> key :: loc :: n :: v :: rest
    | a :: b :: c :: d :: rest -> a :: b :: c :: d :: go rest
    | l -> l
  in
  let header, entries =
    (List.filteri (fun i _ -> i < 4) keys, List.filteri (fun i _ -> i >= 4) keys)
  in
  replace 34735 (Short (Array.of_list (header @ go entries))) base

let refusals =
  cases "refusals name the tag or key" ~id:fst
    [
      ( "no map",
        ( List.filter (fun (t, _) -> t <> 33922) base,
          "ModelTiepoint (33922) or ModelTransformation" ) );
      ( "ground control points",
        (replace 33922 (Double (Array.make 12 0.)) base, "not one tiepoint") );
      ( "a scale and a transformation",
        (base @ [ (34264, Double (Array.make 16 0.)) ], "34264") );
      ( "a tag twice",
        (base @ [ (256, Short [| 4 |]) ], "ImageWidth (256) is given twice") );
      ( "a missing size",
        ( List.filter (fun (t, _) -> t <> 257) base,
          "ImageLength (257) is missing" ) );
      ("a float size", (replace 256 (Double [| 4. |]) base, "256"));
      ("a LONG8 size", (replace 256 (Long8 [| 4L |]) base, "256"));
      ("a user-defined system", (with_key 3072 32767, "user-defined"));
      ("a system jord lacks", (with_key 3072 3395, "3395"));
      ("a geocentric model", (with_key 1024 3, "model type 3"));
      ("a geographic code as projected", (with_key 3072 4269, "NAD83 is not"));
      ("feet", (with_key 3076 9002, "unit 9002"));
      ("another angular unit", (with_key 2054 9105, "unit 9105"));
      ("a reserved raster type", (with_key 1025 3, "raster type 3"));
      ( "version 2",
        ( replace 34735 (Short (Array.append [| 2; 1; 0; 0 |] [||])) base,
          "version" ) );
      ( "a zero scale",
        (replace 33550 (Double [| 10.; 0.; 0. |]) base, "invertible") );
    ]
    (fun (_, (tags, sub)) ->
      match grid tags with Ok _ -> fail "read" | Error e -> contains ~sub e)

let nodata_texts =
  cases "GDAL_NODATA reads as strtod's numbers" ~id:fst
    [
      ("0", Some 0.);
      ("-9999", Some (-9999.));
      ("1e-3", Some 1e-3);
      ("+2.5E+2", Some 250.);
      (".5", Some 0.5);
      ("7.", Some 7.);
      (" 3.25 ", Some 3.25);
      ("-inf", Some Float.neg_infinity);
      ("Infinity", Some Float.infinity);
      ("abc", None);
      ("1,5", None);
      ("0x10", None);
      ("", None);
      ("--1", None);
    ]
    (fun (text, want) ->
      match (nodata [ (42113, Ascii text) ], want) with
      | Ok (Some x), Some w -> equal float_exact w x
      | Error _, None -> ()
      | Ok x, _ ->
          failf "read %s" (Option.fold ~none:"none" ~some:string_of_float x)
      | Error e, Some _ -> fail e)

let nan_nodata =
  test "nan reads as NaN, and no tag as none" (fun () ->
      equal bool true
        (Float.is_nan
           (Option.get (Result.get_ok (nodata [ (42113, Ascii "NaN") ]))));
      equal (result (option float_exact) string) (Ok None) (nodata []))

(* Writing *)

let grids =
  Gen.(
    let* crs =
      map
        (fun z ->
          if z = 0 then Crs.lonlat
          else if z = 61 then Result.get_ok (Crs.of_epsg 3067)
          else Crs.utm ~zone:z ~north:(z mod 2 = 0))
        (int_range 0 61)
    in
    let+ rows = int_range 0 2000
    and+ cols = int_range 0 2000
    and+ ox = float_range (-1e6) 1e7
    and+ oy = float_range (-1e6) 1e7
    and+ d = float_range 0.001 100.
    and+ rotated = bool
    and+ t = float_range (-1.) 1. in
    let row_step, col_step =
      if rotated then ((t *. d, -.d), (d, t *. d)) else ((0., -.d), (d, 0.))
    in
    Grid.v crs ~rows ~cols ~origin:(ox, oy) ~row_step ~col_step)

let round_trip =
  prop "grid reads back the grid tags states" grids (fun g ->
      cover "rotated" (fst g.row_step <> 0.);
      let size = [ (256, Long [| g.cols |]); (257, Long [| g.rows |]) ] in
      equal (result grid_t string) (Ok g) (grid (size @ tags g)))

let floats =
  Gen.(
    frequency
      [
        (8, float);
        ( 1,
          of_list
            [
              0.;
              -0.;
              Float.nan;
              Float.infinity;
              Float.neg_infinity;
              5e-324;
              -9999.;
            ] );
      ])

let nodata_round_trip =
  prop "nodata reads back the text tags writes" floats (fun x ->
      let g =
        Grid.v utm33 ~rows:1 ~cols:1 ~origin:(0., 0.) ~row_step:(0., -1.)
          ~col_step:(1., 0.)
      in
      match nodata (tags ~nodata:x g) with
      | Ok (Some y) when Float.is_nan x -> equal bool true (Float.is_nan y)
      | Ok (Some y) ->
          equal int64 (Int64.bits_of_float x) (Int64.bits_of_float y)
      | _ -> fail "no nodata")

let order =
  test "tags come in increasing order, PixelIsArea" (fun () ->
      let g =
        Grid.v utm33 ~rows:1 ~cols:1 ~origin:(0., 0.) ~row_step:(0., -1.)
          ~col_step:(1., 0.)
      in
      let t = tags ~nodata:0. g in
      equal (list int) [ 33550; 33922; 34735; 42113 ] (List.map fst t);
      match List.assoc 34735 t with
      | Short dir -> equal int 1 dir.(11)
      | _ -> fail "GeoKeyDirectory")

let uncoded =
  test "tags refuses a system without an EPSG code" (fun () ->
      let crs =
        {
          Crs.datum = Crs.Wgs84;
          projection =
            Some
              (Crs.transverse_mercator ~lat0:0. ~lon0:15. ~k0:0.9999
                 ~false_easting:500000. ~false_northing:0.);
        }
      in
      let g =
        Grid.v crs ~rows:1 ~cols:1 ~origin:(0., 0.) ~row_step:(0., -1.)
          ~col_step:(1., 0.)
      in
      raises_match (Exn.invalid_arg ~sub:"no EPSG code") (fun () -> tags g))

(* Through a TIFF file *)

let file =
  test "a saved TIFF's tags read back as its grid and nodata" (fun () ->
      let g =
        Grid.v utm33 ~rows:3 ~cols:5
          ~origin:(596785.7115403633, 5328362.815087351)
          ~row_step:(0., -10.) ~col_step:(10., 0.)
      in
      let path = Filename.concat (temp_dir ()) "grid.tif" in
      Nx_io.save_tiff ~tags:(tags ~nodata:0. g) path
        (Nx.zeros Nx.uint8 [| 3; 5; 1 |]);
      let ifd = List.hd (Nx_io.load_tiff path) in
      equal (result grid_t string) (Ok g) (grid (Nx_io.Tiff.tags ifd));
      equal
        (result (option float_exact) string)
        (Ok (Some 0.))
        (nodata (Nx_io.Tiff.tags ifd)))

(* Overviews *)

(* cog_overviews.tif, a COG of 51 rows and 75 columns of 10 m in UTM 33N
   with GDAL's overviews of 25 x 37, 12 x 18 and 6 x 9 (its dune file names
   the command). GDAL 3.12.2 gives the first one's column step as
   20.270270270270274, 10 times 75/37 rounded, one ulp above 750/37
   rounded, which Requirement 6's formula gives. *)
let cog =
  test "a COG's overviews share its origin, at steps scaled by their sizes"
    (fun () ->
      let ifds =
        List.map Nx_io.Tiff.tags (Nx_io.load_tiff "cog_overviews.tif")
      in
      let full = Result.get_ok (grid (List.hd ifds)) in
      equal grid_t
        (Grid.v utm33 ~rows:51 ~cols:75 ~origin:(500000., 5300000.)
           ~row_step:(0., -10.) ~col_step:(10., 0.))
        full;
      let level rows cols =
        Grid.v utm33 ~rows ~cols ~origin:(500000., 5300000.)
          ~row_step:(0., -510. /. Float.of_int rows)
          ~col_step:(750. /. Float.of_int cols, 0.)
      in
      equal
        (list (result grid_t string))
        [ Ok (level 25 37); Ok (level 12 18); Ok (level 6 9) ]
        (List.map (overview full) (List.tl ifds));
      let refused sub tags =
        match overview full tags with
        | Ok _ -> fail "read"
        | Error e -> contains ~sub e
      in
      refused "NewSubfileType (254) does not mark" (List.hd ifds);
      let o = List.nth ifds 1 in
      refused "NewSubfileType (254) does not mark"
        ((254, Long [| 4 |]) :: List.remove_assoc 254 o);
      refused "ModelPixelScale (33550): an overview states no georeference"
        ((33550, Double [| 20.; 20.; 0. |]) :: o);
      refused "25 x 76 is no overview of 51 x 75"
        ((256, Short [| 76 |]) :: List.remove_assoc 256 o);
      let sized rows cols =
        (256, Short [| cols |]) :: (257, Short [| rows |])
        :: List.remove_assoc 256 (List.remove_assoc 257 o)
      in
      refused "51 x 75 is no overview of 51 x 75" (sized 51 75);
      (* One side may be the image's, as GDAL keeps a side of 1. It makes
         overviews of 1 x 150 to 1 x 9 of a 1 x 300 image. *)
      equal (result grid_t string)
        (Ok
           (Grid.v utm33 ~rows:51 ~cols:37 ~origin:(500000., 5300000.)
              ~row_step:(0., -10.) ~col_step:(750. /. 37., 0.)))
        (overview full (sized 51 37)))

(* Decode then sample *)

(* An image Nx_io.load_tiff reads has a channel axis, [h; w; 1], and
   Band.decode's mask has the DNs' shape; the samplers take that mask as
   the image's cells' mask. *)
let decode_then_sample =
  test "a decoded one-channel image's mask serves the samplers" (fun () ->
      let ifd = List.hd (Nx_io.load_tiff "cog_overviews.tif") in
      let src = Result.get_ok (grid (Nx_io.Tiff.tags ifd)) in
      let dn = Nx.unpack Nx.uint8 (Nx_io.Tiff.samples ifd) in
      equal (array int) [| 51; 75; 1 |] (Nx.shape dn);
      let band =
        {
          Band.name = None;
          common_name = None;
          center_wavelength = None;
          full_width_half_max = None;
          unit = None;
          kind = Band.Measurement;
          scale = 1.;
          offset = 0.;
          nodata = Some 0.;
          saturated = None;
        }
      in
      let v, valid = Band.decode band Nx.float64 dn in
      let dst =
        Grid.v utm33 ~rows:50 ~cols:74 ~origin:(500005., 5299995.)
          ~row_step:(0., -10.) ~col_step:(10., 0.)
      in
      let p = Warp.pixel src (Warp.centres ~at:Nx.Host.on src.crs dst) in
      let flat = Nx.reshape [| 51; 75 |] valid in
      List.iter
        (fun (name, rule) ->
          let a, ok = rule ~valid p v and b, ok' = rule ~valid:flat p v in
          equal ~msg:name (array float_exact) (Nx.to_array b) (Nx.to_array a);
          equal ~msg:name (array bool) (Nx.to_array ok') (Nx.to_array ok))
        [
          ("nearest", fun ~valid p v -> Warp.nearest ~valid src p v);
          ("bilinear", fun ~valid p v -> Warp.bilinear ~valid src p v);
        ];
      raises_match (Exn.invalid_arg ~sub:"mask") (fun () ->
          Warp.bilinear ~valid:(Nx.reshape [| 75; 51 |] flat) src p v);
      (* A mask of a value per element of several channels is refused. *)
      let v2 = Nx.concatenate ~axis:2 [ v; v ] in
      raises_match (Exn.invalid_arg ~sub:"mask") (fun () ->
          Warp.nearest
            ~valid:(Nx.concatenate ~axis:2 [ valid; valid ])
            src p v2))

(* An area rule on the block Warp.area_window gives, with ~row ~col, gives
   the bits it gives on the whole image of a projected system, inside the
   image and across its top left corner, and on a block a column short it
   raises. *)
let area_on_block =
  test "an area rule on its block gives the whole image's bits" (fun () ->
      let ifd = List.hd (Nx_io.load_tiff "cog_overviews.tif") in
      let src = Result.get_ok (grid (Nx_io.Tiff.tags ifd)) in
      let x =
        Nx.cast Nx.float64 (Nx.unpack Nx.uint8 (Nx_io.Tiff.samples ifd))
      in
      let valid = Nx.greater_s x 60. in
      let classes = Nx.cast Nx.int32 (Nx.div_s x 64.) in
      let zone34 = Crs.utm ~zone:34 ~north:true in
      let corners (dx, dy) =
        let x0, y0 = src.origin in
        let o = Nx.create Nx.float64 [| 2 |] [| x0; y0 |] in
        let o = Crs.transform ~src:src.crs ~dst:zone34 o in
        let at i = Float.round (Nx.item [ i ] o) in
        let ox = at 0 +. dx and oy = at 1 +. dy in
        let dst =
          Grid.v zone34 ~rows:3 ~cols:4 ~origin:(ox, oy) ~row_step:(0., -30.)
            ~col_step:(30., 0.)
        in
        Warp.pixel src (Warp.corners ~at:Nx.Host.on src.crs dst)
      in
      let on rule q (row, col, rows, cols) x =
        let cut a = Nx.slice [ R (row, row + rows); R (col, col + cols) ] a in
        rule ~k:6 ?row:(Some row) ?col:(Some col) ?valid:(Some (cut valid)) src
          q (cut x)
      in
      let bits a = Array.map Int64.bits_of_float (Nx.to_array a) in
      let same msg q b =
        let v, f = Warp.average ~k:6 ~valid src q x
        and v', f' = on Warp.average q b x in
        equal ~msg:(msg ^ ": values") (array int64) (bits v) (bits v');
        equal ~msg:(msg ^ ": fractions") (array int64) (bits f) (bits f');
        let m, _ = Warp.mode ~k:6 ~valid src q classes
        and m', _ = on Warp.mode q b classes in
        equal ~msg:(msg ^ ": classes") (array int32) (Nx.to_array m)
          (Nx.to_array m')
      in
      let q = corners (95., -130.) in
      let ((row, col, rows, cols) as b) = Warp.area_window src q in
      is_true ~msg:"inside the image" (row > 0 && col > 0);
      same "inside" q b;
      let q' = corners (-40., 40.) in
      let ((row', col', _, _) as b') = Warp.area_window src q' in
      is_true ~msg:"clipped" (row' = 0 && col' = 0);
      same "clipped" q' b';
      raises_match (Exn.invalid_arg ~sub:"outside the block") (fun () ->
          Nx.to_array (fst (on Warp.average q (row, col, rows, cols - 1) x))))

let () =
  exit
    (run "jord.geotiff"
       [
         group "grid"
           [
             landmasks;
             ms_nfi;
             raster_types;
             transformation;
             geographic;
             base_system;
             integer_types;
             refusals;
           ];
         group "nodata" [ nodata_texts; nan_nodata ];
         group "tags" [ round_trip; nodata_round_trip; order; uncoded; file ];
         modis;
         group "overviews" [ cog ];
         group "decode" [ decode_then_sample ];
         group "windows" [ area_on_block ];
       ])
