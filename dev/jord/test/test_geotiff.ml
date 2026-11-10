(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord
open Geotiff

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
      ("a user-defined system", (with_key 3072 32767, "user-defined"));
      ("a system jord lacks", (with_key 3072 3857, "3857"));
      ("a geocentric model", (with_key 1024 3, "model type 3"));
      ("feet", (with_key 3076 9002, "unit 9002"));
      ("another angular unit", (with_key 2054 9105, "unit 9105"));
      ( "another base system",
        ( replace 34735
            (Short
               [|
                 1; 1; 1; 3; 1024; 0; 1; 1; 2048; 0; 1; 4258; 3072; 0; 1; 32633;
               |])
            base,
          "4258" ) );
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
          if z = 0 then Crs.lonlat else Crs.utm ~zone:z ~north:(z mod 2 = 0))
        (int_range 0 60)
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

let () =
  exit
    (run "jord.geotiff"
       [
         group "grid"
           [
             landmasks;
             raster_types;
             transformation;
             geographic;
             integer_types;
             refusals;
           ];
         group "nodata" [ nodata_texts; nan_nodata ];
         group "tags" [ round_trip; nodata_round_trip; order ];
       ])
