(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord

let points l =
  Nx.create Nx.float64
    [| List.length l; 2 |]
    (Array.of_list (List.concat_map (fun (a, b) -> [ a; b ]) l))

let pairs t =
  let a = Nx.to_array t in
  List.init (Array.length a / 2) (fun k -> (a.(2 * k), a.((2 * k) + 1)))

let utm zone north = Crs.utm ~zone ~north
let fwd c l = pairs (Crs.transform ~src:Crs.lonlat ~dst:c (points l))
let inv c l = pairs (Crs.transform ~src:c ~dst:Crs.lonlat (points l))
let crs = Testable.make ~pp:Crs.pp ~equal:Crs.equal
let wgs84_radius = 6378137.
let deg = Float.pi /. 180.

(* Karney's test set *)

(* The rows of tm_coords.dat: latitude, longitude, easting and northing about
   central meridian 0, without false easting. *)
let karney =
  let ic = open_in "tm_coords.dat" in
  let rec go acc =
    match input_line ic with
    | exception End_of_file ->
        close_in ic;
        List.rev acc
    | l when String.length l > 0 && l.[0] = '#' -> go acc
    | l ->
        go
          (Scanf.sscanf l " %f %f %f %f" (fun la lo x y -> (la, lo, x, y))
          :: acc)
  in
  go []

(* Zone 31's central meridian is 3°E, and its easting carries 500000 m. *)
let cm31 = 3.
let false_easting = 500000.

(* Karney states 5 nm within 3900 km of the central meridian; the domain reaches
   4450 km, where the module states 6 nm. x / k0 bounds the distance from
   above. *)
let bound x = if x /. 0.9996 < 3.9e6 then 5e-9 else 6e-9
let row_id (la, lo, _, _) = Printf.sprintf "%.12g, %.12g" la lo

let forward_karney =
  test "forward is within Karney's values" (fun () ->
      let got =
        fwd (utm 31 true)
          (List.map (fun (la, lo, _, _) -> (lo +. cm31, la)) karney)
      in
      List.iter2
        (fun ((_, _, x, y) as r) (gx, gy) ->
          (* The subtraction of the false easting is exact. *)
          let err = Float.hypot (gx -. false_easting -. x) (gy -. y) in
          less ~msg:(row_id r) float_exact ~than:(bound x) err)
        karney got)

let inverse_karney =
  test "inverse is within Karney's values" (fun () ->
      let got =
        inv (utm 31 true)
          (List.map (fun (_, _, x, y) -> (x +. false_easting, y)) karney)
      in
      List.iter2
        (fun ((la, lo, x, _) as r) (glon, glat) ->
          let dlat = (glat -. la) *. deg *. wgs84_radius in
          let dlon =
            (glon -. cm31 -. lo) *. deg *. wgs84_radius *. Float.cos (la *. deg)
          in
          less ~msg:(row_id r) float_exact ~than:(bound x)
            (Float.hypot dlat dlon))
        karney got)

(* Round trips and symmetry *)

let zone = Gen.int_range 1 60

(* A point within [w] degrees of the zone's central meridian. *)
let near_meridian w =
  Gen.(
    let+ z = zone
    and+ north = bool
    and+ off = float_range (-.w) w
    and+ lat = float_range 0. 89.9 in
    let lon = float_of_int ((6 * z) - 183) +. off in
    let lon =
      if lon > 180. then lon -. 360.
      else if lon < -180. then lon +. 360.
      else lon
    in
    (z, north, lon, if north then lat else -.lat))

let metres (lon, lat) (lon', lat') =
  let dlon = Float.rem (lon' -. lon +. 540.) 360. -. 180. in
  Float.hypot
    ((lat' -. lat) *. deg *. wgs84_radius)
    (dlon *. deg *. wgs84_radius *. Float.cos (lat *. deg))

let round_trip =
  prop "inverse undoes forward within both bounds" (near_meridian 40.)
    (fun (z, north, lon, lat) ->
      cover "across the antimeridian" (z = 1 || z = 60);
      let c = utm z north in
      let back = inv c (fwd c [ (lon, lat) ]) in
      less float_exact ~than:12e-9 (metres (lon, lat) (List.hd back)))

let between_zones =
  prop "a transform between zones passes through longitude and latitude"
    (Gen.pair (near_meridian 3.) (Gen.int_range 1 6))
    (fun ((z, north, lon, lat), d) ->
      (* A zone up to 6 zones away, within 39° of the point. *)
      let z' =
        ((((z - 1 + if d <= 3 then d else 3 - d) mod 60) + 60) mod 60) + 1
      in
      let a = utm z north and b = utm z' north in
      let xy = Crs.transform ~src:Crs.lonlat ~dst:a (points [ (lon, lat) ]) in
      let via =
        Crs.transform ~src:Crs.lonlat ~dst:b
          (Crs.transform ~src:a ~dst:Crs.lonlat xy)
      in
      equal
        (list (pair float_exact float_exact))
        (pairs via)
        (pairs (Crs.transform ~src:a ~dst:b xy)))

let hemispheres =
  prop "a southern point's northing is 10000000 m less the mirrored point's"
    (near_meridian 40.) (fun (z, _, lon, lat) ->
      let lat = Float.abs lat in
      let _, n = List.hd (fwd (utm z true) [ (lon, lat) ]) in
      let _, s = List.hd (fwd (utm z false) [ (lon, -.lat) ]) in
      (* Each northing rounds once at its false northing's magnitude. *)
      less float_exact ~than:4e-9 (Float.abs (s -. (1e7 -. n))))

let identity =
  test "a transform to its own system is its points" (fun () ->
      let p = points [ (500000., 1e7); (Float.nan, 3.) ] in
      let q = Crs.transform ~src:(utm 33 false) ~dst:(utm 33 false) p in
      equal (array float_exact) (Nx.to_array p) (Nx.to_array q))

let shapes =
  test "points keep their batch shape" (fun () ->
      let p = Nx.create Nx.float64 [| 2; 1; 2 |] [| 15.; 48.; 16.; 47. |] in
      equal (array int) [| 2; 1; 2 |]
        (Nx.shape (Crs.transform ~src:Crs.lonlat ~dst:(utm 33 true) p));
      let e = Nx.zeros Nx.float64 [| 0; 2 |] in
      equal (array int) [| 0; 2 |]
        (Nx.shape (Crs.transform ~src:Crs.lonlat ~dst:(utm 33 true) e));
      raises_match (Exn.invalid_arg ~sub:"last axis") (fun () ->
          Crs.transform ~src:Crs.lonlat ~dst:(utm 33 true)
            (Nx.zeros Nx.float64 [| 2; 3 |])))

(* The domain *)

let refuse c src l =
  raises_match (Exn.invalid_arg ~sub:"domain") (fun () ->
      if src then ignore (fwd c l) else ignore (inv c l))

(* A point exactly 40° from the central meridian, at any latitude, poles
   included. *)
let boundary =
  Gen.(
    let+ z = zone
    and+ north = bool
    and+ east = bool
    and+ lat =
      frequency
        [
          (4, float_range (-90.) 90.); (1, of_list [ 90.; -90.; 89.9999999999 ]);
        ]
    in
    let lon = float_of_int ((6 * z) - 183) +. if east then 40. else -40. in
    let lon =
      if lon > 180. then lon -. 360.
      else if lon < -180. then lon +. 360.
      else lon
    in
    (z, north, lon, lat))

let both_ways =
  prop "a point of the boundary goes to UTM and back, and to UTM again" boundary
    (fun (z, north, lon, lat) ->
      cover "a pole" (Float.abs lat = 90.);
      let c = utm z north in
      let xy = fwd c [ (lon, lat) ] in
      let back = inv c xy in
      let again = fwd c back in
      less ~msg:"lon, lat" float_exact ~than:12e-9
        (metres (lon, lat) (List.hd back));
      let (x, y), (x', y') = (List.hd xy, List.hd again) in
      less ~msg:"x, y" float_exact ~than:12e-9 (Float.hypot (x' -. x) (y' -. y)))

(* A latitude's distance from its pole, in metres along a meridian. *)
let from_pole lat = (90. -. Float.abs lat) *. deg *. wgs84_radius

let poles =
  group "poles"
    [
      prop "a pole at any longitude round-trips through any zone"
        Gen.(triple zone bool (float_range (-180.) 180.))
        (fun (z, north, lon) ->
          let lat = if north then 90. else -90. in
          let c = utm z north in
          let back = inv c (fwd c [ (lon, lat) ]) in
          let lat' = snd (List.hd back) in
          equal ~msg:"hemisphere" bool (lat > 0.) (lat' > 0.);
          less ~msg:"from the pole" float_exact ~than:12e-9 (from_pole lat');
          ignore (fwd c back));
      test "a zone's pole moves to the next zone" (fun () ->
          let p = points [ (500000., 9997964.9430209977226) ] in
          let q = Crs.transform ~src:(utm 33 true) ~dst:(utm 34 true) p in
          let lat = snd (List.hd (inv (utm 34 true) (pairs q))) in
          less float_exact ~than:12e-9 (from_pole lat));
      test "11 µm from the pole, 40° out, passes both ways" (fun () ->
          let c = utm 33 true in
          ignore (inv c (fwd c [ (55., 89.9999999999) ])));
      test "near the pole, past 40° by more than 12 nm, is refused" (fun () ->
          refuse (utm 33 true) true [ (55.0001, 89.99) ]);
    ]

let domain =
  group "domain"
    [
      test "40° from the central meridian is inside, past it is not" (fun () ->
          ignore (fwd (utm 33 true) [ (55., 10.); (-25., 0.) ]);
          refuse (utm 33 true) true [ (55.000001, 10.) ];
          refuse (utm 33 true) true [ (-25.000001, 10.) ]);
      test "the domain crosses the antimeridian" (fun () ->
          let x, _ = List.hd (fwd (utm 1 true) [ (179., 10.) ]) in
          less float_exact ~than:false_easting x;
          let x, _ = List.hd (fwd (utm 60 true) [ (-179., 10.) ]) in
          greater float_exact ~than:false_easting x);
      test "a latitude past a pole is refused" (fun () ->
          refuse (utm 33 true) true [ (15., 90.000001) ];
          refuse (utm 33 true) true [ (15., -90.000001) ]);
      test "an infinite coordinate is refused" (fun () ->
          refuse (utm 33 true) true [ (Float.infinity, 10.) ];
          refuse (utm 33 true) false [ (Float.infinity, 1e6) ]);
      test "a NaN coordinate maps to NaN" (fun () ->
          let x, y = List.hd (fwd (utm 33 true) [ (Float.nan, 10.) ]) in
          equal ~msg:"x" bool true (Float.is_nan x);
          equal ~msg:"y" bool true (Float.is_nan y));
      test "a point past 40° by under 12 nm along its parallel is inside"
        (fun () ->
          (* 5e-14° of longitude on the equator is 5.6 nm. *)
          ignore (fwd (utm 33 true) [ (55. +. 5e-14, 0.) ]));
      test "the inverse refuses a point beyond half a meridian" (fun () ->
          refuse (utm 33 true) false [ (500000., 2.1e7) ]);
      test "the inverse refuses a point beyond 40°" (fun () ->
          (* 7000 km east on the equator is 60° of longitude away. *)
          refuse (utm 33 true) false [ (7.5e6, 0.) ]);
      test "the inverse refuses a point past the pole" (fun () ->
          refuse (utm 33 true) false [ (500000., 1.01e7) ]);
      test "the error names the point" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"point [1]") (fun () ->
              fwd (utm 33 true) [ (15., 0.); (80., 0.) ]));
    ]

(* ETRS89 / TM35FIN *)

let tm35fin = Result.get_ok (Crs.of_epsg 3067)

(* PROJ 9.8.1's values, by pyproj 3.8.0's [Transformer.from_crs(4258, 3067,
   always_xy=True)]: longitude, latitude, easting and northing. Finland, then
   points far from the central meridian, 27°E, up to 40° from it. The two
   agree within the module's 6 nm. *)
let proj_3067 =
  [
    (27.0, 0.0, 500000., 0.);
    (27.0, 60.0, 500000., 6651411.1902396968);
    (19.1, 59.7, 56067.374780836166, 6644489.2385817179);
    (31.6, 70.1, 674601.95931652351, 7783619.7566124396);
    (27.1, 67.8, 504216.53648683161, 7520569.7576776287);
    (24.94, 60.17, 385700.4213849057, 6672126.7430107333);
    (21.0, 64.5, 212114.30440762581, 7166354.8078900082);
    (29.9, 62.6, 648861.98541712528, 6944369.3374234671);
    (60.0, 30.0, 3769012.0396767617, 3825335.874026651);
    (-10.0, 45.0, -2402392.4144947589, 5695276.6705876384);
    (67.0, 0.0, 5367577.9377075108, 0.);
    (27.0, 89.0, 500000., 9886315.7555686962);
    (27.0, -45.0, 500000., -4982950.4001068529);
  ]

let tm35fin_id (lon, lat, _, _) = Printf.sprintf "%g, %g" lon lat

let tm35fin_forward =
  cases "forward is PROJ's" ~id:tm35fin_id proj_3067 (fun (lon, lat, x, y) ->
      let gx, gy = List.hd (fwd tm35fin [ (lon, lat) ]) in
      equal ~msg:"easting" (float 6e-9) x gx;
      equal ~msg:"northing" (float 6e-9) y gy)

let tm35fin_inverse =
  cases "inverse is PROJ's" ~id:tm35fin_id proj_3067 (fun (lon, lat, x, y) ->
      let back = List.hd (inv tm35fin [ (x, y) ]) in
      equal ~msg:"metres away" (float 6e-9) 0. (metres (lon, lat) back))

(* PROJ's [Transformer.from_crs(3067, 32635, always_xy=True)], whose pipeline
   is TM35FIN's inverse on GRS80 and UTM 35N's projection on WGS 84, with EPSG
   1149's null step between. Two projections, each within 6 nm. *)
let proj_32635 =
  [
    (500000.0, 6651411.0, 500000., 6651411.0001230193);
    (56067.0, 6644489.0, 56067.000005450624, 6644489.0001228247);
    (674602.0, 7783620.0, 674601.99999745633, 7783620.000114413);
    (504216.5, 7520569.75, 504216.49999994045, 7520569.7501170915);
    (212114.3, 7166354.8, 212114.30000386306, 7166354.8001199793);
  ]

let to_utm35 =
  cases "TM35FIN to UTM 35N is PROJ's null datum step" ~id:tm35fin_id
    proj_32635 (fun (x, y, x', y') ->
      let p = points [ (x, y) ] in
      let gx, gy =
        List.hd (pairs (Crs.transform ~src:tm35fin ~dst:(utm 35 true) p))
      in
      equal ~msg:"easting" (float 12e-9) x' gx;
      equal ~msg:"northing" (float 12e-9) y' gy)

let tm35fin_system =
  test "TM35FIN is not WGS 84's zone 35N" (fun () ->
      equal bool false (Crs.equal tm35fin (utm 35 true));
      equal bool true (Crs.equal tm35fin tm35fin))

let tm35fin_domain =
  test "TM35FIN's domain is 40° from 27°E" (fun () ->
      ignore (fwd tm35fin [ (67., 10.); (-13., 0.) ]);
      refuse tm35fin true [ (67.000001, 10.) ];
      refuse tm35fin true [ (-13.000001, 10.) ])

(* Zones and codes *)

let utm_at =
  cases "utm_at is GN 7-2's zone"
    ~id:(fun (lon, lat, _, _) -> Printf.sprintf "%g, %g" lon lat)
    [
      (0.15, 52.25, 31, true);
      (* Norway's 32V exception does not apply. *)
      (5.05, 60.35, 31, true);
      (* So does Svalbard's 33X. *)
      (15.05, 78.25, 33, true);
      (-58.45, -34.55, 21, false);
      (-180., 0., 1, true);
      (180., 0., 1, true);
      (179.999, 0., 60, true);
      (-0.000001, -0.000001, 30, false);
    ]
    (fun (lon, lat, z, north) -> equal crs (utm z north) (Crs.utm_at ~lon ~lat))

let refusals =
  test "utm and utm_at refuse what is not a zone or a point" (fun () ->
      raises_match (Exn.invalid_arg ~sub:"zone 0") (fun () ->
          Crs.utm ~zone:0 ~north:true);
      raises_match (Exn.invalid_arg ~sub:"zone 61") (fun () ->
          Crs.utm ~zone:61 ~north:true);
      raises_match (Exn.invalid_arg ~sub:"latitude") (fun () ->
          Crs.utm_at ~lon:0. ~lat:90.5);
      raises_match (Exn.invalid_arg ~sub:"latitude") (fun () ->
          Crs.utm_at ~lon:0. ~lat:Float.nan);
      raises_match (Exn.invalid_arg ~sub:"longitude") (fun () ->
          Crs.utm_at ~lon:Float.infinity ~lat:0.))

let every_system =
  Crs.lonlat :: tm35fin
  :: List.concat_map (fun z -> [ utm z true; utm z false ]) (List.init 60 succ)

let epsg =
  group "epsg"
    [
      test "every system's code reads back" (fun () ->
          List.iter
            (fun c ->
              equal (result crs string) (Ok c) (Crs.of_epsg (Crs.epsg c)))
            every_system);
      cases "codes" ~id:string_of_int
        [ 4326; 32601; 32631; 32660; 32701; 32760; 3067 ] (fun code ->
          equal int code (Crs.epsg (Result.get_ok (Crs.of_epsg code))));
      cases "other codes are refused" ~id:string_of_int
        [ 0; 4258; 3857; 32600; 32661; 32700; 32761; 25835 ] (fun code ->
          is_error (Crs.of_epsg code));
    ]

(* WKT *)

(* Esri's Projection Engine WKT, copied from its published lists. *)
let esri_4326 =
  {|GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]|}

let esri_32633 =
  {|PROJCS["WGS_1984_UTM_Zone_33N",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",15.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]|}

let esri_32601 =
  {|PROJCS["WGS_1984_UTM_Zone_1N",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",-177.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]|}

let esri_32760 =
  {|PROJCS["WGS_1984_UTM_Zone_60S",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",10000000.0],PARAMETER["Central_Meridian",177.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]|}

let esri_32733 =
  {|PROJCS["WGS_1984_UTM_Zone_33S",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",10000000.0],PARAMETER["Central_Meridian",15.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]|}

let replace ~sub ~by s =
  let n = String.length sub in
  let rec go i =
    if String.sub s i n = sub then
      String.sub s 0 i ^ by ^ String.sub s (i + n) (String.length s - i - n)
    else go (i + 1)
  in
  go 0

(* PROJ 9.8.1's WKT1_GDAL, copied from its output. *)
let ogc_4326 =
  {|GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]|}

let ogc_32633 =
  {|PROJCS["WGS 84 / UTM zone 33N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",15],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32633"]]|}

let ogc_32701 =
  {|PROJCS["WGS 84 / UTM zone 1S",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-177],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",10000000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32701"]]|}

let wkt_accepts =
  cases "of_wkt reads Esri's WKT" ~id:fst
    [
      ("4326", (esri_4326, Crs.lonlat));
      ("32601", (esri_32601, utm 1 true));
      ("32633", (esri_32633, utm 33 true));
      ("32733", (esri_32733, utm 33 false));
      ("32760", (esri_32760, utm 60 false));
      ("OGC 4326", (ogc_4326, Crs.lonlat));
      ("OGC 32633", (ogc_32633, utm 33 true));
      ("OGC 32701", (ogc_32701, utm 1 false));
      ("OGC case", (String.uppercase_ascii ogc_32633, utm 33 true));
      ( "numbers as numbers",
        (replace ~sub:"15.0" ~by:"1.5E1" esri_32633, utm 33 true) );
      ( "integers",
        (replace ~sub:"500000.0" ~by:"500000" esri_32633, utm 33 true) );
      ("case", (String.lowercase_ascii esri_32633, utm 33 true));
      ( "padding",
        ( String.concat ",\n  " (String.split_on_char ',' esri_32633) ^ " \n",
          utm 33 true ) );
      ( "parentheses",
        ( String.map (function '[' -> '(' | ']' -> ')' | c -> c) esri_32633,
          utm 33 true ) );
    ]
    (fun (_, (s, c)) -> equal (result crs string) (Ok c) (Crs.of_wkt s))

let wkt_refuses =
  cases "of_wkt refuses other trees" ~id:fst
    [
      ( "another scale",
        (replace ~sub:"0.9996" ~by:"0.9999" esri_32633, "WGS_1984_UTM_Zone_33N")
      );
      ( "a zone's name on another meridian",
        (replace ~sub:"15.0" ~by:"9.0" esri_32633, "Zone_33N") );
      ( "another hemisphere's false northing",
        ( replace ~sub:"\"False_Northing\",0.0"
            ~by:"\"False_Northing\",10000000.0" esri_32633,
          "Zone_33N" ) );
      ( "an authority",
        ( replace ~sub:",UNIT[\"Meter\",1.0]]"
            ~by:",UNIT[\"Meter\",1.0],AUTHORITY[\"EPSG\",32633]]" esri_32633,
          "Zone_33N" ) );
      ( "EPSG's name",
        ( replace ~sub:"WGS_1984_UTM_Zone_33N" ~by:"WGS 84 / UTM zone 33N"
            esri_32633,
          "WGS 84 / UTM zone 33N" ) );
      ( "another datum",
        ( replace ~sub:"298.257223563" ~by:"298.257222101" esri_4326,
          "GCS_WGS_1984" ) );
      ("zone 0", (replace ~sub:"Zone_33N" ~by:"Zone_0N" esri_32633, "Zone_0N"));
      ( "an identifier the text contradicts",
        ( replace ~sub:"\"central_meridian\",15" ~by:"\"central_meridian\",21"
            ogc_32633,
          "WGS 84 / UTM zone 33N" ) );
      ( "another zone's identifier",
        (replace ~sub:"\"32633\"" ~by:"\"32634\"" ogc_32633, "zone 33N") );
      ( "no axes",
        ( replace ~sub:",AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH]" ~by:""
            ogc_32633,
          "zone 33N" ) );
      ( "a GEOCCS",
        ( {|GEOCCS["x",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Meter",1.0]]|},
          "GEOCCS" ) );
      ( "mixed delimiters",
        ( replace ~sub:"PRIMEM[\"Greenwich\",0.0]"
            ~by:"PRIMEM(\"Greenwich\",0.0)" esri_4326,
          "syntax" ) );
      ( "trailing text",
        (esri_4326 ^ "]", Printf.sprintf "byte %d" (String.length esri_4326)) );
      ("empty", ("", "byte 0"));
      ("an unclosed name", ({|GEOGCS["GCS_WGS_1984|}, "syntax"));
      ( "a bad number",
        (replace ~sub:"0.0174532925199433" ~by:"1e" esri_4326, "syntax") );
    ]
    (fun (_, (s, sub)) ->
      match Crs.of_wkt s with
      | Ok c -> failf "read as %a" Crs.pp c
      | Error e -> contains ~sub e)

let printing =
  test "pp names a system by its EPSG name" (fun () ->
      equal string "WGS 84 / UTM zone 33N"
        (Format.asprintf "%a" Crs.pp (utm 33 true));
      equal string "WGS 84 / UTM zone 1S"
        (Format.asprintf "%a" Crs.pp (utm 1 false));
      equal string "WGS 84" (Format.asprintf "%a" Crs.pp Crs.lonlat);
      equal string "ETRS89 / TM35FIN(E,N)" (Format.asprintf "%a" Crs.pp tm35fin))

let () =
  exit
    (run "jord.crs"
       [
         group "Karney" [ forward_karney; inverse_karney ];
         group "laws"
           [
             round_trip; between_zones; hemispheres; identity; shapes; both_ways;
           ];
         poles;
         domain;
         group "tm35fin"
           [
             tm35fin_forward;
             tm35fin_inverse;
             to_utm35;
             tm35fin_system;
             tm35fin_domain;
           ];
         group "zones" [ utm_at; refusals ];
         epsg;
         group "wkt" [ wkt_accepts; wkt_refuses ];
         printing;
       ])
