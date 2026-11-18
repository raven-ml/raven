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

(* A transverse Mercator system on [datum], UTM zone 33N's parameters by
   default. *)
let tm ?(datum = Crs.Wgs84) ?(lat0 = 0.) ?(lon0 = 15.) ?(k0 = 0.9996)
    ?(false_easting = 500000.) ?(false_northing = 0.) () =
  {
    Crs.datum;
    projection =
      Some
        (Crs.transverse_mercator ~lat0 ~lon0 ~k0 ~false_easting
           ~false_northing);
  }
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
   above. Both bounds are checked on Karney's published test set. *)
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
          equal ~msg:(row_id r) (float (bound x)) 0. err)
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
          equal ~msg:(row_id r) (float (bound x)) 0. (Float.hypot dlat dlon))
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
      equal (float 12e-9) 0. (metres (lon, lat) (List.hd back)))

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
      equal (float 4e-9) (1e7 -. n) s)

(* HLS's tile 21JVF, south of the equator, on Microsoft's Planetary
   Computer: item HLS.S30.T21JVF.2026306T134221.v2.0 states proj:epsg 32621
   (zone 21N), a 3660 x 3660 grid of 30 m from corner (399960, -3399960),
   and these footprint vertices (fetched 2026-11-16 by POST
   /api/stac/v1/search, collection hls2-s30, filter id LIKE
   'HLS.S30.T21JVF%'). *)
let t21jvf_corners =
  [
    (399960., -3399960.);
    (509760., -3399960.);
    (399960., -3509760.);
    (509760., -3509760.);
  ]

let t21jvf_footprint =
  [
    (-56.933146, -30.732511);
    (-56.898048, -30.732489);
    (-56.896979, -31.723202);
    (-57.224408, -31.723046);
  ]

let negative_northings =
  test "a northern zone's negative northings are southern latitudes"
    (fun () ->
      (* The same points in zone 21S, whose false northing is 10000000 m. *)
      let south = List.map (fun (x, y) -> (x, y +. 1e7)) t21jvf_corners in
      let lonlat = inv (utm 21 true) t21jvf_corners in
      equal
        (list (pair float_exact float_exact))
        (inv (utm 21 false) south) lonlat;
      List.iter (fun (_, lat) -> is_true (lat < -30. && lat > -32.)) lonlat;
      (* The catalogue's footprint lies in the tile, at negative northings,
         to its rounding of 1e-6°, under 0.12 m, and comes back to itself. *)
      let xy = fwd (utm 21 true) t21jvf_footprint in
      let within lo hi v = v >= lo -. 0.12 && v <= hi +. 0.12 in
      List.iter
        (fun (x, y) ->
          is_true (within 399960. 509760. x);
          is_true (within (-3509760.) (-3399960.) y))
        xy;
      List.iter2
        (fun (lon, lat) (lon', lat') ->
          equal (float 1e-12) lon lon';
          equal (float 1e-12) lat lat')
        t21jvf_footprint
        (inv (utm 21 true) xy))

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
      equal ~msg:"lon, lat" (float 12e-9) 0. (metres (lon, lat) (List.hd back));
      let (x, y), (x', y') = (List.hd xy, List.hd again) in
      equal ~msg:"x, y" (float 12e-9) 0. (Float.hypot (x' -. x) (y' -. y)))

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
          equal ~msg:"from the pole" (float 12e-9) 0. (from_pole lat');
          ignore (fwd c back));
      test "a zone's pole moves to the next zone" (fun () ->
          let p = points [ (500000., 9997964.9430209977226) ] in
          let q = Crs.transform ~src:(utm 33 true) ~dst:(utm 34 true) p in
          let lat = snd (List.hd (inv (utm 34 true) (pairs q))) in
          equal (float 12e-9) 0. (from_pole lat));
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
          equal ~msg:"a number" bool false (Float.is_nan x);
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
      (* An unbatched point has no index, and the remedy names the masking
         and, for a zone, the zone that holds the point. *)
      test "the error gives a remedy" (fun () ->
          let p = Nx.create Nx.float64 [| 2 |] [| 60.; 10. |] in
          raises_match
            (Exn.invalid_arg
               ~sub:
                 "Jord.Crs.transform: point (60, 10) is outside WGS 84 / UTM \
                  zone 33N's domain; mask such points with Crs.in_domain, or \
                  take them to a system whose domain holds them, such as the \
                  zone Crs.utm_at gives")
            (fun () ->
              Crs.transform ~src:Crs.lonlat ~dst:(utm 33 true) p));
    ]

(* ETRS89-FIN [EUREF-FIN] / TM35FIN *)

let tm35fin = Result.get_ok (Crs.of_epsg 3067)
let euref_fin = Result.get_ok (Crs.of_epsg 10690)

(* PROJ 9.8.1's values, by pyproj 3.8.0's [Transformer.from_crs(4258, 3067,
   always_xy=True)]: longitude, latitude, easting and northing. Finland, then
   points far from the central meridian, 27°E, up to 40° from it. The two
   agree within the module's 6 nm. The points are EUREF-FIN's own (10690),
   so no datum step bounds them to Finland. *)
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
      let p = points [ (lon, lat) ] in
      let gx, gy =
        List.hd (pairs (Crs.transform ~src:euref_fin ~dst:tm35fin p))
      in
      equal ~msg:"easting" (float 6e-9) x gx;
      equal ~msg:"northing" (float 6e-9) y gy)

let tm35fin_inverse =
  cases "inverse is PROJ's" ~id:tm35fin_id proj_3067 (fun (lon, lat, x, y) ->
      let p = points [ (x, y) ] in
      let back =
        List.hd (pairs (Crs.transform ~src:tm35fin ~dst:euref_fin p))
      in
      equal ~msg:"metres away" (float 6e-9) 0. (metres (lon, lat) back))

(* PROJ's [Transformer.from_crs(3067, 32635, always_xy=True)], whose pipeline
   is TM35FIN's inverse on GRS80 and UTM 35N's projection on WGS 84, with a
   null step between (PROJ names 1149, EPSG now 10701: both are null). Two
   projections, each within 6 nm. *)
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
      let from_euref_fin l =
        ignore (Crs.transform ~src:euref_fin ~dst:tm35fin (points l))
      in
      from_euref_fin [ (67., 10.); (-13., 0.) ];
      raises_match (Exn.invalid_arg ~sub:"domain") (fun () ->
          from_euref_fin [ (67.000001, 10.) ]);
      raises_match (Exn.invalid_arg ~sub:"domain") (fun () ->
          from_euref_fin [ (-13.000001, 10.) ]))

(* in_domain *)

let lon0 (c : Crs.t) =
  match c.projection with
  | Some (Crs.Transverse_mercator p) -> p.lon0
  | Some (Crs.Albers p) -> p.lon0
  | Some (Crs.Pseudo_mercator p) -> p.lon0
  | Some (Crs.Sinusoidal p) -> p.lon0
  | None -> 0.

(* A projected system and a point within 1e-6° of its domain's edge, 40° from
   its central meridian or at a pole, on either side. *)
let near_edge =
  Gen.(
    let+ c =
      frequency
        [
          (1, constant tm35fin);
          (3, map (fun (z, north) -> utm z north) (pair zone bool));
        ]
    and+ side = of_list [ -40.; 40. ]
    and+ d = float_range (-1e-6) 1e-6
    and+ pole = bool
    and+ lat = float_range (-89.) 89. in
    let lon, lat =
      if pole then (lon0 c, Float.copy_sign (90. +. d) lat)
      else (lon0 c +. side +. d, lat)
    in
    (c, (Float.rem (lon +. 540.) 360. -. 180., lat)))

let in_domain_law =
  prop "in_domain is where a transform into the system passes" near_edge
    (fun (c, (lon, lat)) ->
      let p = points [ (lon, lat) ] in
      let inside = Nx.item [ 0 ] (Crs.in_domain ~src:Crs.lonlat ~dst:c p) in
      cover "inside" inside;
      cover "outside" (not inside);
      let passes =
        match Crs.transform ~src:Crs.lonlat ~dst:c p with
        | _ -> true
        | exception Invalid_argument _ -> false
      in
      equal bool inside passes)

(* Systems of each datum, and points of each drawn near the edges of their
   domains: a TM's 40°, the poles, the datum boxes' edges and the
   antimeridian. *)
let pair_systems =
  [
    Crs.lonlat;
    utm 17 true;
    utm 35 true;
    utm 60 false;
    Result.get_ok (Crs.of_epsg 4269);
    Result.get_ok (Crs.of_epsg 26917);
    Result.get_ok (Crs.of_epsg 26904);
    tm35fin;
    Result.get_ok (Crs.of_epsg 10690);
    Result.get_ok (Crs.of_epsg 2193);
    Result.get_ok (Crs.of_epsg 5070);
    Result.get_ok (Crs.of_epsg 3577);
    Result.get_ok (Crs.of_epsg 3857);
    Crs.modis;
  ]

let lonlat_near =
  Gen.(
    let+ lon =
      one_of
        [
          float_range (-180.) 180.;
          of_list [ -81.; -40.; -47.74; -172.54; 31.59; 19.08; 180.; 173. ];
        ]
    and+ lat =
      one_of
        [
          float_range (-90.) 90.;
          of_list [ 23.81; 86.46; 58.84; 70.09; -25.88; 90.; -90.; 95. ];
        ]
    and+ d = of_list [ 0.; 1e-7; -1e-7; 40. ] in
    (lon +. d, lat))

let any_pair =
  Gen.(
    let+ src = of_list pair_systems
    and+ dst = of_list pair_systems
    and+ lon, lat = lonlat_near in
    (src, dst, lon, lat))

let in_domain_pairs =
  prop "in_domain ~src ~dst is where a transform from src to dst passes"
    any_pair (fun (src, dst, lon, lat) ->
      (* A point of [src], made from a longitude and latitude on its datum;
         one outside [src]'s domain gives a point of no system, NaN. *)
      let geo = { src with Crs.projection = None } in
      let ll = points [ (lon, lat) ] in
      let p =
        if src.projection = None then ll
        else
          match Crs.transform ~src:geo ~dst:src ll with
          | p -> p
          | exception Invalid_argument _ -> points [ (lon *. 1e5, lat *. 1e5) ]
      in
      let inside = Nx.item [ 0 ] (Crs.in_domain ~src ~dst p) in
      cover "inside" inside;
      cover "outside" (not inside);
      cover "across datums" (src.datum <> dst.datum);
      if not (Crs.equal src dst) then
        let passes =
          match Crs.transform ~src ~dst p with
          | _ -> true
          | exception Invalid_argument _ -> false
        in
        equal bool inside passes)

let in_domain_equal =
  test "in_domain of equal geographic systems holds a longitude and latitude"
    (fun () ->
      let p = points [ (15., 95.); (Float.infinity, 0.); (15., 45.) ] in
      List.iter
        (fun c ->
          equal ~msg:(Format.asprintf "%a" Crs.pp c) (array bool)
            [| false; false; true |]
            (Nx.to_array (Crs.in_domain ~src:c ~dst:c p)))
        [ Crs.lonlat; Result.get_ok (Crs.of_epsg 4269) ])

let in_domain_points =
  test "in_domain holds no NaN, no infinity and nothing past a pole" (fun () ->
      let p =
        points
          [
            (540., 90.);
            (0., 90.000001);
            (Float.infinity, 0.);
            (Float.nan, 0.);
            (0., Float.nan);
          ]
      in
      List.iter
        (fun c ->
          equal ~msg:(Format.asprintf "%a" Crs.pp c) (array bool)
            [| true; false; false; false; false |]
            (Nx.to_array (Crs.in_domain ~src:Crs.lonlat ~dst:c p)))
        [ Crs.lonlat; utm 33 true; tm35fin ];
      let q = Nx.create Nx.float64 [| 2; 1; 2 |] [| 15.; 48.; 80.; 47. |] in
      equal (array bool) [| true; false |]
        (Nx.to_array (Crs.in_domain ~src:Crs.lonlat ~dst:(utm 33 true) q));
      equal (array int) [| 2; 1 |]
        (Nx.shape (Crs.in_domain ~src:Crs.lonlat ~dst:(utm 33 true) q)))

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

(* The bounds of the grid's exceptions, each on both sides. *)
let utm_grid_at =
  cases "utm_grid_at is the UTM grid's zone, its exceptions included"
    ~id:(fun (lon, lat, _, _) -> Printf.sprintf "%g, %g" lon lat)
    [
      (0.15, 52.25, 31, true);
      (5.05, 60.35, 32, true);
      (2.999, 60., 31, true);
      (3., 56., 32, true);
      (11.999, 63.999, 32, true);
      (12., 60., 33, true);
      (5., 64., 31, true);
      (5., 55.999, 31, true);
      (8.999, 72., 31, true);
      (9., 78., 33, true);
      (20.999, 78., 33, true);
      (21., 78., 35, true);
      (33., 83.999, 37, true);
      (41.999, 78., 37, true);
      (42., 78., 38, true);
      (10., 84., 32, true);
      (10., 71.999, 32, true);
      (365., 60., 32, true);
      (-58.45, -34.55, 21, false);
    ]
    (fun (lon, lat, z, north) ->
      equal crs (utm z north) (Crs.utm_grid_at ~lon ~lat))

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

let nad83_utm zone = Result.get_ok (Crs.of_epsg (26900 + zone))
let nztm = Result.get_ok (Crs.of_epsg 2193)
let nad83 = Result.get_ok (Crs.of_epsg 4269)
let nzgd2000 = Result.get_ok (Crs.of_epsg 4167)

let every_system =
  [
    Crs.lonlat;
    nad83;
    euref_fin;
    tm35fin;
    nzgd2000;
    nztm;
    Result.get_ok (Crs.of_epsg 5070);
    Result.get_ok (Crs.of_epsg 4283);
    Result.get_ok (Crs.of_epsg 3577);
    Result.get_ok (Crs.of_epsg 3857);
  ]
  @ List.concat_map (fun z -> [ utm z true; utm z false ]) (List.init 60 succ)
  @ List.init 23 (fun z -> nad83_utm (z + 1))
  @ List.map (fun c -> Result.get_ok (Crs.of_epsg c)) [ 9712; 3372; 3373 ]

let epsg =
  group "epsg"
    [
      test "every system's code reads back" (fun () ->
          List.iter
            (fun c ->
              let code = Option.get (Crs.epsg c) in
              equal (result crs string) (Ok c) (Crs.of_epsg code))
            every_system);
      cases "codes" ~id:string_of_int
        [
          4326; 32601; 32631; 32660; 32701; 32760; 3067; 10690; 4269; 26901;
          26917; 26923; 9712; 3372; 3373; 4167; 2193; 5070; 4283; 3577; 3857;
        ] (fun code ->
          equal (option int) (Some code)
            (Crs.epsg (Result.get_ok (Crs.of_epsg code))));
      test "no two codes name equal systems" (fun () ->
          List.iteri
            (fun i a ->
              List.iteri
                (fun j b ->
                  if i <> j then
                    equal ~msg:(Format.asprintf "%a, %a" Crs.pp a Crs.pp b)
                      bool false (Crs.equal a b))
                every_system)
            every_system);
      cases "NAD83's zones that EPSG numbers apart" ~id:fst
        [
          ("9712", (9712, 24)); ("3372", (3372, 59)); ("3373", (3373, 60));
        ]
        (fun (_, (code, zone)) ->
          let c = Result.get_ok (Crs.of_epsg code) in
          equal crs { (utm zone true) with datum = Crs.Nad83 } c;
          equal string
            (Printf.sprintf "NAD83 / UTM zone %dN" zone)
            (Format.asprintf "%a" Crs.pp c));
      test "a system the registry does not list has no code" (fun () ->
          equal (option int) None (Crs.epsg (tm ~k0:0.9999 ()));
          equal (option int) None (Crs.epsg (tm ~datum:Crs.Nad83 ())));
      cases "other codes are refused" ~id:string_of_int
        [
          0; 4258; 3395; 32600; 32661; 32700; 32761; 25828; 25835; 25837;
          26900; 26924; 26960;
        ] (fun code ->
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

let esri_4269 =
  {|GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]|}

(* NAD83 / UTM zone 10N as Esri writes it to a shapefile's .prj, wkid 26910
   of the same list. *)
let esri_26910 = In_channel.with_open_bin "esri_26910.prj" In_channel.input_all
let of_code c = Result.get_ok (Crs.of_epsg c)

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
      ("4269", (esri_4269, of_code 4269));
      ("26910 .prj", (esri_26910, of_code 26910));
      ( "26923",
        ( replace ~sub:"Zone_10N" ~by:"Zone_23N"
            (replace ~sub:"-123.0" ~by:"-45.0" esri_26910),
          of_code 26923 ) );
      ( "9712",
        ( replace ~sub:"Zone_10N" ~by:"Zone_24N"
            (replace ~sub:"-123.0" ~by:"-39.0" esri_26910),
          of_code 9712 ) );
      ( "3372",
        ( replace ~sub:"Zone_10N" ~by:"Zone_59N"
            (replace ~sub:"-123.0" ~by:"171.0" esri_26910),
          of_code 3372 ) );
      ( "3373",
        ( replace ~sub:"Zone_10N" ~by:"Zone_60N"
            (replace ~sub:"-123.0" ~by:"177.0" esri_26910),
          of_code 3373 ) );
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
      ( "a NAD83 zone of_epsg does not read",
        ( replace ~sub:"Zone_10N" ~by:"Zone_58N"
            (replace ~sub:"-123.0" ~by:"165.0" esri_26910),
          "Zone_58N" ) );
      ( "a NAD83 zone in the south",
        ( replace ~sub:"Zone_10N" ~by:"Zone_10S"
            (replace ~sub:"\"False_Northing\",0.0"
               ~by:"\"False_Northing\",10000000.0" esri_26910),
          "Zone_10S" ) );
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

(* Systems as data *)

(* A transverse Mercator system of random parameters, each drawn often from
   a few values so that equal systems are drawn, -0. and 180 included. *)
let random_tm =
  Gen.(
    let+ datum = of_list [ Crs.Wgs84; Crs.Nad83; Crs.Euref_fin; Crs.Nzgd2000 ]
    and+ lat0 = of_list [ 0.; -0.; 49.; -41. ]
    and+ lon0 = of_list [ 15.; 180.; -180.; -0. ]
    and+ k0 = of_list [ 0.9996; 0.9999; Float.succ 0.9996 ]
    and+ false_easting = of_list [ 500000.; 0.; -0. ]
    and+ false_northing = of_list [ 0.; 10000000. ] in
    tm ~datum ~lat0 ~lon0 ~k0 ~false_easting ~false_northing ())

let printed c = Format.asprintf "%a" Crs.pp c

let systems =
  group "systems"
    [
      prop "equal is an equivalence" (Gen.triple random_tm random_tm random_tm)
        (fun (a, b, c) ->
          cover "equal" (Crs.equal a b);
          is_true ~msg:"reflexive" (Crs.equal a a);
          equal ~msg:"symmetric" bool (Crs.equal a b) (Crs.equal b a);
          if Crs.equal a b && Crs.equal b c then
            is_true ~msg:"transitive" (Crs.equal a c));
      prop "two systems print alike only when equal"
        (Gen.pair random_tm random_tm) (fun (a, b) ->
          cover "equal" (Crs.equal a b);
          cover "unequal" (not (Crs.equal a b));
          equal bool (Crs.equal a b) (printed a = printed b));
      test "a system the registry lists is its own" (fun () ->
          equal crs (utm 33 true) (tm ());
          equal crs tm35fin (tm ~datum:Crs.Euref_fin ~lon0:27. ()));
      test "lon0 180 is -180, and -0 is 0" (fun () ->
          equal crs (tm ~lon0:(-180.) ()) (tm ~lon0:180. ());
          equal crs (tm ~lon0:0. ~false_easting:0. ())
            (tm ~lon0:(-0.) ~false_easting:(-0.) ());
          equal string
            "WGS 84 / Transverse Mercator (lat0 0, lon0 0, k0 \
             0.99960000000000004, false_easting 0, false_northing 0)"
            (printed (tm ~lon0:(-0.) ~false_easting:(-0.) ())));
      test "a parameter an ulp away is another system" (fun () ->
          let near = tm ~k0:(Float.succ 0.9996) () in
          equal bool false (Crs.equal near (utm 33 true));
          let p = points [ (15., 60.) ] in
          let q = Crs.transform ~src:(utm 33 true) ~dst:near p in
          let moved = Nx.item [ 0; 1 ] q -. Nx.item [ 0; 1 ] p in
          (* 1 ulp of k0, 1.1e-16, times a northing of 6.65e6 m. *)
          equal (float 2e-9) 7.4e-10 moved);
      test "the constructor refuses what is not a projection" (fun () ->
          let refuse sub f = raises_match (Exn.invalid_arg ~sub) f in
          refuse "lat0" (fun () -> tm ~lat0:90.5 ());
          refuse "lon0" (fun () -> tm ~lon0:180.5 ());
          refuse "k0" (fun () -> tm ~k0:0. ());
          refuse "k0" (fun () -> tm ~k0:(-1.) ());
          refuse "false_easting" (fun () -> tm ~false_easting:Float.nan ());
          refuse "false_northing" (fun () ->
              tm ~false_northing:Float.infinity ()));
    ]

(* A latitude of origin. k0 M(φ₀), M the meridian arc to φ₀ on the
   ellipsoid's double flattening, at the double φ₀ and k0, by mpmath at 50
   digits (test/gen/crs_values.py): the northing of the natural origin in
   the system of the same parameters but lat0 = 0. *)
let origins =
  [
    (Crs.Wgs84, 49., 0.9996012717, 5427462.686056628991138588);
    (Crs.Wgs84, -41., 0.9999, -4540119.233900604277969598);
    (Crs.Wgs84, 89.5, 1.0, 9946118.753864446385616081);
    (Crs.Nad83, 49., 0.9996012717, 5427462.685933994606371139);
    (Crs.Nad83, -41., 0.9999, -4540119.233785335473073009);
    (Crs.Nad83, 89.5, 1.0, 9946118.753781266411885906);
  ]

let origin_id (datum, lat0, k0, _) =
  Format.asprintf "%a, lat0 %g, k0 %g" Crs.pp
    { Crs.datum; projection = None }
    lat0 k0

let latitude_of_origin =
  cases "lat0's origin maps to the false easting and northing" ~id:origin_id
    origins (fun (datum, lat0, k0, m) ->
      let c =
        tm ~datum ~lat0 ~lon0:(-2.) ~k0 ~false_easting:400000.
          ~false_northing:(-100000.) ()
      in
      let geo = { Crs.datum; projection = None } in
      let go src dst l = pairs (Crs.transform ~src ~dst (points l)) in
      let x, y = List.hd (go geo c [ (-2., lat0) ]) in
      equal ~msg:"origin's easting" (float 6e-9) 400000. x;
      equal ~msg:"origin's northing" (float 6e-9) (-100000.) y;
      (* The equator's image lies k0 M(φ₀) south of the false northing. *)
      let _, y = List.hd (go geo c [ (-2., 0.) ]) in
      equal ~msg:"equator's northing" (float 6e-9) (-100000. -. m) y;
      let lon, lat = List.hd (go c geo [ (400000., -100000.) ]) in
      equal ~msg:"inverse, metres away" (float 6e-9) 0.
        (metres (-2., lat0) (lon, lat)))

(* Datums *)

(* PROJ 9.8.1's values, by pyproj 3.8.0's [Transformer.from_crs(src, dst,
   always_xy=True)] (test/gen/crs_values.py): NAD83 / UTM zones 17N and 4N
   from NAD83 (4269), and NZTM2000 from NZGD2000 (4167), each on its own
   datum, so no step is taken. *)
let proj_datums =
  [
    ( "26917",
      nad83,
      nad83_utm 17,
      [
        (-81.0, 0.0, 500000.0, 0.0);
        (-81.0, 40.0, 500000.0, 4427757.218624494);
        (-80.34, 26.03, 566035.379317478, 2879173.0682918066);
        (-75.0, 45.0, 972891.7912730454, 5000491.005341886);
        (-87.0, 30.0, -79218.65355600591, 3333984.3712947615);
        (-60.0, 60.0, 1657788.515116156, 6838296.292381035);
      ] );
    ( "26904",
      nad83,
      nad83_utm 4,
      [
        (-157.86, 21.31, 618238.2675954958, 2356884.3296334017);
        (-159.0, 22.0, 500000.0, 2432826.957520062);
      ] );
    ( "2193",
      nzgd2000,
      nztm,
      [
        (173.0, -41.0, 1600000.0, 5461242.938201998);
        (174.78, -41.29, 1749045.8029443629, 5427521.383395424);
        (166.5, -46.0, 1096728.1029922504, 4885373.763791157);
        (178.5, -37.7, 2085034.730668515, 5813208.536515069);
        (-176.5, -44.0, 2441977.480125731, 5074225.226125137);
        (172.0, -34.4, 1508085.5736616908, 6193038.178439911);
      ] );
  ]

let on_datum =
  cases "NAD83 / UTM and NZTM2000 are PROJ's" ~id:(fun (n, _, _, _) -> n)
    proj_datums (fun (_, geo, c, rows) ->
      List.iter
        (fun ((lon, lat, x, y) as r) ->
          let msg = tm35fin_id r in
          let p = points [ (lon, lat) ] in
          let gx, gy = List.hd (pairs (Crs.transform ~src:geo ~dst:c p)) in
          equal ~msg (float 6e-9) x gx;
          equal ~msg (float 6e-9) y gy;
          let back =
            List.hd (pairs (Crs.transform ~src:c ~dst:geo (points [ (x, y) ])))
          in
          equal ~msg (float 6e-9) 0. (metres (lon, lat) back))
        rows)

let to_lonlat c l = pairs (Crs.transform ~src:c ~dst:Crs.lonlat (points l))
let of_lonlat c l = pairs (Crs.transform ~src:Crs.lonlat ~dst:c (points l))

(* The NAIP tile centroid's longitude and latitude, in Florida; Honolulu;
   Unalaska, in the Aleutians; Finland's northernmost point, Nuorgam,
   70°05′32″N, north of extent 1095's box. *)
let naip = (-80.34, 26.03)
let honolulu = (-157.86, 21.31)
let unalaska = (-166.54, 53.88)
let nuorgam = (27.9556, 70.0922)

(* MS-NFI 2023's site-type grid (Luke, kasvupaikka_vmi1x_1923.tif, as
   test_geotiff's tags state it): 42240 × 73472 cells of 16 m from
   (57632, 7778304) in 3067, its corners 15.50°E to 33.13°E, past
   Finland's box both ways. *)
let ms_nfi =
  Grid.v tm35fin ~rows:73472 ~cols:42240 ~origin:(57632., 7778304.)
    ~row_step:(0., -16.) ~col_step:(16., 0.)

(* [g]'s boundary, 22 points between each two corners, as a program densifies
   a grid to find its longitude and latitude bounds. *)
let boundary (g : Grid.t) =
  let r = float_of_int g.rows and c = float_of_int g.cols in
  let edge (r0, c0) (r1, c1) =
    List.init 22 (fun i ->
        let t = float_of_int i /. 22. in
        (r0 +. (t *. (r1 -. r0)), c0 +. (t *. (c1 -. c0))))
  in
  Grid.world g
    (points
       (List.concat
          [
            edge (0., 0.) (0., c);
            edge (0., c) (r, c);
            edge (r, c) (r, 0.);
            edge (r, 0.) (0., 0.);
          ]))

let steps =
  group "datum steps"
    [
      test "a step changes no coordinate, inside its area or out" (fun () ->
          (* Unalaska and the ocean north of Hawaii, where EPSG states
             other NAD83 operations; Paris and Mexico City, outside 1325. *)
          let l =
            [ naip; unalaska; (-160., 24.); (2.35, 48.86); (-99., 19.4) ]
          in
          equal (list (pair float_exact float_exact)) l (to_lonlat nad83 l);
          equal (list (pair float_exact float_exact)) l (of_lonlat nad83 l);
          equal (list (pair float_exact float_exact)) [ nuorgam ]
            (to_lonlat euref_fin [ nuorgam ]));
      test "GDA94's step changes no coordinate, inside its area or out"
        (fun () ->
          (* Sydney, the box's south-west corner, and Paris. *)
          let gda94 = Result.get_ok (Crs.of_epsg 4283) in
          let l = [ (151.21, -33.87); (93.41, -60.55); (2.35, 48.86) ] in
          equal (list (pair float_exact float_exact)) l (to_lonlat gda94 l);
          equal (list (pair float_exact float_exact)) l (of_lonlat gda94 l));
      test "Nuorgam goes to TM35FIN and back" (fun () ->
          let back = to_lonlat tm35fin (of_lonlat tm35fin [ nuorgam ]) in
          equal (float 6e-9) 0. (metres nuorgam (List.hd back)));
      test "MS-NFI's whole boundary goes to WGS 84" (fun () ->
          let ll = to_lonlat tm35fin (pairs (boundary ms_nfi)) in
          let lons = List.map fst ll in
          (* The NW corner, 15.4965°E, and the NE, 33.1272°E (PROJ). *)
          let least = List.fold_left Float.min 180. lons in
          let most = List.fold_left Float.max (-180.) lons in
          equal ~msg:"west" (float 1e-4) 15.4965 least;
          equal ~msg:"east" (float 1e-4) 33.1272 most);
      test "a polygon at Nuorgam moves to UTM 35N" (fun () ->
          let p =
            Polygons.v tm35fin
              (Crs.transform ~src:Crs.lonlat ~dst:tm35fin
                 (points
                    [ nuorgam; (27.96, 70.0922); (27.96, 70.1); nuorgam ]))
              ~rings:[| 0; 4 |] ~features:[| 0; 1 |]
          in
          let q = Polygons.to_crs (utm 35 true) p in
          equal bool false
            (Array.exists Float.is_nan (Nx.to_array q.Polygons.points)));
      test "Honolulu converts within NAD83 and to WGS 84" (fun () ->
          let z4 =
            Crs.transform ~src:nad83 ~dst:(nad83_utm 4) (points [ honolulu ])
          in
          ignore (Crs.transform ~src:(nad83_utm 4) ~dst:(nad83_utm 3) z4);
          equal ~msg:"metres away" (float 6e-9) 0.
            (metres honolulu (List.hd (to_lonlat (nad83_utm 4) (pairs z4)))));
      prop "across datums, a transform composes each datum's"
        Gen.(pair (float_range (-87.) (-75.)) (float_range 25. 49.))
        (fun (lon, lat) ->
          let z17 = nad83_utm 17 in
          let xy = Crs.transform ~src:nad83 ~dst:z17 (points [ (lon, lat) ]) in
          let direct = Crs.transform ~src:z17 ~dst:(utm 17 true) xy in
          let via =
            Crs.transform ~src:Crs.lonlat ~dst:(utm 17 true)
              (Crs.transform ~src:nad83 ~dst:Crs.lonlat
                 (Crs.transform ~src:z17 ~dst:nad83 xy))
          in
          equal (array float_exact) (Nx.to_array via) (Nx.to_array direct));
    ]

(* k0 A *)

(* k0 A for each k0, Karney's eq. (29) to n¹⁰ in exact rationals from the
   double flattening, times the double k0, rounded once
   (test/gen/crs_values.py). *)
let k0_a =
  [
    ( Crs.Wgs84,
      [
        (0.9996, 0x1.847b98aa272e3p+22);
        (0.9999, 0x1.8499719a87d86p+22);
        (1.0, 0x1.84a3649552bbcp+22);
        (0.9996012717, 0x1.847bb90dfa4f1p+22);
      ] );
    ( Crs.Nad83,
      [
        (0.9996, 0x1.847b98aa19754p+22);
        (0.9999, 0x1.8499719a7a1e6p+22);
        (1.0, 0x1.84a3649545016p+22);
        (0.9996012717, 0x1.847bb90dec962p+22);
      ] );
  ]

(* The inverse divides a northing by k0 A once, so a northing of k0 A is the
   rectifying latitude of 1 radian in every system of an ellipsoid, and its
   inverse the same point, bit for bit, where each k0 A is the one stated. *)
let scale =
  cases "k0 A is the product of k0 and A rounded once"
    ~id:(fun (d, _) ->
      Format.asprintf "%a" Crs.pp { Crs.datum = d; projection = None })
    k0_a
    (fun (datum, rows) ->
      let one (k0, y) =
        let c = tm ~datum ~k0 () in
        let geo = { Crs.datum; projection = None } in
        Nx.to_array (Crs.transform ~src:c ~dst:geo (points [ (500000., y) ]))
      in
      let first = one (List.hd rows) in
      List.iter
        (fun ((k0, _) as r) ->
          equal ~msg:(Printf.sprintf "k0 %.17g" k0) (array float_exact) first
            (one r))
        rows)

(* Albers *)

let u = epsilon_float /. 2.

(* Albers' stated error: 144 u D, D the cone's extent (crs.mli). *)
let albers_eps d = 144. *. u *. d

type map_case = {
  name : string;
  system : Crs.t;
  geo : Crs.t;
  eps : float;
  fwd : (float * float * float * float * float * float) list;
  inv : (float * float * float * float * float * float) list;
}

(* The cases of a reference file, each with its method's stated error. *)
let map_cases file =
  let ic = open_in file in
  let num s = if s = "inf" then Float.infinity else float_of_string s in
  let rec go acc =
    match input_line ic with
    | exception End_of_file ->
        close_in ic;
        List.rev acc
    | l when String.length l > 0 && l.[0] = '#' -> go acc
    | l -> (
        match (String.split_on_char ' ' l, acc) with
        | "case" :: name :: meth :: ell :: rest, _ ->
            let datum = if ell = "wgs84" then Crs.Wgs84 else Crs.Nad83 in
            let f = float_of_string in
            let projection, eps =
              match (meth, rest) with
              | "albers", [ lat0; lon0; lat1; lat2; fe; fn; "D"; d ] ->
                  ( Crs.albers ~lat0:(f lat0) ~lon0:(f lon0) ~lat1:(f lat1)
                      ~lat2:(f lat2) ~false_easting:(f fe)
                      ~false_northing:(f fn),
                    albers_eps (f d) )
              | "pm", [ lon0; fe; fn; "D"; _ ] ->
                  ( Crs.pseudo_mercator ~lon0:(f lon0) ~false_easting:(f fe)
                      ~false_northing:(f fn),
                    10e-9 )
              | "sinusoidal", [ r; lon0; fe; fn; "D"; _ ] ->
                  ( Crs.sinusoidal ~radius:(f r) ~lon0:(f lon0)
                      ~false_easting:(f fe) ~false_northing:(f fn),
                    8e-9 )
              | _ -> failwith (file ^ ": " ^ l)
            in
            let system = { Crs.datum; projection = Some projection } in
            let c =
              {
                name;
                system;
                geo = { system with projection = None };
                eps;
                fwd = [];
                inv = [];
              }
            in
            go (c :: acc)
        | [ kind; a; b; c'; d; h; k ], c :: rest ->
            let row = (num a, num b, num c', num d, num h, num k) in
            let c =
              if kind = "fwd" then { c with fwd = row :: c.fwd }
              else { c with inv = row :: c.inv }
            in
            go (c :: rest)
        | _ -> failwith (file ^ ": " ^ l))
  in
  go []

(* The ground's metres of a step (dlon, dlat) at [lat], on WGS 84's a. *)
let on_ground lat dlon dlat =
  let dlon = Float.rem (dlon +. 540.) 360. -. 180. in
  Float.hypot (dlat *. deg *. wgs84_radius)
    (dlon *. deg *. wgs84_radius *. Float.cos (lat *. deg))

(* Each result is within ε of the exact image of a point within ε of the
   input, so within ε (1 + s) of the exact image of the input itself, s the
   map's largest scale there, and the inverse's within ε (1 + 1/s'), s' its
   least. Each row gives the scales h and k along the meridian and the
   parallel, the extremes but for Sinusoidal's shear, whose rows give its
   singular values. *)
let reference file =
  cases "each method is mpmath's to its stated error" ~id:(fun c -> c.name)
    (map_cases file) (fun c ->
      let eps = c.eps in
      List.iter
        (fun (lon, lat, x, y, h, k) ->
          let p = points [ (lon, lat) ] in
          let gx, gy =
            List.hd (pairs (Crs.transform ~src:c.geo ~dst:c.system p))
          in
          let bound = eps *. (1. +. Float.max h k) in
          equal ~msg:(Printf.sprintf "forward %h, %h" lon lat) (float bound) 0.
            (Float.hypot (gx -. x) (gy -. y)))
        c.fwd;
      List.iter
        (fun (x, y, lon, lat, h, k) ->
          let p = points [ (x, y) ] in
          let glon, glat =
            List.hd (pairs (Crs.transform ~src:c.system ~dst:c.geo p))
          in
          let bound = eps *. (1. +. (1. /. Float.min h k)) in
          equal ~msg:(Printf.sprintf "inverse %h, %h" x y) (float bound) 0.
            (on_ground lat (glon -. lon) (glat -. lat)))
        c.inv)

(* PROJ 9.8.1's values (test/gen/crs_values.py), each
   from its system's own geographic system, 4269 and 4283. *)
let proj_albers =
  [
    ( "5070",
      [
        (-96.0, 23.0, 0.0, 0.0);
        (-77.04, 38.9, 1618254.195227809, 1925667.7163729568);
        (-122.42, 37.77, -2275632.970778876, 1955421.6880452568);
        (-70.0, 45.0, 2021877.4783998579, 2724040.2168134125);
        (-150.0, 61.2, -3097230.129993027, 5077869.299215515);
        (-80.34, 26.03, 1574438.9175223738, 461043.1675389774);
      ] );
    ( "3577",
      [
        (132.0, 0.0, 0.0, 0.0);
        (151.21, -33.87, 1761025.1669565553, -3828891.3655631426);
        (115.86, -31.95, -1508261.2956576233, -3577137.7414186457);
        (147.3, -42.88, 1286184.5168893493, -4763847.759537726);
        (130.84, -12.46, -128368.21938505347, -1314806.1986117077);
        (132.0, -27.0, 0.0, -2926820.886839157);
      ] );
  ]

let albers_codes =
  cases "NAD83 / Conus Albers and GDA94 / Australian Albers are PROJ's"
    ~id:fst proj_albers (fun (code, rows) ->
      let c = Result.get_ok (Crs.of_epsg (int_of_string code)) in
      let geo = { c with projection = None } in
      List.iter
        (fun ((lon, lat, x, y) as r) ->
          let p = points [ (lon, lat) ] in
          let gx, gy = List.hd (pairs (Crs.transform ~src:geo ~dst:c p)) in
          (* PROJ's own error, near Snyder's closed form, is a few nm. *)
          equal ~msg:(tm35fin_id r) (float 2e-8) x gx;
          equal ~msg:(tm35fin_id r) (float 2e-8) y gy)
        rows)

let albers_system ?(datum = Crs.Nad83) ~lat0 ~lon0 ~lat1 ~lat2 () =
  {
    Crs.datum;
    projection =
      Some
        (Crs.albers ~lat0 ~lon0 ~lat1 ~lat2 ~false_easting:0.
           ~false_northing:0.);
  }

(* The cone of 5070, 3577, one of a pole as a parallel, and one of nearly
   opposite parallels. *)
let cones =
  [
    albers_system ~lat0:23. ~lon0:(-96.) ~lat1:29.5 ~lat2:45.5 ();
    albers_system ~datum:Crs.Gda94 ~lat0:0. ~lon0:132. ~lat1:(-18.)
      ~lat2:(-36.) ();
    albers_system ~lat0:90. ~lon0:0. ~lat1:60. ~lat2:90. ();
    albers_system ~lat0:0. ~lon0:0. ~lat1:89. ~lat2:(-88.9) ();
  ]

(* The radii of curvature of GRS80 at [lat]: the meridian's M and the prime
   vertical's N. *)
let radii lat =
  let f = 1. /. 298.257222101 in
  let e2 = f *. (2. -. f) in
  let w = 1. -. (e2 *. Float.pow (Float.sin (lat *. deg)) 2.) in
  let m = wgs84_radius *. (1. -. e2) /. (w *. Float.sqrt w) in
  (m, wgs84_radius /. Float.sqrt w)

(* The scales of [c] at (lon, lat): h along the meridian, k along the
   parallel, by forward differentiation of the transform. *)
let scales c lon lat =
  let geo = { c with Crs.projection = None } in
  let j =
    Rune.jacfwd'
      (fun p -> Crs.transform ~src:geo ~dst:c p)
      (Nx.create Nx.float64 [| 2 |] [| lon; lat |])
  in
  let j = Nx.to_array j in
  (* j is [∂E/∂λ; ∂E/∂φ; ∂N/∂λ; ∂N/∂φ] per degree. *)
  let m, n = radii lat in
  let h = Float.hypot j.(1) j.(3) /. (m *. deg) in
  let k = Float.hypot j.(0) j.(2) /. (n *. Float.cos (lat *. deg) *. deg) in
  (h, k)

let away_from_poles =
  Gen.(pair (float_range (-179.) 179.) (float_range (-85.) 85.))

let albers_laws =
  group "Albers"
    [
      prop "Albers is equal-area: h k = 1"
        Gen.(pair (int_range 0 3) away_from_poles)
        (fun (i, (lon, lat)) ->
          let h, k = scales (List.nth cones i) lon lat in
          equal (float 1e-9) 1. (h *. k));
      cases "the scale along a standard parallel is 1"
        ~id:(fun (l1, l2, l) -> Printf.sprintf "%g, %g at %g" l1 l2 l)
        [ (29.5, 45.5, 29.5); (29.5, 45.5, 45.5); (-18., -36., -18.);
          (-18., -36., -36.); (60., 89., 60.); (60., 89., 89.);
          (45., 45., 45.) ]
        (fun (lat1, lat2, lat) ->
          let c = albers_system ~lat0:0. ~lon0:10. ~lat1 ~lat2 () in
          List.iter
            (fun lon ->
              let _, k = scales c lon lat in
              equal ~msg:(Printf.sprintf "at %g" lon) (float 1e-12) 1. k)
            [ 10.; -60.; 100. ]);
      prop "a southern cone is the northern one reflected in the equator"
        Gen.(triple (float_range 1. 89.) (float_range 1. 89.)
               (pair (float_range (-180.) 180.) (float_range (-90.) 90.)))
        (fun (lat1, lat2, (lon, lat)) ->
          let north = albers_system ~lat0:10. ~lon0:(-30.) ~lat1 ~lat2 () in
          let south =
            albers_system ~lat0:(-10.) ~lon0:(-30.) ~lat1:(-.lat1)
              ~lat2:(-.lat2) ()
          in
          let geo = { north with Crs.projection = None } in
          let at c l =
            List.hd (pairs (Crs.transform ~src:geo ~dst:c (points l)))
          in
          let x, y = at north [ (lon, lat) ] in
          let x', y' = at south [ (lon, -.lat) ] in
          equal ~msg:"easting" float_exact x x';
          equal ~msg:"northing" float_exact (-.y) y');
      prop "a pole as a parallel, or as the origin, gives numbers"
        Gen.(pair (float_range (-180.) 180.) (float_range (-90.) 90.))
        (fun (lon, lat) ->
          List.iter
            (fun (lat0, lat1, lat2) ->
              let c = albers_system ~lat0 ~lon0:0. ~lat1 ~lat2 () in
              let geo = { c with Crs.projection = None } in
              let xy = Crs.transform ~src:geo ~dst:c (points [ (lon, lat) ]) in
              let back = Crs.transform ~src:c ~dst:geo xy in
              equal ~msg:"no NaN" bool false
                (Array.exists Float.is_nan
                   (Array.append (Nx.to_array xy) (Nx.to_array back))))
            [
              (90., 60., 90.);
              (-90., -60., -90.);
              (90., 90., 90.);
              (45., 90., 30.);
            ]);
      prop "round trips hold for nearly opposite parallels"
        Gen.(triple (float_range 1. 89.) (float_range 1e-9 1e-3)
               (pair (float_range (-180.) 180.) (float_range (-89.) 89.)))
        (fun (lat1, gap, (lon, lat)) ->
          let c =
            albers_system ~lat0:0. ~lon0:0. ~lat1 ~lat2:(gap -. lat1) ()
          in
          let geo = { c with Crs.projection = None } in
          let back =
            List.hd
              (pairs
                 (Crs.transform ~src:c ~dst:geo
                    (Crs.transform ~src:geo ~dst:c (points [ (lon, lat) ]))))
          in
          (* The cone's extent grows as the parallels near opposite, here to
             at most 1e9 m, and the error as 144 u D with it. *)
          less float_exact ~than:(2. *. albers_eps 1e9 *. 4.)
            (metres (lon, lat) back));
      test "swapped parallels are one system" (fun () ->
          equal crs
            (albers_system ~lat0:23. ~lon0:(-96.) ~lat1:45.5 ~lat2:29.5 ())
            (Result.get_ok (Crs.of_epsg 5070)));
      cases "opposite parallels are the equal-area cylinder"
        ~id:(fun l2 -> Printf.sprintf "30, %h" l2)
        [ -30.; -.Float.pred 30.; -30. +. 8e-15 ]
        (fun lat2 ->
          (* Lambert's cylindrical equal-area on GRS80 (GN 7-2 §3.5.1): x =
             a k λ, y = a q / (2 k), k = cos φs / √(1 - e² sin² φs), q of
             Snyder's (3-12). The cones 1 and 2 ulps from it, whose sines
             may round opposite, differ from it by far less than ε. *)
          let c = albers_system ~lat0:0. ~lon0:0. ~lat1:30. ~lat2 () in
          let geo = { c with Crs.projection = None } in
          let f = 1. /. 298.257222101 in
          let e2 = f *. (2. -. f) and e = Float.sqrt (f *. (2. -. f)) in
          let s30 = Float.sin (30. *. deg) in
          let k =
            Float.cos (30. *. deg) /. Float.sqrt (1. -. (e2 *. s30 *. s30))
          in
          let q phi =
            let s = Float.sin phi in
            let l = Float.log ((1. -. (e *. s)) /. (1. +. (e *. s))) in
            (1. -. e2) *. ((s /. (1. -. (e2 *. s *. s))) -. (l /. (2. *. e)))
          in
          let eps = albers_eps 2.1e7 in
          List.iter
            (fun (lon, lat) ->
              let go src dst p =
                List.hd (pairs (Crs.transform ~src ~dst (points [ p ])))
              in
              let x, y = go geo c (lon, lat) in
              let msg = Printf.sprintf "%g, %g" lon lat in
              equal ~msg (float eps) (wgs84_radius *. k *. lon *. deg) x;
              equal ~msg (float eps)
                (wgs84_radius *. q (lat *. deg) /. (2. *. k))
                y;
              let back = go c geo (x, y) in
              equal ~msg (float (2. *. eps)) 0. (metres (lon, lat) back))
            [ (0., 0.); (100., 45.); (-179.9, -60.); (30., 89.) ]);
      test "a pole on a polar apex is the apex" (fun () ->
          (* The false origin is the pole, so its image is the origin. *)
          List.iter
            (fun (lat0, lat1, lat2) ->
              let c = albers_system ~lat0 ~lon0:0. ~lat1 ~lat2 () in
              let geo = { c with Crs.projection = None } in
              List.iter
                (fun lon ->
                  let p = points [ (lon, lat0) ] in
                  let x, y =
                    List.hd (pairs (Crs.transform ~src:geo ~dst:c p))
                  in
                  let msg = Printf.sprintf "%g on (%g, %g)" lon lat1 lat2 in
                  equal ~msg (float (albers_eps 1.32e7)) 0. (Float.hypot x y))
                [ 0.; 45.; -170.; 180. ])
            [ (90., 60., 90.); (-90., -60., -90.) ]);
      test "the inverse refuses a point beyond a pole's arc or the seam"
        (fun () ->
          let c = Result.get_ok (Crs.of_epsg 5070) in
          let geo = { c with projection = None } in
          let to_plane l = pairs (Crs.transform ~src:geo ~dst:c (points l)) in
          let back msg (x, y) =
            match Crs.transform ~src:c ~dst:geo (points [ (x, y) ]) with
            | _ -> failf "%s: (%.17g, %.17g) is read" msg x y
            | exception Invalid_argument _ -> ()
          in
          (* The south pole on the central meridian, and 1 m beyond it. *)
          let read p = ignore (Crs.transform ~src:c ~dst:geo (points [ p ])) in
          let x, y = List.hd (to_plane [ (-96., -90.) ]) in
          read (x, y);
          back "beyond the pole" (x, y -. 1.);
          (* The seam, 84°E, at the equator (λ - λ₀ = -180°, the side of
             84.001°E), and 1 m past it along the
             parallel, which crosses the seam's meridian at a right angle. *)
          let (x, y), (x', y') =
            match to_plane [ (84., 0.); (84.001, 0.) ] with
            | [ a; b ] -> (a, b)
            | _ -> assert false
          in
          read (x, y);
          let l = Float.hypot (x -. x') (y -. y') in
          back "past the seam" (x +. ((x -. x') /. l), y +. ((y -. y') /. l)));
    ]

(* Pseudo-Mercator and Sinusoidal *)

let pseudo_mercator = Result.get_ok (Crs.of_epsg 3857)

(* GN 7-2 §3.2.1.2's example, at its printed digits: 24° 22′ 54.433″ N,
   100° 20′ 00.000″ W is E -11169055.58 m, N 2800000.00 m, and the point
   10 km north on the grid is 24° 27′ 48.889″ N. *)
let mercator_example =
  test "Pseudo-Mercator is GN 7-2's example" (fun () ->
      let dms d m s = d +. (m /. 60.) +. (s /. 3600.) in
      let lon = -.dms 100. 20. 0. in
      let x, y = List.hd (fwd pseudo_mercator [ (lon, dms 24. 22. 54.433) ]) in
      equal ~msg:"easting" (float 0.005) (-11169055.58) x;
      equal ~msg:"northing" (float 0.005) 2800000. y;
      let lon', lat' =
        List.hd (inv pseudo_mercator [ (-11169055.58, 2810000.) ])
      in
      equal ~msg:"longitude" (float 1e-7) lon lon';
      equal ~msg:"latitude" (float (0.0005 /. 3600.)) (dms 24. 27. 48.889) lat')

(* MODLAND's grid: tiles of 10° at the equator from ULX -20015109.354 m and
   ULY 10007554.677 m (the MCD12 v6.1 user guide), π R and π R / 2 cut to
   millimetres, so within 2 mm, and h35v09's west edge, ULX 18903158.834333
   m (its STAC item on Microsoft's Planetary Computer), 170°E on the
   equator to the same 2 mm, 1.8e-8°. *)
let modis_tiles =
  test "MODIS's tile corners lie on its grid's meridians" (fun () ->
      let x, y = List.hd (fwd Crs.modis [ (-180., 0.); (0., 90.) ]) in
      equal ~msg:"ULX" (float 0.002) (-20015109.354) x;
      equal ~msg:"equator" float_exact 0. y;
      let _, y = List.nth (fwd Crs.modis [ (-180., 0.); (0., 90.) ]) 1 in
      equal ~msg:"ULY" (float 0.002) 10007554.677 y;
      let lon, lat = List.hd (inv Crs.modis [ (18903158.834333, 0.) ]) in
      equal ~msg:"h35's west edge" (float 2e-8) 170. lon;
      equal ~msg:"on the equator" float_exact 0. lat;
      equal (option int) None (Crs.epsg Crs.modis))

let spheres =
  group "laws"
    [
      prop "Sinusoidal is equal-area: |det J| = R² cos φ"
        Gen.(pair (float_range (-179.) 179.) (float_range (-89.) 89.))
        (fun (lon, lat) ->
          let j =
            Nx.to_array
              (Rune.jacfwd'
                 (fun p -> Crs.transform ~src:Crs.lonlat ~dst:Crs.modis p)
                 (Nx.create Nx.float64 [| 2 |] [| lon; lat |]))
          in
          let det = Float.abs ((j.(0) *. j.(3)) -. (j.(1) *. j.(2))) in
          let r = 6371007.181 *. deg in
          equal (float 1e-12) 1. (det /. (r *. r *. Float.cos (lat *. deg))));
      test "Sinusoidal's inverse refuses points outside the sinusoid" (fun () ->
          let half = Float.pi *. 6371007.181 in
          ignore (inv Crs.modis [ (half, 0.); (0., half /. 2.) ]);
          refuse Crs.modis false [ (half +. 1e-6, 0.) ];
          refuse Crs.modis false [ (0., (half /. 2.) +. 1e-6) ];
          (* 60°N, where the parallel's image is half as long. *)
          refuse Crs.modis false [ ((half /. 2.) +. 1e-6, half /. 3.) ]);
      test "Sinusoidal's pole takes the central meridian" (fun () ->
          let c =
            {
              Crs.datum = Crs.Wgs84;
              projection =
                Some
                  (Crs.sinusoidal ~lon0:(-120.) ~radius:6371007.181
                     ~false_easting:1e6 ~false_northing:0.);
            }
          in
          let geo = { c with projection = None } in
          let y = 6371007.181 *. Float.pi /. 2. in
          equal
            (list (pair float_exact float_exact))
            [ (-120., 90.) ]
            (pairs (Crs.transform ~src:c ~dst:geo (points [ (1e6, y) ]))));
      test "Pseudo-Mercator stops at 88°" (fun () ->
          ignore (fwd pseudo_mercator [ (10., 88.); (10., -88.) ]);
          refuse pseudo_mercator true [ (10., 88.000001) ];
          let _, y = List.hd (fwd pseudo_mercator [ (10., 88.) ]) in
          ignore (inv pseudo_mercator [ (0., y) ]);
          refuse pseudo_mercator false [ (0., y +. 1.) ];
          let half = Float.pi *. 6378137. in
          refuse pseudo_mercator false [ (half +. 1e-6, 0.) ]);
    ]

(* WKT2 *)

(* The rows of wkt.dat: source, code and text. *)
let wkt_rows =
  let ic = open_in "wkt.dat" in
  let rec go acc =
    match input_line ic with
    | exception End_of_file ->
        close_in ic;
        List.rev acc
    | l when String.length l > 0 && l.[0] = '#' -> go acc
    | l -> (
        match String.index_from_opt l (String.index l ' ' + 1) ' ' with
        | Some j ->
            let i = String.index l ' ' in
            go
              (( String.sub l 0 i,
                 String.sub l (i + 1) (j - i - 1),
                 String.sub l (j + 1) (String.length l - j - 1) )
              :: acc)
        | None -> failwith ("wkt.dat: " ^ l))
  in
  go []

let wkt source code =
  let _, _, t =
    List.find (fun (s, c, _) -> s = source && c = code) wkt_rows
  in
  t

let read_wkt s =
  match Crs.of_wkt s with Ok c -> c | Error e -> failf "refused: %s" e

let wkt2_reads =
  cases "of_wkt reads each producer's WKT as its code's system"
    ~id:(fun (s, c, _) -> s ^ " " ^ c)
    (List.filter
       (fun (s, c, _) ->
         (not (List.mem c [ "modis"; "25835"; "4258" ]))
         && not (s = "epsg" && c = "5070"))
       wkt_rows)
    (fun (_, code, text) ->
      let c = read_wkt text in
      equal crs (Result.get_ok (Crs.of_epsg (int_of_string code))) c;
      equal (option int) (Some (int_of_string code)) (Crs.epsg c))

let wkt2_modis =
  test "MODIS's sinusoid reads as Crs.modis from GDAL's WKT 1 and PROJ's WKT2"
    (fun () ->
      equal crs Crs.modis (read_wkt (wkt "mpc" "modis"));
      equal crs Crs.modis (read_wkt (wkt "proj" "modis")))

(* The registry stores 5070's parallels in sexagesimal DMS, 29.3 and 45.3,
   which is 29°30′ exactly; its own WKT2 export prints 29.5000000000003. *)
let wkt2_lossy =
  test "EPSG's 5070 export is the system it states, a few nm from 5070"
    (fun () ->
      let c = read_wkt (wkt "epsg" "5070") in
      let c5070 = Result.get_ok (Crs.of_epsg 5070) in
      equal bool false (Crs.equal c c5070);
      equal (option int) None (Crs.epsg c);
      (* Over CONUS the two maps differ by at most 4.3 nm (mpmath). *)
      let geo = { c with projection = None } in
      let l =
        [ (-125., 24.); (-66., 24.); (-125., 50.); (-66., 50.); (-96., 37.) ]
      in
      let a = Crs.transform ~src:geo ~dst:c (points l) in
      let b = Crs.transform ~src:geo ~dst:c5070 (points l) in
      List.iter2
        (fun (x, y) (x', y') ->
          less float_exact ~than:(4.3e-9 +. (2. *. albers_eps 2.1046e7))
            (Float.hypot (x -. x') (y -. y')))
        (pairs a) (pairs b))

let swap ~sub ~by s = replace ~sub ~by s

let wkt2_refuses =
  cases "of_wkt refuses other WKT2" ~id:fst
    [
      ("ETRS89, not a datum here", (wkt "epsg" "25835", "datum"));
      ("ETRS89's geographic system", (wkt "epsg" "4258", "datum"));
      ( "another ellipsoid",
        ( swap ~sub:"298.257222101" ~by:"298.257223563" (wkt "epsg" "26917"),
          "ellipsoid" ) );
      ( "a datum and a system that disagree",
        ( swap ~sub:"ID[\"EPSG\",4269]" ~by:"ID[\"EPSG\",4326]" (wkt "epsg" "26917"),
          "disagree" ) );
      ( "a method by name only",
        ( swap ~sub:",ID[\"EPSG\",9807]" ~by:"" (wkt "epsg" "26917"),
          "no EPSG identifier" ) );
      ( "a parameter by name only",
        ( swap ~sub:",ID[\"EPSG\",8805]" ~by:"" (wkt "epsg" "26917"),
          "parameter" ) );
      ( "a missing parameter",
        ( swap
            ~sub:"PARAMETER[\"Scale factor at natural origin\",0.9996,SCALEUNIT[\"unity\",1,ID[\"EPSG\",9201]],ID[\"EPSG\",8805]],"
            ~by:"" (wkt "epsg" "26917"),
          "parameters" ) );
      ( "Pseudo-Mercator's latitude of origin",
        ( swap ~sub:"Latitude of natural origin\",0" ~by:"Latitude of natural origin\",1"
            (wkt "epsg" "3857"),
          "8801" ) );
      ( "a 3D system",
        (swap ~sub:"CS[Cartesian,2" ~by:"CS[Cartesian,3" (wkt "epsg" "26917"), "2D") );
      ( "feet",
        ( swap ~sub:"CS[Cartesian,2,ID[\"EPSG\",4400]],AXIS[\"Easting (E)\",east],AXIS[\"Northing (N)\",north],LENGTHUNIT[\"metre\",1,ID[\"EPSG\",9001]]"
            ~by:"CS[Cartesian,2],AXIS[\"Easting (E)\",east],AXIS[\"Northing (N)\",north],LENGTHUNIT[\"US survey foot\",0.304800609601219]"
            (wkt "epsg" "26917"),
          "unit" ) );
      ( "a BOUNDCRS",
        ("BOUNDCRS[SOURCECRS[" ^ wkt "epsg" "26917" ^ "]]", "BOUNDCRS") );
      ( "another sphere",
        (swap ~sub:"6371007.181" ~by:"6370997" (wkt "proj" "modis"), "datum") );
    ]
    (fun (_, (s, sub)) ->
      match Crs.of_wkt s with
      | Ok c -> failf "read as %a" Crs.pp c
      | Error e -> contains ~sub e)

let wkt2_anchor =
  test "EPSG's 3067 states an anchor epoch, which is not read" (fun () ->
      contains ~sub:"ANCHOREPOCH[1997]" (wkt "epsg" "3067");
      equal crs tm35fin (read_wkt (wkt "epsg" "3067")))

let printing =
  test "pp names a system by its EPSG name" (fun () ->
      equal string "WGS 84 / UTM zone 33N"
        (Format.asprintf "%a" Crs.pp (utm 33 true));
      equal string "WGS 84 / UTM zone 1S"
        (Format.asprintf "%a" Crs.pp (utm 1 false));
      equal string "WGS 84" (Format.asprintf "%a" Crs.pp Crs.lonlat);
      equal string "ETRS89-FIN [EUREF-FIN] / TM35FIN(E,N)"
        (Format.asprintf "%a" Crs.pp tm35fin);
      equal string "NAD83 / UTM zone 17N"
        (Format.asprintf "%a" Crs.pp (nad83_utm 17));
      equal string "NZGD2000 / New Zealand Transverse Mercator 2000"
        (Format.asprintf "%a" Crs.pp nztm);
      equal string "NAD83" (Format.asprintf "%a" Crs.pp nad83);
      equal string "NAD83 / Conus Albers"
        (Format.asprintf "%a" Crs.pp (Result.get_ok (Crs.of_epsg 5070)));
      equal string "GDA94 / Australian Albers"
        (Format.asprintf "%a" Crs.pp (Result.get_ok (Crs.of_epsg 3577)));
      equal string "WGS 84 / Pseudo-Mercator"
        (Format.asprintf "%a" Crs.pp pseudo_mercator);
      equal string
        "WGS 84 / Sinusoidal (lon0 0, radius 6371007.1809999999, false_easting \
         0, false_northing 0)"
        (Format.asprintf "%a" Crs.pp Crs.modis);
      equal string
        "NAD83 / Albers Equal Area (lat0 23, lon0 -96, lat1 45.5, lat2 \
         29.499999999999996, false_easting 0, false_northing 0)"
        (Format.asprintf "%a" Crs.pp
           (albers_system ~lat0:23. ~lon0:(-96.) ~lat1:(Float.pred 29.5)
              ~lat2:45.5 ())))

let () =
  exit
    (run "jord.crs"
       [
         group "Karney" [ forward_karney; inverse_karney ];
         group "laws"
           [
             round_trip;
             between_zones;
             hemispheres;
             negative_northings;
             identity;
             shapes;
             both_ways;
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
         group "in_domain"
           [
             in_domain_law; in_domain_pairs; in_domain_equal; in_domain_points;
           ];
         group "zones" [ utm_at; utm_grid_at; refusals ];
         epsg;
         group "wkt" [ wkt_accepts; wkt_refuses ];
         group "wkt2"
           [ wkt2_reads; wkt2_modis; wkt2_lossy; wkt2_refuses; wkt2_anchor ];
         systems;
         group "lat0" [ latitude_of_origin ];
         group "k0 A" [ scale ];
         group "Albers reference" [ reference "albers.dat"; albers_codes ];
         group "Pseudo-Mercator and Sinusoidal"
           [ reference "mercator_sinusoidal.dat"; mercator_example; modis_tiles;
             spheres ];
         albers_laws;
         group "datums" [ on_datum ];
         steps;
         printing;
       ])
