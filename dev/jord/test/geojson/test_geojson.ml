(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord

let decode s =
  match Jsont_bytesrw.decode_string Jsont.json s with
  | Ok j -> Jord_geojson.polygons j
  | Error e -> failwith e

let layout (p : Polygons.t) = (Nx.to_array p.points, p.rings, p.features)

let json =
  Testable.make
    ~pp:(fun ppf j -> Jsont.pp_json ppf j)
    ~equal:(fun a b -> Jsont.Json.compare a b = 0)

let collection =
  test "a collection's polygons, multipolygons and null geometries" (fun () ->
      let p, props =
        Result.get_ok
          (decode
             {|{"type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": [
                  {"type": "Feature", "properties": {"crop": "maize"},
                   "geometry": {"type": "Polygon", "coordinates":
                     [[[15, 48], [15.1, 48], [15.1, 48.1], [15, 48]],
                      [[15.01, 48.01, 230.5], [15.02, 48.01, 231], [15.02, 48.02, 231]]]}},
                  {"type": "Feature", "properties": null, "geometry": null},
                  {"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates":
                     [[[[16, 47], [16.1, 47], [16, 47.1]]], [[[17, 46], [17.1, 46], [17, 46.1]]]]}}
                ]}|})
      in
      equal bool true (Crs.equal Crs.lonlat p.crs);
      equal (array float_exact)
        [|
          15.;
          48.;
          15.1;
          48.;
          15.1;
          48.1;
          15.;
          48.;
          15.01;
          48.01;
          15.02;
          48.01;
          15.02;
          48.02;
          16.;
          47.;
          16.1;
          47.;
          16.;
          47.1;
          17.;
          46.;
          17.1;
          46.;
          17.;
          46.1;
        |]
        (Nx.to_array p.points);
      equal (array int) [| 0; 4; 7; 10; 13 |] p.rings;
      equal (array int) [| 0; 2; 2; 4 |] p.features;
      equal int 3 (Array.length props);
      equal json
        (Jsont.Json.object'
           [
             Jsont.Json.mem (Jsont.Json.name "crop") (Jsont.Json.string "maize");
           ])
        props.(0);
      equal json (Jsont.Json.null ()) props.(1);
      equal json (Jsont.Json.null ()) props.(2))

let single =
  test "a Feature or a geometry alone is one feature" (fun () ->
      let poly =
        {|{"type": "Polygon", "coordinates": [[[1, 2], [3, 4], [5, 6]]]}|}
      in
      let a, pa = Result.get_ok (decode poly) in
      let b, _ =
        Result.get_ok
          (decode
             (Printf.sprintf
                {|{"type": "Feature", "properties": {}, "geometry": %s}|} poly))
      in
      equal (array int) [| 0; 1 |] a.features;
      equal json (Jsont.Json.null ()) pa.(0);
      equal bool true (layout a = layout b))

let crs84 =
  cases "a crs naming CRS84 is accepted" ~id:Fun.id
    [ "urn:ogc:def:crs:OGC::CRS84"; "urn:ogc:def:crs:OGC:1.3:CRS84" ]
    (fun name ->
      is_ok
        (decode
           (Printf.sprintf
              {|{"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": %S}}, "features": []}|}
              name)))

let refusals =
  cases "refusals name the feature" ~id:fst
    [
      ( "a projected crs",
        ( {|{"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32633"}}, "features": []}|},
          "CRS84" ) );
      ( "EPSG:4326, whose axes are latitude first",
        ( {|{"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:4326"}}, "features": []}|},
          "CRS84" ) );
      ( "a linked crs",
        ( {|{"type": "FeatureCollection", "crs": {"type": "link", "properties": {"href": "x"}}, "features": []}|},
          "CRS84" ) );
      ( "a crs on a feature",
        ( {|{"type": "FeatureCollection", "features": [{"type": "Feature", "crs": {"type": "name", "properties": {"name": "EPSG:3857"}}, "geometry": null}]}|},
          "feature 0" ) );
      ( "a point",
        ( {|{"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": null}, {"type": "Feature", "geometry": {"type": "Point", "coordinates": [1, 2]}}]}|},
          "feature 1: a Point" ) );
      ( "a short position",
        ( {|{"type": "Polygon", "coordinates": [[[1], [3, 4], [5, 6]]]}|},
          "fewer than two" ) );
      ( "a string coordinate",
        ( {|{"type": "Polygon", "coordinates": [[["1", 2], [3, 4], [5, 6]]]}|},
          "finite" ) );
      ( "a number among the features",
        ( {|{"type": "FeatureCollection", "features": [1, {"type": "Feature", "geometry": null}]}|},
          "feature 0 is not a Feature" ) );
      ( "an element without its type",
        ( {|{"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": null}, {"geometry": null}]}|},
          "feature 1 is not a Feature" ) );
      ( "a line",
        ( {|{"type": "LineString", "coordinates": [[1, 2], [3, 4]]}|},
          "LineString" ) );
      ("no type", ({|{"features": []}|}, "not a GeoJSON object"));
    ]
    (fun (_, (s, sub)) ->
      match decode s with Ok _ -> fail "read" | Error e -> contains ~sub e)

let () =
  exit
    (run "jord.geojson"
       [ group "reading" [ collection; single; crs84; refusals ] ])
