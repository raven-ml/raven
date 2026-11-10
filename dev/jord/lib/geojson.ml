(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

exception Refused of string

let strf = Printf.sprintf
let refuse fmt = Printf.ksprintf (fun s -> raise (Refused s)) fmt
let crs84 = [ "urn:ogc:def:crs:OGC::CRS84"; "urn:ogc:def:crs:OGC:1.3:CRS84" ]

let mem name = function
  | Jsont.Object (ms, _) -> Option.map snd (Jsont.Json.find_mem name ms)
  | _ -> None

let string = function Jsont.String (s, _) -> Some s | _ -> None

(* GeoJSON 2008's named CRS: {"type": "name", "properties": {"name": N}}. *)
let check_crs where obj =
  match mem "crs" obj with
  | None | Some (Jsont.Null _) -> ()
  | Some crs ->
      let name =
        match (Option.bind (mem "type" crs) string, mem "properties" crs) with
        | Some "name", Some props -> Option.bind (mem "name" props) string
        | _ -> None
      in
      if not (Option.fold ~none:false ~some:(fun n -> List.mem n crs84) name)
      then refuse "%s: its crs member does not name OGC CRS84" where

let list where what = function
  | Jsont.Array (l, _) -> l
  | _ -> refuse "%s: %s is not an array" where what

(* Points and offsets, built feature by feature. *)
type acc = {
  mutable coords : float list; (* Reversed: y, x of the last point first. *)
  mutable count : int;
  mutable rings : int list;
  mutable features : int list;
  mutable nrings : int;
}

let number where = function
  | Jsont.Number (x, _) when Float.is_finite x -> x
  | _ -> refuse "%s: a coordinate is not a finite number" where

let ring acc where r =
  List.iter
    (fun p ->
      match list where "a position" p with
      | x :: y :: _ ->
          acc.coords <- number where y :: number where x :: acc.coords;
          acc.count <- acc.count + 1
      | _ -> refuse "%s: a position has fewer than two numbers" where)
    (list where "a ring" r);
  acc.rings <- acc.count :: acc.rings;
  acc.nrings <- acc.nrings + 1

let geometry acc where g =
  (match g with
  | Jsont.Null _ -> ()
  | g -> (
      check_crs where g;
      let coords () =
        match mem "coordinates" g with
        | Some c -> list where "coordinates" c
        | None -> refuse "%s: the geometry has no coordinates" where
      in
      match Option.bind (mem "type" g) string with
      | Some "Polygon" -> List.iter (ring acc where) (coords ())
      | Some "MultiPolygon" ->
          List.iter
            (fun poly ->
              List.iter (ring acc where) (list where "a polygon" poly))
            (coords ())
      | Some t -> refuse "%s: a %s is not a Polygon or a MultiPolygon" where t
      | None -> refuse "%s: the geometry has no type" where));
  acc.features <- acc.nrings :: acc.features

let feature acc i f =
  let where = strf "feature %d" i in
  (match Option.bind (mem "type" f) string with
  | Some "Feature" -> ()
  | _ -> refuse "%s is not a Feature" where);
  check_crs where f;
  geometry acc where
    (Option.value (mem "geometry" f) ~default:(Jsont.Json.null ()));
  Option.value (mem "properties" f) ~default:(Jsont.Json.null ())

let polygons json =
  let acc =
    { coords = []; count = 0; rings = [ 0 ]; features = [ 0 ]; nrings = 0 }
  in
  match
    check_crs "the text" json;
    match Option.bind (mem "type" json) string with
    | Some "FeatureCollection" ->
        let fs =
          match mem "features" json with
          | Some f -> list "the collection" "features" f
          | None -> refuse "the collection has no features"
        in
        Array.of_list (List.mapi (feature acc) fs)
    | Some "Feature" -> [| feature acc 0 json |]
    | Some ("Polygon" | "MultiPolygon") ->
        geometry acc "feature 0" json;
        [| Jsont.Json.null () |]
    | Some t -> refuse "the text is a %s, not a feature or a polygon" t
    | None -> refuse "the text is not a GeoJSON object"
  with
  | props ->
      let points =
        Nx.create Nx.float64 [| acc.count; 2 |]
          (Array.of_list (List.rev acc.coords))
      in
      let rings = Array.of_list (List.rev acc.rings) in
      let features = Array.of_list (List.rev acc.features) in
      Ok (Polygons.v Crs.lonlat points ~rings ~features, props)
  | exception Refused e -> Error e
