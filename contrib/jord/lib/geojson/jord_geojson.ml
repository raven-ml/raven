(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Jord

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

(* A geometry's coordinates. *)
let coordinates where g =
  match mem "coordinates" g with
  | Some c -> c
  | None -> refuse "%s: the geometry has no coordinates" where

(* [x] and [y] of a position, its other numbers dropped. *)
let position where p =
  match list where "a position" p with
  | x :: y :: _ -> (number where x, number where y)
  | _ -> refuse "%s: a position has fewer than two numbers" where

(* The features of the text [json], each geometry given to [geometry], with
   its feature's name, a null geometry included, and each feature's
   properties. A geometry alone is one feature without properties. *)
let features json geometry =
  let feature i f =
    let where = strf "feature %d" i in
    (match Option.bind (mem "type" f) string with
    | Some "Feature" -> ()
    | _ -> refuse "%s is not a Feature" where);
    check_crs where f;
    geometry where
      (Option.value (mem "geometry" f) ~default:(Jsont.Json.null ()));
    Option.value (mem "properties" f) ~default:(Jsont.Json.null ())
  in
  check_crs "the text" json;
  match Option.bind (mem "type" json) string with
  | Some "FeatureCollection" ->
      let fs =
        match mem "features" json with
        | Some f -> list "the collection" "features" f
        | None -> refuse "the collection has no features"
      in
      Array.of_list (List.mapi feature fs)
  | Some "Feature" -> [| feature 0 json |]
  | Some _ ->
      geometry "feature 0" json;
      [| Jsont.Json.null () |]
  | None -> refuse "the text is not a GeoJSON object"

(* A geometry's type, after checking its crs member. *)
let kind where g =
  check_crs where g;
  match Option.bind (mem "type" g) string with
  | Some t -> t
  | None -> refuse "%s: the geometry has no type" where

let geometry acc where g =
  (match g with
  | Jsont.Null _ -> ()
  | g -> (
      let ring r =
        List.iter
          (fun p ->
            let x, y = position where p in
            acc.coords <- y :: x :: acc.coords;
            acc.count <- acc.count + 1)
          (list where "a ring" r);
        acc.rings <- acc.count :: acc.rings;
        acc.nrings <- acc.nrings + 1
      in
      let coords () = list where "coordinates" (coordinates where g) in
      match kind where g with
      | "Polygon" -> List.iter ring (coords ())
      | "MultiPolygon" ->
          List.iter
            (fun poly -> List.iter ring (list where "a polygon" poly))
            (coords ())
      | t -> refuse "%s: a %s is not a Polygon or a MultiPolygon" where t));
  acc.features <- acc.nrings :: acc.features

let polygons json =
  let acc =
    { coords = []; count = 0; rings = [ 0 ]; features = [ 0 ]; nrings = 0 }
  in
  match features json (geometry acc) with
  | props ->
      let points =
        Nx.create Nx.float64 [| acc.count; 2 |]
          (Array.of_list (List.rev acc.coords))
      in
      let rings = Array.of_list (List.rev acc.rings) in
      let features = Array.of_list (List.rev acc.features) in
      Ok (Polygons.v Crs.lonlat points ~rings ~features, props)
  | exception Refused e -> Error e

let points json =
  let coords = ref [] and n = ref 0 in
  let add (x, y) =
    coords := y :: x :: !coords;
    incr n
  in
  let point where = function
    | Jsont.Null _ -> add (Float.nan, Float.nan)
    | g -> (
        match kind where g with
        | "Point" -> (
            match coordinates where g with
            | Jsont.Array ([], _) -> add (Float.nan, Float.nan)
            | c -> add (position where c))
        | t -> refuse "%s: a %s is not a Point" where t)
  in
  match features json point with
  | props ->
      let pts = Array.of_list (List.rev !coords) in
      Ok (Nx.create Nx.float64 [| !n; 2 |] pts, props)
  | exception Refused e -> Error e
