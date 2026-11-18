(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Jord

type view = {
  off_nadir : float option;
  incidence_angle : float option;
  azimuth : float option;
  sun_azimuth : float option;
  sun_elevation : float option;
}

type bitfield = {
  offset : int;
  length : int;
  name : string option;
  description : string option;
  classes : Band.class_ list;
}

type band = {
  band : Band.t;
  grid : (Grid.t, string) result option;
  bitfields : bitfield list;
}

type asset = {
  href : string;
  media_type : string option;
  roles : string list;
  grid : (Grid.t, string) result option;
  bands : band list;
  view : view;
}

type orbit_state = Ascending | Descending | Geostationary | Crossing

type item = {
  id : string;
  collection : string option;
  datetime : float option;
  interval : (float * float) option;
  orbit_state : orbit_state option;
  relative_orbit : int option;
  view : view;
  center_frequency : float option;
  assets : (string * asset) list;
  properties : Jsont.json;
}

exception Refused of string

let refuse fmt = Printf.ksprintf (fun s -> raise (Refused s)) fmt

(* JSON *)

let mem name = function
  | Jsont.Object (ms, _) -> Option.map snd (Jsont.Json.find_mem name ms)
  | _ -> None

(* A member, [None] when absent or [null]. *)
let field name j =
  match mem name j with None | Some (Jsont.Null _) -> None | v -> v

(* The typed value of a member [name], read by [field]. *)
let string where name = function
  | None -> None
  | Some (Jsont.String (s, _)) -> Some s
  | Some _ -> refuse "%s: %s is not a string" where name

let number where name = function
  | None -> None
  | Some (Jsont.Number (x, _)) when Float.is_finite x -> Some x
  | Some _ -> refuse "%s: %s is not a finite number" where name

let list where name = function
  | Jsont.Array (l, _) -> l
  | _ -> refuse "%s: %s is not an array" where name

let array where name v = Option.map (list where name) v

(* An integer an int holds: min_int is a power of two, so both bounds are
   exact floats. *)
let integer where name = function
  | Jsont.Number (x, _)
    when Float.is_integer x
         && Float.of_int min_int <= x
         && x < -.Float.of_int min_int ->
      int_of_float x
  | Jsont.Number (x, _) when Float.is_integer x ->
      refuse "%s: %s holds %g, outside an int's range" where name x
  | _ -> refuse "%s: %s holds a value that is not an integer" where name

let required where name = function
  | Some v -> v
  | None -> refuse "%s: no %s" where name

(* Grids *)

let digits s = s <> "" && String.for_all (fun c -> c >= '0' && c <= '9') s

let code where c =
  match String.split_on_char ':' c with
  | [ "EPSG"; n ] when digits n -> int_of_string n
  | _ -> refuse "%s: proj:code %S is not an EPSG code" where c

(* The member [name] of the first of [levels] that holds it, most specific
   first (a band, its asset, the item's properties); [None] where that
   member is [null]. *)
let lookup levels name =
  let rec first = function
    | [] -> None
    | l :: rest -> (
        match mem name l with
        | Some (Jsont.Null _) -> None
        | Some v -> Some v
        | None -> first rest)
  in
  first levels

(* The system's code, read from the first of [levels] that holds proj:code
   or proj:epsg: one field in two spellings, so they are read together. *)
let epsg where levels =
  let holds l =
    Option.is_some (mem "proj:code" l) || Option.is_some (mem "proj:epsg" l)
  in
  match List.find_opt holds levels with
  | None -> None
  | Some l -> (
      let code =
        Option.map (code where)
          (string where "proj:code" (field "proj:code" l))
      and epsg =
        Option.map (integer where "proj:epsg") (field "proj:epsg" l)
      in
      match (code, epsg) with
      | Some a, Some b when a <> b ->
          refuse "%s: proj:code names EPSG:%d and proj:epsg %d" where a b
      | Some n, _ | None, Some n -> Some n
      | None, None -> None)

(* The system, one fact: read from the first of [levels] that states
   proj:code, proj:epsg or proj:wkt2, its code where it states one, else its
   proj:wkt2. *)
let system where levels =
  let states l =
    List.exists
      (fun f -> Option.is_some (mem f l))
      [ "proj:code"; "proj:epsg"; "proj:wkt2" ]
  in
  match List.find_opt states levels with
  | None -> None
  | Some l -> (
      match epsg where [ l ] with
      | Some n -> (
          match Crs.of_epsg n with
          | Ok c -> Some c
          | Error e -> refuse "%s: %s" where e)
      | None -> (
          match string where "proj:wkt2" (field "proj:wkt2" l) with
          | None -> None
          | Some text -> (
              match Crs.of_wkt text with
              | Ok c -> Some c
              | Error e -> refuse "%s: proj:wkt2: %s" where e)))

(* proj:transform's numbers (a0, a1, a2, a3, a4, a5). *)
let affine where t =
  let num v =
    required where "proj:transform" (number where "proj:transform" (Some v))
  in
  match List.map num t with
  | [ a0; a1; a2; a3; a4; a5 ] | [ a0; a1; a2; a3; a4; a5; 0.; 0.; 1. ] ->
      (a0, a1, a2, a3, a4, a5)
  | [ _; _; _; _; _; _; _; _; _ ] ->
      refuse "%s: proj:transform's last row is not 0, 0, 1" where
  | _ -> refuse "%s: proj:transform has neither 6 nor 9 numbers" where

(* The grid of proj:shape, [[rows, cols]], and proj:transform, read from
   [levels], where both and a system are stated. *)
let grid where levels =
  let get = lookup levels in
  match (get "proj:shape", get "proj:transform") with
  | None, _ | _, None -> None
  | Some shape, Some t -> (
      let read () =
        match system where levels with
        | None -> None
        | Some crs -> (
            let rows, cols =
              match
                List.map (integer where "proj:shape")
                  (list where "proj:shape" shape)
              with
              | [ rows; cols ] when rows >= 0 && cols >= 0 -> (rows, cols)
              | _ -> refuse "%s: proj:shape is not two counts" where
            in
            let a0, a1, a2, a3, a4, a5 =
              affine where (list where "proj:transform" t)
            in
            match
              Grid.v crs ~rows ~cols ~origin:(a2, a5) ~row_step:(a1, a4)
                ~col_step:(a0, a3)
            with
            | g -> Some (Ok g)
            | exception Invalid_argument _ ->
                refuse "%s: proj:transform is not invertible" where)
      in
      try read () with Refused e -> Some (Error e))

(* Bands *)

(* STAC's no-data value: a number, or nan, inf or -inf. *)
let nodata where = function
  | None -> None
  | Some (Jsont.Number (x, _)) -> Some x
  | Some (Jsont.String ("nan", _)) -> Some Float.nan
  | Some (Jsont.String ("inf", _)) -> Some Float.infinity
  | Some (Jsont.String ("-inf", _)) -> Some Float.neg_infinity
  | Some _ -> refuse "%s: nodata is not a number, nan, inf or -inf" where

let boolean where name = function
  | None -> None
  | Some (Jsont.Bool (b, _)) -> Some b
  | Some _ -> refuse "%s: %s is not a boolean" where name

(* A Class Object: classification 1.x requires a description, 2.0 a name. *)
let class_ where c : Band.class_ =
  let value =
    match field "value" c with
    | Some v -> integer where "a class's value" v
    | None -> refuse "%s: a class has no value" where
  in
  let name = string where "name" (field "name" c)
  and description = string where "description" (field "description" c) in
  if name = None && description = None then
    refuse "%s: class %d has neither name nor description" where value;
  let nodata = boolean where "nodata" (field "nodata" c) in
  {
    value;
    name;
    title = string where "title" (field "title" c);
    description;
    nodata = Option.value nodata ~default:false;
  }

(* At least one class, as every version's schema requires, whose values are
   distinct, as jord requires. *)
let classes where name v =
  let cs = List.map (class_ where) (list where name v) in
  if cs = [] then refuse "%s: %s is empty" where name;
  let rec check = function
    | (c : Band.class_) :: rest ->
        if List.exists (fun (d : Band.class_) -> d.value = c.value) rest then
          refuse "%s: %s states value %d twice" where name c.value;
        check rest
    | [] -> ()
  in
  check cs;
  cs

let kind where get : Band.kind =
  match (get "classification:classes", get "classification:bitfields") with
  | Some _, Some _ ->
      refuse "%s: classification:classes and classification:bitfields" where
  | Some v, None -> Classes (classes where "classification:classes" v)
  | None, Some _ -> Bitfield
  | None, None -> Measurement

(* A Bit Field Object, whose classes are its field's values. *)
let bitfield where f =
  let count name =
    match field name f with
    | Some v -> integer where name v
    | None -> refuse "%s: a bit field has no %s" where name
  in
  let offset = count "offset" and length = count "length" in
  if offset < 0 || length < 1 then
    refuse "%s: a bit field has offset %d and length %d" where offset length;
  let classes =
    match field "classes" f with
    | Some v -> classes where "a bit field's classes" v
    | None -> refuse "%s: a bit field has no classes" where
  in
  List.iter
    (fun (c : Band.class_) ->
      if c.value < 0 || (length < Sys.int_size && c.value lsr length <> 0)
      then
        refuse "%s: class %d is outside its %d-bit field" where c.value length)
    classes;
  {
    offset;
    length;
    name = string where "name" (field "name" f);
    description = string where "description" (field "description" f);
    classes;
  }

(* A band from [get], which reads a field by its STAC 1.1 name, on the grid
   [grid]. *)
let band where ~name get grid =
  let num field default =
    Option.value ~default (number where field (get field))
  in
  let band : Band.t =
    {
      name;
      common_name = string where "eo:common_name" (get "eo:common_name");
      center_wavelength =
        number where "eo:center_wavelength" (get "eo:center_wavelength");
      full_width_half_max =
        number where "eo:full_width_half_max"
          (get "eo:full_width_half_max");
      unit = string where "unit" (get "unit");
      kind = kind where get;
      scale = num "raster:scale" 1.;
      offset = num "raster:offset" 0.;
      nodata = nodata where (get "nodata");
      saturated = None;
    }
  in
  let bitfields =
    match get "classification:bitfields" with
    | None -> []
    | Some v -> (
        match list where "classification:bitfields" v with
        | [] -> refuse "%s: classification:bitfields is empty" where
        | fs -> List.map (bitfield where) fs)
  in
  { band; grid; bitfields }

(* The fields a STAC 1.1 asset may state for its one band. *)
let band_fields =
  [
    "nodata";
    "unit";
    "eo:common_name";
    "eo:center_wavelength";
    "eo:full_width_half_max";
    "raster:scale";
    "raster:offset";
    "classification:classes";
    "classification:bitfields";
  ]

(* STAC 1.0's band fields by their STAC 1.1 names: the eo:bands entry [e]
   holds the common name and wavelengths, the raster:bands entry [r] the
   rest, and classes and bitfields may be the asset's. *)
let stac10 asset e r = function
  | "eo:common_name" -> field "common_name" e
  | "eo:center_wavelength" -> field "center_wavelength" e
  | "eo:full_width_half_max" -> field "full_width_half_max" e
  | "raster:scale" -> field "scale" r
  | "raster:offset" -> field "offset" r
  | ("classification:classes" | "classification:bitfields") as name -> (
      match field name r with None -> field name asset | v -> v)
  | name -> field name r

(* An asset's bands, each with its grid: a STAC 1.1 band's own projection
   fields over its asset's, or the asset's [grid]. *)
let bands where ~props ~grid:g asset =
  let own b name = match field name b with None -> field name asset | v -> v in
  let at k = Printf.sprintf "%s, band %d" where k in
  match array where "bands" (field "bands" asset) with
  | Some l ->
      List.mapi
        (fun k b ->
          let name = string (at k) "name" (field "name" b) in
          band (at k) ~name (own b) (grid (at k) [ b; asset; props ]))
        l
  | None -> (
      let eo = array where "eo:bands" (field "eo:bands" asset)
      and raster = array where "raster:bands" (field "raster:bands" asset) in
      match (eo, raster) with
      | None, None ->
          let get name = field name asset in
          if List.exists (fun f -> Option.is_some (get f)) band_fields then
            [ band where ~name:None get g ]
          else []
      | _ ->
          let eo = Option.value eo ~default:[]
          and raster = Option.value raster ~default:[] in
          let n = max (List.length eo) (List.length raster) in
          if eo <> [] && raster <> [] && List.length eo <> List.length raster
          then refuse "%s: eo:bands and raster:bands differ in length" where;
          let empty = Jsont.Json.object' [] in
          let nth l k = Option.value (List.nth_opt l k) ~default:empty in
          List.init n (fun k ->
              let e = nth eo k and r = nth raster k in
              let name = string (at k) "name" (field "name" e) in
              band (at k) ~name (stac10 asset e r) g))

(* Items *)

(* The view extension's angles at the scene centre, in degrees, each read
   from the first of [levels] that holds it. *)
let view where levels =
  let angle name lo hi =
    match number where name (lookup levels name) with
    | Some a when a < lo || a > hi ->
        refuse "%s: %s %g is outside %g to %g" where name a lo hi
    | a -> a
  in
  {
    off_nadir = angle "view:off_nadir" 0. 90.;
    incidence_angle = angle "view:incidence_angle" 0. 90.;
    azimuth = angle "view:azimuth" 0. 360.;
    sun_azimuth = angle "view:sun_azimuth" 0. 360.;
    sun_elevation = angle "view:sun_elevation" (-90.) 90.;
  }

let members where name = function
  | Some (Jsont.Object (ms, _)) -> List.map (fun ((k, _), v) -> (k, v)) ms
  | _ -> refuse "%s: %s is not an object" where name

let asset props (key, a) =
  let where = Printf.sprintf "asset %s" key in
  ignore (members where "the asset" (Some a));
  let grid = grid where [ a; props ] in
  let roles = array where "roles" (field "roles" a) in
  ( key,
    {
      href = required where "href" (string where "href" (field "href" a));
      media_type = string where "type" (field "type" a);
      roles =
        List.map
          (fun r -> required where "roles" (string where "roles" (Some r)))
          (Option.value roles ~default:[]);
      grid;
      bands = bands where ~props ~grid a;
      view = view where [ a; props ];
    } )

let read where json =
  (match string where "type" (field "type" json) with
  | Some "Feature" -> ()
  | _ -> refuse "%s is not a Feature" where);
  let id = required where "id" (string where "id" (field "id" json)) in
  let where = Printf.sprintf "item %s" id in
  let properties = required where "properties" (field "properties" json) in
  ignore (members where "properties" (Some properties));
  let instant name =
    Option.map
      (fun s ->
        match Time.of_rfc3339 s with
        | Ok t -> t
        | Error e -> refuse "%s: %s: %s" where name e)
      (string where name (field name properties))
  in
  let datetime = instant "datetime" in
  let interval =
    match (instant "start_datetime", instant "end_datetime") with
    | Some a, Some b -> Some (a, b)
    | None, None -> None
    | Some _, None -> refuse "%s: start_datetime has no end_datetime" where
    | None, Some _ -> refuse "%s: end_datetime has no start_datetime" where
  in
  if datetime = None && interval = None then
    refuse "%s: no datetime and no interval" where;
  let orbit_state =
    match field "sat:orbit_state" properties with
    | Some (Jsont.String ("ascending", _)) -> Some Ascending
    | Some (Jsont.String ("descending", _)) -> Some Descending
    | Some (Jsont.String ("geostationary", _)) -> Some Geostationary
    | Some (Jsont.String ("crossing", _)) -> Some Crossing
    | _ -> None
  in
  let relative_orbit =
    match field "sat:relative_orbit" properties with
    | None -> None
    | Some v -> (
        match integer where "sat:relative_orbit" v with
        | n when n >= 1 -> Some n
        | n -> refuse "%s: sat:relative_orbit %d is below 1" where n)
  in
  {
    id;
    collection = string where "collection" (field "collection" json);
    datetime;
    interval;
    orbit_state;
    relative_orbit;
    view = view where [ properties ];
    center_frequency =
      number where "sar:center_frequency"
        (field "sar:center_frequency" properties);
    assets =
      List.map (asset properties)
        (members where "assets" (field "assets" json));
    properties;
  }

let item json = try Ok (read "the JSON" json) with Refused e -> Error e

let items json =
  try
    (match string "the JSON" "type" (field "type" json) with
    | Some "FeatureCollection" -> ()
    | _ -> refuse "the JSON is not a FeatureCollection");
    let fs =
      required "the collection" "features"
        (array "the collection" "features" (field "features" json))
    in
    Ok (List.mapi (fun k f -> read (Printf.sprintf "feature %d" k) f) fs)
  with Refused e -> Error e
