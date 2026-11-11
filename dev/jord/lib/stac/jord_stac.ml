(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Jord

type time = { date : int * int * int; seconds : float }

type asset = {
  href : string;
  media_type : string option;
  roles : string list;
  grid : (Grid.t, string) result option;
  bands : (Band.t * (Grid.t, string) result option) list;
}

type item = {
  id : string;
  collection : string option;
  datetime : time option;
  interval : (time * time) option;
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

let integer where name = function
  | Jsont.Number (x, _) when Float.is_integer x -> int_of_float x
  | _ -> refuse "%s: %s holds a value that is not an integer" where name

let required where name = function
  | Some v -> v
  | None -> refuse "%s: no %s" where name

(* Time *)

let leap y = (y mod 4 = 0 && y mod 100 <> 0) || y mod 400 = 0

let days y m =
  match m with
  | 2 -> if leap y then 29 else 28
  | 4 | 6 | 9 | 11 -> 30
  | _ -> 31

let next (y, m, d) =
  if d < days y m then (y, m, d + 1)
  else if m < 12 then (y, m + 1, 1)
  else (y + 1, 1, 1)

let prev (y, m, d) =
  if d > 1 then (y, m, d - 1)
  else if m > 1 then (y, m - 1, days y (m - 1))
  else (y - 1, 12, 31)

(* RFC 3339's date-time: full-date, a separator, partial-time and
   time-offset (§5.6). *)
let time s =
  let n = String.length s in
  let fail () = Error (Printf.sprintf "%S is not an RFC 3339 date-time" s) in
  let digit i = i < n && s.[i] >= '0' && s.[i] <= '9' in
  let int i len =
    if i + len > n || not (List.for_all digit (List.init len (( + ) i))) then
      raise Exit
    else int_of_string (String.sub s i len)
  in
  let char i c = if i >= n || s.[i] <> c then raise Exit in
  match
    let y = int 0 4 in
    char 4 '-';
    let m = int 5 2 in
    char 7 '-';
    let d = int 8 2 in
    if n <= 10 || not (List.mem s.[10] [ 'T'; 't'; ' ' ]) then raise Exit;
    let hh = int 11 2 in
    char 13 ':';
    let mm = int 14 2 in
    char 16 ':';
    let ss = int 17 2 in
    (* The fraction's digits, then the offset. *)
    let i = ref 19 in
    if !i < n && s.[!i] = '.' then begin
      incr i;
      if not (digit !i) then raise Exit;
      while digit !i do
        incr i
      done
    end;
    let frac = float_of_string ("0" ^ String.sub s 19 (!i - 19)) in
    let off =
      if !i < n && (s.[!i] = 'Z' || s.[!i] = 'z') && !i + 1 = n then 0
      else if !i + 6 = n && (s.[!i] = '+' || s.[!i] = '-') then begin
        char (!i + 3) ':';
        let oh = int (!i + 1) 2 and om = int (!i + 4) 2 in
        if oh > 23 || om > 59 then raise Exit;
        (if s.[!i] = '-' then -1 else 1) * ((oh * 60) + om)
      end
      else raise Exit
    in
    if m < 1 || m > 12 || d < 1 || d > days y m then raise Exit;
    if hh > 23 || mm > 59 || ss > 60 then raise Exit;
    (* Taking the offset off may change the day. A leap second is counted
       after its day's last second, so the day comes from second 59. *)
    let local = (hh * 3600) + (mm * 60) + min ss 59 - (off * 60) in
    let date, second =
      if local < 0 then (prev (y, m, d), local + 86400)
      else if local >= 86400 then (next (y, m, d), local - 86400)
      else ((y, m, d), local)
    in
    { date; seconds = float_of_int (second + max 0 (ss - 59)) +. frac }
  with
  | t -> Ok t
  | exception (Exit | Failure _) -> fail ()

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

(* The system, read from the first of [levels] that holds proj:code or
   proj:epsg: one field in two spellings, so they are read together. *)
let system where levels =
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
      let epsg =
        match (code, epsg) with
        | Some a, Some b when a <> b ->
            refuse "%s: proj:code names EPSG:%d and proj:epsg %d" where a b
        | Some n, _ | None, Some n -> Some n
        | None, None -> None
      in
      match epsg with
      | None -> None
      | Some n -> (
          match Crs.of_epsg n with
          | Ok c -> Some c
          | Error e -> refuse "%s: %s" where e))

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

let kind bitfields : Band.kind =
  if Option.is_some bitfields then Bitfield else Measurement

(* A band from [get], which reads a field by its STAC 1.1 name. *)
let band where ~name get : Band.t =
  let num field default =
    Option.value ~default (number where field (get field))
  in
  {
    name;
    common_name = string where "eo:common_name" (get "eo:common_name");
    unit = string where "unit" (get "unit");
    kind = kind (get "classification:bitfields");
    scale = num "raster:scale" 1.;
    offset = num "raster:offset" 0.;
    nodata = nodata where (get "nodata");
    saturated = None;
  }

(* The fields a STAC 1.1 asset may state for its one band. *)
let band_fields =
  [
    "nodata";
    "unit";
    "eo:common_name";
    "raster:scale";
    "raster:offset";
    "classification:bitfields";
  ]

(* STAC 1.0's band fields by their STAC 1.1 names: the eo:bands entry [e]
   holds the common name, the raster:bands entry [r] the rest, and bitfields
   may be the asset's. *)
let stac10 asset e r = function
  | "eo:common_name" -> field "common_name" e
  | "raster:scale" -> field "scale" r
  | "raster:offset" -> field "offset" r
  | "classification:bitfields" as name -> (
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
          (band (at k) ~name (own b), grid (at k) [ b; asset; props ]))
        l
  | None -> (
      let eo = array where "eo:bands" (field "eo:bands" asset)
      and raster = array where "raster:bands" (field "raster:bands" asset) in
      match (eo, raster) with
      | None, None ->
          let get name = field name asset in
          if List.exists (fun f -> Option.is_some (get f)) band_fields then
            [ (band where ~name:None get, g) ]
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
              (band (at k) ~name (stac10 asset e r), g)))

(* Items *)

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
        match time s with
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
  {
    id;
    collection = string where "collection" (field "collection" json);
    datetime;
    interval;
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
