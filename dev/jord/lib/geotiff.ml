(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module T = Nx_io.Tiff

exception Refused of string

(* A product that OCaml's arm64 backend cannot fuse into the addition that uses
   it, as amd64's does not, so the result is the same bits on both. *)
let mul a b = Sys.opaque_identity (a *. b)
let refuse fmt = Printf.ksprintf (fun s -> raise (Refused s)) fmt

(* TIFF 6.0's and GeoTIFF 1.1's tags, GDAL's nodata tag, and the GeoKeys and
   codes this module reads. *)
let image_width = 256
let image_length = 257
let model_pixel_scale = 33550
let model_tiepoint = 33922
let model_transformation = 34264
let geo_key_directory = 34735
let gdal_nodata = 42113
let gt_model_type = 1024
let gt_raster_type = 1025
let geodetic_crs = 2048
let geog_linear_units = 2052
let geog_angular_units = 2054
let projected_crs = 3072
let proj_linear_units = 3076
let model_projected = 1
let model_geographic = 2
let pixel_is_area = 1
let pixel_is_point = 2
let metre = 9001
let degrees = [ 9102; 9122 ]

let tag_names =
  [
    (image_width, "ImageWidth");
    (image_length, "ImageLength");
    (model_pixel_scale, "ModelPixelScale");
    (model_tiepoint, "ModelTiepoint");
    (model_transformation, "ModelTransformation");
    (geo_key_directory, "GeoKeyDirectory");
    (gdal_nodata, "GDAL_NODATA");
  ]

let named t =
  Printf.sprintf "%s (%d)"
    (Option.value (List.assoc_opt t tag_names) ~default:"tag")
    t

let find tags t =
  match List.filter (fun (k, _) -> k = t) tags with
  | [] -> None
  | [ (_, v) ] -> Some v
  | _ -> refuse "%s is given twice" (named t)

let ints t = function
  | T.Byte a | T.Short a | T.Long a -> a
  | _ -> refuse "%s is not of an unsigned integer type" (named t)

let floats t = function
  | T.Float a | T.Double a -> a
  | _ -> refuse "%s is not of a floating-point type" (named t)

let one t tags =
  match find tags t with
  | None -> refuse "%s is missing" (named t)
  | Some v -> (
      match ints t v with
      | [| n |] when n >= 0 -> n
      | [| n |] -> refuse "%s: %d is negative" (named t) n
      | _ -> refuse "%s has not one value" (named t))

(* GeoKeys: the SHORT keys of the directory, by key. A key whose location is 0
   holds its value; one at the directory's own tag points into it. *)
let geo_keys tags =
  let dir =
    match find tags geo_key_directory with
    | None -> refuse "%s is missing" (named geo_key_directory)
    | Some (T.Short a) -> a
    | Some _ -> refuse "%s is not SHORT" (named geo_key_directory)
  in
  let what = named geo_key_directory in
  let n = Array.length dir in
  if n < 4 then refuse "%s has no header" what;
  if dir.(0) <> 1 || dir.(1) <> 1 || not (dir.(2) = 0 || dir.(2) = 1) then
    refuse "%s: version %d.%d.%d is not 1.1.0 or 1.1.1" what dir.(0) dir.(1)
      dir.(2);
  let keys = dir.(3) in
  if n < 4 + (4 * keys) then refuse "%s: its %d keys leave the tag" what keys;
  List.init keys (fun k ->
      let e = 4 + (4 * k) in
      let key = dir.(e) and loc = dir.(e + 1) and count = dir.(e + 2) in
      let v = dir.(e + 3) in
      let value =
        if loc = 0 then Some v
        else if loc = geo_key_directory && count = 1 && v < n then Some dir.(v)
        else None
      in
      (key, value))

let key keys k =
  match List.filter (fun (k', _) -> k' = k) keys with
  | [] -> None
  | [ (_, Some v) ] -> Some v
  | [ (_, None) ] -> refuse "GeoKey %d is not one SHORT value" k
  | _ -> refuse "GeoKey %d is given twice" k

let crs keys =
  let code k =
    match key keys k with
    | Some 32767 -> refuse "GeoKey %d states a user-defined system" k
    | Some c -> (
        match Crs.of_epsg c with
        | Ok c -> c
        | Error e -> refuse "GeoKey %d: %s" k e)
    | None -> refuse "GeoKey %d is missing" k
  in
  let unit k ok name =
    match key keys k with
    | Some u when not (List.mem u ok) ->
        refuse "GeoKey %d: unit %d is not %s" k u name
    | _ -> ()
  in
  unit proj_linear_units [ metre ] "the metre (9001)";
  unit geog_linear_units [ metre ] "the metre (9001)";
  unit geog_angular_units degrees "the degree (9102, 9122)";
  match key keys gt_model_type with
  | Some m when m = model_projected ->
      let c = code projected_crs in
      if Crs.equal c Crs.lonlat then
        refuse "GeoKey %d: 4326 is not projected" projected_crs;
      c
  | Some m when m = model_geographic ->
      let c = code geodetic_crs in
      if not (Crs.equal c Crs.lonlat) then
        refuse "GeoKey %d is not geographic" geodetic_crs;
      c
  | Some m ->
      refuse "GeoKey %d: model type %d is not projected (1) or geographic (2)"
        gt_model_type m
  | None -> refuse "GeoKey %d is missing" gt_model_type

(* The map from raster space (I, J) as x = a I + b J + c, y = d I + e J + f. *)
let affine tags =
  let get t n =
    Option.map
      (fun v ->
        let a = floats t v in
        if Array.length a < n then
          refuse "%s has %d values" (named t) (Array.length a);
        a)
      (find tags t)
  in
  match
    ( get model_transformation 16,
      get model_pixel_scale 3,
      find tags model_tiepoint )
  with
  | Some m, None, None ->
      if Array.length m <> 16 then
        refuse "%s has not 16 values" (named model_transformation);
      if not (m.(12) = 0. && m.(13) = 0. && m.(14) = 0. && m.(15) = 1.) then
        refuse "%s: its last row is not 0 0 0 1" (named model_transformation);
      (m.(0), m.(1), m.(3), m.(4), m.(5), m.(7))
  | Some _, _, _ ->
      refuse "%s is given with %s or %s"
        (named model_transformation)
        (named model_pixel_scale) (named model_tiepoint)
  | None, Some s, Some tp ->
      let tp = floats model_tiepoint tp in
      if Array.length tp <> 6 then
        refuse "%s holds %d values, not one tiepoint" (named model_tiepoint)
          (Array.length tp);
      let i = tp.(0) and j = tp.(1) and x = tp.(3) and y = tp.(4) in
      (s.(0), 0., x -. mul i s.(0), 0., -.s.(1), y +. mul j s.(1))
  | None, None, Some _ ->
      refuse "%s has no %s" (named model_tiepoint) (named model_pixel_scale)
  | None, _, None ->
      refuse "%s or %s is missing" (named model_tiepoint)
        (named model_transformation)

let finite_all l = List.for_all Float.is_finite l

let grid tags =
  match
    let cols = one image_width tags and rows = one image_length tags in
    let keys = geo_keys tags in
    let crs = crs keys in
    let a, b, c, d, e, f = affine tags in
    let shift =
      match key keys gt_raster_type with
      | None -> 0.
      | Some t when t = pixel_is_area -> 0.
      | Some t when t = pixel_is_point -> 0.5
      | Some t ->
          refuse "GeoKey %d: raster type %d is not PixelIsArea or PixelIsPoint"
            gt_raster_type t
    in
    (* Pixel point (r, c) is raster point (c - shift, r - shift). *)
    let origin =
      (c -. mul shift a -. mul shift b, f -. mul shift d -. mul shift e)
    in
    if not (finite_all [ a; b; c; d; e; f ]) then
      refuse "the map has a number that is not finite";
    if mul b d -. mul a e = 0. then refuse "the map is not invertible";
    Grid.v crs ~rows ~cols ~origin ~row_step:(b, e) ~col_step:(a, d)
  with
  | g -> Ok g
  | exception Refused e -> Error e

(* Nodata *)

let is_digit c = '0' <= c && c <= '9'

(* strtod's decimal syntax, or nan, inf, infinity with a sign. The digits go to
   [float_of_string], which is C's strtod; glibc's, Apple's and musl's round
   correctly, so they agree, as their [%.17g] printing in [text] does. *)
let number text =
  let s = String.lowercase_ascii text in
  let n = String.length s in
  let pos = ref 0 in
  let span ok =
    let start = !pos in
    while !pos < n && ok s.[!pos] do
      incr pos
    done;
    String.sub s start (!pos - start)
  in
  let sign () =
    if !pos < n && (s.[!pos] = '+' || s.[!pos] = '-') then (
      incr pos;
      String.make 1 s.[!pos - 1])
    else ""
  in
  let sgn = sign () in
  match String.sub s !pos (n - !pos) with
  | "nan" -> Some Float.nan
  | "inf" | "infinity" ->
      Some (if sgn = "-" then Float.neg_infinity else Float.infinity)
  | _ -> (
      let whole = span is_digit in
      let frac =
        if !pos < n && s.[!pos] = '.' then (
          incr pos;
          span is_digit)
        else ""
      in
      let exp =
        if !pos < n && s.[!pos] = 'e' then begin
          incr pos;
          let esgn = sign () in
          let d = span is_digit in
          if d = "" then None else Some ("e" ^ esgn ^ d)
        end
        else Some ""
      in
      let digits d = if d = "" then "0" else d in
      match exp with
      | Some exp when (whole <> "" || frac <> "") && !pos = n ->
          (* Written out in full: OCaml's syntax wants digits before the
             period. *)
          Some (float_of_string (sgn ^ digits whole ^ "." ^ digits frac ^ exp))
      | _ -> None)

let nodata tags =
  match find tags gdal_nodata with
  | None -> Ok None
  | Some (T.Ascii text) -> (
      let text = String.trim (List.hd (String.split_on_char '\000' text)) in
      match number text with
      | Some x -> Ok (Some x)
      | None ->
          Error
            (Printf.sprintf "%s: %S is not a number" (named gdal_nodata) text))
  | Some _ -> Error (Printf.sprintf "%s is not ASCII" (named gdal_nodata))
  | exception Refused e -> Error e

(* Writing *)

let text x =
  if Float.is_nan x then "nan"
  else if Float.is_integer x && Float.abs x < 1e15 then Printf.sprintf "%.0f" x
  else
    let rec go p =
      let s = Printf.sprintf "%.*g" p x in
      if p >= 17 || float_of_string s = x then s else go (p + 1)
    in
    if Float.is_finite x then go 1 else if x > 0. then "inf" else "-inf"

let tags ?nodata (g : Grid.t) =
  let (ox, oy), (rx, ry), (cx, cy) = (g.origin, g.row_step, g.col_step) in
  let map =
    if rx = 0. && ry < 0. && cx > 0. && cy = 0. then
      [
        (model_pixel_scale, T.Double [| cx; -.ry; 0. |]);
        (model_tiepoint, T.Double [| 0.; 0.; 0.; ox; oy; 0. |]);
      ]
    else
      [
        ( model_transformation,
          T.Double
            [| cx; rx; 0.; ox; cy; ry; 0.; oy; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 1. |]
        );
      ]
  in
  let model, key =
    match g.crs with
    | Crs.Lonlat -> (model_geographic, geodetic_crs)
    | Crs.Utm _ | Crs.Tm35fin -> (model_projected, projected_crs)
  in
  let dir =
    [|
      1;
      1;
      1;
      3;
      gt_model_type;
      0;
      1;
      model;
      gt_raster_type;
      0;
      1;
      pixel_is_area;
      key;
      0;
      1;
      Crs.epsg g.crs;
    |]
  in
  let nodata =
    match nodata with
    | None -> []
    | Some x -> [ (gdal_nodata, T.Ascii (text x)) ]
  in
  map @ [ (geo_key_directory, T.Short dir) ] @ nodata
