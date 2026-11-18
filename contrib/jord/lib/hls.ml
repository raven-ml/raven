(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Products *)

type product = L30 | S30

let product_name = function L30 -> "L30" | S30 -> "S30"

(* Bands *)

type band =
  | Coastal
  | Blue
  | Green
  | Red
  | Red_edge1
  | Red_edge2
  | Red_edge3
  | Nir_broad
  | Nir_narrow
  | Swir16
  | Swir22
  | Water_vapour
  | Cirrus
  | Lwir11
  | Lwir12
  | Fmask
  | Sun_zenith
  | Sun_azimuth
  | View_zenith
  | View_azimuth

let angles = [ Sun_zenith; Sun_azimuth; View_zenith; View_azimuth ]

(* Tables 6 and 7, in order, then Table 8. *)
let bands = function
  | L30 ->
      [
        Coastal; Blue; Green; Red; Nir_narrow; Swir16; Swir22; Cirrus; Lwir11;
        Lwir12; Fmask;
      ]
      @ angles
  | S30 ->
      [
        Coastal; Blue; Green; Red; Red_edge1; Red_edge2; Red_edge3; Nir_broad;
        Nir_narrow; Water_vapour; Cirrus; Swir16; Swir22; Fmask;
      ]
      @ angles

(* Table 3's HLS band codes. *)
let code p b =
  match (p, b) with
  | _, Coastal -> Some "B01"
  | _, Blue -> Some "B02"
  | _, Green -> Some "B03"
  | _, Red -> Some "B04"
  | L30, Nir_narrow -> Some "B05"
  | L30, Swir16 -> Some "B06"
  | L30, Swir22 -> Some "B07"
  | L30, Cirrus -> Some "B09"
  | L30, Lwir11 -> Some "B10"
  | L30, Lwir12 -> Some "B11"
  | S30, Red_edge1 -> Some "B05"
  | S30, Red_edge2 -> Some "B06"
  | S30, Red_edge3 -> Some "B07"
  | S30, Nir_broad -> Some "B08"
  | S30, Nir_narrow -> Some "B8A"
  | S30, Water_vapour -> Some "B09"
  | S30, Cirrus -> Some "B10"
  | S30, Swir16 -> Some "B11"
  | S30, Swir22 -> Some "B12"
  | _, Fmask -> Some "Fmask"
  | _, Sun_zenith -> Some "SZA"
  | _, Sun_azimuth -> Some "SAA"
  | _, View_zenith -> Some "VZA"
  | _, View_azimuth -> Some "VAA"
  | _ -> None

let name_of fn p b =
  match code p b with
  | Some n -> n
  | None ->
      invalid_arg
        (strf "Jord.Hls.%s: %s products have no such band" fn (product_name p))

let name p b = name_of "name" p b

(* STAC's electro-optical extension, "Common Band Names". *)
let common_name = function
  | Coastal -> Some "coastal"
  | Blue -> Some "blue"
  | Green -> Some "green"
  | Red -> Some "red"
  | Red_edge1 -> Some "rededge071"
  | Red_edge2 -> Some "rededge075"
  | Red_edge3 -> Some "rededge078"
  | Nir_broad -> Some "nir"
  | Nir_narrow -> Some "nir08"
  | Water_vapour -> Some "nir09"
  | Cirrus -> Some "cirrus"
  | Swir16 -> Some "swir16"
  | Swir22 -> Some "swir22"
  | Lwir11 -> Some "lwir11"
  | Lwir12 -> Some "lwir12"
  | Fmask | Sun_zenith | Sun_azimuth | View_zenith | View_azimuth -> None

(* Unit, scale and fill of each layer (Tables 6 to 8). *)
let table = function
  | Lwir11 | Lwir12 -> (Some "degree_Celsius", 0.01, -9999.)
  | Fmask -> (None, 1., 255.)
  | Sun_zenith | Sun_azimuth | View_zenith | View_azimuth ->
      (Some "degree", 0.01, 40000.)
  | _ -> (None, 0.0001, -9999.)

let band p b =
  let name = name_of "band" p b in
  let unit, scale, nodata = table b in
  {
    Band.name = Some name;
    common_name = common_name b;
    center_wavelength = None;
    full_width_half_max = None;
    unit;
    kind = (if b = Fmask then Bitfield else Measurement);
    scale;
    offset = 0.;
    nodata = Some nodata;
    saturated = None;
  }

(* Quality bits *)

module Fmask = struct
  let fn name = "Jord.Hls.Fmask." ^ name

  type flag = Cloud | Adjacent | Cloud_shadow | Snow_ice | Water

  let bit = function
    | Cloud -> 1
    | Adjacent -> 2
    | Cloud_shadow -> 3
    | Snow_ice -> 4
    | Water -> 5

  let any fs x = Dn.any (fn "any") (List.map bit fs) x

  type aerosol = Climatology | Low | Medium | High

  let value = function Climatology -> 0 | Low -> 1 | Medium -> 2 | High -> 3

  let aerosol ls x =
    Dn.field (fn "aerosol") ~offset:6 ~length:2 (List.map value ls) x

  let fill x = Dn.field (fn "fill") ~offset:0 ~length:8 [ 255 ] x
end

(* Identity *)

type id = { product : product; tile : string; stamp : float }

let digits s = s <> "" && String.for_all (fun c -> c >= '0' && c <= '9') s
let upper s = s <> "" && String.for_all (fun c -> c >= 'A' && c <= 'Z') s

let tile_name t =
  String.length t = 5 && digits (String.sub t 0 2) && upper (String.sub t 2 3)

(* yyyydddThhmmss, a year's day from 001 and a UTC time. *)
let stamp s =
  let int i n = int_of_string (String.sub s i n) in
  if
    not
      (String.length s = 14
      && digits (String.sub s 0 7)
      && s.[7] = 'T'
      && digits (String.sub s 8 6))
  then None
  else
    let y = int 0 4 and d = int 4 3 in
    let hh = int 8 2 and mm = int 10 2 and ss = int 12 2 in
    let days = Time.day_of_year (Time.of_date (y, 12, 31)) in
    if d < 1 || d > days || hh > 23 || mm > 59 || ss > 59 then None
    else
      Some
        (Time.of_date (y, 1, 1)
        +. Float.of_int (((d - 1) * 86400) + (hh * 3600) + (mm * 60) + ss))

let id s =
  let error () = Error (strf "%S is not an HLS v2.0 identifier" s) in
  match String.split_on_char '.' s with
  | [ "HLS"; p; t; when_; ("v2" | "V2"); "0" ] -> (
      let product =
        match p with "L30" -> Some L30 | "S30" -> Some S30 | _ -> None
      in
      let tile =
        if String.length t = 6 && t.[0] = 'T' && tile_name (String.sub t 1 5)
        then Some (String.sub t 1 5)
        else None
      in
      match (product, tile, stamp when_) with
      | Some product, Some tile, Some stamp -> Ok { product; tile; stamp }
      | _ -> error ())
  | _ -> error ()
