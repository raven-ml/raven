(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let ( let* ) = Result.bind

(* Bands *)

type band =
  | B01
  | B02
  | B03
  | B04
  | B05
  | B06
  | B07
  | B08
  | B8A
  | B09
  | B10
  | B11
  | B12

let bands =
  [ B01; B02; B03; B04; B05; B06; B07; B08; B8A; B09; B10; B11; B12 ]

let index = function
  | B01 -> 0
  | B02 -> 1
  | B03 -> 2
  | B04 -> 3
  | B05 -> 4
  | B06 -> 5
  | B07 -> 6
  | B08 -> 7
  | B8A -> 8
  | B09 -> 9
  | B10 -> 10
  | B11 -> 11
  | B12 -> 12

let name = function
  | B01 -> "B01"
  | B02 -> "B02"
  | B03 -> "B03"
  | B04 -> "B04"
  | B05 -> "B05"
  | B06 -> "B06"
  | B07 -> "B07"
  | B08 -> "B08"
  | B8A -> "B8A"
  | B09 -> "B09"
  | B10 -> "B10"
  | B11 -> "B11"
  | B12 -> "B12"

let resolution = function
  | B02 | B03 | B04 | B08 -> 10
  | B05 | B06 | B07 | B8A | B11 | B12 -> 20
  | B01 | B09 | B10 -> 60

(* Reflectance *)

type wavelength = { central : float; min : float; max : float }

type metadata = {
  baseline : string;
  quantification : float;
  offsets : (band * int) list;
  nodata : int;
  saturated : int;
  wavelengths : (band * wavelength) list;
}

(* The first processing baseline whose products shift their range by
   BOA_ADD_OFFSET. *)
let shifted = (4, 0)

let digits s = s <> "" && String.for_all (fun c -> c >= '0' && c <= '9') s

let baseline s =
  let two d = String.length d = 2 && digits d in
  match String.split_on_char '.' s with
  | [ a; b ] when two a && two b -> Ok (int_of_string a, int_of_string b)
  | _ -> Error (strf "processing baseline %S is not NN.NN" s)

let offset m b =
  let* pb = baseline m.baseline in
  match List.assoc_opt b m.offsets with
  | Some o -> Ok o
  | None when pb < shifted -> Ok 0
  | None ->
      Error
        (strf "baseline %s states offsets; none is given for %s"
           m.baseline (name b))

let reflectance m b =
  let q = m.quantification in
  if not (Float.is_finite q && q > 0. && Float.is_finite (1. /. q)) then
    invalid_arg
      (strf "Jord.Sentinel2.reflectance: quantification %g is not positive" q);
  let* offset = offset m b in
  let w = List.assoc_opt b m.wavelengths in
  let um f = Option.map (fun w -> f w /. 1000.) w in
  Ok
    {
      Band.name = Some (name b);
      common_name = None;
      center_wavelength = um (fun w -> w.central);
      full_width_half_max = um (fun w -> w.max -. w.min);
      unit = None;
      kind = Measurement;
      scale = 1. /. q;
      offset = float offset /. q;
      nodata = Some (float m.nodata);
      saturated = Some (float m.saturated);
    }

(* Scene classification *)

module Scl = struct
  type t =
    | No_data
    | Saturated_or_defective
    | Cast_shadows
    | Cloud_shadows
    | Vegetation
    | Not_vegetated
    | Water
    | Unclassified
    | Cloud_medium_probability
    | Cloud_high_probability
    | Thin_cirrus
    | Snow_or_ice

  let code = function
    | No_data -> 0
    | Saturated_or_defective -> 1
    | Cast_shadows -> 2
    | Cloud_shadows -> 3
    | Vegetation -> 4
    | Not_vegetated -> 5
    | Water -> 6
    | Unclassified -> 7
    | Cloud_medium_probability -> 8
    | Cloud_high_probability -> 9
    | Thin_cirrus -> 10
    | Snow_or_ice -> 11

  let mem cs x =
    Dn.mem "Jord.Sentinel2.Scl.mem" (List.map (fun c -> float (code c)) cs) x
end

(* Identity *)

type key = { platform : string; tile : string; sensing_start : string }

let upper s = s <> "" && String.for_all (fun c -> c >= 'A' && c <= 'Z') s

(* A UTC time as the naming convention writes it, yyyymmddThhmmss. *)
let stamp s =
  String.length s = 15
  && digits (String.sub s 0 8)
  && s.[8] = 'T'
  && digits (String.sub s 9 6)

(* The fixed fields of a datastrip identifier, mission (3), file class (4),
   file type (10), processing centre (4) and processing time (15), joined by
   underscores, as in S2A_OPER_MSI_L2A_DS_MSFT_20240629T161811. The fields
   after them each start with a letter. These are the positions of the
   identifiers catalogues give. ESA's naming text in its Products
   Specification Document is not followed. *)
let fixed = 40

let tile_name t =
  String.length t = 5 && digits (String.sub t 0 2) && upper (String.sub t 2 3)

let datastrip id =
  String.length id > fixed
  && String.starts_with ~prefix:"S2" id
  && List.for_all (fun i -> id.[i] = '_') [ 3; 8; 19; 24; fixed ]
  && stamp (String.sub id 25 15)

let key ~tile id =
  if not (tile_name tile) then
    Error (strf "tile %S is not two digits and three letters" tile)
  else if not (datastrip id) then
    Error (strf "%S is not a Sentinel-2 datastrip identifier" id)
  else
    let rest = String.sub id (fixed + 1) (String.length id - fixed - 1) in
    let sensing f = String.length f = 16 && f.[0] = 'S' in
    match List.find_opt sensing (String.split_on_char '_' rest) with
    | Some f when stamp (String.sub f 1 15) ->
        let platform = String.sub id 0 3 in
        Ok { platform; tile; sensing_start = String.sub f 1 15 }
    | _ -> Error (strf "%S states no sensing start" id)
