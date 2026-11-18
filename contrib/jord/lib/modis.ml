(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Products *)

type product = Mod13q1 | Mod09a1

let product_name = function Mod13q1 -> "MOD13Q1" | Mod09a1 -> "MOD09A1"

(* Bands *)

type band =
  | Ndvi
  | Evi
  | Vi_quality
  | Red
  | Nir
  | Blue
  | Green
  | Swir12
  | Swir16
  | Swir22
  | Qc
  | State
  | View_zenith
  | Sun_zenith
  | Relative_azimuth
  | Day_of_year
  | Reliability

(* MOD13 Table 1 and MOD09 Table 6, in order. *)
let bands = function
  | Mod13q1 ->
      [
        Ndvi; Evi; Vi_quality; Red; Nir; Blue; Swir22; View_zenith; Sun_zenith;
        Relative_azimuth; Day_of_year; Reliability;
      ]
  | Mod09a1 ->
      [
        Red; Nir; Blue; Green; Swir12; Swir16; Swir22; Qc; Sun_zenith;
        View_zenith; Relative_azimuth; State; Day_of_year;
      ]

(* The MODIS band of a reflectance layer. *)
let modis_band = function
  | Red -> Some 1
  | Nir -> Some 2
  | Blue -> Some 3
  | Green -> Some 4
  | Swir12 -> Some 5
  | Swir16 -> Some 6
  | Swir22 -> Some 7
  | _ -> None

(* The science data sets' names, as LP DAAC lists them. *)
let sds p b =
  match (p, b) with
  | Mod13q1, Ndvi -> Some "250m 16 days NDVI"
  | Mod13q1, Evi -> Some "250m 16 days EVI"
  | Mod13q1, Vi_quality -> Some "250m 16 days VI Quality"
  | Mod13q1, Red -> Some "250m 16 days red reflectance"
  | Mod13q1, Nir -> Some "250m 16 days NIR reflectance"
  | Mod13q1, Blue -> Some "250m 16 days blue reflectance"
  | Mod13q1, Swir22 -> Some "250m 16 days MIR reflectance"
  | Mod13q1, View_zenith -> Some "250m 16 days view zenith angle"
  | Mod13q1, Sun_zenith -> Some "250m 16 days sun zenith angle"
  | Mod13q1, Relative_azimuth -> Some "250m 16 days relative azimuth angle"
  | Mod13q1, Day_of_year -> Some "250m 16 days composite day of the year"
  | Mod13q1, Reliability -> Some "250m 16 days pixel reliability"
  | Mod09a1, (Red | Nir | Blue | Green | Swir12 | Swir16 | Swir22) ->
      Option.map (strf "sur_refl_b%02d") (modis_band b)
  | Mod09a1, Qc -> Some "sur_refl_qc_500m"
  | Mod09a1, State -> Some "sur_refl_state_500m"
  | Mod09a1, Sun_zenith -> Some "sur_refl_szen"
  | Mod09a1, View_zenith -> Some "sur_refl_vzen"
  | Mod09a1, Relative_azimuth -> Some "sur_refl_raz"
  | Mod09a1, Day_of_year -> Some "sur_refl_day_of_year"
  | _ -> None

let name_of fn p b =
  match sds p b with
  | Some n -> n
  | None ->
      invalid_arg
        (strf "Jord.Modis.%s: %s products have no such band" fn
           (product_name p))

let name p b = name_of "name" p b

(* STAC's electro-optical extension, "Common Band Names", its MODIS column. *)
let common_name = function
  | Red -> Some "red"
  | Nir -> Some "nir"
  | Blue -> Some "blue"
  | Green -> Some "green"
  | Swir16 -> Some "swir16"
  | Swir22 -> Some "swir22"
  | _ -> None

(* Unit, scale, fill and valid range of each layer (MOD13 Table 1, MOD09
   Table 6). *)
let table p b =
  let reflectance = (None, 0.0001) and angle = (Some "degree", 0.01) in
  match (p, b) with
  | Mod13q1, (Ndvi | Evi) -> (reflectance, -3000., (-2000, 10000))
  | Mod13q1, Vi_quality -> ((None, 1.), 65535., (0, 65534))
  | Mod13q1, (Red | Nir | Blue | Swir22) -> (reflectance, -1000., (0, 10000))
  | Mod13q1, (View_zenith | Sun_zenith) -> (angle, -10000., (0, 18000))
  | Mod13q1, Relative_azimuth -> (angle, -4000., (-18000, 18000))
  | Mod13q1, Day_of_year -> ((None, 1.), -1., (1, 366))
  | Mod13q1, Reliability -> ((None, 1.), -1., (0, 3))
  (* MOD09 Table 6 states NA; LP DAAC's Variables table states this. *)
  | Mod09a1, Qc -> ((None, 1.), 4294967295., (0, 4294966531))
  | Mod09a1, State -> ((None, 1.), 65535., (0, 57343))
  | Mod09a1, (View_zenith | Sun_zenith) -> (angle, 0., (0, 18000))
  | Mod09a1, Relative_azimuth -> (angle, 0., (-18000, 18000))
  | Mod09a1, Day_of_year -> ((None, 1.), 65535., (1, 366))
  | _ -> (reflectance, -28672., (-100, 16000))

(* Pixel reliability *)

module Reliability = struct
  type t = Fill | Good | Marginal | Snow_ice | Cloudy

  let code = function
    | Fill -> -1
    | Good -> 0
    | Marginal -> 1
    | Snow_ice -> 2
    | Cloudy -> 3

  (* MOD13 Table 4's summary and description. *)
  let row = function
    | Fill -> ("Fill/No Data", "Not Processed")
    | Good -> ("Good Data", "Use with confidence")
    | Marginal -> ("Marginal data", "Useful, but look at other QA information")
    | Snow_ice -> ("Snow/Ice", "Target covered with snow/ice")
    | Cloudy -> ("Cloudy", "Target not visible, covered with cloud")

  let classes =
    List.map
      (fun r ->
        let title, description = row r in
        {
          Band.value = code r;
          name = None;
          title = Some title;
          description = Some description;
          nodata = r = Fill;
        })
      [ Fill; Good; Marginal; Snow_ice; Cloudy ]

  let mem rs x =
    Dn.mem "Jord.Modis.Reliability.mem"
      (List.map (fun r -> Float.of_int (code r)) rs)
      x
end

let band p b =
  let name = name_of "band" p b in
  let (unit, scale), nodata, _ = table p b in
  let kind : Band.kind =
    match b with
    | Vi_quality | Qc | State -> Bitfield
    | Reliability -> Classes Reliability.classes
    | _ -> Measurement
  in
  {
    Band.name = Some name;
    common_name = common_name b;
    center_wavelength = None;
    full_width_half_max = None;
    unit;
    kind;
    scale;
    offset = 0.;
    nodata = Some nodata;
    saturated = None;
  }

let valid_range p b =
  ignore (name_of "valid_range" p b);
  let _, _, range = table p b in
  range

(* Grid *)

(* MODLAND's tile calculator, tilemap3.c's SIN entry (lines 103-106 of
   release 4.0): the grid's upper left corner is (-u, u / 2), and 36 × 18
   tiles of side u / 18 cover it. *)
let ulx = 20015109.354

let tile p ~h ~v =
  if h < 0 || h > 35 || v < 0 || v > 17 then
    invalid_arg (strf "Jord.Modis.tile: (%d, %d) is not a tile" h v);
  let n = match p with Mod13q1 -> 4800 | Mod09a1 -> 2400 in
  let d = ulx /. Float.of_int (18 * n) in
  let t = ulx /. 18. in
  let x = t *. Float.of_int (h - 18) and y = t *. Float.of_int (9 - v) in
  Grid.v Crs.modis ~rows:n ~cols:n ~origin:(x, y) ~row_step:(0., -.d)
    ~col_step:(d, 0.)

(* Quality *)

type modland =
  | Good_quality
  | Other_quality
  | Not_produced_cloud
  | Not_produced_other

let modland_value = function
  | Good_quality -> 0
  | Other_quality -> 1
  | Not_produced_cloud -> 2
  | Not_produced_other -> 3

type land_water =
  | Shallow_ocean
  | Land
  | Shoreline
  | Shallow_inland_water
  | Ephemeral_water
  | Deep_inland_water
  | Moderate_ocean
  | Deep_ocean

let land_water_value = function
  | Shallow_ocean -> 0
  | Land -> 1
  | Shoreline -> 2
  | Shallow_inland_water -> 3
  | Ephemeral_water -> 4
  | Deep_inland_water -> 5
  | Moderate_ocean -> 6
  | Deep_ocean -> 7

let fn name = "Jord.Modis." ^ name

(* Where the field of [length] bits at [offset] holds one of [values]. *)
let field name ~offset ~length value vs x =
  Dn.field (fn name) ~offset ~length (List.map value vs) x

let all_set name width x = Dn.all (fn name) (List.init width Fun.id) x

module Vi_quality = struct
  let modland ms x =
    field "Vi_quality.modland" ~offset:0 ~length:2 modland_value ms x

  let usefulness us x =
    field "Vi_quality.usefulness" ~offset:2 ~length:4 Fun.id us x

  type aerosol = Climatology | Low | Intermediate | High

  let aerosol_value = function
    | Climatology -> 0
    | Low -> 1
    | Intermediate -> 2
    | High -> 3

  let aerosol as_ x =
    field "Vi_quality.aerosol" ~offset:6 ~length:2 aerosol_value as_ x

  type flag = Adjacent_cloud | Brdf_corrected | Mixed_clouds | Snow_ice | Shadow

  let bit = function
    | Adjacent_cloud -> 8
    | Brdf_corrected -> 9
    | Mixed_clouds -> 10
    | Snow_ice -> 14
    | Shadow -> 15

  let any fs x = Dn.any (fn "Vi_quality.any") (List.map bit fs) x

  let land_water ls x =
    field "Vi_quality.land_water" ~offset:11 ~length:3 land_water_value ls x

  let fill x = all_set "Vi_quality.fill" 16 x
end

module Qc = struct
  let modland ms x = field "Qc.modland" ~offset:0 ~length:2 modland_value ms x

  type quality =
    | Highest
    | Noisy_detector
    | Dead_detector
    | Sun_zenith_86
    | Sun_zenith_85
    | Missing_input
    | Internal_constant
    | Out_of_bounds
    | L1b_faulty
    | Not_processed

  let quality_value = function
    | Highest -> 0
    | Noisy_detector -> 7
    | Dead_detector -> 8
    | Sun_zenith_86 -> 9
    | Sun_zenith_85 -> 10
    | Missing_input -> 11
    | Internal_constant -> 12
    | Out_of_bounds -> 13
    | L1b_faulty -> 14
    | Not_processed -> 15

  let quality b qs x =
    match modis_band b with
    | Some k ->
        field "Qc.quality" ~offset:(2 + (4 * (k - 1))) ~length:4 quality_value
          qs x
    | None -> invalid_arg (fn "Qc.quality: not a reflectance band of MOD09A1")

  type flag = Atmospheric_correction | Adjacency_correction

  let bit = function Atmospheric_correction -> 30 | Adjacency_correction -> 31
  let any fs x = Dn.any (fn "Qc.any") (List.map bit fs) x
  let fill x = all_set "Qc.fill" 32 x
end

module State = struct
  type cloud = Clear | Cloudy | Mixed | Not_set

  let cloud_value = function
    | Clear -> 0
    | Cloudy -> 1
    | Mixed -> 2
    | Not_set -> 3

  let cloud cs x = field "State.cloud" ~offset:0 ~length:2 cloud_value cs x

  let land_water ls x =
    field "State.land_water" ~offset:3 ~length:3 land_water_value ls x

  type aerosol = Climatology | Low | Average | High

  let aerosol_value = function
    | Climatology -> 0
    | Low -> 1
    | Average -> 2
    | High -> 3

  let aerosol as_ x =
    field "State.aerosol" ~offset:6 ~length:2 aerosol_value as_ x

  type cirrus = No_cirrus | Small_cirrus | Average_cirrus | High_cirrus

  let cirrus_value = function
    | No_cirrus -> 0
    | Small_cirrus -> 1
    | Average_cirrus -> 2
    | High_cirrus -> 3

  let cirrus cs x = field "State.cirrus" ~offset:8 ~length:2 cirrus_value cs x

  type flag =
    | Cloud_shadow
    | Internal_cloud
    | Fire
    | Snow_ice
    | Adjacent_cloud
    | Salt_pan
    | Internal_snow

  let bit = function
    | Cloud_shadow -> 2
    | Internal_cloud -> 10
    | Fire -> 11
    | Snow_ice -> 12
    | Adjacent_cloud -> 13
    | Salt_pan -> 14
    | Internal_snow -> 15

  let any fs x = Dn.any (fn "State.any") (List.map bit fs) x
  let fill x = all_set "State.fill" 16 x
end
