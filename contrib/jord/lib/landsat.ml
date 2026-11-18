(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Satellites and sensors *)

type satellite = Landsat_4 | Landsat_5 | Landsat_7 | Landsat_8 | Landsat_9
type sensor = Tm | Etm_plus | Oli_tirs

let sensor = function
  | Landsat_4 | Landsat_5 -> Tm
  | Landsat_7 -> Etm_plus
  | Landsat_8 | Landsat_9 -> Oli_tirs

let satellite_name = function
  | Landsat_4 -> "Landsat 4"
  | Landsat_5 -> "Landsat 5"
  | Landsat_7 -> "Landsat 7"
  | Landsat_8 -> "Landsat 8"
  | Landsat_9 -> "Landsat 9"

let satellite id =
  let field =
    if String.length id >= 4 && (String.length id = 4 || id.[4] = '_') then
      String.sub id 0 4
    else ""
  in
  match field with
  | "LT04" -> Ok Landsat_4
  | "LT05" -> Ok Landsat_5
  | "LE07" -> Ok Landsat_7
  | "LC08" | "LO08" -> Ok Landsat_8
  | "LC09" | "LO09" -> Ok Landsat_9
  | _ -> Error (strf "%S is not a Landsat product identifier" id)

(* Bands *)

type band =
  | Coastal
  | Blue
  | Green
  | Red
  | Nir
  | Swir16
  | Swir22
  | Lwir
  | St_qa
  | St_trad
  | St_urad
  | St_drad
  | St_atran
  | St_emis
  | St_emsd
  | St_cdist
  | Sr_atmos_opacity
  | Qa_pixel
  | Qa_radsat
  | Sr_qa_aerosol
  | Sr_cloud_qa

(* LSDS-1619 Table 6-1 and LSDS-1618 Table 5-1, in order. *)
let bands s =
  match sensor s with
  | Oli_tirs ->
      [
        Coastal; Blue; Green; Red; Nir; Swir16; Swir22; Lwir; Qa_pixel;
        Sr_qa_aerosol; Qa_radsat; St_qa; St_trad; St_urad; St_drad; St_atran;
        St_emis; St_emsd; St_cdist;
      ]
  | Tm | Etm_plus ->
      [
        Blue; Green; Red; Nir; Swir16; Lwir; Swir22; Sr_atmos_opacity;
        Qa_pixel; Qa_radsat; Sr_cloud_qa; St_qa; St_trad; St_urad; St_drad;
        St_atran; St_emis; St_emsd; St_cdist;
      ]

(* The sensor's number of a reflective or thermal band. *)
let number s b =
  match (sensor s, b) with
  | Oli_tirs, Coastal -> Some 1
  | Oli_tirs, Blue -> Some 2
  | Oli_tirs, Green -> Some 3
  | Oli_tirs, Red -> Some 4
  | Oli_tirs, Nir -> Some 5
  | Oli_tirs, Swir16 -> Some 6
  | Oli_tirs, Swir22 -> Some 7
  | Oli_tirs, Lwir -> Some 10
  | (Tm | Etm_plus), Blue -> Some 1
  | (Tm | Etm_plus), Green -> Some 2
  | (Tm | Etm_plus), Red -> Some 3
  | (Tm | Etm_plus), Nir -> Some 4
  | (Tm | Etm_plus), Swir16 -> Some 5
  | (Tm | Etm_plus), Lwir -> Some 6
  | (Tm | Etm_plus), Swir22 -> Some 7
  | _ -> None

let check fn s b =
  if not (List.mem b (bands s)) then
    invalid_arg
      (strf "Jord.Landsat.%s: %s products have no such band" fn
         (satellite_name s))

let name_of fn s b =
  check fn s b;
  match (b, number s b) with
  | Lwir, Some n -> strf "ST_B%d" n
  | _, Some n -> strf "SR_B%d" n
  | St_qa, None -> "ST_QA"
  | St_trad, None -> "ST_TRAD"
  | St_urad, None -> "ST_URAD"
  | St_drad, None -> "ST_DRAD"
  | St_atran, None -> "ST_ATRAN"
  | St_emis, None -> "ST_EMIS"
  | St_emsd, None -> "ST_EMSD"
  | St_cdist, None -> "ST_CDIST"
  | Sr_atmos_opacity, None -> "SR_ATMOS_OPACITY"
  | Qa_pixel, None -> "QA_PIXEL"
  | Qa_radsat, None -> "QA_RADSAT"
  | Sr_qa_aerosol, None -> "SR_QA_AEROSOL"
  | Sr_cloud_qa, None -> "SR_CLOUD_QA"
  | (Coastal | Blue | Green | Red | Nir | Swir16 | Swir22 | Lwir), None ->
      assert false (* [check] refused it. *)

let name s b = name_of "name" s b

(* STAC's electro-optical extension, "Common Band Names". *)
let common_name s b =
  match (sensor s, b) with
  | _, Coastal -> Some "coastal"
  | _, Blue -> Some "blue"
  | _, Green -> Some "green"
  | _, Red -> Some "red"
  | Oli_tirs, Nir -> Some "nir08"
  | (Tm | Etm_plus), Nir -> Some "nir"
  | _, Swir16 -> Some "swir16"
  | _, Swir22 -> Some "swir22"
  | Oli_tirs, Lwir -> Some "lwir11"
  | (Tm | Etm_plus), Lwir -> Some "lwir"
  | _ -> None

let radiance = "watt/steradian/square_meter/micrometer"

(* Unit, scale, offset and fill of each band (the tables' NA is none). *)
let table = function
  | Coastal | Blue | Green | Red | Nir | Swir16 | Swir22 ->
      (None, 2.75e-5, -0.2, Some 0.)
  | Lwir -> (Some "kelvin", 0.00341802, 149., Some 0.)
  | St_qa -> (Some "kelvin", 0.01, 0., Some (-9999.))
  | St_trad | St_urad | St_drad -> (Some radiance, 0.001, 0., Some (-9999.))
  | St_atran | St_emis | St_emsd -> (None, 0.0001, 0., Some (-9999.))
  | St_cdist -> (Some "kilometer", 0.01, 0., Some (-9999.))
  | Sr_atmos_opacity -> (None, 0.001, 0., Some (-9999.))
  | Qa_pixel | Sr_qa_aerosol -> (None, 1., 0., Some 1.)
  | Qa_radsat -> (None, 1., 0., None)
  | Sr_cloud_qa -> (None, 1., 0., Some 0.)

(* Each satellite's centre wavelength and full width at half maximum, in
   micrometres, of the USGS's band-average relative spectral responses
   (landsat.usgs.gov/spectral-characteristics-viewer, files
   L<n>_<sensor>_RSR.xlsx and HANDBOOK_LandsatThermalRSRs.spectral.dat.xlsx):
   Landsat 9's and Landsat 8's TIRS from the files' stated 50% points, the
   others from the responses' outermost crossings of half their peak,
   interpolated linearly (test/landsat_rsr.dat holds the samples); written
   to 0.1 nm. *)
let spectral s b =
  match (s, b) with
  | Landsat_4, Blue -> Some (0.4849, 0.0662)
  | Landsat_4, Green -> Some (0.5691, 0.0806)
  | Landsat_4, Red -> Some (0.6587, 0.0687)
  | Landsat_4, Nir -> Some (0.8406, 0.1291)
  | Landsat_4, Swir16 -> Some (1.6781, 0.2170)
  | Landsat_4, Lwir -> Some (11.0823, 1.3164)
  | Landsat_4, Swir22 -> Some (2.2223, 0.2502)
  | Landsat_5, Blue -> Some (0.4851, 0.0654)
  | Landsat_5, Green -> Some (0.5686, 0.0813)
  | Landsat_5, Red -> Some (0.6598, 0.0668)
  | Landsat_5, Nir -> Some (0.8404, 0.1281)
  | Landsat_5, Swir16 -> Some (1.6758, 0.2167)
  | Landsat_5, Lwir -> Some (11.4393, 1.9792)
  | Landsat_5, Swir22 -> Some (2.2231, 0.2519)
  | Landsat_7, Blue -> Some (0.4776, 0.0726)
  | Landsat_7, Green -> Some (0.5600, 0.0814)
  | Landsat_7, Red -> Some (0.6613, 0.0614)
  | Landsat_7, Nir -> Some (0.8348, 0.1264)
  | Landsat_7, Swir16 -> Some (1.6476, 0.2011)
  | Landsat_7, Lwir -> Some (11.3384, 2.0506)
  | Landsat_7, Swir22 -> Some (2.2050, 0.2812)
  | Landsat_8, Coastal -> Some (0.4429, 0.0160)
  | Landsat_8, Blue -> Some (0.4821, 0.0601)
  | Landsat_8, Green -> Some (0.5615, 0.0574)
  | Landsat_8, Red -> Some (0.6546, 0.0375)
  | Landsat_8, Nir -> Some (0.8646, 0.0282)
  | Landsat_8, Swir16 -> Some (1.6088, 0.0847)
  | Landsat_8, Swir22 -> Some (2.2007, 0.1867)
  | Landsat_8, Lwir -> Some (10.8960, 0.5880)
  | Landsat_9, Coastal -> Some (0.4428, 0.0155)
  | Landsat_9, Blue -> Some (0.4819, 0.0599)
  | Landsat_9, Green -> Some (0.5610, 0.0565)
  | Landsat_9, Red -> Some (0.6543, 0.0369)
  | Landsat_9, Nir -> Some (0.8646, 0.0288)
  | Landsat_9, Swir16 -> Some (1.6082, 0.0861)
  | Landsat_9, Swir22 -> Some (2.2002, 0.1894)
  | Landsat_9, Lwir -> Some (10.8250, 0.7500)
  | _ -> None

let band s b =
  let name = name_of "band" s b in
  let unit, scale, offset, nodata = table b in
  let kind : Band.kind =
    match b with
    | Qa_pixel | Qa_radsat | Sr_qa_aerosol | Sr_cloud_qa -> Bitfield
    | _ -> Measurement
  in
  {
    Band.name = Some name;
    common_name = common_name s b;
    center_wavelength = Option.map fst (spectral s b);
    full_width_half_max = Option.map snd (spectral s b);
    unit;
    kind;
    scale;
    offset;
    nodata;
    saturated = None;
  }

let valid_range s b =
  check "valid_range" s b;
  match (sensor s, b) with
  | _, (Coastal | Blue | Green | Red | Nir | Swir16 | Swir22) -> (7273, 43636)
  | _, Lwir -> (293, 61440)
  | _, St_qa -> (0, 32767)
  | _, St_trad -> (0, 22000)
  | _, (St_urad | St_drad) -> (0, 28000)
  | _, (St_atran | St_emis | St_emsd) -> (0, 10000)
  | _, St_cdist -> (0, 24000)
  | _, Sr_atmos_opacity -> (0, 1000)
  | Oli_tirs, Qa_pixel -> (21824, 65534)
  | (Tm | Etm_plus), Qa_pixel -> (5440, 16382)
  | _, Sr_qa_aerosol -> (1, 255)
  | _, Qa_radsat -> (0, 3829)
  | _, Sr_cloud_qa -> (1, 63)

(* Bits *)

let fn name = "Jord.Landsat." ^ name
let any_bit name bits x = Dn.any (fn name) bits x

(* Where the two-bit field at [offset] of [x] holds one of [values]. *)
let field name offset values x = Dn.field (fn name) ~offset ~length:2 values x

module Qa_pixel = struct
  type flag =
    | Fill
    | Dilated_cloud
    | Cirrus
    | Cloud
    | Cloud_shadow
    | Snow
    | Clear
    | Water

  let bit = function
    | Fill -> 0
    | Dilated_cloud -> 1
    | Cirrus -> 2
    | Cloud -> 3
    | Cloud_shadow -> 4
    | Snow -> 5
    | Clear -> 6
    | Water -> 7

  let any fs x = any_bit "Qa_pixel.any" (List.map bit fs) x
  let fill x = any_bit "Qa_pixel.fill" [ bit Fill ] x

  type confidence =
    | Cloud_confidence
    | Cloud_shadow_confidence
    | Snow_ice_confidence
    | Cirrus_confidence

  type level = Unset | Low | Medium | High

  let offset = function
    | Cloud_confidence -> 8
    | Cloud_shadow_confidence -> 10
    | Snow_ice_confidence -> 12
    | Cirrus_confidence -> 14

  let value = function Unset -> 0 | Low -> 1 | Medium -> 2 | High -> 3

  let confidence c ls x =
    field "Qa_pixel.confidence" (offset c) (List.map value ls) x
end

module Qa_radsat = struct
  let saturated s b x =
    let name = "Qa_radsat.saturated" in
    let bits =
      match (sensor s, number s b) with
      | Oli_tirs, Some 10 ->
          invalid_arg
            (strf "%s: QA_RADSAT states no saturation of TIRS" (fn name))
      | Etm_plus, Some 6 -> [ 5; 8 ]
      | _, Some n -> [ n - 1 ]
      | _, None ->
          invalid_arg
            (strf "%s: no reflective or thermal band of %s" (fn name)
               (satellite_name s))
    in
    (* Saturated where every bit is set: ETM+'s temperature where both
       gains are. *)
    Dn.all (fn name) bits x

  let occluded x = any_bit "Qa_radsat.occluded" [ 11 ] x
  let dropped x = any_bit "Qa_radsat.dropped" [ 9 ] x
end

module Sr_qa_aerosol = struct
  type flag = Fill | Valid_retrieval | Water | Interpolated

  let bit = function
    | Fill -> 0
    | Valid_retrieval -> 1
    | Water -> 2
    | Interpolated -> 5

  let any fs x = any_bit "Sr_qa_aerosol.any" (List.map bit fs) x
  let fill x = any_bit "Sr_qa_aerosol.fill" [ bit Fill ] x

  type level = Climatology | Low | Medium | High

  let value = function Climatology -> 0 | Low -> 1 | Medium -> 2 | High -> 3
  let level ls x = field "Sr_qa_aerosol.level" 6 (List.map value ls) x
end

module Sr_cloud_qa = struct
  type flag = Ddv | Cloud | Cloud_shadow | Adjacent_cloud | Snow | Water

  let bit = function
    | Ddv -> 0
    | Cloud -> 1
    | Cloud_shadow -> 2
    | Adjacent_cloud -> 3
    | Snow -> 4
    | Water -> 5

  let any fs x = any_bit "Sr_cloud_qa.any" (List.map bit fs) x

  (* DN 0, every bit unset, read as the other bits are. *)
  let fill x =
    Nx.logical_not (any_bit "Sr_cloud_qa.fill" (List.init 64 Fun.id) x)
end
