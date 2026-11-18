(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord
open Landsat

let floats x = Nx.to_array (Nx.cast Nx.float64 x)
let bools x = Nx.to_array x

(* [x] rounded once to float32. *)
let single x = Int32.float_of_bits (Int32.bits_of_float x)

(* Every 16-bit DN, as uint16. *)
let every16 = Array.init 65536 Fun.id
let dns16 = Nx.create Nx.uint16 [| 65536 |] every16

let satellites = [ Landsat_4; Landsat_5; Landsat_7; Landsat_8; Landsat_9 ]

let satellite_t =
  Testable.make
    ~pp:(fun ppf s ->
      Format.pp_print_string ppf
        (match s with
        | Landsat_4 -> "Landsat 4"
        | Landsat_5 -> "Landsat 5"
        | Landsat_7 -> "Landsat 7"
        | Landsat_8 -> "Landsat 8"
        | Landsat_9 -> "Landsat 9"))
    ~equal:( = )

(* Satellites and sensors *)

(* LSDS-1619 Section 5 and LSDS-1618 Section 4, and the identifiers of
   Microsoft's Planetary Computer's items, which omit the processing
   date. *)
let identifiers =
  cases "a product identifier names its satellite and sensor"
    ~id:(fun (id, _, _) -> id)
    [
      ("LC08_L2SP_039037_20150728_20200318_02_T1", Landsat_8, Oli_tirs);
      ("LC09_L2SP_193024_20240629_02_T1", Landsat_9, Oli_tirs);
      ("LO08_L2SR_039037_20150728_20200318_02_T2", Landsat_8, Oli_tirs);
      ("LE07_L2SP_039037_20080728_20200319_02_T1", Landsat_7, Etm_plus);
      ("LT05_L2SP_192024_20000628_02_T1", Landsat_5, Tm);
      ("LT04_L2SP_039037_19880728_20200917_02_T1", Landsat_4, Tm);
      ("LT05", Landsat_5, Tm);
    ]
    (fun (id, s, sn) ->
      equal (result satellite_t string) (Ok s) (satellite id);
      is_true (sensor s = sn))

let bad_identifiers =
  cases "an identifier of no Level-2 sensor is an error"
    ~id:(Printf.sprintf "%S")
    [
      "";
      "LC0";
      "LC08L2SP";
      "lc08_L2SP_039037_20150728_20200318_02_T1";
      "LE08_L2SP_039037_20150728_20200318_02_T1";
      "LC07_L2SP_039037_20080728_20200319_02_T1";
      "LT06_L2SP_039037_19930728_20200319_02_T1";
      "LM05_L2SP_039037_19930728_20200319_02_T1";
      (* TIRS alone, which LSDS-1619 Section 3.2 cannot process. *)
      "LT08_L1GT_039037_20150728_20200318_02_T2";
      "LT09_L1GT_039037_20230728_20230729_02_T2";
      "S2A_MSIL2A_20240629T102021_R065_T33UUU";
    ]
    (fun id -> is_error (satellite id))

(* Bands *)

(* LSDS-1619 Table 6-1 and LSDS-1618 Table 5-1, row by row: name, unit,
   scale, offset, fill, valid range. *)
let oli_rows =
  [
    (Coastal, "SR_B1", None, 2.75e-5, -0.2, Some 0., (7273, 43636));
    (Blue, "SR_B2", None, 2.75e-5, -0.2, Some 0., (7273, 43636));
    (Green, "SR_B3", None, 2.75e-5, -0.2, Some 0., (7273, 43636));
    (Red, "SR_B4", None, 2.75e-5, -0.2, Some 0., (7273, 43636));
    (Nir, "SR_B5", None, 2.75e-5, -0.2, Some 0., (7273, 43636));
    (Swir16, "SR_B6", None, 2.75e-5, -0.2, Some 0., (7273, 43636));
    (Swir22, "SR_B7", None, 2.75e-5, -0.2, Some 0., (7273, 43636));
    (Lwir, "ST_B10", Some "kelvin", 0.00341802, 149., Some 0., (293, 61440));
    (Qa_pixel, "QA_PIXEL", None, 1., 0., Some 1., (21824, 65534));
    (Sr_qa_aerosol, "SR_QA_AEROSOL", None, 1., 0., Some 1., (1, 255));
    (Qa_radsat, "QA_RADSAT", None, 1., 0., None, (0, 3829));
  ]

let tm_rows =
  [
    (Blue, "SR_B1", None, 2.75e-5, -0.2, Some 0., (7273, 43636));
    (Green, "SR_B2", None, 2.75e-5, -0.2, Some 0., (7273, 43636));
    (Red, "SR_B3", None, 2.75e-5, -0.2, Some 0., (7273, 43636));
    (Nir, "SR_B4", None, 2.75e-5, -0.2, Some 0., (7273, 43636));
    (Swir16, "SR_B5", None, 2.75e-5, -0.2, Some 0., (7273, 43636));
    (Lwir, "ST_B6", Some "kelvin", 0.00341802, 149., Some 0., (293, 61440));
    (Swir22, "SR_B7", None, 2.75e-5, -0.2, Some 0., (7273, 43636));
    (Sr_atmos_opacity, "SR_ATMOS_OPACITY", None, 0.001, 0., Some (-9999.),
     (0, 1000));
    (Qa_pixel, "QA_PIXEL", None, 1., 0., Some 1., (5440, 16382));
    (Qa_radsat, "QA_RADSAT", None, 1., 0., None, (0, 3829));
    (Sr_cloud_qa, "SR_CLOUD_QA", None, 1., 0., Some 0., (1, 63));
  ]

let radiance = Some "watt/steradian/square_meter/micrometer"

(* The surface temperature's intermediate bands, alike in both tables. *)
let st_rows =
  [
    (St_qa, "ST_QA", Some "kelvin", 0.01, 0., Some (-9999.), (0, 32767));
    (St_trad, "ST_TRAD", radiance, 0.001, 0., Some (-9999.), (0, 22000));
    (St_urad, "ST_URAD", radiance, 0.001, 0., Some (-9999.), (0, 28000));
    (St_drad, "ST_DRAD", radiance, 0.001, 0., Some (-9999.), (0, 28000));
    (St_atran, "ST_ATRAN", None, 0.0001, 0., Some (-9999.), (0, 10000));
    (St_emis, "ST_EMIS", None, 0.0001, 0., Some (-9999.), (0, 10000));
    (St_emsd, "ST_EMSD", None, 0.0001, 0., Some (-9999.), (0, 10000));
    (St_cdist, "ST_CDIST", Some "kilometer", 0.01, 0., Some (-9999.),
     (0, 24000));
  ]

let rows s =
  match sensor s with
  | Oli_tirs -> oli_rows @ st_rows
  | Tm | Etm_plus -> tm_rows @ st_rows

let quality = function
  | Qa_pixel | Qa_radsat | Sr_qa_aerosol | Sr_cloud_qa -> true
  | _ -> false

(* STAC's electro-optical extension, "Common Band Names". *)
let common s b =
  match (sensor s, b) with
  | _, Coastal -> Some "coastal"
  | _, Blue -> Some "blue"
  | _, Green -> Some "green"
  | _, Red -> Some "red"
  | Oli_tirs, Nir -> Some "nir08"
  | _, Nir -> Some "nir"
  | _, Swir16 -> Some "swir16"
  | _, Swir22 -> Some "swir22"
  | Oli_tirs, Lwir -> Some "lwir11"
  | _, Lwir -> Some "lwir"
  | _ -> None

let band_t =
  Testable.make
    ~pp:(fun ppf (b : Band.t) ->
      let str = Option.value ~default:"-" in
      let opt = Option.fold ~none:"-" ~some:string_of_float in
      Format.fprintf ppf "{%s %s %s %s scale %.17g offset %.17g nodata %s %s}"
        (str b.name) (str b.common_name) (str b.unit)
        (match b.kind with
        | Measurement -> "measurement"
        | Bitfield -> "bits"
        | Classes l -> Printf.sprintf "%d classes" (List.length l))
        b.scale b.offset (opt b.nodata) (opt b.saturated))
    ~equal:(fun a b -> compare a b = 0)

let tables =
  cases "each band as its guide's table states it"
    ~id:(fun s -> Format.asprintf "%a" (Testable.pp satellite_t) s)
    satellites
    (fun s ->
      List.iter
        (fun (b, n, unit, scale, offset, nodata, range) ->
          let want =
            {
              Band.name = Some n;
              common_name = common s b;
              center_wavelength = None;
              full_width_half_max = None;
              unit;
              kind = (if quality b then Bitfield else Measurement);
              scale;
              offset;
              nodata;
              saturated = None;
            }
          in
          equal string n (name s b);
          let ours =
            {
              (band s b) with
              center_wavelength = None;
              full_width_half_max = None;
            }
          in
          equal band_t want ours;
          equal (pair int int) range (valid_range s b))
        (rows s))

(* The 50% points the USGS's RSR files state in their band summaries, in
   micrometres: L9_OLI2_RSR.xlsx and L9_TIRS2_RSR.xlsx ("lower 50% point",
   "upper 50% point"), L8_TIRS_RSR.xlsx (band 10), from
   landsat.usgs.gov/spectral-characteristics-viewer, fetched 2026-11-16. *)
let stated =
  [
    (Landsat_9, Coastal, 0.435055, 0.450566);
    (Landsat_9, Blue, 0.451955, 0.511823);
    (Landsat_9, Green, 0.532705, 0.589202);
    (Landsat_9, Red, 0.635882, 0.672761);
    (Landsat_9, Nir, 0.850258, 0.879027);
    (Landsat_9, Swir16, 1.565111, 1.651232);
    (Landsat_9, Swir22, 2.105439, 2.294874);
    (Landsat_9, Lwir, 10.45, 11.2);
    (Landsat_8, Lwir, 10.602, 11.19);
  ]

(* STAC's electro-optical extension's band range of each common name. *)
let eo_range = function
  | "coastal" -> (0.40, 0.45)
  | "blue" -> (0.45, 0.53)
  | "green" -> (0.51, 0.60)
  | "red" -> (0.62, 0.69)
  | "nir" -> (0.76, 1.00)
  | "nir08" -> (0.80, 0.90)
  | "swir16" -> (1.55, 1.75)
  | "swir22" -> (2.08, 2.35)
  | "lwir" -> (10.4, 12.5)
  | "lwir11" -> (10.5, 11.5)
  | n -> failf "no range for %s" n

let spectral =
  test "each satellite's bands state its instrument's centre and width"
    (fun () ->
      List.iter
        (fun (s, b, lo, hi) ->
          let ours = band s b in
          (* Rounded to 0.1 nm. *)
          equal (option (float 5e-5)) (Some ((lo +. hi) /. 2.))
            ours.center_wavelength;
          equal (option (float 5e-5)) (Some (hi -. lo))
            ours.full_width_half_max)
        stated;
      List.iter
        (fun s ->
          List.iter
            (fun b ->
              let ours = band s b in
              match (ours.common_name, ours.center_wavelength) with
              | None, None -> is_true (ours.full_width_half_max = None)
              | Some n, Some c ->
                  let lo, hi = eo_range n in
                  let w = Option.get ours.full_width_half_max in
                  is_true ~msg:n (lo <= c && c <= hi && w > 0.)
              | _ -> failf "%s: a centre without a common name" (name s b))
            (bands s))
        satellites;
      (* One sensor, two instruments: Landsat 8's OLI and 9's OLI-2. *)
      let red s = (band s Red).center_wavelength in
      equal (option float_exact) (Some 0.6546) (red Landsat_8);
      equal (option float_exact) (Some 0.6543) (red Landsat_9))

(* landsat_rsr.dat: the USGS's responses around each half-maximum crossing
   of the bands whose values are derived. Each value is the crossings'
   linear interpolation, rounded to 0.1 nm. *)
let derived =
  test "each derived centre and width interpolates its RSR's crossings"
    (fun () ->
      let satellite = function
        | "Landsat_4" -> Landsat_4
        | "Landsat_5" -> Landsat_5
        | "Landsat_7" -> Landsat_7
        | "Landsat_8" -> Landsat_8
        | s -> failf "satellite %s" s
      in
      let band = function
        | "Coastal" -> Coastal
        | "Blue" -> Blue
        | "Green" -> Green
        | "Red" -> Red
        | "Nir" -> Nir
        | "Swir16" -> Swir16
        | "Swir22" -> Swir22
        | "Lwir" -> Lwir
        | b -> failf "band %s" b
      in
      let round x = Float.round (x *. 1e4) /. 1e4 in
      let rows =
        In_channel.with_open_text "landsat_rsr.dat" In_channel.input_lines
        |> List.filter (fun l -> l <> "" && l.[0] <> '#')
      in
      equal int 28 (List.length rows);
      List.iter
        (fun l ->
          match String.split_on_char ' ' l with
          | s :: b :: peak :: samples ->
              let h = float_of_string peak /. 2. in
              let cross = function
                | [ w0; r0; w1; r1 ] ->
                    w0 +. ((h -. r0) *. (w1 -. w0) /. (r1 -. r0))
                | _ -> failf "%s" l
              in
              let v = List.map float_of_string samples in
              let lo = cross (List.filteri (fun i _ -> i < 4) v) in
              let hi = cross (List.filteri (fun i _ -> i >= 4) v) in
              let ours = Landsat.band (satellite s) (band b) in
              equal ~msg:l (option float_exact)
                (Some (round ((lo +. hi) /. 2.))) ours.center_wavelength;
              equal ~msg:l (option float_exact) (Some (round (hi -. lo)))
                ours.full_width_half_max
          | _ -> failf "row %S" l)
        rows)

let orders =
  test "bands lists each sensor's bands in its table's order" (fun () ->
      List.iter
        (fun s ->
          equal (list string)
            (List.map (fun (_, n, _, _, _, _, _) -> n) (rows s))
            (List.map (name s) (bands s)))
        satellites)

let absent =
  test "a band a sensor's products lack raises" (fun () ->
      List.iter
        (fun (s, b) ->
          raises_match (Exn.invalid_arg ~sub:"no such band") (fun () ->
              name s b);
          raises_match (Exn.invalid_arg ~sub:"no such band") (fun () ->
              band s b);
          raises_match (Exn.invalid_arg ~sub:"no such band") (fun () ->
              valid_range s b))
        [
          (Landsat_4, Coastal);
          (Landsat_7, Coastal);
          (Landsat_5, Sr_qa_aerosol);
          (Landsat_8, Sr_cloud_qa);
          (Landsat_9, Sr_atmos_opacity);
        ])

(* Quantities *)

(* Every 16-bit DN of band [b] decodes as [scale · DN + offset] rounded to
   float32, with DN 0 masked as fill. *)
let every s b scale offset =
  let b = band s b in
  let v, ok = Band.decode b Nx.float32 dns16 in
  let want d =
    if d = 0 then 0. else single ((scale *. Float.of_int d) +. offset)
  in
  equal (array float_exact) (Array.map want every16) (floats v);
  equal (array bool) (Array.map (fun d -> d <> 0) every16) (bools ok);
  is_false (Array.exists Fun.id (bools (Band.saturated b dns16)))

let reflectance =
  test "every DN decodes to 2.75e-5 · DN - 0.2, DN 0 fill" (fun () ->
      List.iter
        (fun s -> List.iter (fun b -> every s b 2.75e-5 (-0.2)) [ Red; Swir22 ])
        satellites)

let temperature =
  test "every DN decodes to 0.00341802 · DN + 149 kelvin, DN 0 fill"
    (fun () -> List.iter (fun s -> every s Lwir 0.00341802 149.) satellites)

(* The extremes LSDS-1619's example metadata states (Appendix B,
   LEVEL2_SURFACE_REFLECTANCE_PARAMETERS and ..._TEMPERATURE_...), to its
   six decimals, and the valid ranges' ends, which the guides round. *)
let extremes =
  cases "the guide's stated extremes" ~id:(fun (n, _, _, _, _) -> n)
    [
      ("REFLECTANCE_MINIMUM, DN 1", Red, 1, -0.199972, 1e-6);
      ("REFLECTANCE_MAXIMUM, DN 65535", Red, 65535, 1.602213, 1e-6);
      ("TEMPERATURE_MINIMUM, DN 1", Lwir, 1, 149.003418, 1e-6);
      ("TEMPERATURE_MAXIMUM, DN 65535", Lwir, 65535, 372.999941, 1e-6);
      ("valid reflectance from 0", Red, 7273, 0., 1e-5);
      ("valid reflectance to 1", Red, 43636, 1., 1e-5);
      ("valid temperature from 150 K", Lwir, 293, 150., 0.005);
      ("valid temperature to 359 K", Lwir, 61440, 359., 0.005);
    ]
    (fun (_, b, dn, want, eps) ->
      let v, _ =
        Band.decode (band Landsat_8 b) Nx.float64
          (Nx.create Nx.uint16 [| 1 |] [| dn |])
      in
      equal (float eps) want (floats v).(0))

let intermediates =
  test "an intermediate band masks -9999 and scales the rest" (fun () ->
      let x = Nx.create Nx.int16 [| 4 |] [| -9999; 0; 2731; 32767 |] in
      let v, ok = Band.decode (band Landsat_9 St_qa) Nx.float64 x in
      equal (array (float 1e-12)) [| 0.; 0.; 27.31; 327.67 |] (floats v);
      equal (array bool) [| false; true; true; true |] (bools ok))

let bitfields =
  test "a quality band is read as bits, never decoded" (fun () ->
      List.iter
        (fun s ->
          List.iter
            (fun b ->
              if quality b then
                raises_match (Exn.invalid_arg ~sub:"bitfield") (fun () ->
                    Band.decode (band s b) Nx.float32 dns16))
            (bands s))
        satellites)

(* Bits *)

(* [d]'s field of [length] bits at [offset], STAC's classification
   extension's [data >> offset & (1 << length) - 1]. *)
let field d offset length = (d lsr offset) land ((1 lsl length) - 1)

let set d b = field d b 1 = 1

(* LSDS-1619 Table 6-2 and LSDS-1618 Table 5-5. *)
let pixel_flags =
  Qa_pixel.
    [
      (Fill, 0);
      (Dilated_cloud, 1);
      (Cirrus, 2);
      (Cloud, 3);
      (Cloud_shadow, 4);
      (Snow, 5);
      (Clear, 6);
      (Water, 7);
    ]

let pixel_bits =
  test "each QA_PIXEL flag's bit, on every 16-bit DN" (fun () ->
      List.iter
        (fun (f, b) ->
          equal int b (Qa_pixel.bit f);
          equal (array bool)
            (Array.map (fun d -> set d b) every16)
            (bools (Qa_pixel.any [ f ] dns16)))
        pixel_flags)

(* QA_PIXEL's and SR_QA_AEROSOL's bit 0 marks fill (LSDS-1619 Tables 6-2
   and 6-5, LSDS-1618 Table 5-5), and each band's fill value is 1, that bit
   alone. *)
let fills =
  test "each layer's fill is the DNs its guide reads as fill" (fun () ->
      let want = Array.map (fun d -> set d 0) every16 in
      equal ~msg:"QA_PIXEL" (array bool) want (bools (Qa_pixel.fill dns16));
      equal ~msg:"SR_QA_AEROSOL" (array bool) want
        (bools (Sr_qa_aerosol.fill dns16));
      List.iter
        (fun b ->
          equal (option float_exact) (Some 1.) (band Landsat_8 b).nodata)
        [ Qa_pixel; Sr_qa_aerosol ];
      (* SR_CLOUD_QA's fill is DN 0 (LSDS-1618 Table 5-4), outside its valid
         range of 1 to 63 (Table 5-1). *)
      equal ~msg:"SR_CLOUD_QA" (array bool)
        (Array.map (fun d -> d = 0) every16)
        (bools (Sr_cloud_qa.fill dns16));
      equal (option float_exact) (Some 0.) (band Landsat_7 Sr_cloud_qa).nodata;
      equal (pair int int) (1, 63) (valid_range Landsat_7 Sr_cloud_qa))

let pixel_any =
  test "any is the union of its flags; none is nowhere" (fun () ->
      let fs = Qa_pixel.[ Fill; Dilated_cloud; Cloud; Cloud_shadow ] in
      equal (array bool)
        (Array.map (fun d -> List.exists (set d) [ 0; 1; 3; 4 ]) every16)
        (bools (Qa_pixel.any fs dns16));
      is_false (Array.exists Fun.id (bools (Qa_pixel.any [] dns16))))

let levels = Qa_pixel.[ (Unset, 0); (Low, 1); (Medium, 2); (High, 3) ]

let pixel_confidences =
  cases "each QA_PIXEL confidence field, on every 16-bit DN"
    ~id:(fun (_, o) -> Printf.sprintf "bits %d-%d" o (o + 1))
    Qa_pixel.
      [
        (Cloud_confidence, 8);
        (Cloud_shadow_confidence, 10);
        (Snow_ice_confidence, 12);
        (Cirrus_confidence, 14);
      ]
    (fun (c, o) ->
      List.iter
        (fun (l, v) ->
          equal (array bool)
            (Array.map (fun d -> field d o 2 = v) every16)
            (bools (Qa_pixel.confidence c [ l ] dns16)))
        levels;
      equal (array bool)
        (Array.map (fun d -> field d o 2 >= 2) every16)
        (bools (Qa_pixel.confidence c Qa_pixel.[ Medium; High ] dns16)))

(* LSDS-1619 Table 6-3 and LSDS-1618 Table 5-6: a value, its flags set, and
   its cloud, cloud shadow, snow and cirrus confidences. *)
let interpretations =
  Qa_pixel.
    [
      (1, [ Fill ], (Unset, Unset, Unset, Unset));
      (21824, [ Clear ], (Low, Low, Low, Low));
      (21826, [ Dilated_cloud; Clear ], (Low, Low, Low, Low));
      (21888, [ Water ], (Low, Low, Low, Low));
      (21890, [ Dilated_cloud; Water ], (Low, Low, Low, Low));
      (22080, [ Clear ], (Medium, Low, Low, Low));
      (22144, [ Water ], (Medium, Low, Low, Low));
      (22280, [ Cloud ], (High, Low, Low, Low));
      (23888, [ Cloud_shadow; Clear ], (Low, High, Low, Low));
      (23952, [ Cloud_shadow; Water ], (Low, High, Low, Low));
      (24088, [ Cloud; Cloud_shadow ], (Medium, High, Low, Low));
      (24216, [ Cloud; Cloud_shadow; Water ], (Medium, High, Low, Low));
      (24344, [ Cloud; Cloud_shadow ], (High, High, Low, Low));
      (24472, [ Cloud; Cloud_shadow; Water ], (High, High, Low, Low));
      (30048, [ Snow; Clear ], (Low, Low, High, Low));
      (54596, [ Cirrus; Clear ], (Low, Low, Low, High));
      (54852, [ Cirrus; Clear ], (Medium, Low, Low, High));
      (55052, [ Cirrus; Cloud ], (High, Low, Low, High));
      (* Landsat 4-7, whose cirrus bits are unused. *)
      (5440, [ Clear ], (Low, Low, Low, Unset));
      (5442, [ Dilated_cloud; Clear ], (Low, Low, Low, Unset));
      (5504, [ Water ], (Low, Low, Low, Unset));
      (5506, [ Dilated_cloud; Water ], (Low, Low, Low, Unset));
      (5696, [ Clear ], (Medium, Low, Low, Unset));
      (5760, [ Water ], (Medium, Low, Low, Unset));
      (5896, [ Cloud ], (High, Low, Low, Unset));
      (7440, [ Cloud_shadow ], (Low, High, Low, Unset));
      (7568, [ Cloud_shadow; Water ], (Low, High, Low, Unset));
      (7696, [ Cloud_shadow ], (Medium, High, Low, Unset));
      (7824, [ Cloud_shadow; Water ], (Medium, High, Low, Unset));
      (7960, [ Cloud; Cloud_shadow ], (High, High, Low, Unset));
      (8088, [ Cloud; Cloud_shadow; Water ], (High, High, Low, Unset));
      (13664, [ Snow; Clear ], (Low, Low, High, Unset));
    ]

let pixel_table =
  cases "the guides' QA_PIXEL values read as their tables say"
    ~id:(fun (v, _, _) -> string_of_int v)
    interpretations
    (fun (v, fs, (cloud, shadow, snow, cirrus)) ->
      let x = Nx.create Nx.uint16 [| 1 |] [| v |] in
      List.iter
        (fun (f, _) ->
          equal bool (List.mem f fs) (bools (Qa_pixel.any [ f ] x)).(0))
        pixel_flags;
      List.iter
        (fun (c, l) ->
          is_true (bools (Qa_pixel.confidence c [ l ] x)).(0))
        Qa_pixel.
          [
            (Cloud_confidence, cloud);
            (Cloud_shadow_confidence, shadow);
            (Snow_ice_confidence, snow);
            (Cirrus_confidence, cirrus);
          ])

(* LSDS-1619 Table 6-4 and LSDS-1618 Table 5-7: each band's saturation
   bits. *)
let radsat_bits =
  [
    ( Landsat_8,
      [
        (Coastal, [ 0 ]);
        (Blue, [ 1 ]);
        (Green, [ 2 ]);
        (Red, [ 3 ]);
        (Nir, [ 4 ]);
        (Swir16, [ 5 ]);
        (Swir22, [ 6 ]);
      ] );
    ( Landsat_5,
      [
        (Blue, [ 0 ]);
        (Green, [ 1 ]);
        (Red, [ 2 ]);
        (Nir, [ 3 ]);
        (Swir16, [ 4 ]);
        (Lwir, [ 5 ]);
        (Swir22, [ 6 ]);
      ] );
    ( Landsat_7,
      [
        (Blue, [ 0 ]);
        (Green, [ 1 ]);
        (Red, [ 2 ]);
        (Nir, [ 3 ]);
        (Swir16, [ 4 ]);
        (Lwir, [ 5; 8 ]);
        (Swir22, [ 6 ]);
      ] );
  ]

let radsat =
  test "each band's QA_RADSAT saturation, on every 16-bit DN" (fun () ->
      List.iter
        (fun (s, bs) ->
          List.iter
            (fun (b, bits) ->
              equal (array bool)
                (Array.map (fun d -> List.for_all (set d) bits) every16)
                (bools (Qa_radsat.saturated s b dns16)))
            bs)
        radsat_bits;
      equal (array bool)
        (Array.map (fun d -> set d 11) every16)
        (bools (Qa_radsat.occluded dns16));
      equal (array bool)
        (Array.map (fun d -> set d 9) every16)
        (bools (Qa_radsat.dropped dns16)))

let radsat_refused =
  test "QA_RADSAT refuses bands it does not flag" (fun () ->
      List.iter
        (fun (s, b, sub) ->
          raises_match (Exn.invalid_arg ~sub) (fun () ->
              Qa_radsat.saturated s b dns16))
        [
          (Landsat_9, Lwir, "TIRS");
          (Landsat_4, Coastal, "no reflective or thermal band");
          (Landsat_8, St_qa, "no reflective or thermal band");
          (Landsat_7, Qa_pixel, "no reflective or thermal band");
        ])

(* Every 8-bit DN, as uint8. *)
let every8 = Array.init 256 Fun.id
let dns8 = Nx.create Nx.uint8 [| 256 |] every8

let aerosol_bits =
  test "each SR_QA_AEROSOL flag's bit and its level, on every 8-bit DN"
    (fun () ->
      List.iter
        (fun (f, b) ->
          equal int b (Sr_qa_aerosol.bit f);
          equal (array bool)
            (Array.map (fun d -> set d b) every8)
            (bools (Sr_qa_aerosol.any [ f ] dns8)))
        Sr_qa_aerosol.
          [ (Fill, 0); (Valid_retrieval, 1); (Water, 2); (Interpolated, 5) ];
      List.iter
        (fun (l, v) ->
          equal (array bool)
            (Array.map (fun d -> field d 6 2 = v) every8)
            (bools (Sr_qa_aerosol.level [ l ] dns8)))
        Sr_qa_aerosol.[ (Climatology, 0); (Low, 1); (Medium, 2); (High, 3) ])

(* LSDS-1619 Table 6-6: the values the guide lists for each attribute. *)
let aerosol_table =
  test "the guide's SR_QA_AEROSOL values read as its table says" (fun () ->
      let has values m =
        let m = bools m in
        List.iter (fun v -> is_true ~msg:(string_of_int v) m.(v)) values
      in
      let open Sr_qa_aerosol in
      has [ 1 ] (any [ Fill ] dns8);
      has [ 2; 66; 130; 194 ] (any [ Valid_retrieval ] dns8);
      has [ 4; 68; 100; 132; 164; 196; 228 ] (any [ Water ] dns8);
      has [ 32; 96; 100; 160; 164; 224; 228 ] (any [ Interpolated ] dns8);
      has [ 66; 68; 96; 100 ] (level [ Low ] dns8);
      has [ 130; 132; 160; 164 ] (level [ Medium ] dns8);
      has [ 192; 194; 196; 224; 228 ] (level [ High ] dns8);
      has [ 2; 4; 32 ] (level [ Climatology ] dns8))

(* LSDS-1618 Tables 5-2 and 5-3. *)
let cloud_qa =
  test "each SR_CLOUD_QA flag's bit, and the guide's values" (fun () ->
      let open Sr_cloud_qa in
      let flags =
        [
          (Ddv, 0, [ 1; 9 ]);
          (Cloud, 1, [ 2; 34 ]);
          (Cloud_shadow, 2, [ 4; 12; 20; 36; 52 ]);
          (Adjacent_cloud, 3, [ 8; 12; 24; 40; 56 ]);
          (Snow, 4, [ 16; 20; 24; 48; 52; 56 ]);
          (Water, 5, [ 32; 34; 36; 40; 48; 52; 56 ]);
        ]
      in
      List.iter
        (fun (f, b, values) ->
          equal int b (bit f);
          let m = bools (any [ f ] dns8) in
          equal (array bool) (Array.map (fun d -> set d b) every8) m;
          List.iter (fun v -> is_true ~msg:(string_of_int v) m.(v)) values)
        flags)

let dtypes =
  test "bits read from any integer dtype, and nothing else" (fun () ->
      let want = [| false; true; false; true |] in
      let cloud = Qa_pixel.[ Cloud ] in
      let x = Nx.create Nx.int32 [| 4 |] [| 0l; 8l; 7l; 22280l |] in
      equal (array bool) want (bools (Qa_pixel.any cloud x));
      equal (array bool) want
        (bools
           (Qa_pixel.any cloud (Nx.create Nx.int16 [| 4 |] [| 0; 8; 7; -1 |])));
      List.iter
        (fun f ->
          raises_match
            (Exn.invalid_arg
               ~sub:
                 "not an integer dtype; mask NaN DNs as fill (Nx.isnan), then \
                  cast the rest to the file's integer dtype")
            f)
        [
          (fun () -> Qa_pixel.any cloud (Nx.zeros Nx.float32 [| 2 |]));
          (fun () -> Qa_radsat.dropped (Nx.zeros Nx.bool [| 2 |]));
          (fun () -> Sr_qa_aerosol.level [] (Nx.zeros Nx.float64 [| 2 |]));
        ])

(* A DN's bits as its dtype holds them: an int8 has no bit 8, and bit 7 is
   its sign. *)
let widths =
  test "a bit at or past a dtype's width is unset" (fun () ->
      let every = Array.init 256 (fun d -> d - 128) in
      let x = Nx.create Nx.int8 [| 256 |] every in
      let nowhere = Array.make 256 false in
      equal (array bool)
        (Array.map (fun d -> d < 0) every)
        (bools (Qa_pixel.any Qa_pixel.[ Water ] x));
      equal (array bool)
        (Array.map (fun d -> d < 0 || set d 3) every)
        (bools (Qa_pixel.any Qa_pixel.[ Cloud; Water ] x));
      equal (array bool) (Array.make 256 true)
        (bools (Qa_pixel.confidence Cloud_confidence Qa_pixel.[ Unset ] x));
      equal (array bool) nowhere (bools (Qa_radsat.dropped x));
      equal (array bool) nowhere
        (bools (Qa_radsat.saturated Landsat_7 Lwir x));
      (* Bits 14 and 15 of an int16 are those of the uint16 of its bits. *)
      let int16 d = if d < 32768 then d else d - 65536 in
      let x = Nx.create Nx.int16 [| 65536 |] (Array.map int16 every16) in
      equal (array bool)
        (Array.map (fun d -> field d 14 2 = 3) every16)
        (bools (Qa_pixel.confidence Cirrus_confidence Qa_pixel.[ High ] x)))

let () =
  exit
    (run "jord.landsat"
       [
         group "sensors" [ identifiers; bad_identifiers ];
         group "bands" [ tables; spectral; derived; orders; absent ];
         group "quantities"
           [ reflectance; temperature; extremes; intermediates; bitfields ];
         group "qa_pixel"
           [ pixel_bits; pixel_any; pixel_confidences; pixel_table ];
         group "fill" [ fills ];
         group "qa_radsat" [ radsat; radsat_refused ];
         group "sr_qa_aerosol" [ aerosol_bits; aerosol_table ];
         group "sr_cloud_qa" [ cloud_qa ];
         group "dtypes" [ dtypes; widths ];
       ])
