(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord
open Hls

let bools x = Nx.to_array x

let product_t =
  Testable.make
    ~pp:(fun ppf p ->
      Format.pp_print_string ppf (match p with L30 -> "L30" | S30 -> "S30"))
    ~equal:( = )

(* Bands *)

let reflectance = (None, 0.0001, -9999.)
let angle = (Some "degree", 0.01, 40000.)
let angles =
  [
    ("SZA", Sun_zenith);
    ("SAA", Sun_azimuth);
    ("VZA", View_zenith);
    ("VAA", View_azimuth);
  ]

(* HLS v2.0 User Guide, Tables 6 and 7, then Table 8: each layer's name,
   common name, unit, scale and fill. *)
let rows = function
  | L30 ->
      [
        (Coastal, "B01", Some "coastal", reflectance);
        (Blue, "B02", Some "blue", reflectance);
        (Green, "B03", Some "green", reflectance);
        (Red, "B04", Some "red", reflectance);
        (Nir_narrow, "B05", Some "nir08", reflectance);
        (Swir16, "B06", Some "swir16", reflectance);
        (Swir22, "B07", Some "swir22", reflectance);
        (Cirrus, "B09", Some "cirrus", reflectance);
        (Lwir11, "B10", Some "lwir11", (Some "degree_Celsius", 0.01, -9999.));
        (Lwir12, "B11", Some "lwir12", (Some "degree_Celsius", 0.01, -9999.));
        (Fmask, "Fmask", None, (None, 1., 255.));
      ]
      @ List.map (fun (n, b) -> (b, n, None, angle)) angles
  | S30 ->
      [
        (Coastal, "B01", Some "coastal", reflectance);
        (Blue, "B02", Some "blue", reflectance);
        (Green, "B03", Some "green", reflectance);
        (Red, "B04", Some "red", reflectance);
        (Red_edge1, "B05", Some "rededge071", reflectance);
        (Red_edge2, "B06", Some "rededge075", reflectance);
        (Red_edge3, "B07", Some "rededge078", reflectance);
        (Nir_broad, "B08", Some "nir", reflectance);
        (Nir_narrow, "B8A", Some "nir08", reflectance);
        (Water_vapour, "B09", Some "nir09", reflectance);
        (Cirrus, "B10", Some "cirrus", reflectance);
        (Swir16, "B11", Some "swir16", reflectance);
        (Swir22, "B12", Some "swir22", reflectance);
        (Fmask, "Fmask", None, (None, 1., 255.));
      ]
      @ List.map (fun (n, b) -> (b, n, None, angle)) angles

let tables =
  cases "each layer as the guide's tables state it"
    ~id:(fun p -> Format.asprintf "%a" (Testable.pp product_t) p)
    [ L30; S30 ]
    (fun p ->
      equal int (List.length (rows p)) (List.length (bands p));
      List.iter2
        (fun (b, n, common_name, (unit, scale, nodata)) b' ->
          is_true ~msg:n (b = b');
          equal string n (name p b);
          let want =
            {
              Band.name = Some n;
              common_name;
              center_wavelength = None;
              full_width_half_max = None;
              unit;
              kind = (if b = Fmask then Bitfield else Measurement);
              scale;
              offset = 0.;
              nodata = Some nodata;
              saturated = None;
            }
          in
          is_true ~msg:n (band p b = want))
        (rows p) (bands p))

let absent =
  test "a layer a product lacks raises" (fun () ->
      List.iter
        (fun (p, b) ->
          raises_match (Exn.invalid_arg ~sub:"have no such band") (fun () ->
              name p b);
          raises_match (Exn.invalid_arg ~sub:"have no such band") (fun () ->
              band p b))
        [
          (L30, Red_edge1); (L30, Nir_broad); (L30, Water_vapour);
          (S30, Lwir11); (S30, Lwir12);
        ])

let decode =
  test "a layer's DNs decode by its scale, its fill masked" (fun () ->
      let dn = Nx.create Nx.int16 [| 3 |] [| -9999; 0; 2345 |] in
      let v, ok = Band.decode (band S30 Red) Nx.float64 dn in
      equal (array bool) [| false; true; true |] (bools ok);
      equal (array float_exact) [| 0.; 0.; 2345. *. 0.0001 |] (Nx.to_array v);
      let dn = Nx.create Nx.uint16 [| 2 |] [| 40000; 4512 |] in
      let v, ok = Band.decode (band L30 Sun_zenith) Nx.float64 dn in
      equal (array bool) [| false; true |] (bools ok);
      equal (array float_exact) [| 0.; 4512. *. 0.01 |] (Nx.to_array v);
      raises_match (Exn.invalid_arg ~sub:"bitfield") (fun () ->
          Band.decode (band L30 Fmask) Nx.float32 (Nx.zeros Nx.uint8 [| 2 |])))

(* Quality bits *)

let every8 = Array.init 256 Fun.id
let dns8 = Nx.create Nx.uint8 [| 256 |] every8
let set d b = (d lsr b) land 1 = 1

let flags =
  test "each Fmask flag's bit, on every byte (Table 9)" (fun () ->
      List.iter
        (fun (f, b) ->
          equal int b (Fmask.bit f);
          equal (array bool)
            (Array.map (fun d -> set d b) every8)
            (bools (Fmask.any [ f ] dns8)))
        Fmask.
          [
            (Cloud, 1); (Adjacent, 2); (Cloud_shadow, 3); (Snow_ice, 4);
            (Water, 5);
          ];
      (* Bit 0, cirrus, is reserved: no flag reads it. *)
      let all = Fmask.[ Cloud; Adjacent; Cloud_shadow; Snow_ice; Water ] in
      equal (array bool)
        (Array.map (fun d -> d land 0b111110 <> 0) every8)
        (bools (Fmask.any all dns8)))

let aerosol =
  test "the aerosol level is bits 6 and 7 as a number (Appendix A)"
    (fun () ->
      List.iter
        (fun (l, v) ->
          equal (array bool)
            (Array.map (fun d -> d lsr 6 = v) every8)
            (bools (Fmask.aerosol [ l ] dns8)))
        Fmask.[ (Climatology, 0); (Low, 1); (Medium, 2); (High, 3) ];
      (* DN 100, binary 01100100: low aerosol, water and adjacent to cloud. *)
      let x = Nx.create Nx.uint8 [| 1 |] [| 100 |] in
      equal (array bool) [| true |] (bools (Fmask.aerosol [ Low ] x));
      equal (array bool) [| true |] (bools (Fmask.any [ Water ] x));
      equal (array bool) [| true |] (bools (Fmask.any [ Adjacent ] x));
      equal (array bool) [| false |]
        (bools (Fmask.any [ Cloud; Cloud_shadow; Snow_ice ] x)))

let fill =
  test "fill is the byte 255, which sets every flag" (fun () ->
      equal (array bool) (Array.map (fun d -> d = 255) every8)
        (bools (Fmask.fill dns8));
      let x = Nx.create Nx.uint8 [| 1 |] [| 255 |] in
      equal (array bool) [| true |] (bools (Fmask.any [ Fmask.Cloud ] x));
      raises_match (Exn.invalid_arg ~sub:"not an integer dtype") (fun () ->
          Fmask.fill (Nx.zeros Nx.float32 [| 2 |])))

(* Identity *)

let id_t =
  Testable.make
    ~pp:(fun ppf { product; tile; stamp } ->
      Format.fprintf ppf "{%a %s %h}" (Testable.pp product_t) product tile
        stamp)
    ~equal:( = )

let at y m d h mi s =
  Time.of_date (y, m, d) +. Float.of_int ((h * 3600) + (mi * 60) + s)

let identifiers =
  test "an identifier states its product, tile and stamp (Section 6.1)"
    (fun () ->
      List.iter
        (fun (s, want) -> equal (result id_t string) (Ok want) (id s))
        [
          (* The guide's examples: day 209 of 2020 is 27 July, day 117 is
             26 April. *)
          ( "HLS.L30.T17SLU.2020209T155956.V2.0",
            { product = L30; tile = "17SLU"; stamp = at 2020 7 27 15 59 56 } );
          ( "HLS.S30.T17SLU.2020117T160901.V2.0",
            { product = S30; tile = "17SLU"; stamp = at 2020 4 26 16 9 1 } );
          (* Microsoft's Planetary Computer's item south of the equator. *)
          ( "HLS.S30.T21JVF.2026306T134221.v2.0",
            { product = S30; tile = "21JVF"; stamp = at 2026 11 2 13 42 21 } );
          (* The last second of a leap year's day 366. *)
          ( "HLS.L30.T21JVF.2024366T235959.v2.0",
            { product = L30; tile = "21JVF"; stamp = at 2024 12 31 23 59 59 } );
        ])

let bad_identifiers =
  cases ~id:Fun.id "texts that are not identifiers"
    [
      "HLS.L30.T17SLU.2020209T155956.v1.4";
      "HLS.L30.T17SLU.2020209T155956.v2.0.B04.tif";
      "HLS.X30.T17SLU.2020209T155956.v2.0";
      "HLS.L30.17SLU.2020209T155956.v2.0";
      "HLS.L30.T17slu.2020209T155956.v2.0";
      "HLS.L30.T17SLU.2023366T155956.v2.0";
      "HLS.L30.T17SLU.2020000T155956.v2.0";
      "HLS.L30.T17SLU.2020209T245956.v2.0";
      "HLS.L30.T17SLU.2020209T155960.v2.0";
      "HLS.L30.T17SLU.2020209155956.v2.0";
      "HLS.L30.T17SLU.2020209T155956";
      "";
    ]
    (fun s ->
      match id s with
      | Ok _ -> failf "%S read" s
      | Error e -> contains ~sub:"HLS v2.0 identifier" e)

let () =
  exit
    (run "jord.hls"
       [
         group "bands" [ tables; absent; decode ];
         group "fmask" [ flags; aerosol; fill ];
         group "identity" [ identifiers; bad_identifiers ];
       ])
