(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord
open Sentinel2

let floats x = Nx.to_array (Nx.cast Nx.float64 x)

(* [x] rounded once to float32. *)
let single x = Int32.float_of_bits (Int32.bits_of_float x)

(* Bands *)

(* ESA's names and native resolutions, from the product metadata's
   Spectral_Information list and SentiWiki's "S2 Products". *)
let table =
  cases "each band's name and resolution" ~id:(fun (_, n, _) -> n)
    [
      (B01, "B01", 60);
      (B02, "B02", 10);
      (B03, "B03", 10);
      (B04, "B04", 10);
      (B05, "B05", 20);
      (B06, "B06", 20);
      (B07, "B07", 20);
      (B08, "B08", 10);
      (B8A, "B8A", 20);
      (B09, "B09", 60);
      (B11, "B11", 20);
      (B12, "B12", 20);
    ]
    (fun (b, n, r) ->
      equal string n (name b);
      equal int r (resolution b))

let order =
  test "bands lists the 12 bands in ESA's order" (fun () ->
      equal (list string)
        [
          "B01"; "B02"; "B03"; "B04"; "B05"; "B06"; "B07"; "B08"; "B8A";
          "B09"; "B11"; "B12";
        ]
        (List.map name bands))

(* Reflectance *)

let metadata ?(baseline = "05.10") ?(offsets = []) () =
  { baseline; quantification = 10000.; offsets; nodata = 0; saturated = 65535 }

let all = List.map (fun b -> (b, -1000)) bands

let band =
  Testable.make
    ~pp:(fun ppf (b : Band.t) ->
      let opt = Option.fold ~none:"-" ~some:string_of_float in
      Format.fprintf ppf "{%s scale %.17g offset %.17g nodata %s saturated %s}"
        (Option.value ~default:"-" b.name)
        b.scale b.offset (opt b.nodata) (opt b.saturated))
    ~equal:(fun a b -> compare a b = 0)

let stated =
  test "a stated offset gives scale 1/Q and offset BOA_ADD_OFFSET/Q"
    (fun () ->
      let want =
        {
          Band.name = Some "B04";
          common_name = None;
          unit = None;
          kind = Measurement;
          scale = 1. /. 10000.;
          offset = -1000. /. 10000.;
          nodata = Some 0.;
          saturated = Some 65535.;
        }
      in
      equal (result band string) (Ok want)
        (reflectance (metadata ~offsets:all ()) B04);
      (* A product before 04.00 that states an offset is read by it. *)
      equal (result band string) (Ok want)
        (reflectance (metadata ~baseline:"03.00" ~offsets:all ()) B04))

let unshifted =
  test "before baseline 04.00 an offset not stated is 0" (fun () ->
      let b = require_ok (reflectance (metadata ~baseline:"03.01" ()) B8A) in
      equal float_exact 0. b.offset;
      equal (option string) (Some "B8A") b.name)

let unknown =
  test "from baseline 04.00 an offset not stated is an error" (fun () ->
      is_error (reflectance (metadata ~baseline:"04.00" ()) B02);
      let some = [ (B02, -1000) ] in
      is_ok (reflectance (metadata ~offsets:some ()) B02);
      is_error (reflectance (metadata ~offsets:some ()) B03))

let baselines =
  cases "a baseline not of the form NN.NN is an error"
    ~id:(Printf.sprintf "%S")
    [ ""; "5.10"; "05.1"; "05,10"; "0510"; "+5.10"; "05.10.1"; "N05.10" ]
    (fun baseline ->
      is_error (reflectance (metadata ~baseline ~offsets:all ()) B04))

let quantification =
  test "a quantification or inverse not positive and finite raises"
    (fun () ->
      List.iter
        (fun q ->
          raises_match (Exn.invalid_arg ~sub:"quantification") (fun () ->
              reflectance { (metadata ()) with quantification = q } B04))
        [ 0.; -10000.; Float.nan; Float.infinity; 1e-320 ])

(* ESA's rule, (DN + offset) / Q rounded to float32, on every 16-bit DN, with
   NODATA and SATURATED masked. *)
let esa offset =
  test (Printf.sprintf "every DN decodes as ESA's rule at offset %d" offset)
    (fun () ->
      let m = metadata ~offsets:[ (B11, offset) ] () in
      let b = require_ok (reflectance m B11) in
      let dns = Array.init 65536 Fun.id in
      let x = Nx.create Nx.uint16 [| 65536 |] dns in
      let v, ok = Band.decode b Nx.float32 x in
      let special d = d = 0 || d = 65535 in
      let want d =
        if special d then 0. else single (Float.of_int (d + offset) /. 10000.)
      in
      equal (array float_exact) (Array.map want dns) (floats v);
      equal (array bool)
        (Array.map (fun d -> not (special d)) dns)
        (Nx.to_array ok))

(* Scene classification *)

(* ESA's Table 3. *)
let classes =
  Scl.
    [
      (No_data, 0);
      (Saturated_or_defective, 1);
      (Cast_shadows, 2);
      (Cloud_shadows, 3);
      (Vegetation, 4);
      (Not_vegetated, 5);
      (Water, 6);
      (Unclassified, 7);
      (Cloud_medium_probability, 8);
      (Cloud_high_probability, 9);
      (Thin_cirrus, 10);
      (Snow_or_ice, 11);
    ]

let codes =
  cases "each class's code" ~id:(fun (_, c) -> string_of_int c) classes
    (fun (c, n) -> equal int n (Scl.code c))

(* Subsets of the classes, drawn often empty, whole or of one class. *)
let subsets =
  Gen.(
    let any = list (map fst (of_list classes)) in
    frequency
      [
        (4, any);
        (1, constant []);
        (1, constant (List.map fst classes));
        (1, map (fun c -> [ c ]) (map fst (of_list classes)));
      ])

let mem =
  prop "mem is membership of each byte's class" subsets (fun cs ->
      let bytes = Array.init 256 Fun.id in
      let want b = List.exists (fun c -> Scl.code c = b) cs in
      equal (array bool) (Array.map want bytes)
        (Nx.to_array (Scl.mem cs (Nx.create Nx.uint8 [| 256 |] bytes))))

let reals =
  test "mem compares codes of any real dtype" (fun () ->
      let x =
        Nx.create Nx.float32 [| 6 |] [| 2.; 2.5; Float.nan; -0.; 11.; -2. |]
      in
      equal (array bool)
        [| true; false; false; true; false; false |]
        (Nx.to_array (Scl.mem Scl.[ No_data; Cast_shadows ] x));
      let x = Nx.create Nx.int16 [| 3 |] [| -1; 3; 256 + 3 |] in
      equal (array bool) [| false; true; false |]
        (Nx.to_array (Scl.mem [ Scl.Cloud_shadows ] x));
      raises_match (Exn.invalid_arg ~sub:"not a real dtype") (fun () ->
          Scl.mem [ Scl.Water ] (Nx.zeros Nx.bool [| 2 |])))

(* Identity *)

let key_t =
  Testable.make
    ~pp:(fun ppf k ->
      Format.fprintf ppf "{%s %s %s}" k.platform k.tile k.sensing_start)
    ~equal:( = )

(* Datastrips of Microsoft's Planetary Computer's items (test/safe/) and of
   SentiWiki's "S2 Products". *)
let keys =
  cases "a key is the platform, the tile and the sensing start"
    ~id:(fun (_, id, _) -> id)
    [
      ( "33UUU",
        "S2A_OPER_MSI_L2A_DS_MSFT_20240629T161811_S20240629T102556_N05.10",
        ("S2A", "20240629T102556") );
      ( "33UUU",
        "S2B_OPER_MSI_L2A_DS_ESRI_20210628T031703_S20210627T100555_N03.00",
        ("S2B", "20210627T100555") );
      ( "60HXB",
        "S2A_OPER_MSI_L1C_DS_S2RP_20231019T133207_S20160418T212546_N05.00",
        ("S2A", "20160418T212546") );
      (* A processing centre of three letters ends with an underscore. *)
      ( "01CCV",
        "S2C_OPER_MSI_L2A_DS_MPC__20250101T000000_S20241231T235959_N05.11",
        ("S2C", "20241231T235959") );
    ]
    (fun (tile, id, (platform, sensing_start)) ->
      equal (result key_t string)
        (Ok { platform; tile; sensing_start })
        (key ~tile id))

(* A Sentinel-2A Level-2A datastrip identifier from its processing centre
   on. *)
let ds rest = "S2A_OPER_MSI_L2A_DS_" ^ rest

let reprocessed =
  test "reprocessing keeps the key; another datastrip changes it" (fun () ->
      let k id = require_ok (key ~tile:"32TPT" id) in
      let a = k (ds "ESRI_20211222T150000_S20211222T102436_N03.00")
      and b = k (ds "S2RP_20230301T120000_S20211222T102436_N05.00")
      and c = k (ds "ESRI_20211222T150000_S20211222T102744_N03.00") in
      equal key_t a b;
      not_equal key_t a c)

let bad_keys =
  let good = ds "MSFT_20240629T161811_S20240629T102556_N05.10" in
  cases "a tile or identifier not of the naming convention is an error"
    ~id:(fun (t, id) -> t ^ " " ^ id)
    [
      ("T33UUU", good);
      ("33uuu", good);
      ("3UUU", good);
      ("33UUU", ds "MSFT_20240629T161811_N05.10");
      ("33UUU", ds "MSFT_20240629T161811_S20240629X102556_N05.10");
      ("33UUU", ds "MSFT_20240629T161811");
      ("33UUU", ds "MSFT_2024062XT161811_S20240629T102556_N05.10");
      ("33UUU", "S1A" ^ String.sub good 3 (String.length good - 3));
      ("33UUU", "");
    ]
    (fun (tile, id) -> is_error (key ~tile id))

let () =
  exit
    (run "jord.sentinel2"
       [
         group "bands" [ table; order ];
         group "reflectance"
           [
             stated;
             unshifted;
             unknown;
             baselines;
             quantification;
             esa 0;
             esa (-1000);
           ];
         group "scl" [ codes; mem; reals ];
         group "key" [ keys; reprocessed; bad_keys ];
       ])
