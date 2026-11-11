(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord

let read file = In_channel.with_open_bin file In_channel.input_all

(* Baseline 03.00, Esri's processing: no offsets. *)
let n0300 = read "mtd_n0300.xml"

(* Baseline 05.10, Microsoft's processing: -1000 for every band. *)
let n0510 = read "mtd_n0510.xml"

let metadata =
  Testable.make
    ~pp:(fun ppf (m : Sentinel2.metadata) ->
      Format.fprintf ppf "{%s Q %g [%s] nodata %d saturated %d}" m.baseline
        m.quantification
        (String.concat "; "
           (List.map
              (fun (b, o) -> Printf.sprintf "%s %d" (Sentinel2.name b) o)
              m.offsets))
        m.nodata m.saturated)
    ~equal:( = )

let stated ?(offsets = []) baseline =
  {
    Sentinel2.baseline;
    quantification = 10000.;
    offsets;
    nodata = 0;
    saturated = 65535;
  }

let all = List.map (fun b -> (b, -1000)) Sentinel2.bands

(* [s] with its one occurrence of [a] replaced by [b]. *)
let edit a b s =
  let n = String.length a in
  let rec find i =
    if i + n > String.length s then failf "%S is not in the fixture" a
    else if String.sub s i n = a then i
    else find (i + 1)
  in
  let i = find 0 in
  String.sub s 0 i ^ b ^ String.sub s (i + n) (String.length s - i - n)

let products =
  test "a product's baseline, quantification, special values and offsets"
    (fun () ->
      equal (result metadata string)
        (Ok (stated "03.00"))
        (Jord_safe.l2a n0300);
      equal (result metadata string)
        (Ok (stated ~offsets:all "05.10"))
        (Jord_safe.l2a n0510))

let reflectance =
  test "the metadata decodes each product's DNs by ESA's rule" (fun () ->
      let dn = Nx.create Nx.uint16 [| 4 |] [| 0; 1000; 2234; 65535 |] in
      let decode xml =
        let m = require_ok (Jord_safe.l2a xml) in
        let b = require_ok (Sentinel2.reflectance m Sentinel2.B04) in
        let v, ok = Band.decode b Nx.float32 dn in
        (Nx.to_array (Nx.cast Nx.float64 v), Nx.to_array ok)
      in
      let single x = Int32.float_of_bits (Int32.bits_of_float x) in
      let mask = [| false; true; true; false |] in
      equal (pair (array float_exact) (array bool))
        ([| 0.; single 0.1; single 0.2234; 0. |], mask)
        (decode n0300);
      equal (pair (array float_exact) (array bool))
        ([| 0.; 0.; single 0.1234; 0. |], mask)
        (decode n0510))

let left_out =
  test "an offset \"Not applicable\" is left out" (fun () ->
      let xml =
        edit {|<BOA_ADD_OFFSET band_id="3">-1000|}
          {|<BOA_ADD_OFFSET band_id="3">Not applicable|} n0510
      in
      let offsets = List.filter (fun (b, _) -> b <> Sentinel2.B04) all in
      equal (result metadata string)
        (Ok (stated ~offsets "05.10"))
        (Jord_safe.l2a xml))

(* The 05.10 file with BOA_QUANTIFICATION_VALUE [q]. *)
let quantification q =
  edit {|<BOA_QUANTIFICATION_VALUE unit="none">10000|}
    ({|<BOA_QUANTIFICATION_VALUE unit="none">|} ^ q)
    n0510

let quantified q =
  ("a quantification of " ^ q, "BOA_QUANTIFICATION_VALUE", quantification q)

let decimals =
  cases "a quantification in plain decimal form reads" ~id:fst
    [ ("10000.0", 10000.); ("1.0E4", 10000.); ("1e+4", 10000.); ("2.5", 2.5) ]
    (fun (q, v) ->
      let m = require_ok (Jord_safe.l2a (quantification q)) in
      equal float_exact v m.Sentinel2.quantification)

(* Each row breaks the 05.10 file in one place, and the error names it. *)
let refusals =
  cases "a file that breaks the format is an error" ~id:(fun (n, _, _) -> n)
    [
      ("not XML", "line", String.sub n0510 0 1000);
      ("text after the root", "follows", n0510 ^ "<x/>");
      ( "another product level",
        "Level-1C_User_Product",
        edit "<n1:Level-2A_User_Product" "<n1:Level-1C_User_Product"
          (edit "</n1:Level-2A_User_Product>" "</n1:Level-1C_User_Product>"
             n0510) );
      ( "no baseline",
        "PROCESSING_BASELINE",
        edit "<PROCESSING_BASELINE>05.10</PROCESSING_BASELINE>" "" n0510 );
      ( "no quantification",
        "BOA_QUANTIFICATION_VALUE",
        edit
          {|<BOA_QUANTIFICATION_VALUE unit="none">10000</BOA_QUANTIFICATION_VALUE>|}
          "" n0510 );
      quantified "0";
      quantified "1_0000";
      quantified "0x2710";
      quantified "1e-320";
      quantified "1e400";
      quantified "-10000";
      quantified ".5";
      quantified "1e";
      quantified "1.0E+";
      quantified "inf";
      ( "NODATA twice",
        "NODATA",
        edit "<SPECIAL_VALUE_TEXT>SATURATED" "<SPECIAL_VALUE_TEXT>NODATA" n0510
      );
      ( "a special value not an integer",
        "SPECIAL_VALUE_INDEX",
        edit "<SPECIAL_VALUE_INDEX>65535" "<SPECIAL_VALUE_INDEX>0x10" n0510 );
      ( "an offset twice",
        "twice for B01",
        edit {|<BOA_ADD_OFFSET band_id="1">|} {|<BOA_ADD_OFFSET band_id="0">|}
          n0510 );
      ( "an offset of no band",
        "band_id \"13\"",
        edit {|<BOA_ADD_OFFSET band_id="12">|}
          {|<BOA_ADD_OFFSET band_id="13">|} n0510 );
      ( "an offset not an integer",
        "-1e3",
        edit {|<BOA_ADD_OFFSET band_id="2">-1000|}
          {|<BOA_ADD_OFFSET band_id="2">-1e3|} n0510 );
      ( "an offset with no band_id",
        "band_id",
        edit {|<BOA_ADD_OFFSET band_id="2">|} {|<BOA_ADD_OFFSET>|} n0510 );
      ( "a band not Sentinel-2's",
        "B13",
        edit {|physicalBand="B8A"|} {|physicalBand="B13"|} n0510 );
    ]
    (fun (_, sub, xml) -> contains ~sub (require_error (Jord_safe.l2a xml)))

let () =
  exit
    (run "jord.safe"
       [ group "l2a" [ products; reflectance; left_out; decimals; refusals ] ])
