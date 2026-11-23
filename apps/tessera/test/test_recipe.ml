(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* TESSERA's rules, Sentinel2's and Sentinel1's, against the upstream
   reference's own functions on the same inputs (upstream/l1.npz; its dune file
   names the command that cut it). *)

open Windtrap
open Tessera

let fixture = lazy (Nx_io.load_npz "upstream/l1.npz")

let get dt name =
  match Nx_io.Archive.find name (Lazy.force fixture) with
  | Some x -> Nx.unpack dt x
  | None -> failwith ("l1.npz: no " ^ name)

let ints x = Array.map Fun.id (Nx.to_array (Nx.cast Nx.int32 x))
let roi name = Nx.not_equal_s (get Nx.uint8 name) 0
let items x = List.init (Nx.dim 0 x) (fun k -> Nx.slice [ Nx.I k ] x)
let flag name = Nx.item [] (get Nx.bool name)

let sentinel2 =
  group "Sentinel2 against TESSERA's functions"
    [
      test "first_valid is process_scl's tile selection" (fun () ->
          let choice =
            Sentinel2.first_valid ~roi:(roi "s2_roi")
              (items (get Nx.uint8 "s2_scl"))
          in
          equal (array int32) (ints (get Nx.int32 "s2_choice")) (ints choice));
      test "mosaic is smart_mosaic where a product is chosen" (fun () ->
          let choice = get Nx.int32 "s2_choice" in
          let all = Nx.ones Nx.bool (Nx.shape choice) in
          let products =
            List.map (fun b -> Some (b, all)) (items (get Nx.int16 "s2_bands"))
          in
          let f = Sentinel2.mosaic choice products in
          equal (array int32) (ints (get Nx.int16 "s2_mosaic")) (ints f.bands);
          equal (array int32)
            (ints (Nx.greater_equal_s choice 0l))
            (ints f.mask));
      test "covered decides as process_scl's coverage at 0.01 %" (fun () ->
          for i = 0 to 5 do
            let case = Printf.sprintf "s2cov%d_" i in
            let roi = roi (case ^ "roi") in
            let choice =
              Sentinel2.first_valid ~roi [ get Nx.uint8 (case ^ "scl") ]
            in
            equal ~msg:case bool
              (flag (case ^ "kept"))
              (Sentinel2.covered ~roi choice)
          done);
    ]

(* [Sentinel1.code] floors [(20 log10 g + 50) 200] in the C library's log10,
   which C leaves inexact. The fixture's codes come from numpy's log10 on
   glibc, which states it within [log10_ulps] ulps; any other C library is
   allowed as much. An error of [k] ulps in log10 is [4000 k] ulps of log10 in
   the value, and each of its three roundings is at most an ulp of a value
   below [2{^15}]. Two libraries then floor to different codes only where the
   value is within the sum of their errors of an integer, and there to
   neighbours. *)
let log10_ulps = 2.
let roundings = 3.
let ulp x = Float.succ (Float.abs x) -. Float.abs x

let edge g =
  let log10 = 4000. *. log10_ulps *. ulp (Float.log10 g) in
  2. *. (log10 +. (roundings *. ulp 32768.))

let sentinel1 =
  group "Sentinel1 against TESSERA's functions"
    [
      test "code is amplitude_to_db, but next to an integer" (fun () ->
          let gamma = get Nx.float64 "s1_gamma" in
          let want = ints (get Nx.int16 "s1_code") in
          let got = ints (Sentinel1.code gamma) in
          let gamma = Nx.to_array gamma in
          equal int (Array.length want) (Array.length got);
          Array.iteri
            (fun i w ->
              let g = got.(i) in
              if g <> w then begin
                let msg = Printf.sprintf "gamma %h" gamma.(i) in
                equal ~msg int32 1l (Int32.abs (Int32.sub g w));
                let value = (4000. *. Float.log10 gamma.(i)) +. 10000. in
                let integer = Int32.to_float (Int32.max g w) in
                less ~msg float_exact ~than:(edge gamma.(i))
                  (Float.abs (value -. integer))
              end)
            want);
      test "covered decides as analyze_coverage at 0.01 %" (fun () ->
          for i = 0 to 4 do
            let case = Printf.sprintf "s1cov%d_" i in
            equal ~msg:case bool
              (flag (case ^ "kept"))
              (Sentinel1.covered
                 ~roi:(roi (case ^ "roi"))
                 (get Nx.float64 (case ^ "gamma")))
          done);
      test "merge is rasterio's first merge with nodata 0" (fun () ->
          equal (array int32)
            (ints (get Nx.int16 "s1_merge_out"))
            (ints (Sentinel1.merge (items (get Nx.int16 "s1_merge_in")))));
    ]

(* Items *)

let s2_assets =
  "product-metadata" :: "SCL"
  :: [ "B04"; "B02"; "B03"; "B08"; "B8A"; "B05"; "B06"; "B07"; "B11"; "B12" ]

let item ?(assets = s2_assets) ?(props = []) ~id ~time () =
  let props =
    String.concat ","
      (Printf.sprintf {|"datetime":%S|} time
      :: List.map (fun (k, v) -> Printf.sprintf "%S:%S" k v) props)
  in
  let assets =
    String.concat ","
      (List.map
         (fun k -> Printf.sprintf {|%S:{"href":"https://x/%s"}|} k k)
         assets)
  in
  let json =
    Printf.sprintf
      {|{"type":"Feature","id":%S,"geometry":null,"properties":{%s},"assets":{%s}}|}
      id props assets
  in
  match
    Result.bind (Jsont_bytesrw.decode_string Jsont.json json) Jord_stac.item
  with
  | Ok it -> it
  | Error e -> failwith e

let s2 ~id ~time ~strip ~baseline =
  item ~id ~time
    ~props:
      [
        ("s2:mgrs_tile", "31UDQ");
        ( "s2:datastrip_id",
          Printf.sprintf "S2A_OPER_MSI_L2A_DS_ESRI_20230101T000000_S%s_N%s"
            strip baseline );
        ("s2:processing_baseline", baseline);
      ]
    ()

let ids = List.map (fun (it : Jord_stac.item) -> it.id)

let items_tests =
  group "items"
    [
      test "distinct keeps the highest baseline of an acquisition" (fun () ->
          let a =
            s2 ~id:"a" ~time:"2021-12-22T10:30:00Z" ~strip:"20211222T102436"
              ~baseline:"03.00"
          and b =
            s2 ~id:"b" ~time:"2021-12-22T10:30:00Z" ~strip:"20211222T102436"
              ~baseline:"05.10"
          and c =
            s2 ~id:"c" ~time:"2021-12-22T10:30:00Z" ~strip:"20211222T102744"
              ~baseline:"03.00"
          and d =
            s2 ~id:"d" ~time:"2021-12-20T10:30:00Z" ~strip:"20211220T102436"
              ~baseline:"04.00"
          in
          equal (list string) [ "d"; "b"; "c" ]
            (ids (Sentinel2.distinct [ a; c; b; d ])));
      test "days groups by UTC date and leaves out items lacking an asset"
        (fun () ->
          let a =
            s2 ~id:"a" ~time:"2025-07-01T23:59:59Z" ~strip:"20250701T235959"
              ~baseline:"05.11"
          and b =
            s2 ~id:"b" ~time:"2025-07-02T00:00:00Z" ~strip:"20250702T000000"
              ~baseline:"05.11"
          and c =
            item ~assets:[ "B04" ] ~id:"c" ~time:"2025-07-02T00:00:00Z" ()
          in
          let days, left = Sentinel2.days [ b; c; a ] in
          equal
            (list (pair (triple int int int) (list string)))
            [ ((2025, 7, 1), [ "a" ]); ((2025, 7, 2), [ "b" ]) ]
            (List.map (fun (d, l) -> (d, ids l)) days);
          equal (list string) [ "c" ] (ids left));
      test "days puts a leap second on the next day" (fun () ->
          let a =
            s2 ~id:"a" ~time:"2016-12-31T23:59:59Z" ~strip:"20161231T235959"
              ~baseline:"05.11"
          and b =
            s2 ~id:"b" ~time:"2016-12-31T23:59:60Z" ~strip:"20161231T235960"
              ~baseline:"05.11"
          in
          let days, _ = Sentinel2.days [ b; a ] in
          equal
            (list (pair (triple int int int) (list string)))
            [ ((2016, 12, 31), [ "a" ]); ((2017, 1, 1), [ "b" ]) ]
            (List.map (fun (d, l) -> (d, ids l)) days));
      test "Sentinel1.days groups by date and orbit" (fun () ->
          let s1 id time orbit =
            item ~assets:[ "vv"; "vh" ]
              ~props:[ ("sat:orbit_state", orbit) ]
              ~id ~time ()
          in
          let days, left =
            Sentinel1.days
              [
                s1 "b" "2025-07-01T17:40:00Z" "ascending";
                s1 "a" "2025-07-01T05:50:00Z" "descending";
                s1 "c" "2025-07-01T17:40:00Z" "ascending";
                item ~assets:[ "vv" ]
                  ~props:[ ("sat:orbit_state", "ascending") ]
                  ~id:"d" ~time:"2025-07-01T17:40:00Z" ();
              ]
          in
          let orbit = function
            | Jord_stac.Ascending -> "a"
            | Descending -> "d"
            | Geostationary -> "g"
            | Crossing -> "c"
          in
          equal
            (list (pair string (list string)))
            [ ("d", [ "a" ]); ("a", [ "b"; "c" ]) ]
            (List.map (fun ((_, o), l) -> (orbit o, ids l)) days);
          equal (list string) [ "d" ] (ids left));
      test "Sentinel1.days refuses an item of no polar orbit state" (fun () ->
          let s1 ?props id =
            item ~assets:[ "vv"; "vh" ] ?props ~id ~time:"2025-07-01T00:00:00Z"
              ()
          in
          let geo = [ ("sat:orbit_state", "geostationary") ] in
          raises_match (Exn.failure ~sub:"x") (fun () ->
              ignore (Sentinel1.days [ s1 ~props:geo "x" ]));
          raises_match (Exn.failure ~sub:"y") (fun () ->
              ignore (Sentinel1.days [ s1 "y" ]));
          let crossing = [ ("sat:orbit_state", "crossing") ] in
          raises_match (Exn.failure ~sub:"z") (fun () ->
              ignore (Sentinel1.days [ s1 ~props:crossing "z" ])));
      test "distinct compares baselines as numbers and refuses another form"
        (fun () ->
          let a =
            s2 ~id:"a" ~time:"2021-12-22T10:30:00Z" ~strip:"20211222T102436"
              ~baseline:"10.00"
          and b =
            s2 ~id:"b" ~time:"2021-12-22T10:30:00Z" ~strip:"20211222T102436"
              ~baseline:"09.99"
          in
          equal (list string) [ "a" ] (ids (Sentinel2.distinct [ b; a ]));
          let c =
            s2 ~id:"c" ~time:"2021-12-22T10:30:00Z" ~strip:"20211222T102744"
              ~baseline:"5.10"
          in
          raises_match (Exn.failure ~sub:"c") (fun () ->
              ignore (Sentinel2.distinct [ c ])));
    ]

let () = exit (run "tessera.recipe" [ sentinel2; sentinel1; items_tests ])
