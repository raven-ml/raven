(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord
open Modis

let bools x = Nx.to_array x

let product_t =
  Testable.make
    ~pp:(fun ppf p ->
      Format.pp_print_string ppf
        (match p with Mod13q1 -> "MOD13Q1" | Mod09a1 -> "MOD09A1"))
    ~equal:( = )

(* Bands *)

let reflectance = (None, 0.0001)
let angle = (Some "degree", 0.01)
let count = (None, 1.)

(* MOD13 Table 1 and MOD09 Table 6, with LP DAAC's names: each layer's name,
   common name, unit and scale, fill and valid range. *)
let rows = function
  | Mod13q1 ->
      let l = "250m 16 days " in
      [
        (Ndvi, l ^ "NDVI", None, reflectance, -3000., (-2000, 10000));
        (Evi, l ^ "EVI", None, reflectance, -3000., (-2000, 10000));
        (Vi_quality, l ^ "VI Quality", None, count, 65535., (0, 65534));
        ( Red,
          l ^ "red reflectance",
          Some "red",
          reflectance,
          -1000.,
          (0, 10000) );
        ( Nir,
          l ^ "NIR reflectance",
          Some "nir",
          reflectance,
          -1000.,
          (0, 10000) );
        ( Blue,
          l ^ "blue reflectance",
          Some "blue",
          reflectance,
          -1000.,
          (0, 10000) );
        ( Swir22,
          l ^ "MIR reflectance",
          Some "swir22",
          reflectance,
          -1000.,
          (0, 10000) );
        ( View_zenith,
          l ^ "view zenith angle",
          None,
          angle,
          -10000.,
          (0, 18000) );
        ( Sun_zenith,
          l ^ "sun zenith angle",
          None,
          angle,
          -10000.,
          (0, 18000) );
        ( Relative_azimuth,
          l ^ "relative azimuth angle",
          None,
          angle,
          -4000.,
          (-18000, 18000) );
        ( Day_of_year,
          l ^ "composite day of the year",
          None,
          count,
          -1.,
          (1, 366) );
        (Reliability, l ^ "pixel reliability", None, count, -1., (0, 3));
      ]
  | Mod09a1 ->
      let refl b n common =
        (b, "sur_refl_b0" ^ n, common, reflectance, -28672., (-100, 16000))
      in
      [
        refl Red "1" (Some "red");
        refl Nir "2" (Some "nir");
        refl Blue "3" (Some "blue");
        refl Green "4" (Some "green");
        refl Swir12 "5" None;
        refl Swir16 "6" (Some "swir16");
        refl Swir22 "7" (Some "swir22");
        (Qc, "sur_refl_qc_500m", None, count, 4294967295., (0, 4294966531));
        (Sun_zenith, "sur_refl_szen", None, angle, 0., (0, 18000));
        (View_zenith, "sur_refl_vzen", None, angle, 0., (0, 18000));
        ( Relative_azimuth,
          "sur_refl_raz",
          None,
          angle,
          0.,
          (-18000, 18000) );
        (State, "sur_refl_state_500m", None, count, 65535., (0, 57343));
        ( Day_of_year,
          "sur_refl_day_of_year",
          None,
          count,
          65535.,
          (1, 366) );
      ]

let kind = function
  | Vi_quality | Qc | State -> `Bits
  | Reliability -> `Classes
  | _ -> `Measurement

let tables =
  cases "each layer as the guide's tables state it"
    ~id:(fun p -> Format.asprintf "%a" (Testable.pp product_t) p)
    [ Mod13q1; Mod09a1 ]
    (fun p ->
      equal int (List.length (rows p)) (List.length (bands p));
      List.iter2
        (fun (b, n, common_name, (unit, scale), nodata, range) b' ->
          is_true ~msg:n (b = b');
          equal string n (name p b);
          equal ~msg:n (pair int int) range (valid_range p b);
          let got = band p b in
          let k : Band.kind =
            match (kind b, got.kind) with
            | `Bits, _ -> Bitfield
            | `Classes, (Classes _ as k) -> k
            | `Classes, _ -> failf "%s: not a classified band" n
            | `Measurement, _ -> Measurement
          in
          let want =
            {
              Band.name = Some n;
              common_name;
              center_wavelength = None;
              full_width_half_max = None;
              unit;
              kind = k;
              scale;
              offset = 0.;
              nodata = Some nodata;
              saturated = None;
            }
          in
          is_true ~msg:n (got = want))
        (rows p) (bands p))

let absent =
  test "a layer a product lacks raises" (fun () ->
      List.iter
        (fun (p, b) ->
          let raises f =
            raises_match (Exn.invalid_arg ~sub:"have no such band") f
          in
          raises (fun () -> name p b);
          raises (fun () -> band p b);
          raises (fun () -> valid_range p b))
        [
          (Mod13q1, Green); (Mod13q1, Swir12); (Mod13q1, Swir16);
          (Mod13q1, Qc); (Mod13q1, State); (Mod09a1, Ndvi); (Mod09a1, Evi);
          (Mod09a1, Vi_quality); (Mod09a1, Reliability);
        ])

(* MOD13 Table 4. *)
let reliability =
  test "pixel reliability: Table 4's ranks, -1 its fill" (fun () ->
      let classes =
        match (band Mod13q1 Reliability).kind with
        | Classes l -> l
        | _ -> fail "not a classified band"
      in
      equal (list int) [ -1; 0; 1; 2; 3 ]
        (List.map (fun (c : Band.class_) -> c.value) classes);
      equal (list bool) [ true; false; false; false; false ]
        (List.map (fun (c : Band.class_) -> c.nodata) classes);
      equal (list (option string))
        [
          Some "Fill/No Data"; Some "Good Data"; Some "Marginal data";
          Some "Snow/Ice"; Some "Cloudy";
        ]
        (List.map (fun (c : Band.class_) -> c.title) classes);
      let dn = Nx.create Nx.int8 [| 6 |] [| -1; 0; 1; 2; 3; 4 |] in
      equal (array bool)
        [| false; true; true; false; false; false |]
        (bools (Reliability.mem [ Good; Marginal ] dn));
      equal (array bool)
        [| true; false; false; false; true; false |]
        (bools (Reliability.mem [ Fill; Cloudy ] dn)))

let decode =
  test "a layer's DNs decode by its scale, its fill masked" (fun () ->
      let dn = Nx.create Nx.int16 [| 3 |] [| -3000; -2000; 8123 |] in
      let v, ok = Band.decode (band Mod13q1 Ndvi) Nx.float64 dn in
      equal (array bool) [| false; true; true |] (bools ok);
      equal (array float_exact)
        [| 0.; -2000. *. 0.0001; 8123. *. 0.0001 |]
        (Nx.to_array v);
      (* Fills inside the valid range are fill. Decoding drops the
         observations that lie there: on Microsoft's Planetary Computer's
         tile h18v04 of 2023-08-29, 2050 MOD13Q1 pixels of valid NDVI hold a
         relative azimuth of -40.00°, as often as -39.99° and -40.01° occur,
         and 11 MOD09A1 pixels of valid reflectance a relative azimuth of
         0.00°. *)
      let dn = Nx.create Nx.int16 [| 3 |] [| -4000; -3999; 18000 |] in
      let v, ok = Band.decode (band Mod13q1 Relative_azimuth) Nx.float64 dn in
      equal (array bool) [| false; true; true |] (bools ok);
      equal (array float_exact)
        [| 0.; -3999. *. 0.01; 18000. *. 0.01 |]
        (Nx.to_array v);
      let dn = Nx.create Nx.int16 [| 2 |] [| 0; 1 |] in
      let _, ok = Band.decode (band Mod09a1 Sun_zenith) Nx.float64 dn in
      equal (array bool) [| false; true |] (bools ok);
      let dn = Nx.create Nx.int16 [| 2 |] [| -28672; -100 |] in
      let _, ok = Band.decode (band Mod09a1 Swir12) Nx.float64 dn in
      equal (array bool) [| false; true |] (bools ok);
      raises_match (Exn.invalid_arg ~sub:"bitfield") (fun () ->
          Band.decode (band Mod09a1 Qc) Nx.float32
            (Nx.zeros Nx.uint32 [| 2 |])))

(* Grid *)

let ulx = 20015109.354
let r = 6371007.181

let corner g (row, col) =
  let p = Nx.create Nx.float64 [| 2 |] [| row; col |] in
  let x = Nx.to_array (Grid.world g p) in
  (x.(0), x.(1))

let xy = pair float_exact float_exact
(* To an ulp of the plane's largest coordinates, 3.7e-9 m. *)
let near = pair (float 4e-9) (float 4e-9)

let corners =
  test "tiles cover the sinusoid from MODLAND's corner" (fun () ->
      let g = tile Mod13q1 ~h:0 ~v:0 in
      is_true (Crs.equal Crs.modis g.crs);
      equal int 4800 g.rows;
      equal int 4800 g.cols;
      equal xy (-.ulx, ulx /. 2.) g.origin;
      equal xy (0., -.(ulx /. 86400.)) g.row_step;
      equal xy (ulx /. 86400., 0.) g.col_step;
      equal near (ulx, -.ulx /. 2.)
        (let g = tile Mod13q1 ~h:35 ~v:17 in
         corner g (4800., 4800.));
      (* The tile of MPC's fixtures, h18v04, whose corner is on the central
         meridian, and its 500 m grid. *)
      let g = tile Mod09a1 ~h:18 ~v:4 in
      equal int 2400 g.rows;
      equal xy (0., 5559752.598333333) g.origin;
      equal xy (463.31271652777775, 0.) g.col_step;
      (* Neighbours share their edges. *)
      for h = 0 to 34 do
        for v = 0 to 16 do
          let g = tile Mod09a1 ~h ~v in
          equal near (corner g (0., 2400.)) (tile Mod09a1 ~h:(h + 1) ~v).origin;
          equal near (corner g (2400., 0.)) (tile Mod09a1 ~h ~v:(v + 1)).origin
        done
      done;
      List.iter
        (fun (h, v) ->
          raises_match (Exn.invalid_arg ~sub:"is not a tile") (fun () ->
              tile Mod13q1 ~h ~v))
        [ (-1, 0); (36, 0); (0, -1); (0, 18) ])

(* The corner's x is πR truncated to the millimetre, so the tiles are 10° of
   latitude, and of longitude at the equator, to the 1.8 mm it falls short
   by: under 2e-8°. *)
let degrees =
  test "tiles are 10° of the sphere, short by πR's truncation" (fun () ->
      let short = (Float.pi *. r) -. ulx in
      is_true ~msg:"1.8 mm" (short > 0.0017974 && short < 0.0017975);
      let lonlat x y =
        let p = Nx.create Nx.float64 [| 1; 2 |] [| x; y |] in
        let q = Nx.to_array (Crs.transform ~src:Crs.modis ~dst:Crs.lonlat p) in
        (q.(0), q.(1))
      in
      let deg = pair (float 2e-8) (float 2e-8) in
      for v = 0 to 17 do
        let x, y = (tile Mod13q1 ~h:18 ~v).origin in
        equal deg (0., 90. -. (10. *. Float.of_int v)) (lonlat x y)
      done;
      for h = 0 to 35 do
        let x, _ = (tile Mod13q1 ~h ~v:9).origin in
        equal deg ((10. *. Float.of_int h) -. 180., 0.) (lonlat x 0.)
      done;
      (* MODLAND's table of tile bounds gives h18v04 latitudes 40° to 50°
         and longitudes 0° to 15.5702°: 7°E, 45°N lies inside it. *)
      let p = Nx.create Nx.float64 [| 1; 2 |] [| 7.; 45. |] in
      let q =
        Nx.to_array
          (Grid.pixel (tile Mod13q1 ~h:18 ~v:4)
             (Crs.transform ~src:Crs.lonlat ~dst:Crs.modis p))
      in
      is_true (q.(0) >= 0. && q.(0) < 4800. && q.(1) >= 0. && q.(1) < 4800.))

(* Quality *)

let u16 l = Nx.create Nx.uint16 [| List.length l |] (Array.of_list l)

let vi_quality =
  test "VI quality: Table 5's fields" (fun () ->
      (* Bits 0-1 01, usefulness 0010, aerosol 01, mixed clouds, land 001,
         shadow: 1 + 2·4 + 1·64 + 1024 + 1·2048 + 32768. *)
      let dn = u16 [ 1 + 8 + 64 + 1024 + 2048 + 32768; 0; 65535 ] in
      equal (array bool) [| true; false; false |]
        (bools (Vi_quality.modland [ Other_quality ] dn));
      equal (array bool) [| false; true; false |]
        (bools (Vi_quality.modland [ Good_quality ] dn));
      equal (array bool) [| true; false; false |]
        (bools (Vi_quality.usefulness [ 2 ] dn));
      equal (array bool) [| false; false; true |]
        (bools (Vi_quality.usefulness [ 15; 16; -1 ] dn));
      equal (array bool) [| true; false; false |]
        (bools (Vi_quality.aerosol [ Low ] dn));
      equal (array bool) [| true; false; true |]
        (bools (Vi_quality.any [ Mixed_clouds ] dn));
      equal (array bool) [| true; false; true |]
        (bools (Vi_quality.any [ Shadow ] dn));
      equal (array bool) [| false; false; true |]
        (bools
           (Vi_quality.any [ Adjacent_cloud; Brdf_corrected; Snow_ice ] dn));
      equal (array bool) [| true; false; false |]
        (bools (Vi_quality.land_water [ Land ] dn));
      equal (array bool) [| false; true; false |]
        (bools (Vi_quality.land_water [ Shallow_ocean ] dn));
      equal (array bool) [| false; false; true |]
        (bools (Vi_quality.land_water [ Deep_ocean ] dn));
      equal (array bool) [| false; false; true |] (bools (Vi_quality.fill dn));
      equal (array bool) [| true |]
        (bools (Vi_quality.fill (Nx.create Nx.int16 [| 1 |] [| -1 |]))))

let qc =
  test "reflectance quality: Table 10's fields" (fun () ->
      (* MODLAND 10, band 1 at 1001 (9, the Sun at 86° or more), band 7 at
         1111, the atmospheric correction run. *)
      let dn =
        Nx.create Nx.uint32 [| 3 |]
          [|
            Int32.of_int (2 + (9 lsl 2) + (15 lsl 26) + (1 lsl 30));
            0l;
            -1l;
          |]
      in
      equal (array bool) [| true; false; false |]
        (bools (Qc.modland [ Not_produced_cloud ] dn));
      equal (array bool) [| true; false; false |]
        (bools (Qc.quality Red [ Sun_zenith_86 ] dn));
      equal (array bool) [| false; true; false |]
        (bools (Qc.quality Red [ Highest ] dn));
      equal (array bool) [| true; true; false |]
        (bools (Qc.quality Nir [ Highest ] dn));
      equal (array bool) [| true; false; true |]
        (bools (Qc.quality Swir22 [ Not_processed ] dn));
      equal (array bool) [| true; false; true |]
        (bools (Qc.any [ Atmospheric_correction ] dn));
      equal (array bool) [| false; false; true |]
        (bools (Qc.any [ Adjacency_correction ] dn));
      equal (array bool) [| false; false; true |] (bools (Qc.fill dn));
      (* A 16-bit DN holds no bit 30 and is never the 32-bit fill. *)
      equal (array bool) [| false |] (bools (Qc.fill (u16 [ 65535 ])));
      raises_match (Exn.invalid_arg ~sub:"not a reflectance band") (fun () ->
          Qc.quality State [ Highest ] dn))

let state =
  test "state flags: Table 13's fields" (fun () ->
      (* Cloudy 01, shadow, land 001, aerosol high 11, cirrus small 01,
         internal cloud, adjacent to cloud. *)
      let dn =
        u16
          [
            1 + 4 + (1 lsl 3) + (3 lsl 6) + (1 lsl 8) + (1 lsl 10) + (1 lsl 13);
            0;
          ]
      in
      equal (array bool) [| true; false |] (bools (State.cloud [ Cloudy ] dn));
      equal (array bool) [| false; true |] (bools (State.cloud [ Clear ] dn));
      equal (array bool) [| true; false |]
        (bools (State.land_water [ Land ] dn));
      equal (array bool) [| true; false |] (bools (State.aerosol [ High ] dn));
      equal (array bool) [| false; true |]
        (bools (State.aerosol [ Climatology ] dn));
      equal (array bool) [| true; false |]
        (bools (State.cirrus [ Small_cirrus ] dn));
      equal (array bool) [| true; false |]
        (bools (State.any [ Cloud_shadow ] dn));
      equal (array bool) [| true; false |]
        (bools (State.any [ Internal_cloud; Adjacent_cloud ] dn));
      equal (array bool) [| false; false |]
        (bools (State.any [ Fire; Snow_ice; Salt_pan; Internal_snow ] dn));
      equal (array bool) [| false; false |] (bools (State.fill dn));
      equal (array bool) [| true |] (bools (State.fill (u16 [ 65535 ]))))

(* Table 10's named values, each alone in band 1's field. MOD09 Section 4.1
   calls 1011 "band quality value 11", missing input, which read from the
   lowest bit would be 13, so the combinations are written from their
   highest bit. Values 1 to 6 are named by none. *)
let qc_values =
  test "each named data quality is its value" (fun () ->
      let named =
        Qc.
          [
            (0, Highest); (7, Noisy_detector); (8, Dead_detector);
            (9, Sun_zenith_86); (10, Sun_zenith_85); (11, Missing_input);
            (12, Internal_constant); (13, Out_of_bounds); (14, L1b_faulty);
            (15, Not_processed);
          ]
      in
      let all = List.map snd named in
      List.iter
        (fun v ->
          let dn = Nx.create Nx.uint32 [| 1 |] [| Int32.of_int (v lsl 2) |] in
          List.iter
            (fun (v', q) ->
              equal
                ~msg:(Printf.sprintf "%d as %d" v v')
                (array bool) [| v = v' |]
                (bools (Qc.quality Red [ q ] dn)))
            named;
          if v >= 1 && v <= 6 then
            equal (array bool) [| false |] (bools (Qc.quality Red all dn)))
        (List.init 16 Fun.id))

(* Each one-bit flag alone, at the position its table gives: no other flag
   reads it. *)
let single_bits =
  test "each flag is its own bit" (fun () ->
      let check ~msg ~dtype ~any flags =
        List.iter
          (fun (b, f) ->
            let dn = Nx.create dtype [| 1 |] [| b |] in
            List.iter
              (fun (_, g) ->
                equal ~msg:(Printf.sprintf "%s bit %d" msg b) (array bool)
                  [| f = g |]
                  (bools (any [ g ] dn)))
              flags)
          flags
      in
      let alone bits = List.map (fun (b, f) -> (1 lsl b, f)) bits in
      check ~msg:"VI quality" ~dtype:Nx.uint16 ~any:Vi_quality.any
        (alone
           Vi_quality.
             [
               (8, Adjacent_cloud); (9, Brdf_corrected); (10, Mixed_clouds);
               (14, Snow_ice); (15, Shadow);
             ]);
      check ~msg:"state" ~dtype:Nx.uint16 ~any:State.any
        (alone
           State.
             [
               (2, Cloud_shadow); (10, Internal_cloud); (11, Fire);
               (12, Snow_ice); (13, Adjacent_cloud); (14, Salt_pan);
               (15, Internal_snow);
             ]);
      (* Bit 30 alone, then bit 31 alone. *)
      let qc = Nx.create Nx.uint32 [| 2 |] [| 0x40000000l; Int32.min_int |] in
      equal (array bool) [| true; false |]
        (bools (Qc.any [ Atmospheric_correction ] qc));
      equal (array bool) [| false; true |]
        (bools (Qc.any [ Adjacency_correction ] qc));
      (* A fill sets every bit: one bit short is not fill. *)
      equal (array bool) [| false; true |]
        (bools (Vi_quality.fill (u16 [ 32767; 65535 ])));
      equal (array bool) [| false; true |]
        (bools (State.fill (u16 [ 32767; 65535 ])));
      equal (array bool) [| false; true |]
        (bools
           (Qc.fill (Nx.create Nx.uint32 [| 2 |] [| 0x7FFFFFFFl; -1l |])));
      (* State's aerosol alone, without the cirrus above it. *)
      equal (array bool) [| true |]
        (bools (State.aerosol [ Average ] (u16 [ 2 lsl 6 ]))))

let () =
  exit
    (run "jord.modis"
       [
         group "bands" [ tables; absent; reliability; decode ];
         group "grid" [ corners; degrees ];
         group "quality" [ vi_quality; qc; state; qc_values; single_bits ];
       ])
