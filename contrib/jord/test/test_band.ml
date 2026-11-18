(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord

let band ?(kind = Band.Measurement) ?(scale = 1.) ?(offset = 0.) ?nodata
    ?saturated () =
  {
    Band.name = None;
    common_name = None;
    center_wavelength = None;
    full_width_half_max = None;
    unit = None;
    kind;
    scale;
    offset;
    nodata;
    saturated;
  }

let floats x = Nx.to_array (Nx.cast Nx.float64 x)

(* [scale · dn + offset] in two roundings, as nx computes it: a product that
   OCaml's arm64 backend cannot fuse into the addition. *)
let affine scale offset dn = Sys.opaque_identity (dn *. scale) +. offset

(* Landsat Collection 2's surface reflectance: scale 2.75e-5, offset -0.2,
   fill 0. Sentinel-2 L2A from baseline 04.00: (DN - 1000) / 10000, NODATA 0
   and SATURATED 65535. *)
let landsat = band ~scale:2.75e-5 ~offset:(-0.2) ~nodata:0. ()

let sentinel2 =
  band ~scale:1e-4 ~offset:(-0.1) ~nodata:0. ~saturated:65535. ()

let products =
  test "a product's DNs decode by its scale, offset and special values"
    (fun () ->
      let dn = Nx.create Nx.uint16 [| 4 |] [| 0; 7273; 43636; 65535 |] in
      let r = affine 2.75e-5 (-0.2) in
      let v, ok = Band.decode landsat Nx.float64 dn in
      equal (array float_exact)
        [| 0.; r 7273.; r 43636.; r 65535. |]
        (floats v);
      equal (array bool) [| false; true; true; true |] (Nx.to_array ok);
      let v, ok = Band.decode sentinel2 Nx.float64 dn in
      let r = affine 1e-4 (-0.1) in
      equal (array float_exact) [| 0.; r 7273.; r 43636.; 0. |] (floats v);
      equal (array bool) [| false; true; true; false |] (Nx.to_array ok);
      equal (array bool) [| false; false; false; true |]
        (Nx.to_array (Band.saturated sentinel2 dn));
      equal (array bool) [| false; false; false; false |]
        (Nx.to_array (Band.saturated landsat dn)))

let specials =
  test "NaN and infinite special values match themselves alone" (fun () ->
      let x =
        Nx.create Nx.float32 [| 4 |]
          [| Float.nan; Float.infinity; Float.neg_infinity; -32768. |]
      in
      let mask b = Nx.to_array (snd (Band.decode b Nx.float32 x)) in
      equal (array bool) [| false; true; true; true |]
        (mask (band ~nodata:Float.nan ()));
      equal (array bool) [| true; false; true; true |]
        (mask (band ~nodata:Float.infinity ()));
      (* Sentinel-1 RTC on Microsoft's Planetary Computer: a NaN is data. *)
      let v, ok = Band.decode (band ~nodata:(-32768.) ()) Nx.float32 x in
      equal (array bool) [| true; true; true; false |] (Nx.to_array ok);
      equal (array float_exact)
        [| Float.nan; Float.infinity; Float.neg_infinity; 0. |]
        (floats v))

(* DNs of every 16-bit value, the special ones often, and a band. *)
let dns =
  Gen.(
    let dn =
      frequency
        [ (6, int_range (-32768) 32767); (1, of_list [ 0; -1; -32768 ]) ]
    in
    let special = option (of_list [ 0.; -1.; -32768.; 7. ]) in
    let+ x = array ~size:(int_range 0 12) dn
    and+ scale = float_range (-2.) 2.
    and+ offset = float_range (-1000.) 1000.
    and+ nodata = special
    and+ saturated = special in
    (x, band ~scale ~offset ?nodata ?saturated ()))

let law =
  prop "decode is scale · DN + offset off the special values" dns
    (fun (x, b) ->
      let dn = Nx.create Nx.int16 [| Array.length x |] x in
      let special d =
        let d = Some (float_of_int d) in
        d = b.nodata || d = b.saturated
      in
      let want =
        Array.map
          (fun d ->
            if special d then 0. else affine b.scale b.offset (float_of_int d))
          x
      in
      let v, ok = Band.decode b Nx.float64 dn in
      equal ~msg:"values" (array float_exact) want (floats v);
      equal ~msg:"mask" (array bool)
        (Array.map (fun d -> not (special d)) x)
        (Nx.to_array ok);
      equal ~msg:"float32 rounds once" (array float_exact)
        (floats (Nx.cast Nx.float32 v))
        (floats (fst (Band.decode b Nx.float32 dn))))

let exact =
  test "a DN is special where it equals the value in its own dtype" (fun () ->
      (* 2^53 + 1 is no float64, whose nearest is 2^53. *)
      let x =
        Nx.create Nx.int64 [| 3 |]
          [| 0x20000000000000L; 0x20000000000001L; 0L |]
      in
      let b = band ~nodata:0x1p53 ~saturated:0x1p53 () in
      let valid b x = Nx.to_array (snd (Band.decode b Nx.float64 x)) in
      equal (array bool) [| false; true; true |] (valid b x);
      equal (array bool) [| true; false; false |]
        (Nx.to_array (Band.saturated b x));
      (* int64's largest, 2^63 - 1, rounds to 2^63 in float64; no int64 is
         2^63. *)
      let x = Nx.create Nx.int64 [| 1 |] [| Int64.max_int |] in
      equal (array bool) [| true |] (valid (band ~nodata:0x1p63 ()) x);
      (* No int8 is -1.5. *)
      let x = Nx.create Nx.int8 [| 2 |] [| -1; -2 |] in
      equal (array bool) [| true; true |] (valid (band ~nodata:(-1.5) ()) x);
      (* No uint16 is -9999 and no float32 is 0.1. *)
      let x = Nx.create Nx.uint16 [| 2 |] [| 0; 55537 |] in
      equal (array bool) [| true; true |] (valid (band ~nodata:(-9999.) ()) x);
      let x = Nx.create Nx.float32 [| 1 |] [| 0.1 |] in
      equal (array bool) [| true |] (valid (band ~nodata:0.1 ()) x))

let spectral =
  test "Sentinel-2's bands state no centre or width their metadata does not"
    (fun () ->
      let m =
        {
          Sentinel2.baseline = "03.01";
          quantification = 10000.;
          offsets = [];
          nodata = 0;
          saturated = 65535;
          wavelengths = [];
        }
      in
      let b = Result.get_ok (Sentinel2.reflectance m Sentinel2.B01) in
      is_true (b.center_wavelength = None && b.full_width_half_max = None))

let refusals =
  test "decode refuses bitfield and classified bands, and dtypes of no real \
        numbers" (fun () ->
      let qa = band ~kind:Bitfield ~nodata:1. () in
      raises_match
        (Exn.invalid_arg
           ~sub:
             "read as bits; mask its fill with its product module's fill, \
              or with Band.nodata where the module has none")
        (fun () -> Band.decode qa Nx.float32 (Nx.zeros Nx.uint16 [| 2 |]));
      let water =
        {
          Band.value = 80;
          name = None;
          title = None;
          description = Some "Water";
          nodata = false;
        }
      in
      let map = band ~kind:(Classes [ water ]) ~nodata:0. () in
      raises_match
        (Exn.invalid_arg ~sub:"are classes; mask its fill with Band.nodata")
        (fun () -> Band.decode map Nx.float32 (Nx.zeros Nx.uint8 [| 2 |]));
      raises_match (Exn.invalid_arg ~sub:"not a real dtype") (fun () ->
          Band.decode landsat Nx.float32 (Nx.zeros Nx.bool [| 2 |])))

(* A bitfield or classified band's fill is masked without decoding, in
   the DNs' own dtype: Landsat's QA_PIXEL fill 1, and a land cover map's 0
   that its uint8 holds, where -9999 is held by no DN. *)
let nodata_mask =
  test "nodata masks a band of any kind, in the DNs' dtype" (fun () ->
      let qa = band ~kind:Bitfield ~nodata:1. () in
      let dn = Nx.create Nx.uint16 [| 3 |] [| 1; 21824; 0 |] in
      equal (array bool) [| true; false; false |]
        (Nx.to_array (Band.nodata qa dn));
      equal (array bool) [| true; false |]
        (Nx.to_array
           (Band.nodata (band ~nodata:0. ())
              (Nx.create Nx.uint16 [| 2 |] [| 0; 5 |])));
      let map = band ~kind:(Classes []) ~nodata:0. () in
      let codes = Nx.create Nx.uint8 [| 2 |] [| 80; 0 |] in
      equal (array bool) [| false; true |]
        (Nx.to_array (Band.nodata map codes));
      equal (array bool) [| false; false |]
        (Nx.to_array
           (Band.nodata (band ~nodata:(-9999.) ())
              (Nx.create Nx.uint16 [| 2 |] [| 0; 55537 |])));
      equal (array bool) [| false |]
        (Nx.to_array (Band.nodata (band ()) (Nx.zeros Nx.uint8 [| 1 |])));
      raises_match (Exn.invalid_arg ~sub:"Jord.Band.nodata") (fun () ->
          Band.nodata qa (Nx.zeros Nx.bool [| 1 |])))

let () =
  exit
    (run "jord.band"
       [
         group "decode"
           [ products; specials; exact; spectral; law; refusals; nodata_mask ];
       ])
