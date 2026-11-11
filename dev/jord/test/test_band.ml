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

let refusals =
  test "decode refuses bitfield bands and dtypes of no real numbers"
    (fun () ->
      let qa = band ~kind:Bitfield ~nodata:1. () in
      raises_match (Exn.invalid_arg ~sub:"bitfield") (fun () ->
          Band.decode qa Nx.float32 (Nx.zeros Nx.uint16 [| 2 |]));
      raises_match (Exn.invalid_arg ~sub:"not a real dtype") (fun () ->
          Band.decode landsat Nx.float32 (Nx.zeros Nx.bool [| 2 |])))

let () =
  exit
    (run "jord.band"
       [ group "decode" [ products; specials; law; refusals ] ])
