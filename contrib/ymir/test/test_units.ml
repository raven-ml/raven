(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Astronomy's units, against their definitions: the IAU 2015 parsec, 648000/π
   au, the Julian year, 365.25 days of 86400 s, and the AB system's maggy,
   10^(−48.6/2.5) erg s^-1 cm^-2 Hz^-1 (Oke & Gunn 1983). Expected numbers are
   the exact values rounded once, computed with mpmath at 400 bits. *)

open Windtrap
open Ymir

let unit = Testable.make ~pp:Unit.pp ~equal:Unit.equal

let definitions =
  group "definitions"
    [
      test "the parsec subtends one arcsecond per astronomical unit" (fun () ->
          equal unit
            Unit.(astronomical_unit / arcsecond)
            Unit.(Units.parsec / radian));
      test "the parsec in metres, rounded once" (fun () ->
          (* IAU 2015 Resolution B2: 3.08567758149136727e16 m. *)
          equal float_exact 3.0856775814913673e16
            (Unit.ratio Nx.float64 Units.parsec Unit.metre));
      test "the Julian year is 31557600 s" (fun () ->
          equal unit Unit.(int 31557600 * second) Units.julian_year);
      test "the Julian year is 365.25 days" (fun () ->
          equal float_exact 365.25
            (Unit.ratio Nx.float64 Units.julian_year Unit.day));
    ]

(* Photometry *)

let jansky = Unit.(decimal "1e-26" * watt / (metre ** 2) / hertz)
let erg = Unit.(decimal "1e-7" * joule)
let cgs = Unit.(erg / second / (centi metre ** 2) / hertz)
let maggy = Units.maggy
let nanomaggy = Unit.nano maggy
let zero_point = Units.zero_point

(* [ten_to n d] is 10^(n/d). *)
let ten_to n d = Unit.(root d (int 10) ** n)

let maggies =
  group "the maggy"
    [
      test "is 10^(−561/25) W m^-2 Hz^-1, held exactly" (fun () ->
          (* W m^-2 Hz^-1 is kg s^-2; a prime with a fractional exponent is its
             own item of the canonical text. *)
          equal string "2^-561/25 5^-561/25 kg s^-2" (Unit.to_string maggy));
      test "is 10^(−48.6/2.5) erg s^-1 cm^-2 Hz^-1" (fun () ->
          equal unit Unit.(ten_to (-486) 25 * cgs) maggy);
      test "is 3630.7805477010133 Jy, 10^3.56 rounded once" (fun () ->
          equal float_exact 3630.7805477010133
            (Unit.ratio Nx.float64 maggy jansky));
    ]

let zero_points =
  cases "zero_point m is 10^(−0.4 m) maggy" ~id:fst
    [
      ("0", maggy);
      ("-0", maggy);
      ("22.5", nanomaggy);
      ("2.25e1", nanomaggy);
      ("225e-1", nanomaggy);
      ("25", Unit.(decimal "1e-10" * maggy));
      ("27", Unit.(ten_to (-54) 5 * maggy));
      ("-2.5", Unit.(int 10 * maggy));
      ("1e1", Unit.(decimal "1e-4" * maggy));
      ("0.0001", Unit.(ten_to (-1) 25000 * maggy));
      ("-48.6", cgs);
    ]
    (fun (m, u) -> equal unit u (zero_point m))

(* Values in float64 on zero point [from] read in zero point [into]: one
   multiply by 10^(−0.4 (from − into)), the factor and the product each rounded
   once. *)
let conversions =
  cases "Quantity.value converts between zero points"
    ~id:(fun (from, into, x, _) -> Printf.sprintf "%s to %s, %h" from into x)
    [
      ("27", "22.5", 1., 0.015848931924611134);
      ("27", "22.5", 1234.5, 19.565506460932447);
      ("27", "22.5", -7.25, -0.11490475645343072);
      ("27", "25", 1., 0.15848931924611134);
      ("31.4", "22.5", 1., 0.0002754228703338166);
      ("23.5", "22.5", 3., 1.1943215116604917);
      ("28.9", "22.5", 0.5, 0.0013771143516690832);
      ("22.5", "27", 1., 63.09573444801932);
      ("22.5", "25", 1., 10.);
      ("0", "22.5", 1., 1e9);
      ("-48.6", "0", 1., 2.7542287033381663e+19);
    ]
    (fun (from, into, x, expected) ->
      let q = Quantity.v (zero_point from) (Nx.scalar Nx.float64 x) in
      equal float_exact expected
        (Nx.item [] (Quantity.value (zero_point into) q)))

(* HSC coadds are on zero point 27, and AION-1's codec rescales them to the
   Legacy Surveys' 22.5 by multiplying float32 pixels by the float32 reciprocal
   of 10^1.8, 0x3c81d59f: f32(10^−1.8) correctly rounded, and (float)(1.0 /
   (double)b) for b = f32(10^1.8) = 0x427c6208. *)
let hsc_factor = 0x3c81d59fl
let hsc_scale = 0x427c6208l
let bits_of x = Int32.bits_of_float x

(* Float32 values: any bit pattern, specials, and subnormals of either sign. *)
let float32s =
  let special =
    Gen.of_list
      [ nan; infinity; neg_infinity; 0.; -0.; 0x1p-149; -0x1p-149; 0x1p-126 ]
  in
  let subnormal =
    Gen.map
      (fun b -> Int32.float_of_bits (Int32.logand b 0x807fffffl))
      Gen.int32
  in
  let any = Gen.map Int32.float_of_bits Gen.int32 in
  Gen.list ~size:(Gen.constant 64)
    (Gen.frequency [ (6, any); (1, special); (1, subnormal) ])

let hsc =
  group "HSC's rescale"
    [
      test "the factor from zero point 27 to 22.5 is 0x3c81d59f in float32"
        (fun () ->
          equal int32 hsc_factor
            (bits_of (Unit.ratio Nx.float32 (zero_point "27") nanomaggy));
          equal int32 hsc_scale
            (bits_of (Unit.ratio Nx.float32 nanomaggy (zero_point "27")));
          equal int32 hsc_factor (bits_of (1. /. Int32.float_of_bits hsc_scale)));
      prop "Quantity.value multiplies float32 pixels by 0x3c81d59f" float32s
        (fun xs ->
          let factor = Int32.float_of_bits hsc_factor in
          (* A product of two float32 values is exact in float64, so one
             rounding to float32 gives the float32 product. *)
          let product x = Int32.float_of_bits (bits_of (x *. factor)) in
          let tiny x = Float.abs (product x) < 0x1p-126 && product x <> 0. in
          cover "NaN" (List.exists Float.is_nan xs);
          cover "infinite" (List.exists (fun x -> Float.abs x = infinity) xs);
          cover "a subnormal product" (List.exists tiny xs);
          let pixels =
            Nx.create Nx.float32 [| List.length xs |] (Array.of_list xs)
          in
          let q = Quantity.v (zero_point "27") pixels in
          equal (array float_exact)
            (Array.of_list (List.map product xs))
            (Nx.to_array (Quantity.value nanomaggy q)));
    ]

let grammar =
  cases "zero_point raises outside its grammar" ~id:(Printf.sprintf "%S")
    [
      "";
      "-";
      "+27";
      "27.";
      ".5";
      "2e";
      "e1";
      "27 ";
      "--1";
      "1.5.2";
      "1E2";
      "1e+2";
    ] (fun m ->
      raises_match (Exn.invalid_arg ~sub:"Units.zero_point") (fun () ->
          zero_point m))

(* Magnitudes whose mantissa or exponent of 10 leaves int: 0.4 m is past int, or
   m's own digits are. *)
let past_int =
  cases "zero_point raises past int" ~id:(Printf.sprintf "%S")
    [
      "99999999999999999999";
      "1e99999999999999999999";
      "1e100000000";
      "25e4611686018427387903";
      "-1e-4611686018427387903";
      "1e-4611686018427387903";
    ] (fun m ->
      raises_match (Exn.invalid_arg ~sub:"Units.zero_point") (fun () ->
          zero_point m))

let () =
  exit
    (run "Units"
       [
         definitions; maggies; zero_points; conversions; hsc; grammar; past_int;
       ])
