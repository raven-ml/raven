(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

(* Codes

   A float32's code is held in the low 32 bits of an int64, a float64's in all
   64. [wide] picks float64. *)

let low32 = 0xFFFFFFFFL

let code wide v =
  if wide then Int64.bits_of_float v
  else Int64.logand (Int64.of_int32 (Int32.bits_of_float v)) low32

let value wide c =
  if wide then Int64.float_of_bits c else Int32.float_of_bits (Int64.to_int32 c)

let sign wide = if wide then Int64.min_int else 0x80000000L
let mmask wide = if wide then 0x000FFFFFFFFFFFFFL else 0x007FFFFFL
let pick wide (a, b) = if wide then b else a
let is_nan wide c = Float.is_nan (value wide c)

(* The zeros, the least and greatest subnormals, the least normals, the greatest
   finite floats and the infinities of both signs, then NaNs: quiet, negative,
   with a payload and signalling. *)
let edges wide =
  let pos =
    if wide then
      [
        0L;
        1L;
        0x000FFFFFFFFFFFFFL;
        0x0010000000000000L;
        0x7FEFFFFFFFFFFFFFL;
        0x7FF0000000000000L;
      ]
    else [ 0L; 1L; 0x007FFFFFL; 0x00800000L; 0x7F7FFFFFL; 0x7F800000L ]
  in
  let nans =
    if wide then
      [
        0x7FF8000000000000L;
        0xFFF8000000000000L;
        0x7FF80000DEADBEEFL;
        0x7FF0000000000001L;
      ]
    else [ 0x7FC00000L; 0xFFC00000L; 0x7FC12345L; 0x7F800001L ]
  in
  List.concat_map (fun c -> [ c; Int64.logor c (sign wide) ]) pos @ nans

(* Kinds

   A spec gives one operand's draws at one dtype: the header's switch points and
   edges of the kind, positive, and the bounds of its uniform ranges. A kind of
   two operands may draw pairs too, as codes at float64 if its argument holds,
   else float32, such as operands that overflow its naive formula. A cover names
   a class of an element's operands. A kind's fixed elements, at float64 if
   their argument holds, else float32, open its example. *)

type spec = { points : float list; ranges : float list }
type f1 = { f1 : 's 'd. (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t }

type f2 = {
  f2 :
    's 'd. (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t;
}

type fn = One of f1 | Two of f2 * (bool -> int64 array Windtrap.Gen.t)

type kind = {
  name : string;
  fn : fn;
  specs : (spec * spec) array;  (** Per operand, at float32 and float64. *)
  covers : (string * (bool -> float array -> bool)) list;
  fixed : bool -> int64 array list;
  generic : bool;  (** Whether the law covers the special values. *)
}

let name k = k.name
let none _ = []

(* A Weyl sequence over the code space, whose codes spread over every exponent
   with significands of every pattern. *)
let weyl wide i =
  if wide then Int64.mul (Int64.of_int i) 0x9E3779B97F4A7C15L
  else Int64.logand (Int64.mul (Int64.of_int i) 0x9E3779B9L) low32

(* Pairs of Annex F's operands and small integers, each of either sign. *)
let annex_pairs wide =
  let vs =
    [ 0.; -0.; 1.; -1.; 2.; 0.5; 3.; Float.infinity; Float.neg_infinity ]
    @ [ Float.nan ]
  in
  List.concat_map
    (fun a -> List.map (fun b -> [| code wide a; code wide b |]) vs)
    vs

let arity k = Array.length k.specs
let spec_of wide k i = pick wide k.specs.(i)

(* Trigonometric kinds: nx_kinds.h's switch to Payne and Hanek's reduction, 2^15
   or 2^28, and the cutoff below which sin and tan are x, 2^-12 or 2^-27;
   multiples of pi/2 on either side of the switch, where the reduced argument
   cancels, among them 355 and 103993 times pi/2, whose ratios approach pi's;
   and large values to the greatest finite. *)

let half_pi = Float.pi /. 2.
let trig_big = (0x1p15, 0x1p28)
let trig_tiny = (0x1p-12, 0x1p-27)

let trig_spec wide =
  let big = pick wide trig_big in
  let below, past = if wide then (1e8, 2e8) else (20000., 20861.) in
  {
    points =
      ([
         pick wide trig_tiny; big; 2. *. big; below *. half_pi; past *. half_pi;
       ]
      @ List.map (fun k -> k *. half_pi) [ 1.; 2.; 3.; 4.; 5.; 355.; 103993. ]
      @
      if wide then [ 0x1.6ac5b262ca1ffp+849; 1e22; 1e300 ]
      else [ 105414350.; 2.1904577e38; 1e30 ]);
    ranges = [ Float.pi; (if wide then 1e9 else 1e5) ];
  }

let trig_covers =
  let mag wide e = Float.abs (value wide (code wide e.(0))) in
  [
    ( "below the cutoff, not zero",
      fun w e -> e.(0) <> 0. && mag w e < pick w trig_tiny );
    ( "from the cutoff to the switch",
      fun w e -> mag w e >= pick w trig_tiny && mag w e < pick w trig_big );
    ( "past the switch, finite",
      fun w e -> mag w e >= pick w trig_big && Float.is_finite e.(0) );
    ( "beside a multiple of pi/2",
      fun _ e ->
        let x = e.(0) in
        let q = Float.round (x /. half_pi) in
        Float.abs x < 1e9
        && q <> 0.
        && Float.abs (x -. (q *. half_pi)) <= 0x1p-20 *. Float.abs x );
  ]

(* Every window of nx_two_over_pi that Payne and Hanek's reduction reads, [g >>
   5] for [g] the exponent less 120 (float32) or 1045 (float64): the least and
   greatest exponent field past the switch reading each, with a Weyl
   significand, of both signs. *)
let trig_windows wide =
  let mbits, offset, first, last =
    if wide then (52, 1045, 1051, 2046) else (23, 120, 142, 254)
  in
  let window w =
    let lo = max first (offset + (32 * w))
    and hi = min last (offset + (32 * w) + 31) in
    List.concat_map
      (fun e ->
        let frac = Int64.logand (weyl wide (e + (7 * w))) (mmask wide) in
        let c = Int64.logor (Int64.shift_left (Int64.of_int e) mbits) frac in
        [ [| c |]; [| Int64.logor c (sign wide) |] ])
      [ lo; hi ]
  in
  List.concat_map window (List.init (((last - offset) / 32) + 1) Fun.id)

let trig name f =
  {
    name;
    fn = One f;
    specs = [| (trig_spec false, trig_spec true) |];
    covers = trig_covers;
    fixed = trig_windows;
    generic = true;
  }

let sin = trig "sin" { f1 = Nx.sin }
let cos = trig "cos" { f1 = Nx.cos }
let tan = trig "tan" { f1 = Nx.tan }

(* Exponentials: nx_kinds.h's clamps (EXP_HI and EXP_LO, EXP2_HI and EXP2_LO,
   EXPM1_LO, HYP_HI, TANH_HI), exp's near range (EXP_NEAR), the cutoffs below
   which expm1 is x (EXPM1_TINY) and tanh its polynomial (TANH_SMALL), where k
   passes its cap (EXPM1_KCAP and HYP_KCAP times ln2), where results overflow,
   turn subnormal and round to zero, and ln2/2, where the reduction's r is
   largest. *)

let half_ln2 = 0.34657359

let unary name f s32 s64 covers =
  {
    name;
    fn = One f;
    specs = [| (s32, s64) |];
    covers;
    fixed = none;
    generic = true;
  }

let spec points ranges = { points; ranges }
let past (a, b) w v = v > pick w (a, b) && Float.is_finite v
let below (a, b) w v = v < pick w (a, b) && Float.is_finite v

let exp =
  unary "exp" { f1 = Nx.exp }
    (spec
       [
         89.; 104.; 86.; 88.72284; 87.33655; 103.27893; 103.97208; half_ln2; 1.;
       ]
       [ 1.; 100. ])
    (spec
       [
         710.;
         746.;
         708.;
         709.782712893384;
         708.3964185322641;
         744.4400719213812;
         745.1332191019411;
         half_ln2;
         1.;
       ]
       [ 1.; 800. ])
    [
      ( "past a clamp",
        fun w e -> past (89., 710.) w e.(0) || below (-104., -746.) w e.(0) );
      ("an overflowing result", fun w e -> past (88.7229, 709.783) w e.(0));
      ( "a subnormal result",
        fun w e ->
          e.(0) < pick w (-87.34, -708.4) && e.(0) > pick w (-103.97, -745.13)
      );
      ( "past the near range, within the clamps",
        fun w e ->
          let a = Float.abs e.(0) in
          a > pick w (86., 708.)
          && e.(0) < pick w (89., 710.)
          && e.(0) > pick w (-104., -746.) );
    ]

let exp2 =
  unary "exp2" { f1 = Nx.exp2 }
    (spec [ 128.; 129.; 126.; 149.; 150.; 151.; 0.5; 1.5; 3. ] [ 1.; 160. ])
    (spec
       [ 1024.; 1025.; 1022.; 1074.; 1075.; 1076.; 0.5; 1.5; 3. ]
       [ 1.; 1100. ])
    [
      ( "an integer in range, not zero",
        fun w e ->
          Float.is_integer e.(0)
          && e.(0) <> 0.
          && Float.abs e.(0) < pick w (128., 1024.) );
      ( "a half-integer",
        fun _ e ->
          Float.is_finite e.(0) && Float.abs (e.(0) -. Float.round e.(0)) = 0.5
      );
      ( "past a clamp",
        fun w e -> past (129., 1025.) w e.(0) || below (-151., -1076.) w e.(0)
      );
      ( "a subnormal result",
        fun w e ->
          e.(0) < pick w (-126., -1022.) && e.(0) > pick w (-150., -1075.) );
    ]

let expm1 =
  let kcap = (64. *. 0.6931471805599453, 128. *. 0.6931471805599453) in
  unary "expm1" { f1 = Nx.expm1 }
    (spec
       [ 26.; 0x1p-25; 89.; 88.72284; fst kcap; half_ln2; 1.; 0.5 ]
       [ 1.; 50. ])
    (spec
       [ 40.; 0x1p-54; 710.; 709.782712893384; snd kcap; half_ln2; 1.; 0.5 ]
       [ 1.; 720. ])
    [
      ( "below the tiny cutoff, not zero",
        fun w e -> e.(0) <> 0. && Float.abs e.(0) < pick w (0x1p-25, 0x1p-54) );
      ("below the low clamp", fun w e -> below (-26., -40.) w e.(0));
      ("past k's cap", fun w e -> past kcap w e.(0));
      ("an overflowing result", fun w e -> past (88.7229, 709.783) w e.(0));
      ( "within 1/2 of zero, past the cutoff",
        fun w e ->
          Float.abs e.(0) < 0.5 && Float.abs e.(0) >= pick w (0x1p-25, 0x1p-54)
      );
    ]

let hyp_points w =
  let kcap = pick w (32., 64.) *. 0.6931471805599453 in
  [
    pick w (90., 712.);
    pick w (89.41599, 710.4758600739439);
    kcap;
    2. *. kcap;
    half_ln2;
    0x1p-30;
    6.;
    1.;
  ]

let hyp_covers =
  let a e = Float.abs e.(0) in
  [
    ( "k past its cap",
      fun w e ->
        a e > pick w (32., 64.) *. 0.6931471805599453 && Float.is_finite e.(0)
    );
    ( "an overflowing result",
      fun w e -> a e > pick w (89.416, 710.476) && Float.is_finite e.(0) );
    ("within 1/2 of zero, not zero", fun _ e -> a e < 0.5 && e.(0) <> 0.);
  ]

let sinh =
  unary "sinh" { f1 = Nx.sinh }
    (spec (hyp_points false) [ 1.; 100. ])
    (spec (hyp_points true) [ 1.; 720. ])
    hyp_covers

let cosh =
  unary "cosh" { f1 = Nx.cosh }
    (spec (hyp_points false) [ 1.; 100. ])
    (spec (hyp_points true) [ 1.; 720. ])
    hyp_covers

let tanh =
  let a e = Float.abs e.(0) in
  unary "tanh" { f1 = Nx.tanh }
    (spec [ 0.625; 9.5; 0x1p-30; 0.3; 1.; 2. ] [ 1.; 25. ])
    (spec [ 0.625; 19.5; 19.06154746539849; 0x1p-54; 0.3; 1.; 2. ] [ 1.; 25. ])
    [
      ("below the polynomial's split", fun _ e -> a e < 0.625);
      ( "from the split to the clamp",
        fun w e -> a e >= 0.625 && a e < pick w (9.5, 19.5) );
      ( "past the clamp, finite",
        fun w e -> a e >= pick w (9.5, 19.5) && Float.is_finite e.(0) );
    ]

(* Inverse trigonometric kinds: asin's switch at 1/2 and the edge of its domain,
   1; atan's at 1 (float32) and 7/16, 11/16, 19/16 and 39/16 (float64); atan2's
   swap where |y| passes |x|, its quadrants, and quotients that overflow,
   underflow or are scaled. *)

let near_one w e = Float.abs (Float.abs e.(0) -. 1.) < pick w (0x1p-21, 0x1p-50)

let asin_spec w =
  spec
    [
      0.5;
      1.;
      0.25;
      0.75;
      2.;
      pick w (0x1p-12, 0x1p-27);
      pick w (0x1p-30, 0x1p-60);
    ]
    [ 1.; 2. ]

let asin_covers =
  let a e = Float.abs e.(0) in
  [
    ("|x| at most 1/2", fun _ e -> a e <= 0.5);
    ("|x| past 1/2, below 1", fun _ e -> a e > 0.5 && a e < 1.);
    ("within 3 ranks of ±1", near_one);
    ("|x| past 1, finite", fun _ e -> a e > 1. && Float.is_finite e.(0));
  ]

let asin =
  unary "asin" { f1 = Nx.asin } (asin_spec false) (asin_spec true) asin_covers

let acos =
  unary "acos" { f1 = Nx.acos } (asin_spec false) (asin_spec true) asin_covers

let atan_switches = [ 0.4375; 0.6875; 1.1875; 2.4375 ]

let atan_spec w =
  spec
    (atan_switches
    @ [
        1.;
        pick w (0x1p-12, 0x1p-27);
        pick w (0x1p24, 0x1p54);
        pick w (1e30, 1e300);
      ]
    @ if w then [ 0x1p-960 ] else [])
    [ 1.; 10. ]

let atan =
  let a e = Float.abs e.(0) in
  let between lo hi = fun _ e -> a e >= lo && a e < hi in
  unary "atan" { f1 = Nx.atan } (atan_spec false) (atan_spec true)
    [
      ("|x| below 7/16", fun _ e -> a e < 0.4375);
      ("|x| in [7/16, 11/16)", between 0.4375 0.6875);
      ("|x| in [11/16, 1]", fun _ e -> a e >= 0.6875 && a e <= 1.);
      ("|x| in (1, 19/16)", fun _ e -> a e > 1. && a e < 1.1875);
      ("|x| in [19/16, 39/16)", between 1.1875 2.4375);
      ( "|x| from 39/16, finite",
        fun _ e -> a e >= 2.4375 && Float.is_finite e.(0) );
    ]

(* atan2's pairs: |y| and |x| within a rank of each other; one huge and the
   other tiny; both below the scale's switch, 2^-960 at float64 and 2^-120 at
   float32 likewise; and zeros and infinities, each of either sign. *)
let atan2_pairs wide =
  let open Gen in
  let signed c =
    let+ neg = bool in
    if neg then Int64.logor c (sign wide) else c
  in
  let finite = if wide then 0x7FEFFFFFFFFFFFFFL else 0x7F7FFFFFL in
  let near =
    let* c = int64_range 1L finite and+ d = int_range (-1) 1 in
    let+ y = signed c and+ x = signed (Int64.add c (Int64.of_int d)) in
    [| y; x |]
  in
  let tiny_top = code wide (pick wide (0x1p-120, 0x1p-960)) in
  let tiny = int64_range 1L tiny_top in
  let huge =
    of_list
      (List.map (code wide)
         [ pick wide (1e30, 1e300); pick wide (0x1p127, 0x1p1023) ])
  in
  let extreme =
    let* h = huge and+ t = tiny and+ swap = bool in
    let+ h = signed h and+ t = signed t in
    if swap then [| t; h |] else [| h; t |]
  in
  let both_tiny =
    let* y = tiny and+ x = tiny in
    let+ y = signed y and+ x = signed x in
    [| y; x |]
  in
  let special =
    let s = of_list [ 0L; code wide Float.infinity ] in
    let* y = s and+ x = s in
    let+ y = signed y and+ x = signed x in
    [| y; x |]
  in
  frequency [ (2, near); (1, extreme); (1, both_tiny); (1, special) ]

let atan2_spec w =
  spec (atan_switches @ [ 1.; 3.; pick w (1e30, 1e300) ]) [ 1.; 1e4 ]

let atan2 =
  let finite e = Float.is_finite e.(0) && Float.is_finite e.(1) in
  {
    name = "atan2";
    fn = Two ({ f2 = Nx.atan2 }, atan2_pairs);
    fixed = annex_pairs;
    generic = true;
    specs =
      [|
        (atan2_spec false, atan2_spec true); (atan2_spec false, atan2_spec true);
      |];
    covers =
      [
        ( "|y| within a rank of |x|, finite, not zero",
          fun w e ->
            finite e
            && e.(1) <> 0.
            && Float.abs (Float.abs (e.(0) /. e.(1)) -. 1.)
               < pick w (0x1p-21, 0x1p-50) );
        ("|y| above |x|", fun _ e -> Float.abs e.(0) > Float.abs e.(1));
        ( "x's sign bit set",
          fun _ e -> Float.sign_bit e.(1) && not (Float.is_nan e.(1)) );
        ( "both infinite",
          fun _ e ->
            Float.abs e.(0) = Float.infinity && Float.abs e.(1) = Float.infinity
        );
        ("both zero", fun _ e -> e.(0) = 0. && e.(1) = 0.);
        ( "a quotient past the greatest finite or below the least normal",
          fun w e ->
            let q = Float.abs (e.(0) /. e.(1)) in
            finite e
            && e.(0) <> 0.
            && e.(1) <> 0.
            && (q > pick w (0x1.fffffep127, Float.max_float)
               || q < pick w (0x1p-126, Float.min_float)) );
        ( "both below the scale's switch, not zero",
          fun w e ->
            let t = pick w (0x1p-120, 0x1p-960) in
            e.(0) <> 0.
            && e.(1) <> 0.
            && Float.abs e.(0) < t
            && Float.abs e.(1) < t );
      ];
  }

(* erf: the switch at 1 and, at float32, the tail's cutoff 2^-100 and the clamp
   at 4; at float64 the switch between the near and far ranges at 2.5, the clamp
   at 5.95, and the expansions' centres 1.75 and 4.25. *)
let erf =
  let a e = Float.abs e.(0) in
  let clamp w = pick w (4., 5.95) in
  unary "erf" { f1 = Nx.erf }
    (spec [ 1.; 2.5; 4.; 0x1p-100; 0.5; 3. ] [ 1.; 7. ])
    (spec
       [ 1.; 2.5; 1.75; 4.25; 5.95; 6.; 0.5; 0x1p-100; 5.921587195794507 ]
       [ 1.; 7. ])
    [
      ("|x| below 2^-100, not zero", fun _ e -> e.(0) <> 0. && a e < 0x1p-100);
      ("|x| from 2^-100 to 1", fun _ e -> a e >= 0x1p-100 && a e < 1.);
      ("|x| from 1 to 2.5", fun _ e -> a e >= 1. && a e <= 2.5);
      ("|x| past 2.5, to the clamp", fun w e -> a e > 2.5 && a e <= clamp w);
      ( "|x| past the clamp, finite",
        fun w e -> a e > clamp w && Float.is_finite e.(0) );
    ]

(* pow: nx_pow's clamps of y log2|x| (POW_HI, POW_LO), the results that
   overflow, turn subnormal or round to zero there, y's parity past NX_INTEGRAL,
   2^23 or 2^52, and twice it, x near 1 with large exponents, negative bases,
   and Annex F's operands. Its pairs: y set from a drawn x to put y log2|x| at a
   threshold; x within 2^-12 of 1 with |y| up to 2^60; a negative x with an
   integer y; y an integer about NX_INTEGRAL; and pairs of Annex F's
   operands. *)

let pow_pairs wide =
  let open Gen in
  let c v = code wide v in
  let signed v =
    let+ neg = bool in
    if neg then -.v else v
  in
  let integral = pick wide (0x1p23, 0x1p52) in
  let threshold =
    let ts =
      if wide then [ 1023.99; -1021.9; -1074.5; 1030.; -1080. ]
      else [ 127.99; -125.9; -149.5; 130.; -160. ]
    in
    let+ x = float_range 1.0001 100. and+ t = of_list ts and+ neg = bool in
    let x = value wide (c x) in
    let y = t /. Float.log2 x in
    [| c x; c (if neg then -.y else y) |]
  in
  let near_one =
    let+ u = float_range (-1.) 1.
    and+ k = int_range 1 12
    and+ v = float_range (-1.) 1.
    and+ e = int_range 10 60 in
    [| c (1. +. Float.ldexp u (-k)); c (Float.ldexp v e) |]
  in
  let negative =
    let+ x = float_range 0.5 4. and+ y = int_range (-100) 100 in
    [| c (-.x); c (Float.of_int y) |]
  in
  let parity =
    let* x = of_list [ -1.5; -1.; -0.5; -3. ]
    and+ base = of_list [ integral; 2. *. integral ]
    and+ d = int_range (-2) 2 in
    let+ y = signed (base +. Float.of_int d) in
    [| c x; c y |]
  in
  let special =
    let xs =
      [ 0.; -0.; 1.; -1.; Float.infinity; Float.neg_infinity; Float.nan ]
    in
    let ys =
      [ 0.; -0.; Float.infinity; Float.neg_infinity; 3.; 2.; -3.; Float.nan ]
    in
    let+ x = of_list xs and+ y = of_list ys in
    [| c x; c y |]
  in
  frequency
    [ (2, threshold); (1, near_one); (1, negative); (1, parity); (2, special) ]

let pow =
  let x_spec w =
    spec
      [
        1.; 2.; 0.5; 10.; 1.5; pick w (0x1p-126, 0x1p-1022); pick w (1e30, 1e300);
      ]
      [ 4.; 1e3 ]
  in
  let y_spec w =
    let i = pick w (0x1p23, 0x1p52) in
    spec
      [
        1.;
        2.;
        3.;
        0.5;
        i;
        2. *. i;
        1e9;
        pick w (130., 1030.);
        pick w (160., 1080.);
      ]
      [ 4.; 200. ]
  in
  let r _ e = Float.pow (Float.abs e.(0)) e.(1) in
  let finite e =
    Float.is_finite e.(0) && Float.is_finite e.(1) && e.(0) <> 0.
  in
  let integer v = Float.is_integer v in
  {
    name = "pow";
    fn = Two ({ f2 = Nx.pow }, pow_pairs);
    fixed = annex_pairs;
    generic = true;
    specs = [| (x_spec false, x_spec true); (y_spec false, y_spec true) |];
    covers =
      [
        ( "an overflowing result",
          fun w e -> finite e && r w e > pick w (0x1.fffffep127, Float.max_float)
        );
        ( "a subnormal result",
          fun w e ->
            finite e && r w e > 0. && r w e < pick w (0x1p-126, Float.min_float)
        );
        ( "a result that rounds to zero",
          fun w e -> finite e && if w then r w e = 0. else r w e <= 0x1p-150 );
        ( "a negative base, an odd integer exponent",
          fun _ e -> e.(0) < 0. && integer e.(1) && Float.rem e.(1) 2. <> 0. );
        ( "a negative base, a non-integer exponent",
          fun _ e ->
            e.(0) < 0.
            && Float.is_finite e.(0)
            && Float.is_finite e.(1)
            && not (integer e.(1)) );
        ( "|y| past NX_INTEGRAL",
          fun w e ->
            Float.abs e.(1) >= pick w (0x1p23, 0x1p52) && Float.is_finite e.(1)
        );
        ( "x within 2^-10 of 1, |y| past 2^10",
          fun _ e ->
            e.(0) <> 1.
            && Float.abs (e.(0) -. 1.) < 0x1p-10
            && Float.abs e.(1) > 1024. );
        ( "x = ±1, y infinite",
          fun _ e -> Float.abs e.(0) = 1. && Float.abs e.(1) = Float.infinity );
        ("y = ±0", fun _ e -> e.(1) = 0.);
      ];
  }

(* mod: exact, so any exact algorithm gives nx_mod_f32's long division's bits
   and fmod's. Its pairs: exponent gaps of up to the whole range, a subnormal
   divisor, an exact multiple, and Annex F's operands. *)

let mod_pairs wide =
  let open Gen in
  let c v = code wide v in
  let signed v =
    let+ neg = bool in
    if neg then -.v else v
  in
  let top = if wide then 1023 else 127 and low = if wide then -1074 else -149 in
  let gap =
    let* ex = int_range (top - 60) top
    and+ ey = int_range low (low + 60)
    and+ mx = float_range 1. 2.
    and+ my = float_range 1. 2. in
    let+ x = signed (Float.ldexp mx ex) and+ y = signed (Float.ldexp my ey) in
    [| c x; c y |]
  in
  let sub_divisor =
    let* x = float_range (-1e6) 1e6 and+ y = int64_range 1L (mmask wide) in
    let+ neg = bool in
    [| c x; (if neg then Int64.logor y (sign wide) else y) |]
  in
  let multiple =
    let* y = float_range 0.5 4. and+ k = int_range 1 1000 in
    let y = value wide (c y) in
    let+ x = signed (Float.of_int k *. y) in
    [| c x; c y |]
  in
  let special =
    let xs = [ 0.; -0.; 1.; Float.infinity; Float.neg_infinity; Float.nan ] in
    let ys = [ 0.; -0.; 1.; Float.infinity; Float.nan ] in
    let+ x = of_list xs and+ y = of_list ys in
    [| c x; c y |]
  in
  frequency [ (2, gap); (1, sub_divisor); (1, multiple); (1, special) ]

let mod_ =
  let s w =
    spec [ 7.; 7.5; 3.; 0.75; 0.1; 1.; pick w (1e30, 1e300) ] [ 1e3; 10. ]
  in
  let finite e = Float.is_finite e.(0) && Float.is_finite e.(1) in
  {
    name = "mod";
    fn = Two ({ f2 = Nx.mod_ }, mod_pairs);
    fixed = annex_pairs;
    generic = true;
    specs = [| (s false, s true); (s false, s true) |];
    covers =
      [
        ( "an exponent gap past 50 binades",
          fun _ e ->
            finite e
            && e.(1) <> 0.
            && Float.abs e.(0) > Float.ldexp (Float.abs e.(1)) 50 );
        ( "a subnormal divisor",
          fun w e ->
            e.(1) <> 0. && Float.abs e.(1) < pick w (0x1p-126, Float.min_float)
        );
        ( "an exact multiple, not zero",
          fun _ e ->
            finite e && e.(0) <> 0. && e.(1) <> 0. && Float.rem e.(0) e.(1) = 0.
        );
        ( "|x| below |y|, not zero",
          fun _ e -> e.(0) <> 0. && Float.abs e.(0) < Float.abs e.(1) );
        ("a zero divisor", fun _ e -> e.(1) = 0.);
        ("an infinite dividend", fun _ e -> Float.abs e.(0) = Float.infinity);
      ];
  }

(* Logarithms: nx_log_reduce's switch, where a significand passes sqrt(1/2), in
   several binades; the least normal, below which it scales a subnormal up; 1,
   around which log cancels; powers of two, where log2 is exact; for log1p, -1,
   the arguments whose 1 + x crosses sqrt(1/2) or 2, and those past which 1 + x
   rounds to x. *)

let sqrt_half = (0x1.6a09e6p-1, 0x1.6a09e667f3bcdp-1)

let log_points wide =
  let h = pick wide sqrt_half in
  [
    pick wide (0x1p-126, 0x1p-1022);
    h;
    2. *. h;
    Float.ldexp h (-100);
    Float.ldexp h 60;
    1.;
    2.;
    0.5;
    3.;
    10.;
    1024.;
    pick wide (1e-40, 1e-310);
    pick wide (1e30, 1e300);
  ]
  @
  if wide then
    [ 0x1.481e0a20c684p+0; 0x1.203eace7848f4p+0; 0x1.936dc3c250523p-1 ]
  else []

let log_spec wide = { points = log_points wide; ranges = [ 2.; 1e4 ] }
let finite_pos v = v > 0. && Float.is_finite v

let log_covers =
  [
    ("a negative number", fun _ e -> e.(0) < 0. && Float.is_finite e.(0));
    ( "within 2^-10 of 1, not 1",
      fun _ e -> e.(0) <> 1. && Float.abs (e.(0) -. 1.) < 0x1p-10 );
    ( "a significand beside sqrt(1/2)",
      fun _ e ->
        finite_pos e.(0)
        && Float.abs (fst (Float.frexp e.(0)) -. 0.70710678) < 0x1p-20 );
  ]

let log_kind name f covers =
  {
    name;
    fn = One f;
    specs = [| (log_spec false, log_spec true) |];
    covers = log_covers @ covers;
    fixed = none;
    generic = true;
  }

let log = log_kind "log" { f1 = Nx.log } []

let log2 =
  log_kind "log2" { f1 = Nx.log2 }
    [
      ( "a power of two but 1",
        fun _ e ->
          finite_pos e.(0) && e.(0) <> 1. && fst (Float.frexp e.(0)) = 0.5 );
    ]

let log1p_spec wide =
  let h = pick wide sqrt_half in
  {
    points =
      [
        1.;
        0.5;
        1. -. h;
        (2. *. h) -. 1.;
        2.;
        10.;
        pick wide (0x1p-25, 0x1p-54);
        pick wide (0x1p24, 0x1p53);
        pick wide (1e-40, 1e-310);
        pick wide (1e30, 1e300);
      ];
    ranges = [ 1.; 1e4 ];
  }

let log1p =
  let big w = pick w (0x1p24, 0x1p53) in
  {
    name = "log1p";
    fn = One { f1 = Nx.log1p };
    specs = [| (log1p_spec false, log1p_spec true) |];
    fixed = none;
    generic = true;
    covers =
      [
        ("below -1", fun _ e -> e.(0) < -1.);
        ( "within 3 ranks of -1",
          fun w e -> Float.abs (e.(0) +. 1.) < pick w (0x1p-21, 0x1p-50) );
        ("in (-1, -1/2)", fun _ e -> e.(0) > -1. && e.(0) < -0.5);
        ( "below 2^-24 of 1, not zero",
          fun w e -> e.(0) <> 0. && Float.abs e.(0) < pick w (0x1p-24, 0x1p-53)
        );
        ( "where 1 + x rounds to x, finite",
          fun w e -> e.(0) >= big w && Float.is_finite e.(0) );
      ];
  }

(* Cases *)

type case = { f32 : int64 array array; f64 : int64 array array }

(* Elements per dtype in a case: one compiled program per dtype and kind. *)
let n = 4096

(* One operand's code at [wide], drawn from five classes by weight out of 12: a
   switch point stepped up to 3 ranks either way, of either sign (4); an edge
   (2); a uniform bit pattern (2); a uniform value of a range (3); a subnormal
   of either sign (1).

   Rates. Windtrap repeats an earlier draw of a generator with probability 1/8,
   so at least 7/8 of the elements are fresh draws. An edge, each of 16 equally
   likely, is then drawn with probability at least 7/8 * 2/12 / 16 = 1/110: a
   block of 4096 elements misses it with probability (1 - 1/110)^4096 < 1e-16. A
   subnormal (1/12 at least) and a cover of a switch point's side (with p
   points, 7/8 * 4/12 / p * 3/7 at least, int_range drawing its uniform values
   and one corner in ten among -3, 3, 0, -1 and 1) are drawn more often for
   every kind's p <= 16. A fault that 0.1% of the drawn elements reach escapes
   12 cases of 4096 with probability (1 - 7/8 * 0.001)^49152 < 1e-18. *)
let operand wide s =
  let open Gen in
  let signed g =
    let+ c = g and+ neg = bool in
    if neg then Int64.logor c (sign wide) else c
  in
  let point =
    signed
      (let+ p = of_list s.points and+ d = int_range (-3) 3 in
       Int64.add (code wide p) (Int64.of_int d))
  in
  let bits =
    if wide then int64_range Int64.min_int (Int64.pred Int64.max_int)
    else int64_range 0L (Int64.pred low32)
  in
  let range =
    let* r = of_list s.ranges in
    map (code wide) (float_range (-.r) r)
  in
  let sub = signed (int64_range 1L (mmask wide)) in
  frequency
    [ (4, point); (2, of_list (edges wide)); (2, bits); (3, range); (1, sub) ]

(* An element's operands, each drawn alone, or for two operands a pair the kind
   draws in 2 of 5. *)
let element wide k =
  let ops = List.init (arity k) (fun i -> operand wide (spec_of wide k i)) in
  let alone =
    Gen.map Array.of_list
      (List.fold_right
         (fun g acc ->
           Gen.(
             let+ v = g and+ vs = acc in
             v :: vs))
         ops (Gen.constant []))
  in
  match k.fn with
  | One _ -> alone
  | Two (_, pairs) -> Gen.frequency [ (3, alone); (2, pairs wide) ]

let gen k =
  let block wide = Gen.array ~size:(Gen.constant n) (element wide k) in
  Gen.with_pp
    (fun ppf _ ->
      Format.fprintf ppf "%d elements of %s at float32 and float64" n k.name)
    Gen.(
      let+ f32 = block false and+ f64 = block true in
      { f32; f64 })

(* The example's operand codes: the edges, then each point and its neighbours up
   to 3 ranks of both signs, then a Weyl sequence over the code space, [m] in
   all. *)
let example_codes wide s m =
  let around =
    List.concat_map
      (fun p ->
        List.concat_map
          (fun d ->
            let c = Int64.add (code wide p) (Int64.of_int d) in
            [ c; Int64.logor c (sign wide) ])
          [ -3; -2; -1; 0; 1; 2; 3 ])
      s.points
  in
  let fixed = Array.of_list (edges wide @ around) in
  let k = Array.length fixed in
  if k > m then invalid_arg "Kinds.example: more fixed codes than elements";
  Array.init m (fun i -> if i < k then fixed.(i) else weyl wide (i - k + 1))

(* The example: the kind's fixed elements, then elements pairing the codes of
   each operand at the same position, the second operand's list turned by one,
   [n] in all. *)
let example k =
  let block wide =
    let fixed = Array.of_list (k.fixed wide) in
    let m = n - Array.length fixed in
    if m < 0 then invalid_arg "Kinds.example: more fixed elements than n";
    let cols =
      Array.init (arity k) (fun j ->
          let c = example_codes wide (spec_of wide k j) m in
          Array.init m (fun i -> c.((i + j) mod m)))
    in
    Array.append fixed (Array.init m (fun i -> Array.map (fun c -> c.(i)) cols))
  in
  { f32 = block false; f64 = block true }

(* [clamped k ~lo ~hi covers] is [k] of its operand clamped to [lo, hi] by two
   selections on comparisons inside the compiled program, [x < hi] then [lo <
   x], a NaN taking [hi]. Its covers read the clamped operand, as the program
   computes it at the dtype. *)
let clamped k ~lo ~hi covers =
  match k.fn with
  | Two _ -> invalid_arg "Kinds.clamped: a kind of two operands"
  | One { f1 } ->
      let clamp x =
        let c v = Nx.full_like x v in
        let x = Nx.where (Nx.less x (c hi)) x (c hi) in
        Nx.where (Nx.less (c lo) x) x (c lo)
      in
      let at w v =
        let r b = value w (code w b) in
        let v = if v < r hi then v else r hi in
        if r lo < v then v else r lo
      in
      let cover (label, f) = (label, fun w e -> f w [| at w e.(0) |]) in
      {
        k with
        name =
          Printf.sprintf "%s of an operand clamped to [%g, %g]" k.name lo hi;
        fn = One { f1 = (fun x -> f1 (clamp x)) };
        covers = List.map cover covers;
        generic = false;
      }

(* Under this clamp tolk folds only exp's NaN selection, merging it with the
   clamp's comparisons; exp's own clamps at 89 and -104 stay. *)
let clamped_exp =
  clamped exp ~lo:(-80.) ~hi:80.
    [
      ("the upper bound", fun _ e -> e.(0) = 80.);
      ("the lower bound", fun _ e -> e.(0) = -80.);
      ("within ln2/2 of 0", fun _ e -> Float.abs e.(0) < half_ln2);
      ( "past 1 within the bounds",
        fun _ e -> Float.abs e.(0) > 1. && Float.abs e.(0) < 80. );
    ]

(* Under this clamp tolk folds log's selections of 0, a negative and infinity;
   its subnormal scaling, which compares bits, stays. *)
let clamped_log =
  clamped log ~lo:1e-30 ~hi:1e4
    [
      ("the upper bound", fun _ e -> e.(0) = 1e4);
      ("near the lower bound", fun _ e -> e.(0) < 1e-29);
      ( "within 2^-10 of 1, not 1",
        fun _ e -> e.(0) <> 1. && Float.abs (e.(0) -. 1.) < 0x1p-10 );
      ( "a significand beside sqrt(1/2)",
        fun _ e -> Float.abs (fst (Float.frexp e.(0)) -. 0.70710678) < 0x1p-20
      );
    ]

(* Laws *)

let show wide c =
  Printf.sprintf "%h (%#Lx)" (value wide c)
    (if wide then c else Int64.logand c low32)

let same wide a b = Int64.equal a b || (is_nan wide a && is_nan wide b)

let compiled (type s d) (module S : Nx.Devices with type d = d) k
    (xs : (float, s, Nx.host) Nx.t list) =
  let before = Rune.evaluations () in
  let on x = Nx.place S.on x in
  let y =
    match (k.fn, xs) with
    | One { f1 }, [ x ] ->
        Rune.jit Nx.Ptree.(tensor @-> returns tensor) f1 (on x)
    | Two ({ f2 }, _), [ x; y ] ->
        Rune.jit
          Nx.Ptree.(tensor @-> tensor @-> returns tensor)
          f2 (on x) (on y)
    | _ -> invalid_arg "Kinds: operands of another arity"
  in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  Nx.place Nx.Host.on y

let eager k xs =
  match (k.fn, xs) with
  | One { f1 }, [ x ] -> f1 x
  | Two ({ f2 }, _), [ x; y ] -> f2 x y
  | _ -> invalid_arg "Kinds: operands of another arity"

(* A block's operands as host tensors and a result's codes. *)
let tensor32 cs =
  Nx.bitcast Nx.float32
    (Nx.create Nx.int32 [| Array.length cs |] (Array.map Int64.to_int32 cs))

let codes32 y =
  Array.map
    (fun c -> Int64.logand (Int64.of_int32 c) low32)
    (Nx.to_array (Nx.bitcast Nx.int32 y))

let tensor64 cs =
  Nx.bitcast Nx.float64 (Nx.create Nx.int64 [| Array.length cs |] cs)

let codes64 y = Nx.to_array (Nx.bitcast Nx.int64 y)
let column es j = Array.map (fun e -> e.(j)) es

(* The elements where compiled and eager differ, at most eight, then their
   count. *)
let misses k wide es ~eager ~compiled =
  let out = ref [] and count = ref 0 in
  Array.iteri
    (fun i e ->
      if not (same wide eager.(i) compiled.(i)) then begin
        incr count;
        if !count <= 8 then
          out :=
            Printf.sprintf "%s(%s) at %s: eager %s, compiled %s" k.name
              (String.concat ", " (Array.to_list (Array.map (show wide) e)))
              (if wide then "float64" else "float32")
              (show wide eager.(i))
              (show wide compiled.(i))
            :: !out
      end)
    es;
  if !count = 0 then []
  else
    List.rev !out
    @ [ Printf.sprintf "%d of %d elements differ" !count (Array.length es) ]

let run (type d) (module S : Nx.Devices with type d = d) k wide es =
  let cols = List.init (arity k) (column es) in
  if wide then
    let xs = List.map tensor64 cols in
    misses k wide es
      ~eager:(codes64 (eager k xs))
      ~compiled:(codes64 (compiled (module S) k xs))
  else
    let xs = List.map tensor32 cols in
    misses k wide es
      ~eager:(codes32 (eager k xs))
      ~compiled:(codes32 (compiled (module S) k xs))

(* Classes of an operand every kind's law covers. *)
let generic =
  let any p wide e = Array.exists (fun v -> p wide v) e in
  let payload wide v =
    Float.is_nan v
    && Int64.logand (code wide v) (Int64.shift_right_logical (mmask wide) 1)
       <> 0L
  in
  let sub wide v =
    v <> 0. && Float.abs v < if wide then Float.min_float else 0x1p-126
  in
  let greatest wide v =
    Float.abs v = if wide then Float.max_float else 0x1.fffffep127
  in
  [
    ("a NaN", any (fun _ v -> Float.is_nan v));
    ("a NaN with a payload", any payload);
    ("+0.", any (fun _ v -> v = 0. && not (Float.sign_bit v)));
    ("-0.", any (fun _ v -> v = 0. && Float.sign_bit v));
    ("+inf", any (fun _ v -> v = Float.infinity));
    ("-inf", any (fun _ v -> v = Float.neg_infinity));
    ("a subnormal", any sub);
    ("a greatest finite magnitude", any greatest);
  ]

let covers k wide es =
  let vs = Array.map (Array.map (value wide)) es in
  let dt = if wide then "float64: " else "float32: " in
  List.iter
    (fun (label, f) -> cover (dt ^ label) (Array.exists (f wide) vs))
    ((if k.generic then generic else []) @ k.covers)

let law (type d) (module S : Nx.Devices with type d = d) ~f64 k c =
  covers k false c.f32;
  if f64 then covers k true c.f64;
  let bad =
    run (module S) k false c.f32
    @ if f64 then run (module S) k true c.f64 else []
  in
  equal (list string) [] bad

let law (module S : Nx.Devices) = law (module S)

(* Narrow floats *)

type narrow = Narrow : string * (float, 's) Nx.dtype * int -> narrow

let narrows =
  Nx.
    [
      Narrow ("float16", float16, 16);
      Narrow ("bfloat16", bfloat16, 16);
      Narrow ("float8_e4m3fn", float8_e4m3fn, 8);
      Narrow ("float8_e5m2", float8_e5m2, 8);
      Narrow ("float4_e2m1fn", float4_e2m1fn, 4);
    ]

(* float4_e2m1fn's codes are its 16 values. *)
let float4 = [ 0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6. ]
let float4 = Array.of_list (float4 @ List.map Float.neg float4)

(* [codes dt bits is] is the codes of [dt] numbered [is]. *)
let narrow_codes (type s) (dt : (float, s) Nx.dtype) bits is =
  let m = Array.length is in
  match bits with
  | 16 -> Nx.bitcast dt (Nx.create Nx.uint16 [| m |] is)
  | 8 -> Nx.bitcast dt (Nx.create Nx.uint8 [| m |] is)
  | _ ->
      Nx.cast dt
        (Nx.create Nx.float64 [| m |] (Array.map (Array.get float4) is))

(* Every code, or for two operands every pair of codes, but for the 16-bit
   floats a grid: every 257th code and the zeros, ones and infinities of both
   signs. *)
let numbers k bits =
  let all = Array.init (if bits = 16 then 0 else 1 lsl bits) Fun.id in
  let grid =
    Array.of_list
      (List.init 256 (fun i -> i * 257)
      @ [
          0x8000; 0x3C00; 0xBC00; 0x3F80; 0xBF80; 0x7C00; 0xFC00; 0x7F80; 0xFF80;
        ])
  in
  match arity k with
  | 1 -> [ (if bits = 16 then Array.init 0x10000 Fun.id else all) ]
  | _ ->
      let base = if bits = 16 then grid else all in
      let m = Array.length base in
      [
        Array.init (m * m) (fun i -> base.(i / m));
        Array.init (m * m) (fun i -> base.(i mod m));
      ]

let narrow (type d) (module S : Nx.Devices with type d = d) k =
  let check (Narrow (label, dt, bits)) =
    let xs = List.map (narrow_codes dt bits) (numbers k bits) in
    let floats y = Nx.to_array (Nx.cast Nx.float64 y) in
    let e = floats (eager k xs) and c = floats (compiled (module S) k xs) in
    let ins = List.map floats xs in
    let out = ref [] and count = ref 0 in
    Array.iteri
      (fun i v ->
        let w = c.(i) in
        let ok =
          Int64.equal (Int64.bits_of_float v) (Int64.bits_of_float w)
          || (Float.is_nan v && Float.is_nan w)
        in
        if not ok then begin
          incr count;
          if !count <= 4 then
            out :=
              Printf.sprintf "%s(%s) at %s: eager %h, compiled %h" k.name
                (String.concat ", "
                   (List.map (fun a -> Printf.sprintf "%h" a.(i)) ins))
                label v w
              :: !out
        end)
      e;
    if !count = 0 then []
    else
      List.rev !out
      @ [
          Printf.sprintf "%d of %d elements differ at %s" !count
            (Array.length e) label;
        ]
  in
  equal (list string) [] (List.concat_map check narrows)

let narrow (module S : Nx.Devices) = narrow (module S)

(* Folded constants *)

(* The constants of one operand: each point and its neighbours, of both signs,
   and the edges but the NaNs. *)
let constants wide s =
  let around =
    List.concat_map
      (fun p ->
        List.concat_map
          (fun d ->
            let v = value wide (Int64.add (code wide p) (Int64.of_int d)) in
            [ v; -.v ])
          [ -1; 0; 1 ])
      s.points
  in
  List.filter_map
    (fun c ->
      let v = value wide c in
      if Float.is_nan v then None else Some v)
    (edges wide)
  @ around @ [ Float.nan ]

let folded_at (type s d) (module S : Nx.Devices with type d = d) k
    (dt : (float, s) Nx.dtype) wide =
  let cs = List.init (arity k) (fun i -> constants wide (spec_of wide k i)) in
  let tuples =
    match cs with
    | [ a ] -> List.map (fun v -> [ v ]) a
    | [ a; b ] ->
        let b = Array.of_list b in
        List.mapi (fun i v -> [ v; b.(((7 * i) + 3) mod Array.length b) ]) a
    | _ -> invalid_arg "Kinds: operands of another arity"
  in
  let at x vs =
    let full v = Nx.full_like x v in
    let y =
      match (k.fn, vs) with
      | One { f1 }, [ a ] -> f1 (full a)
      | Two ({ f2 }, _), [ a; b ] -> f2 (full a) (full b)
      | _ -> invalid_arg "Kinds: operands of another arity"
    in
    Nx.mul x y
  in
  let f x = Nx.concatenate ~axis:0 (List.map (at x) tuples) in
  let x = Nx.full dt [| 1 |] 1. in
  let e = Nx.to_array (f x) in
  let before = Rune.evaluations () in
  let c = Nx.to_array (Nx.place Nx.Host.on (Rune.jit' f (Nx.place S.on x))) in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  let code v = code wide v in
  List.concat
    (List.mapi
       (fun i vs ->
         if same wide (code e.(i)) (code c.(i)) then []
         else
           [
             Printf.sprintf "%s(%s) folded at %s: eager %h, compiled %h" k.name
               (String.concat ", " (List.map (Printf.sprintf "%h") vs))
               (if wide then "float64" else "float32")
               e.(i) c.(i);
           ])
       tuples)

let folded (type d) (module S : Nx.Devices with type d = d) ~f64 k =
  let bad =
    folded_at (module S) k Nx.float32 false
    @ if f64 then folded_at (module S) k Nx.float64 true else []
  in
  equal (list string) [] bad

let folded (module S : Nx.Devices) = folded (module S)
