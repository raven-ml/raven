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

type f3 = {
  f3 :
    's 'd.
    (float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t;
}

type c1 = { c1 : 's 'd. (Complex.t, 's, 'd) Nx.t -> (Complex.t, 's, 'd) Nx.t }

type c2 = {
  c2 :
    's 'd.
    (Complex.t, 's, 'd) Nx.t ->
    (Complex.t, 's, 'd) Nx.t ->
    (Complex.t, 's, 'd) Nx.t;
}

(* A complex kind's element holds its operands' parts, each real part then its
   imaginary part, as codes of the part type: float32 for complex64, float64 for
   complex128. Its pairs draw whole elements. *)
type fn =
  | One of f1
  | Two of f2 * (bool -> int64 array Windtrap.Gen.t)
  | Three of f3 * (bool -> int64 array Windtrap.Gen.t)
  | Cone of c1 * (bool -> int64 array Windtrap.Gen.t)
  | Ctwo of c2 * (bool -> int64 array Windtrap.Gen.t)

type kind = {
  name : string;
  fn : fn;
  specs : (spec * spec) array;
      (** Per operand, or per part of a complex kind's operands, at float32 and
          float64. *)
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

(* Annex G's parts: the zeros, ones and infinities of both signs, 2, 1/2 and
   NaNs of both signs. *)
let annex_parts =
  [ 0.; -0.; 1.; -1.; 2.; 0.5; Float.infinity; Float.neg_infinity ]
  @ [ Float.nan; -.Float.nan ]

(* Every part of an element one of Annex G's parts, each of the 10 alike. *)
let annex_g wide parts =
  Gen.(
    let part = of_list (List.map (code wide) annex_parts) in
    map Array.of_list (list ~size:(constant parts) part))

(* Every complex number of two of Annex G's parts, for a kind of one operand; of
   two operands, every pair of numbers whose parts are among the zeros, the
   ones, the infinities and a NaN. *)
let annex_numbers parts wide =
  let vs =
    if parts = 2 then annex_parts
    else [ 0.; -0.; 1.; -1.; Float.infinity; Float.neg_infinity; Float.nan ]
  in
  let rec tuples n =
    if n = 0 then [ [] ]
    else
      List.concat_map (fun t -> List.map (fun v -> v :: t) vs) (tuples (n - 1))
  in
  List.map (fun t -> Array.of_list (List.map (code wide) t)) (tuples parts)

let arity k = Array.length k.specs

let is_complex k =
  match k.fn with Cone _ | Ctwo _ -> true | One _ | Two _ | Three _ -> false

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

(* Complex kinds

   Each kind's specs give its parts' draws, its pairs whole elements: the
   switches of nx_kinds_complex.h's function and the operands that overflow or
   underflow its naive formula. Its covers read an element's parts. A kind's
   example opens with every number of Annex G's parts (two operands: every pair
   of numbers of its zeros, ones, infinities and a NaN).

   Rates. An element's parts are drawn each as {!operand} draws one in 3 of 6
   elements, all of Annex G's parts in 1 of 6, and as the kind's pairs in 2 of
   6, each pair's class weighted 1 in 2 to 1 in 8 and a divisor's own class 1 in
   6 of its pairs'. Measured over 12 drawn cases of each complex kind, 49152
   elements per dtype, each cover below and each of Annex G's classes is reached
   by at least 5.6 elements in 1000, the least two infinite parts (csin,
   float32), then 7.3 (csign's both parts zero): a block of 4096 misses it with
   probability below 1e-10. A fault that 0.1% of the drawn elements reach, of
   which at least 7/8 are fresh draws, escapes 12 cases of 4096 with probability
   below 1e-18. *)

let least wide = pick wide (0x1p-126, Float.min_float)
let greatest wide = pick wide (0x1.fffffep127, Float.max_float)
let finite_all e = Array.for_all Float.is_finite e

(* A code of either sign, of magnitude [2^e m] with [m] uniform in [1, 2). *)
let scaled wide e =
  Gen.(
    let+ m = float_range 1. 2. and+ neg = bool in
    let v = Float.ldexp m e in
    code wide (if neg then -.v else v))

(* A code of either sign and uniform value in [-r, r]. *)
let uniform wide r = Gen.map (code wide) (Gen.float_range (-.r) r)

(* The distance in ranks between the magnitudes of two finite parts. *)
let ranks wide a b =
  let m v = Int64.logand (code wide v) (Int64.lognot (sign wide)) in
  Int64.abs (Int64.sub (m a) (m b))

(* log |x + iy| of finite parts, which no rounding overflows. *)
let log_mod x y =
  let a = Float.max (Float.abs x) (Float.abs y)
  and b = Float.min (Float.abs x) (Float.abs y) in
  if a = 0. then Float.neg_infinity
  else Float.log a +. (0.5 *. Float.log1p (b /. a *. (b /. a)))

let complex1 name f specs pairs covers =
  {
    name;
    fn = Cone (f, pairs);
    specs;
    covers;
    fixed = annex_numbers 2;
    generic = true;
  }

let complex2 name f specs pairs covers =
  {
    name;
    fn = Ctwo (f, pairs);
    specs;
    covers;
    fixed = annex_numbers 4;
    generic = true;
  }

let parts2 s = [| s; s |]

(* Products: parts whose products cancel, a = (x, y) times (y, x) or (x, -y);
   and parts of exponents [ea] and [eb] whose product passes the greatest finite
   or lands below the least normal, the other parts uniform. *)

let mul_pairs wide =
  let open Gen in
  let c = code wide in
  let top = pick wide (127, 1023) and low = pick wide (-149, -1074) in
  let cancel =
    let+ x = float_range (-4.) 4. and+ y = float_range (-4.) 4. and+ b = bool in
    let x = value wide (c x) and y = value wide (c y) in
    if b then [| c x; c y; c y; c x |] else [| c x; c y; c x; c (-.y) |]
  in
  let extreme =
    let near_top = int_range (top - 1) (top + 1)
    and near_low = int_range (low - 1) (low + 22) in
    let* t = frequency [ (1, near_top); (1, near_low) ] in
    let* ea = int_range ((t / 2) - 20) ((t / 2) + 20) in
    let* a = scaled wide ea and+ b = scaled wide (t - ea) in
    let+ u = uniform wide 4. and+ v = uniform wide 4. and+ at = int_range 0 3 in
    (* the extreme product at ar br, ai bi, ar bi or ai br *)
    match at with
    | 0 -> [| a; u; b; v |]
    | 1 -> [| u; a; v; b |]
    | 2 -> [| a; u; v; b |]
    | _ -> [| u; a; b; v |]
  in
  frequency [ (1, cancel); (1, extreme) ]

let products e =
  [ e.(0) *. e.(2); e.(1) *. e.(3); e.(0) *. e.(3); e.(1) *. e.(2) ]

let cmul =
  let s w =
    spec
      [
        1.;
        0.5;
        3.;
        pick w (0x1p64, 0x1p512);
        pick w (0x1p-64, 0x1p-512);
        pick w (0x1p-75, 0x1p-537);
      ]
      [ 1.; 1e3 ]
  in
  let sp = (s false, s true) in
  complex2 "cmul" { c2 = Nx.mul } [| sp; sp; sp; sp |] mul_pairs
    [
      ( "a product of finite parts past the greatest finite",
        fun w e ->
          finite_all e
          && List.exists (fun p -> Float.abs p > greatest w) (products e) );
      ( "a subnormal product of parts",
        fun w e ->
          List.exists (fun p -> p <> 0. && Float.abs p < least w) (products e)
      );
      ( "a part whose two products cancel, not zero",
        fun _ e ->
          finite_all e
          && e.(0) <> 0.
          && e.(1) <> 0.
          && (Float.abs (e.(0) *. e.(2)) = Float.abs (e.(1) *. e.(3))
             || Float.abs (e.(0) *. e.(3)) = Float.abs (e.(1) *. e.(2))) );
    ]

(* Quotients: nx_fdiv's switch to scaling a dividend whose parts are below
   TINY_DIVIDEND, 2^-102 or 2^-969, and Smith's switch where the divisor's parts
   have equal magnitudes; divisors of zero, subnormal or huge parts, and
   quotients that overflow or underflow. *)

let tiny_dividend = (0x1p-102, 0x1p-969)

(* A divisor's parts: equal magnitudes, or a rank apart; zeros; subnormals; near
   the greatest finite; uniform. *)
let divisor wide =
  let open Gen in
  let finite = pick wide (0x7F7FFFFFL, 0x7FEFFFFFFFFFFFFFL) in
  let signed c =
    let+ neg = bool in
    if neg then Int64.logor c (sign wide) else c
  in
  let equal =
    let* m = int64_range 1L (Int64.pred finite) and+ d = int_range (-1) 1 in
    let* re = signed m and+ im = signed (Int64.add m (Int64.of_int d)) in
    let+ swap = bool in
    if swap then [| im; re |] else [| re; im |]
  in
  let zero =
    let+ re = signed 0L and+ im = signed 0L in
    [| re; im |]
  in
  let sub =
    let* a = int64_range 1L (mmask wide) and+ b = int64_range 0L (mmask wide) in
    let+ re = signed a and+ im = signed b in
    [| re; im |]
  in
  let huge =
    let top = pick wide (127, 1023) in
    let+ re = scaled wide top and+ im = scaled wide (top - 1) in
    [| re; im |]
  in
  let any =
    let+ re = uniform wide 4. and+ im = uniform wide 4. in
    [| re; im |]
  in
  frequency [ (2, equal); (1, zero); (1, sub); (1, huge); (1, any) ]

let div_pairs wide =
  let open Gen in
  let tiny_top = code wide (pick wide tiny_dividend) in
  let signed g =
    let+ c = g and+ neg = bool in
    if neg then Int64.logor c (sign wide) else c
  in
  let tiny =
    let* re = signed (int64_range 0L tiny_top)
    and+ im = signed (int64_range 0L tiny_top)
    and+ e = int_range (-40) 0 in
    let+ br = scaled wide e and+ bi = uniform wide (Float.ldexp 1. e) in
    [| re; im; br; bi |]
  in
  let any =
    let+ a = uniform wide 4. and+ b = uniform wide 4. and+ d = divisor wide in
    [| a; b; d.(0); d.(1) |]
  in
  let top = pick wide (127, 1023) and low = pick wide (-126, -1022) in
  let over =
    let* e = int_range (top - 4) top and+ f = int_range (low - 20) (-4) in
    let+ a = scaled wide e and+ b = uniform wide 1. and+ d = scaled wide f in
    [| a; b; d; d |]
  in
  let under =
    let* e = int_range (low - 20) (low + 4) and+ f = int_range (top - 20) top in
    let+ a = scaled wide e and+ b = scaled wide e and+ d = scaled wide f in
    [| a; b; d; d |]
  in
  frequency [ (2, tiny); (2, any); (1, over); (1, under) ]

let divisor_covers j =
  let b e = (e.(j), e.(j + 1)) in
  [
    ( "a divisor of parts of equal magnitude, not zero",
      fun _ e ->
        let x, y = b e in
        Float.is_finite x && x <> 0. && Float.abs x = Float.abs y );
    ( "a divisor of parts a rank apart",
      fun w e ->
        let x, y = b e in
        Float.is_finite x && Float.is_finite y && ranks w x y = 1L );
    ( "a zero divisor",
      fun _ e ->
        let x, y = b e in
        x = 0. && y = 0. );
    ( "a divisor whose larger part is subnormal",
      fun w e ->
        let x, y = b e in
        let m = Float.max (Float.abs x) (Float.abs y) in
        m > 0. && m < least w );
  ]

let dividend_spec w =
  spec
    [ pick w tiny_dividend; 1.; 3.; 0.5; pick w (0x1p100, 0x1p900) ]
    [ 1.; 1e3 ]

let divisor_spec w =
  spec [ 1.; 3.; 0.5; least w; pick w (0x1p-100, 0x1p-900) ] [ 1.; 1e3 ]

let cdiv =
  let tiny w = pick w tiny_dividend in
  complex2 "cdiv" { c2 = Nx.div }
    [|
      (dividend_spec false, dividend_spec true);
      (dividend_spec false, dividend_spec true);
      (divisor_spec false, divisor_spec true);
      (divisor_spec false, divisor_spec true);
    |]
    div_pairs
    ([
       ( "a dividend of parts below the scaling's switch, not zero",
         fun w e ->
           Float.abs e.(0) < tiny w
           && Float.abs e.(1) < tiny w
           && (e.(0) <> 0. || e.(1) <> 0.) );
       ( "a dividend part within 3 ranks of the scaling's switch",
         fun w e -> ranks w e.(0) (tiny w) <= 3L || ranks w e.(1) (tiny w) <= 3L
       );
       ( "a quotient past the greatest finite",
         fun w e ->
           finite_all e
           && log_mod e.(0) e.(1) -. log_mod e.(2) e.(3)
              > Float.log (greatest w) );
       ( "a quotient below the least normal, not zero",
         fun w e ->
           finite_all e
           && (e.(0) <> 0. || e.(1) <> 0.)
           && log_mod e.(0) e.(1) -. log_mod e.(2) e.(3) < Float.log (least w)
       );
     ]
    @ divisor_covers 2)

let crecip =
  complex1 "crecip" { c1 = Nx.recip }
    (parts2 (divisor_spec false, divisor_spec true))
    divisor
    (( "a reciprocal past the greatest finite",
       fun w e ->
         finite_all e
         && (e.(0) <> 0. || e.(1) <> 0.)
         && -.log_mod e.(0) e.(1) > Float.log (greatest w) )
    :: divisor_covers 0)

(* Numbers of one operand, drawn as pairs of codes [x; y]. *)

let number gx gy =
  Gen.(
    let+ x = gx and+ y = gy in
    [| x; y |])

let zero wide = Gen.map (fun neg -> if neg then sign wide else 0L) Gen.bool

(* A code of either sign of magnitude uniform in [lo, hi]. *)
let between wide lo hi =
  Gen.(
    let+ v = float_range lo hi and+ neg = bool in
    code wide (if neg then -.v else v))

(* A code of either sign within 3 ranks of [p]. *)
let around wide p =
  Gen.(
    let+ d = int_range (-3) 3 and+ neg = bool in
    let c = Int64.add (code wide p) (Int64.of_int d) in
    if neg then Int64.logor c (sign wide) else c)

(* A code of either sign whose magnitude is [2^e m] for [e] in [lo, hi]. *)
let binade wide lo hi =
  Gen.(
    let* e = int_range lo hi in
    scaled wide e)

(* The numbers on the real axis beyond [lo], each side of the cut: [y] a zero of
   either sign. *)
let on_cut wide lo hi = number (between wide lo hi) (zero wide)

(* [x + iy] read in the coordinates a kind turns it to: [(x, y)] itself, or
   [(-y, x)] for the circular kinds of the hyperbolic ones, [(y, -x)] for asinh
   of asin. *)
type turn = Plain | Circular | Asinh

let coords t e =
  match t with
  | Plain -> (e.(0), e.(1))
  | Circular -> (-.e.(1), e.(0))
  | Asinh -> (e.(1), -.e.(0))

(* The codes of [x + iy] from the coordinates [(a, b)] a kind reads. *)
let uncoords wide t c =
  let neg v = Int64.logxor v (sign wide) in
  match t with
  | Plain -> c
  | Circular -> [| c.(1); neg c.(0) |]
  | Asinh -> [| neg c.(1); c.(0) |]

let turned wide t g = Gen.map (uncoords wide t) g

(* Covers on a kind's coordinates. *)
let on t covers =
  List.map (fun (l, f) -> (l, fun w e -> f w (coords t e))) covers

(* Specs on a kind's coordinates: the parts' specs of [x + iy]. *)
let specs_of t (sa, sb) =
  match t with Plain -> [| sa; sb |] | Circular | Asinh -> [| sb; sa |]

(* Roots and signs: nx_sqrt_raw's switches at an infinite or a zero part and x's
   sign, its scaling by an even power of two, x near -|z| where y / 2t takes
   over; nx_sign_raw's scaling and its hypot near overflow and underflow. *)

let root_spec w =
  spec
    [
      1.;
      4.;
      0.5;
      2.;
      least w;
      pick w (0x1p64, 0x1p512);
      pick w (0x1p-64, 0x1p-512);
      pick w (0x1p100, 0x1p1000);
    ]
    [ 4.; 1e30 ]

let extreme_parts wide =
  let top = pick wide (127, 1023) and low = pick wide (-149, -1074) in
  Gen.frequency
    [
      (1, number (binade wide (top - 10) top) (binade wide (top - 40) top));
      (1, number (binade wide low (low + 30)) (binade wide low (low + 60)));
      (1, number (binade wide (top - 10) top) (binade wide low (low + 30)));
    ]

let sqrt_pairs wide =
  let top = pick wide (127, 1023) in
  Gen.frequency
    [
      (2, on_cut wide 0. 1e3);
      (1, number (between wide 0. 1e3) (zero wide));
      (* x near -|z|: y / 2t is the larger part *)
      ( 2,
        number
          (Gen.map (fun c -> Int64.logor c (sign wide)) (between wide 1. 1e3))
          (binade wide (-60) (-10)) );
      (2, extreme_parts wide);
      (1, number (binade wide (top - 2) top) (binade wide (top - 2) top));
    ]

let csqrt =
  let sp = (root_spec false, root_spec true) in
  complex1 "csqrt" { c1 = Nx.sqrt } (parts2 sp) sqrt_pairs
    [
      ( "on the cut, y = +0, x negative",
        fun _ e -> e.(0) < 0. && e.(1) = 0. && not (Float.sign_bit e.(1)) );
      ( "on the cut, y = -0, x negative",
        fun _ e -> e.(0) < 0. && e.(1) = 0. && Float.sign_bit e.(1) );
      ("y = 0, x positive", fun _ e -> e.(0) > 0. && e.(1) = 0.);
      ( "x negative, |y| below 2^-10 |x|, not zero",
        fun _ e ->
          e.(0) < 0.
          && e.(1) <> 0.
          && Float.abs e.(1) < Float.abs e.(0) *. 0x1p-10 );
      ( "a finite part past 2^-10 the greatest finite",
        fun w e ->
          finite_all e
          && Float.max (Float.abs e.(0)) (Float.abs e.(1))
             > greatest w *. 0x1p-10 );
      ( "both parts below 2^30 the least normal, not zero",
        fun w e ->
          (e.(0) <> 0. || e.(1) <> 0.)
          && Float.abs e.(0) < least w *. 0x1p30
          && Float.abs e.(1) < least w *. 0x1p30 );
    ]

let sign_pairs wide =
  Gen.frequency
    [ (3, extreme_parts wide); (1, number (uniform wide 4.) (uniform wide 4.)) ]

let csign =
  let sp = (root_spec false, root_spec true) in
  let rt w = Float.sqrt (greatest w) and lt w = Float.sqrt (least w) in
  complex1 "csign" { c1 = Nx.sign } (parts2 sp) sign_pairs
    [
      ("both parts zero", fun _ e -> e.(0) = 0. && e.(1) = 0.);
      ( "a finite part past the square root of the greatest finite",
        fun w e ->
          finite_all e && Float.max (Float.abs e.(0)) (Float.abs e.(1)) > rt w
      );
      ( "parts below the square root of the least normal, not zero",
        fun w e ->
          (e.(0) <> 0. || e.(1) <> 0.)
          && Float.abs e.(0) < lt w
          && Float.abs e.(1) < lt w );
      ( "finite parts more than 2^30 apart, not zero",
        fun _ e ->
          finite_all e
          && e.(0) <> 0.
          && e.(1) <> 0.
          && (Float.abs e.(0) > 0x1p30 *. Float.abs e.(1)
             || Float.abs e.(1) > 0x1p30 *. Float.abs e.(0)) );
    ]

(* Logarithms: nx_log_abs's switch where the larger part a is in [1/2, 2] and
   the smaller b not zero, the unit circle where a^2 + b^2 - 1 cancels, b / a
   underflowing; atan2's cut on the negative real axis; nx_log2_raw's axes. *)

let log_spec w =
  spec
    [
      0.5;
      2.;
      1.;
      0.6;
      0.8;
      least w;
      pick w (1e30, 1e300);
      pick w (0x1p-30, 0x1p-60);
    ]
    [ 2.; 1e4 ]

(* Within a few ranks of the unit circle: (cos t, sin t) rounded, of any
   signs. *)
let unit_circle wide =
  Gen.(
    let+ t = float_range 0. (Float.pi /. 2.)
    and+ sx = bool
    and+ sy = bool
    and+ d = int_range (-2) 2 in
    let s b v = if b then -.v else v in
    [|
      Int64.add (code wide (s sx (Float.cos t))) (Int64.of_int d);
      code wide (s sy (Float.sin t));
    |])

let log_pairs wide =
  Gen.frequency
    [
      (3, unit_circle wide);
      (1, on_cut wide 0. 1e3);
      (1, number (zero wide) (between wide 0. 1e3));
      (1, number (between wide 0. 1e3) (zero wide));
      (1, number (around wide 0.5) (binade wide (-40) (-2)));
      (1, number (around wide 2.) (binade wide (-40) (-2)));
      (1, number (between wide 1. 2.) (binade wide (-60) (-20)));
      (1, extreme_parts wide);
    ]

let log_covers =
  let a e = Float.max (Float.abs e.(0)) (Float.abs e.(1))
  and b e = Float.min (Float.abs e.(0)) (Float.abs e.(1)) in
  let near w x v = ranks w x v <= 3L in
  [
    ( "the larger part in [1/2, 2], the smaller not zero",
      fun _ e -> a e >= 0.5 && a e <= 2. && b e <> 0. );
    ( "the larger part within 3 ranks of 1/2 or 2, the smaller not zero",
      fun w e -> b e <> 0. && (near w (a e) 0.5 || near w (a e) 2.) );
    ( "within 2^-20 of the unit circle, off the axes",
      fun _ e ->
        b e <> 0.
        && Float.abs ((e.(0) *. e.(0)) +. (e.(1) *. e.(1)) -. 1.) < 0x1p-20 );
    ( "the smaller part below 2^-20 the larger, not zero",
      fun _ e -> b e <> 0. && Float.is_finite (a e) && b e < a e *. 0x1p-20 );
    ( "on the cut, y = +0",
      fun _ e -> e.(0) < 0. && e.(1) = 0. && not (Float.sign_bit e.(1)) );
    ( "on the cut, y = -0",
      fun _ e -> e.(0) < 0. && e.(1) = 0. && Float.sign_bit e.(1) );
    ( "x = 0, y finite, not zero",
      fun _ e -> e.(0) = 0. && e.(1) <> 0. && Float.is_finite e.(1) );
    ( "y = 0, x finite, not zero",
      fun _ e -> e.(1) = 0. && e.(0) <> 0. && Float.is_finite e.(0) );
  ]

let clog =
  let sp = (log_spec false, log_spec true) in
  complex1 "clog" { c1 = Nx.log } (parts2 sp) log_pairs log_covers

let clog2 =
  let sp = (log_spec false, log_spec true) in
  complex1 "clog2" { c1 = Nx.log2 } (parts2 sp) log_pairs log_covers

(* Exponentials: nx_exp_raw's switches at y = 0, a non-finite part, x past
   EXP_NEAR, 86 or 708, to the clamp CEXP_HI, 194 or 1456, where exp_times
   scales cos y and sin y, and the trigonometric kinds' switch in y;
   nx_exp2_raw's at y ln2 past that switch, read from 2 ln2 / pi, and its clamps
   CEXP2_HI, 280 or 2101, and EXP2_LO. *)

let exp_near = (86., 708.)
let cexp_hi = (194., 1456.)
let cexp2_hi = (280., 2101.)
let ln2 = 0.6931471805599453

let phase_spec w =
  let big = pick w trig_big in
  spec
    ([ 0.; 1.; big; 2. *. big; pick w trig_tiny ]
    @ List.map (fun k -> k *. half_pi) [ 1.; 2.; 3.; 355. ])
    [ Float.pi; pick w (1e5, 1e9) ]

let exp_spec w =
  spec
    [
      pick w exp_near;
      pick w cexp_hi;
      pick w (88.72284, 709.782712893384);
      pick w (103.97208, 745.1332191019411);
      pick w (87.33655, 708.3964185322641);
      pick w (104., 746.);
      1.;
      half_ln2;
    ]
    [ 1.; pick w (300., 2000.) ]

(* y where cos y or sin y is small: within a rank of a multiple of pi/2. *)
let near_axis wide =
  Gen.(
    let+ k = int_range 1 1000 and+ d = int_range (-1) 1 and+ neg = bool in
    let c =
      Int64.add (code wide (Float.of_int k *. half_pi)) (Int64.of_int d)
    in
    if neg then Int64.logor c (sign wide) else c)

let exp_pairs wide =
  let near = pick wide exp_near and hi = pick wide cexp_hi in
  Gen.frequency
    [
      (2, number (between wide near hi) (uniform wide Float.pi));
      (1, number (around wide near) (uniform wide Float.pi));
      (1, number (between wide hi (2. *. hi)) (uniform wide Float.pi));
      (1, number (between wide 80. hi) (near_axis wide));
      (1, number (uniform wide 4.) (between wide 1. (pick wide (1e5, 1e9))));
      (1, number (uniform wide 4.) (zero wide));
    ]

let exp_covers =
  let near w = pick w exp_near and hi w = pick w cexp_hi in
  let t w e = Float.abs e.(1) >= pick w trig_big && Float.is_finite e.(1) in
  [
    ( "x past the near range, to the clamp",
      fun w e -> e.(0) > near w && e.(0) <= hi w );
    ( "x within 3 ranks of the near range's end",
      fun w e -> ranks w e.(0) (near w) <= 3L && e.(0) > 0. );
    ( "x past the clamp, finite",
      fun w e -> e.(0) > hi w && Float.is_finite e.(0) );
    ( "e^x subnormal",
      fun w e ->
        e.(0) < pick w (-87.34, -708.4) && e.(0) > pick w (-103.97, -745.13) );
    ( "x past 80, |cos y| or |sin y| below 2^-10",
      fun w e ->
        e.(0) > 80.
        && e.(0) <= hi w
        && Float.abs (Float.cos e.(1) *. Float.sin e.(1)) < 0x1p-10 );
    ("y past the trigonometric switch, finite", t);
    ("y = 0, x finite", fun _ e -> e.(1) = 0. && Float.is_finite e.(0));
  ]

let cexp =
  complex1 "cexp" { c1 = Nx.exp }
    [| (exp_spec false, exp_spec true); (phase_spec false, phase_spec true) |]
    exp_pairs exp_covers

let exp2_spec w =
  spec
    [
      pick w cexp2_hi;
      pick w (128., 1024.);
      pick w (129., 1025.);
      pick w (150., 1075.);
      pick w (151., 1076.);
      pick w (126., 1022.);
      0.5;
      1.;
    ]
    [ 1.; pick w (400., 3000.) ]

(* y whose phase y ln2 lies past or near the trigonometric switch. *)
let exp2_phase_spec w =
  let s = pick w trig_big /. ln2 in
  spec
    [ s; 2. *. s; 1.; 0.; half_pi /. ln2; 355. *. half_pi /. ln2 ]
    [ 4.; pick w (1e6, 1e10) ]

let exp2_pairs wide =
  let hi = pick wide cexp2_hi and s = pick wide trig_big /. ln2 in
  Gen.frequency
    [
      (2, number (uniform wide 4.) (between wide (s *. 0.999) (s *. 1.001)));
      (1, number (uniform wide 4.) (between wide s (pick wide (1e30, 1e300))));
      (1, number (between wide (hi *. 0.5) (hi *. 1.5)) (uniform wide 4.));
      (1, number (between wide 100. 2000.) (uniform wide 4.));
      (1, number (uniform wide 4.) (zero wide));
    ]

let cexp2 =
  let hi w = pick w cexp2_hi and s w = pick w trig_big /. ln2 in
  complex1 "cexp2" { c1 = Nx.exp2 }
    [|
      (exp2_spec false, exp2_spec true);
      (exp2_phase_spec false, exp2_phase_spec true);
    |]
    exp2_pairs
    [
      ( "y ln2 past the trigonometric switch, finite",
        fun w e -> Float.abs e.(1) >= s w && Float.is_finite e.(1) );
      ( "y ln2 within 2^-12 of the switch",
        fun w e -> Float.abs ((Float.abs e.(1) /. s w) -. 1.) < 0x1p-12 );
      ( "x past the clamp, finite",
        fun w e -> e.(0) > hi w && Float.is_finite e.(0) );
      ( "x below the low clamp, finite",
        fun w e -> e.(0) < pick w (-151., -1076.) && Float.is_finite e.(0) );
      ( "2^x subnormal",
        fun w e ->
          e.(0) < pick w (-126., -1022.) && e.(0) > pick w (-150., -1075.) );
      ("y = 0, x finite", fun _ e -> e.(1) = 0. && Float.is_finite e.(0));
    ]

(* Powers: nx_pow_raw's switches at w = 0, z on the positive real axis with w
   real, w real or imaginary, log z's cut, and results past the greatest finite
   or below the least normal. *)

let pow_pairs wide =
  let open Gen in
  let any = uniform wide 4. in
  let z = number any any and w = number any any in
  let cat a b =
    let+ a = a and+ b = b in
    Array.append a b
  in
  let positive = number (between wide 0. 8.) (zero wide) in
  let real = number any (zero wide) and imag = number (zero wide) any in
  frequency
    [
      (1, cat z (number (zero wide) (zero wide)));
      (1, cat positive real);
      (1, cat z real);
      (1, cat z imag);
      (1, cat (on_cut wide 0. 8.) w);
      (1, cat (number (zero wide) (zero wide)) w);
      (* |w log z| large: the result overflows or underflows *)
      ( 2,
        cat
          (number (between wide 1e3 1e6) any)
          (number (between wide 10. 60.) any) );
    ]

let cpow =
  let zs w = spec [ 1.; 2.; 0.5; pick w (1e30, 1e300); least w ] [ 4.; 1e3 ] in
  let ws w = spec [ 1.; 2.; 0.5; 3.; pick w (130., 1030.) ] [ 4.; 200. ] in
  let wlog e =
    let l =
      { Complex.re = log_mod e.(0) e.(1); im = Float.atan2 e.(1) e.(0) }
    in
    Complex.mul { Complex.re = e.(2); im = e.(3) } l
  in
  (* Whether w log z, its parts multiplied as nx_pow_raw multiplies them, is
     -inf + i NaN: 222 of the 10^4 elements of Annex G's parts are, drawn in
     7/8 * 1/6 of the elements at least, so that a block of 4096 misses the
     class with probability below 2e-6 and the 12 blocks of a run below 1e-60;
     the example holds it. *)
  let vanishing e =
    let zr = e.(0) and zi = e.(1) and wr = e.(2) and wi = e.(3) in
    let lr = Float.log (Float.hypot zr zi) and li = Float.atan2 zi zr in
    let pr, pi =
      if wi = 0. then (wr *. lr, wr *. li)
      else if wr = 0. then (-.(wi *. li), wi *. lr)
      else ((wr *. lr) -. (wi *. li), (wr *. li) +. (wi *. lr))
    in
    (wr <> 0. || wi <> 0.)
    && (not (zi = 0. && wi = 0. && zr > 0.))
    && pr = Float.neg_infinity
    && Float.is_nan pi
  in
  complex2 "cpow" { c2 = Nx.pow }
    [|
      (zs false, zs true);
      (zs false, zs true);
      (ws false, ws true);
      (ws false, ws true);
    |]
    pow_pairs
    [
      ("w = 0", fun _ e -> e.(2) = 0. && e.(3) = 0.);
      ( "z on the positive real axis, w real",
        fun _ e -> e.(0) > 0. && e.(1) = 0. && e.(3) = 0. );
      ( "w real, not zero, z off the positive real axis",
        fun _ e -> e.(3) = 0. && e.(2) <> 0. && (e.(1) <> 0. || e.(0) < 0.) );
      ("w imaginary, not zero", fun _ e -> e.(2) = 0. && e.(3) <> 0.);
      ("w of two non-zero parts", fun _ e -> e.(2) <> 0. && e.(3) <> 0.);
      ( "z = 0, w not zero",
        fun _ e -> e.(0) = 0. && e.(1) = 0. && (e.(2) <> 0. || e.(3) <> 0.) );
      ( "z on the cut, y = ±0",
        fun _ e -> e.(0) < 0. && e.(1) = 0. && (e.(2) <> 0. || e.(3) <> 0.) );
      ("w log z = -inf + i NaN", fun _ e -> vanishing e);
      ( "a result past the greatest finite",
        fun w e ->
          finite_all e && e.(0) <> 0. && (wlog e).re > Float.log (greatest w) );
      ( "a result below the least normal, not zero",
        fun w e ->
          finite_all e && e.(0) <> 0. && (wlog e).re < Float.log (least w) );
    ]

(* Hyperbolic and circular kinds, on the hyperbolic coordinates (x, y): sinh and
   cosh's switch at HYP_BIG, 12 or 22, past which exp_times scales cos y and sin
   y, its clamp CEXP_HI, the overflow of e^|x| / 2, y = 0, the trigonometric
   switch in y; tanh's at HYP_BIG and tan y's poles. sin, cos and tan read sinh,
   cosh and tanh at iz. *)

let hyp_big = (12., 22.)

let hyp_spec w =
  spec
    [
      pick w hyp_big;
      pick w cexp_hi;
      pick w (89.41599, 710.4758600739439);
      1.;
      half_ln2;
      0x1p-30;
      6.;
    ]
    [ 2.; pick w (200., 1600.) ]

let hyp_pairs t wide =
  let big = pick wide hyp_big and hi = pick wide cexp_hi in
  turned wide t
    (Gen.frequency
       [
         (1, number (around wide big) (uniform wide Float.pi));
         (2, number (between wide big hi) (uniform wide Float.pi));
         (1, number (between wide hi (2. *. hi)) (uniform wide Float.pi));
         (1, number (between wide 0. hi) (near_axis wide));
         (1, number (uniform wide 4.) (between wide 1. (pick wide (1e5, 1e9))));
         (1, number (uniform wide 4.) (zero wide));
       ])

let hyp_covers =
  let big w = pick w hyp_big and hi w = pick w cexp_hi in
  let ax (x, _) = Float.abs x in
  [
    ("|x| past HYP_BIG, to the clamp", fun w c -> ax c > big w && ax c <= hi w);
    ("|x| within 3 ranks of HYP_BIG", fun w (x, _) -> ranks w x (big w) <= 3L);
    ( "|x| past the clamp, finite",
      fun w (x, _) -> Float.abs x > hi w && Float.is_finite x );
    ( "|x| past the overflow of e^|x| / 2, to the clamp",
      fun w c -> ax c > pick w (89.416, 710.476) && ax c <= hi w );
    ( "|x| past HYP_BIG, |cos y| or |sin y| below 2^-10",
      fun w (x, y) ->
        Float.abs x > big w
        && Float.abs x <= hi w
        && Float.abs (Float.cos y *. Float.sin y) < 0x1p-10 );
    ( "y past the trigonometric switch, finite",
      fun w (_, y) -> Float.abs y >= pick w trig_big && Float.is_finite y );
    ("y = 0, x finite", fun _ (x, y) -> y = 0. && Float.is_finite x);
  ]

let hyperbolic name f t =
  let sp = (hyp_spec false, hyp_spec true)
  and pp = (phase_spec false, phase_spec true) in
  complex1 name f (specs_of t (sp, pp)) (hyp_pairs t) (on t hyp_covers)

let tanh_pairs t wide =
  let big = pick wide hyp_big in
  turned wide t
    (Gen.frequency
       [
         (1, number (around wide big) (uniform wide Float.pi));
         (1, number (between wide big 50.) (uniform wide Float.pi));
         (2, number (uniform wide 4.) (near_axis wide));
         (1, number (uniform wide 4.) (between wide 1. (pick wide (1e5, 1e9))));
         (1, number (uniform wide 4.) (zero wide));
       ])

let tanh_covers =
  let big w = pick w hyp_big in
  [
    ( "|x| past HYP_BIG, finite",
      fun w (x, _) -> Float.abs x > big w && Float.is_finite x );
    ("|x| within 3 ranks of HYP_BIG", fun w (x, _) -> ranks w x (big w) <= 3L);
    ( "|x| at most HYP_BIG, |cos y| below 2^-10",
      fun w (x, y) -> Float.abs x <= big w && Float.abs (Float.cos y) < 0x1p-10
    );
    ( "y past the trigonometric switch, finite",
      fun w (_, y) -> Float.abs y >= pick w trig_big && Float.is_finite y );
    ("y = 0, x finite", fun _ (x, y) -> y = 0. && Float.is_finite x);
  ]

let tanh_spec w =
  spec [ pick w hyp_big; 1.; 0.5; 2.; 0x1p-30; pick w (9.5, 19.5) ] [ 2.; 50. ]

let tanh_kind name f t =
  let sp = (tanh_spec false, tanh_spec true)
  and pp = (phase_spec false, phase_spec true) in
  complex1 name f (specs_of t (sp, pp)) (tanh_pairs t) (on t tanh_covers)

let csinh = hyperbolic "csinh" { c1 = Nx.sinh } Plain
let ccosh = hyperbolic "ccosh" { c1 = Nx.cosh } Plain
let csin = hyperbolic "csin" { c1 = Nx.sin } Circular
let ccos = hyperbolic "ccos" { c1 = Nx.cos } Circular
let ctanh = tanh_kind "ctanh" { c1 = Nx.tanh } Plain
let ctan = tanh_kind "ctan" { c1 = Nx.tan } Circular

(* Inverse kinds, on the coordinates of asin, acos or atanh: asin and acos's
   switch at ASIN_BIG, 2^13 or 2^28, the real axis within [-1, 1], the cuts
   beyond ±1 on either side, the branch points; atanh's at ATANH_BIG, 2^62 or
   2^510, x = 0, |x| = 1, its cuts. atan reads atanh at iz, asinh asin at -iz,
   and acosh acos at x + i|y|. *)

let asin_big = (0x1p13, 0x1p28)
let atanh_big = (0x1p62, 0x1p510)

let inv_spec big w =
  spec [ 1.; 0.5; 2.; pick w big; 0.; pick w (0x1p-30, 0x1p-60) ] [ 2.; 1e4 ]

let inv_pairs big t wide =
  let b = pick wide big in
  turned wide t
    (Gen.frequency
       [
         (2, on_cut wide 1. 1e3);
         (1, number (between wide 0. 1.) (zero wide));
         (1, number (around wide 1.) (binade wide (-60) (-10)));
         (1, number (around wide b) (uniform wide 4.));
         (1, number (uniform wide 4.) (around wide b));
         (1, number (between wide b (b *. 1e3)) (between wide 0. (b *. 1e3)));
         (1, number (zero wide) (uniform wide 4.));
       ])

let inv_covers big =
  let b w = pick w big in
  let pos v = not (Float.sign_bit v) in
  [
    ( "a part past the far switch, finite",
      fun w (x, y) ->
        Float.is_finite x && Float.is_finite y
        && (Float.abs x > b w || Float.abs y > b w) );
    ( "a part within 3 ranks of the far switch",
      fun w (x, y) -> ranks w x (b w) <= 3L || ranks w y (b w) <= 3L );
    ( "on the real axis within [-1, 1]",
      fun _ (x, y) -> y = 0. && Float.abs x <= 1. );
    ( "on a cut, y = +0",
      fun _ (x, y) -> Float.abs x > 1. && Float.is_finite x && y = 0. && pos y
    );
    ( "on a cut, y = -0",
      fun _ (x, y) ->
        Float.abs x > 1. && Float.is_finite x && y = 0. && not (pos y) );
    ( "within 2^-20 of ±1, y not zero",
      fun _ (x, y) -> Float.abs (Float.abs x -. 1.) < 0x1p-20 && y <> 0. );
    ( "|x| = 1, y not zero",
      fun _ (x, y) -> Float.abs x = 1. && y <> 0. && Float.is_finite y );
    ("x = 0, y finite", fun _ (x, y) -> x = 0. && Float.is_finite y);
  ]

let inverse name f big t =
  let sp = (inv_spec big false, inv_spec big true) in
  complex1 name f
    (specs_of t (sp, sp))
    (inv_pairs big t)
    (on t (inv_covers big))

let casin = inverse "casin" { c1 = Nx.asin } asin_big Plain
let cacos = inverse "cacos" { c1 = Nx.acos } asin_big Plain
let casinh = inverse "casinh" { c1 = Nx.asinh } asin_big Asinh
let catanh = inverse "catanh" { c1 = Nx.atanh } atanh_big Plain
let catan = inverse "catan" { c1 = Nx.atan } atanh_big Circular

(* acosh reads acos at x + i|y|: the cuts are below 1, y's sign naming the
   side. *)
let cacosh = inverse "cacosh" { c1 = Nx.acosh } asin_big Plain

(* Composed functions

   Functions nx composes from kinds and exact operations, one program each
   (nx.ml, special.ml), at float32 and float64: their points are the switches
   their formulas take where they state one (asinh's 2^28 and 2^-28, acosh's 2,
   atanh's 1/2, hypot's least normal, the inverses' tails) and their domains'
   edges, their ranges their domains. A function of several operands draws
   tuples from its domain in 2 of 5 elements: concentrations and arguments
   spread over [1e-3, 1e4] on a logarithmic scale, probabilities uniform. *)

let big = 0x1p28
let tiny = 0x1p-28
let mag e = Float.abs e.(0)

let tuple doms wide =
  Gen.(
    map Array.of_list
      (List.fold_right
         (fun g acc ->
           let+ v = g and+ vs = acc in
           code wide v :: vs)
         doms (constant [])))

let spread lo hi =
  Gen.map Stdlib.exp (Gen.float_range (Stdlib.log lo) (Stdlib.log hi))

let prob = Gen.float_range 0. 1.

let composed1 name f points ranges covers =
  unary name f (spec points ranges) (spec points ranges) covers

let composed n name fn pts covers =
  let s = spec (fst pts) (snd pts) in
  {
    name;
    fn;
    specs = Array.make n (s, s);
    covers;
    fixed = (if n = 2 then annex_pairs else none);
    generic = true;
  }

let asinh =
  composed1 "asinh" { f1 = Nx.asinh } [ big; tiny; 1.; 0.5 ] [ 1.; 1e4 ]
    [
      ("|x| below 2^-28, not zero", fun _ e -> e.(0) <> 0. && mag e < tiny);
      ("|x| from 2^-28 below 2^28", fun _ e -> mag e >= tiny && mag e < big);
      ("|x| from 2^28, finite", fun _ e -> mag e >= big && finite_pos (mag e));
    ]

let sinc =
  composed1 "sinc" { f1 = Nx.sinc } [ 0.25; 0.; 1.; 0.5 ] [ 1.; 1e4 ]
    [
      ("|x| below 1/4", fun _ e -> mag e < 0.25);
      ("|x| from 1/4, finite", fun _ e -> mag e >= 0.25 && finite_pos (mag e));
    ]

let acosh =
  composed1 "acosh" { f1 = Nx.acosh } [ 1.; 2.; big; 1.5 ] [ 2.; 1e4 ]
    [
      ("x below 1", fun _ e -> e.(0) < 1.);
      ("x in [1, 2)", fun _ e -> e.(0) >= 1. && e.(0) < 2.);
      ("x in [2, 2^28)", fun _ e -> e.(0) >= 2. && e.(0) < big);
      ("x from 2^28, finite", fun _ e -> e.(0) >= big && Float.is_finite e.(0));
    ]

let atanh =
  composed1 "atanh" { f1 = Nx.atanh } [ 0.5; 1.; tiny; 0.25 ] [ 1.; 2. ]
    [
      ("|x| below 2^-28, not zero", fun _ e -> e.(0) <> 0. && mag e < tiny);
      ("|x| in [2^-28, 1/2)", fun _ e -> mag e >= tiny && mag e < 0.5);
      ("|x| in [1/2, 1)", fun _ e -> mag e >= 0.5 && mag e < 1.);
      ("|x| = 1", fun _ e -> mag e = 1.);
    ]

let rsqrt =
  composed1 "rsqrt" { f1 = Nx.rsqrt } [ 1.; 4.; 0.25 ] [ 1.; 1e4 ]
    [ ("x positive, finite", fun _ e -> finite_pos e.(0)) ]

let wide_spread = spread 1e-300 1e300

let hypot =
  composed 2 "hypot"
    (Two ({ f2 = Nx.hypot }, tuple [ wide_spread; wide_spread ]))
    ([ 1.; 0x1p-1022; 0x1p-126; 3. ], [ 1.; 1e30 ])
    [
      ( "|x| = |y|, finite, not zero",
        fun _ e -> mag e = Float.abs e.(1) && finite_pos (mag e) );
      ( "both below the least normal, one not zero",
        fun w e ->
          let least = pick w (0x1p-126, 0x1p-1022) in
          mag e < least
          && Float.abs e.(1) < least
          && (e.(0) <> 0. || e.(1) <> 0.) );
    ]

let erfc =
  composed1 "erfc" { f1 = Nx.erfc }
    [ 0.5; 1.; 2.; 4.; 10.; 26.; 27. ]
    [ 1.; 30. ]
    [
      ("x below -1", fun _ e -> e.(0) < -1.);
      ("|x| below 1", fun _ e -> mag e < 1.);
      ("x in [1, 10]", fun _ e -> e.(0) >= 1. && e.(0) <= 10.);
      ("x past 10, finite", fun _ e -> e.(0) > 10. && Float.is_finite e.(0));
    ]

let erfinv =
  composed1 "erfinv" { f1 = Nx.erfinv }
    [ 1.; 0.5; 0.9; 0.99; tiny ]
    [ 1. ]
    [
      ("|p| below 1/2", fun _ e -> mag e < 0.5);
      ("|p| in [1/2, 1)", fun _ e -> mag e >= 0.5 && mag e < 1.);
      ("|p| = 1", fun _ e -> mag e = 1.);
      ("|p| past 1", fun _ e -> mag e > 1.);
    ]

let ndtr =
  composed1 "ndtr" { f1 = Nx.ndtr } [ 1.; 5.; 14.; 38.5; 0.5 ] [ 1.; 40. ]
    [
      ("x below -14", fun _ e -> e.(0) < -14.);
      ("x in [-14, 0)", fun _ e -> e.(0) >= -14. && e.(0) < 0.);
      ("x positive", fun _ e -> e.(0) > 0.);
    ]

let log_ndtr =
  composed1 "log_ndtr" { f1 = Nx.log_ndtr }
    [ 1.; 5.; 38.5; 1e5; 2.6e19; 1.9e154 ]
    [ 10.; 1e5 ]
    [
      ("x below -38.5", fun _ e -> e.(0) < -38.5);
      ("x in [-38.5, 5]", fun _ e -> e.(0) >= -38.5 && e.(0) <= 5.);
      ("x past 5, finite", fun _ e -> e.(0) > 5. && Float.is_finite e.(0));
    ]

let ndtri =
  composed1 "ndtri" { f1 = Nx.ndtri }
    [ 0.5; 0.25; 1e-10; 1e-30; 0.75; 1. ]
    [ 1. ]
    [
      ("p below 1e-10, not zero", fun _ e -> e.(0) > 0. && e.(0) < 1e-10);
      ("p in [1e-10, 1/2]", fun _ e -> e.(0) >= 1e-10 && e.(0) <= 0.5);
      ("p in (1/2, 1)", fun _ e -> e.(0) > 0.5 && e.(0) < 1.);
      ("p outside [0, 1]", fun _ e -> e.(0) < 0. || e.(0) > 1.);
    ]

let gammas name f =
  composed1 name f
    [ 1.; 2.; 3.; 0.5; 1.4616321449683622; 8.; 1e6; 0x1p52 ]
    [ 10.; 1e4 ]
    [
      ( "x negative, not an integer",
        fun _ e ->
          e.(0) < 0. && Float.is_finite e.(0) && not (Float.is_integer e.(0)) );
      ("x a negative integer", fun _ e -> e.(0) < 0. && Float.is_integer e.(0));
      ("x in (0, 3)", fun _ e -> e.(0) > 0. && e.(0) < 3.);
      ("x past 8, finite", fun _ e -> e.(0) > 8. && Float.is_finite e.(0));
    ]

let lgamma = gammas "lgamma" { f1 = Nx.lgamma }
let digamma = gammas "digamma" { f1 = Nx.digamma }
let shapes = [ 1.; 0.5; 0x1p20; 0x1p-61; 1e10 ]

let lbeta =
  composed 2 "lbeta"
    (Two ({ f2 = Nx.lbeta }, tuple [ spread 1e-3 1e10; spread 1e-3 1e10 ]))
    (shapes, [ 10.; 1e4 ])
    [
      ( "both positive and finite",
        fun _ e -> finite_pos e.(0) && finite_pos e.(1) );
      ("a or b past 1e5", fun _ e -> e.(0) > 1e5 || e.(1) > 1e5);
    ]

let gamma_args = tuple [ spread 1e-3 1e4; spread 1e-3 1e4 ]

let incomplete name f =
  composed 2 name
    (Two (f, gamma_args))
    (shapes @ [ 10.; 100. ], [ 10.; 1e4 ])
    [
      ( "0 < x < a",
        fun _ e -> e.(1) > 0. && e.(1) < e.(0) && Float.is_finite e.(0) );
      ( "x > a > 0",
        fun _ e -> e.(0) > 0. && e.(1) > e.(0) && Float.is_finite e.(1) );
      ("a in (0, 1)", fun _ e -> e.(0) > 0. && e.(0) < 1.);
    ]

let gammainc = incomplete "gammainc" { f2 = Nx.gammainc }
let gammaincc = incomplete "gammaincc" { f2 = Nx.gammaincc }
let log_gammainc = incomplete "log_gammainc" { f2 = Nx.log_gammainc }
let log_gammaincc = incomplete "log_gammaincc" { f2 = Nx.log_gammaincc }

let tails =
  [
    ("p in (0, 1/2)", fun _ e -> e.(1) > 0. && e.(1) < 0.5);
    ("p in (1/2, 1)", fun _ e -> e.(1) > 0.5 && e.(1) < 1.);
    ("a in (0, 1)", fun _ e -> e.(0) > 0. && e.(0) < 1.);
  ]

let gamma_inverse name f =
  composed 2 name
    (Two (f, tuple [ spread 1e-3 1e4; prob ]))
    (shapes @ [ 0.25; 0.75; 1e-10 ], [ 1.; 1e4 ])
    tails

let gammaincinv = gamma_inverse "gammaincinv" { f2 = Nx.gammaincinv }
let gammainccinv = gamma_inverse "gammainccinv" { f2 = Nx.gammainccinv }
let beta_args = tuple [ spread 1e-3 1e4; spread 1e-3 1e4; prob ]

let beta_covers =
  [
    ( "x below a / (a + b)",
      fun _ e -> e.(2) > 0. && e.(2) < e.(0) /. (e.(0) +. e.(1)) );
    ( "x above a / (a + b)",
      fun _ e -> e.(2) < 1. && e.(2) > e.(0) /. (e.(0) +. e.(1)) );
    ( "a or b in (0, 1)",
      fun _ e -> (e.(0) > 0. && e.(0) < 1.) || (e.(1) > 0. && e.(1) < 1.) );
  ]

let incomplete_beta name f =
  composed 3 name
    (Three (f, beta_args))
    (shapes @ [ 0.25; 0.75; 1e-10 ], [ 1.; 1e4 ])
    beta_covers

let betainc = incomplete_beta "betainc" { f3 = Nx.betainc }
let betaincc = incomplete_beta "betaincc" { f3 = Nx.betaincc }
let log_betainc = incomplete_beta "log_betainc" { f3 = Nx.log_betainc }
let log_betaincc = incomplete_beta "log_betaincc" { f3 = Nx.log_betaincc }
let betaincinv = incomplete_beta "betaincinv" { f3 = Nx.betaincinv }
let betainccinv = incomplete_beta "betainccinv" { f3 = Nx.betainccinv }

let bessel name f =
  composed1 name f
    [ 8.; 500.; 714.; 92.; 1. ]
    [ 10.; 1e3 ]
    [
      ("|x| below 8", fun _ e -> mag e < 8.);
      ("|x| from 8, finite", fun _ e -> mag e >= 8. && Float.is_finite e.(0));
    ]

let i0e = bessel "i0e" { f1 = Nx.i0e }
let i1e = bessel "i1e" { f1 = Nx.i1e }

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
  | Two (_, pairs) | Three (_, pairs) ->
      Gen.frequency [ (3, alone); (2, pairs wide) ]
  | Cone (_, pairs) | Ctwo (_, pairs) ->
      Gen.frequency [ (3, alone); (1, annex_g wide (arity k)); (2, pairs wide) ]

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
  | Two _ | Three _ | Cone _ | Ctwo _ ->
      invalid_arg "Kinds.clamped: a kind of several operands or complex"
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
    | Three ({ f3 }, _), [ x; y; z ] ->
        Rune.jit
          Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor)
          f3 (on x) (on y) (on z)
    | _ -> invalid_arg "Kinds: operands of another arity"
  in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  Nx.place Nx.Host.on y

let eager k xs =
  match (k.fn, xs) with
  | One { f1 }, [ x ] -> f1 x
  | Two ({ f2 }, _), [ x; y ] -> f2 x y
  | Three ({ f3 }, _), [ x; y; z ] -> f3 x y z
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
   count. A result is its parts' codes: one for a float, two for a complex
   number, each NaN part equal to any NaN. *)
let misses k wide es ~eager ~compiled =
  let out = ref [] and count = ref 0 in
  let codes r = String.concat ", " (Array.to_list (Array.map (show wide) r)) in
  Array.iteri
    (fun i e ->
      if not (Array.for_all2 (same wide) eager.(i) compiled.(i)) then begin
        incr count;
        if !count <= 8 then
          out :=
            Printf.sprintf "%s(%s) at %s: eager %s, compiled %s" k.name
              (codes e)
              (if wide then "float64" else "float32")
              (codes eager.(i))
              (codes compiled.(i))
            :: !out
      end)
    es;
  if !count = 0 then []
  else
    List.rev !out
    @ [ Printf.sprintf "%d of %d elements differ" !count (Array.length es) ]

let ones cs = Array.map (fun c -> [| c |]) cs

(* A complex kind's operands, each of two columns of parts, and its result's
   parts. *)
let ccompiled (type c d) (module S : Nx.Devices with type d = d) k
    (xs : (Complex.t, c, Nx.host) Nx.t list) =
  let before = Rune.evaluations () in
  let on x = Nx.place S.on x in
  let y =
    match (k.fn, xs) with
    | Cone ({ c1 }, _), [ x ] ->
        Rune.jit Nx.Ptree.(tensor @-> returns tensor) c1 (on x)
    | Ctwo ({ c2 }, _), [ x; y ] ->
        Rune.jit
          Nx.Ptree.(tensor @-> tensor @-> returns tensor)
          c2 (on x) (on y)
    | _ -> invalid_arg "Kinds: operands of another arity"
  in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  Nx.place Nx.Host.on y

let ceager k xs =
  match (k.fn, xs) with
  | Cone ({ c1 }, _), [ x ] -> c1 x
  | Ctwo ({ c2 }, _), [ x; y ] -> c2 x y
  | _ -> invalid_arg "Kinds: operands of another arity"

let crun (type d) (module S : Nx.Devices with type d = d) k wide es =
  let col = column es in
  let ops = arity k / 2 in
  let pair a b = Array.map2 (fun r i -> [| r; i |]) a b in
  if wide then
    let xs =
      List.init ops (fun j ->
          Nx.complex Nx.complex128
            ~re:(tensor64 (col (2 * j)))
            ~im:(tensor64 (col ((2 * j) + 1))))
    in
    let parts y =
      pair (codes64 (Nx.real Nx.float64 y)) (codes64 (Nx.imag Nx.float64 y))
    in
    misses k wide es
      ~eager:(parts (ceager k xs))
      ~compiled:(parts (ccompiled (module S) k xs))
  else
    let xs =
      List.init ops (fun j ->
          Nx.complex Nx.complex64
            ~re:(tensor32 (col (2 * j)))
            ~im:(tensor32 (col ((2 * j) + 1))))
    in
    let parts y =
      pair (codes32 (Nx.real Nx.float32 y)) (codes32 (Nx.imag Nx.float32 y))
    in
    misses k wide es
      ~eager:(parts (ceager k xs))
      ~compiled:(parts (ccompiled (module S) k xs))

let run (type d) (module S : Nx.Devices with type d = d) k wide es =
  let cols = List.init (arity k) (column es) in
  if is_complex k then crun (module S) k wide es
  else if wide then
    let xs = List.map tensor64 cols in
    misses k wide es
      ~eager:(ones (codes64 (eager k xs)))
      ~compiled:(ones (codes64 (compiled (module S) k xs)))
  else
    let xs = List.map tensor32 cols in
    misses k wide es
      ~eager:(ones (codes32 (eager k xs)))
      ~compiled:(ones (codes32 (compiled (module S) k xs)))

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

(* Classes of a complex operand, a pair of parts, every complex kind's law
   covers: Annex G's special values lie where a part is infinite or a NaN. *)
let generic_complex =
  let any p _ e =
    let rec go j =
      j + 1 < Array.length e && (p e.(j) e.(j + 1) || go (j + 2))
    in
    go 0
  in
  let inf v = Float.abs v = Float.infinity and fin = Float.is_finite in
  let both p q x y = (p x && q y) || (q x && p y) in
  [
    ("an infinite part beside a NaN", any (both inf Float.is_nan));
    ("two infinite parts", any (fun x y -> inf x && inf y));
    ("an infinite part beside a finite one", any (both inf fin));
    ("a NaN part beside a finite one", any (both Float.is_nan fin));
    ("two NaN parts", any (fun x y -> Float.is_nan x && Float.is_nan y));
    ("an infinite part beside a zero", any (both inf (fun v -> v = 0.)));
  ]

let covers k wide es =
  let vs = Array.map (Array.map (value wide)) es in
  let dt = if wide then "float64: " else "float32: " in
  let generic =
    if not k.generic then []
    else if is_complex k then generic @ generic_complex
    else generic
  in
  List.iter
    (fun (label, f) -> cover (dt ^ label) (Array.exists (f wide) vs))
    (generic @ k.covers)

let law (type d) ?(f32 = true) (module S : Nx.Devices with type d = d) ~f64 k
    c =
  if f32 then covers k false c.f32;
  if f64 then covers k true c.f64;
  let bad =
    (if f32 then run (module S) k false c.f32 else [])
    @ if f64 then run (module S) k true c.f64 else []
  in
  equal (list string) [] bad

let law ?f32 (module S : Nx.Devices) = law ?f32 (module S)

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
  if is_complex k then invalid_arg "Kinds.narrow: a complex kind";
  if arity k > 2 then invalid_arg "Kinds.narrow: a kind of three operands";
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

(* The tuples of constants, one per operand or part: the first's every constant,
   the [j]th's from its [3j]th at a stride of [6j + 1], so that the columns pair
   differently. *)
let tuples wide k =
  let cs =
    List.init (arity k) (fun i ->
        Array.of_list (constants wide (spec_of wide k i)))
  in
  List.init
    (Array.length (List.hd cs))
    (fun i ->
      List.mapi
        (fun j c -> c.(((i * ((6 * j) + 1)) + (3 * j)) mod Array.length c))
        cs)

let mismatch k wide vs ~eager ~compiled =
  Printf.sprintf "%s(%s) folded at %s: eager %s, compiled %s" k.name
    (String.concat ", " (List.map (Printf.sprintf "%h") vs))
    (if wide then "float64" else "float32")
    eager compiled

let folded_at (type s d) (module S : Nx.Devices with type d = d) k
    (dt : (float, s) Nx.dtype) wide =
  let tuples = tuples wide k in
  let at x vs =
    let full v = Nx.full_like x v in
    let y =
      match (k.fn, vs) with
      | One { f1 }, [ a ] -> f1 (full a)
      | Two ({ f2 }, _), [ a; b ] -> f2 (full a) (full b)
      | Three ({ f3 }, _), [ a; b; c ] -> f3 (full a) (full b) (full c)
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
             mismatch k wide vs
               ~eager:(Printf.sprintf "%h" e.(i))
               ~compiled:(Printf.sprintf "%h" c.(i));
           ])
       tuples)

(* A complex kind's constants: its parts' tuples, then every number of Annex G's
   parts, or for two operands each number of the zeros, ones, infinities and a
   NaN beside two others, so that one compile folds 98. The kind reads constants
   alone, which the compiler may fold; its result is then selected by a
   comparison of [x] with itself, which keeps its bits, so that the function
   reads [x]. *)
let cfolded_at (type c d) (module S : Nx.Devices with type d = d) k
    (dt : (Complex.t, c) Nx.dtype) wide =
  let number t = List.map (value wide) (Array.to_list t) in
  let annex =
    if arity k = 2 then List.map number (annex_numbers 2 wide)
    else
      (* each of 49 numbers beside two others, so that a compile folds 98 *)
      let vs = [ 0.; -0.; 1.; -1.; Float.infinity; Float.neg_infinity ] in
      let vs = vs @ [ Float.nan ] in
      let zs =
        Array.of_list
          (List.concat_map (fun re -> List.map (fun im -> [ re; im ]) vs) vs)
      in
      List.concat_map
        (fun s -> List.init 49 (fun i -> zs.(i) @ zs.(((s * i) + 3) mod 49)))
        [ 7; 11 ]
  in
  let tuples = tuples wide k @ annex in
  let at x vs =
    let full re im = Nx.full_like x { Complex.re; im } in
    let y =
      match (k.fn, vs) with
      | Cone ({ c1 }, _), [ a; b ] -> c1 (full a b)
      | Ctwo ({ c2 }, _), [ a; b; c; d ] -> c2 (full a b) (full c d)
      | _ -> invalid_arg "Kinds: operands of another arity"
    in
    Nx.where (Nx.equal x x) y x
  in
  let f x = Nx.concatenate ~axis:0 (List.map (at x) tuples) in
  let x = Nx.full dt [| 1 |] Complex.one in
  let e = Nx.to_array (f x) in
  let before = Rune.evaluations () in
  let c = Nx.to_array (Nx.place Nx.Host.on (Rune.jit' f (Nx.place S.on x))) in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  let code v = code wide v in
  let show (z : Complex.t) = Printf.sprintf "(%h, %h)" z.re z.im in
  List.concat
    (List.mapi
       (fun i vs ->
         let e = e.(i) and c = c.(i) in
         if
           same wide (code e.re) (code c.re)
           && same wide (code e.im) (code c.im)
         then []
         else [ mismatch k wide vs ~eager:(show e) ~compiled:(show c) ])
       tuples)

let folded (type d) (module S : Nx.Devices with type d = d) ~f64 k =
  let bad =
    if is_complex k then
      cfolded_at (module S) k Nx.complex64 false
      @ if f64 then cfolded_at (module S) k Nx.complex128 true else []
    else
      folded_at (module S) k Nx.float32 false
      @ if f64 then folded_at (module S) k Nx.float64 true else []
  in
  equal (list string) [] bad

let folded (module S : Nx.Devices) = folded (module S)
