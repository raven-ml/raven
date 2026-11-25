(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The decimal digits munin writes numbers with. The laws decide whether a
   decimal reads back as a float, and where a float lies against a decimal, with
   [Exact], in integers, so that no C library takes part. *)

open Windtrap
module F = Munin_float_text.Float_text

let digits = pair string int
let text (d, e) = Printf.sprintf "%se%d" d e
let single x = Int32.float_of_bits (Int32.bits_of_float x)

(* The float32s next to [x] above and below, for a float32 [x >= 0]. *)
let next32 x = Int32.float_of_bits (Int32.succ (Int32.bits_of_float x))
let prev32 x = Int32.float_of_bits (Int32.pred (Int32.bits_of_float x))

(* The value of [f] that [de] reads as, from [guess], the C library's reading,
   which is at most a unit in the last place off. *)
let nearest f ~up ~down de guess =
  let near = [ guess; up guess; down guess ] in
  match List.find_opt (fun x -> Exact.reads f x de) near with
  | Some x -> x
  | None -> failf "%s reads as no value next to %h" (text de) guess

let nearest64 de =
  nearest F.binary64 ~up:Float.succ ~down:Float.pred de
    (float_of_string (text de))

let nearest32 de =
  nearest F.binary32 ~up:next32 ~down:prev32 de
    (single (float_of_string (text de)))

(* [leading x] is the exponent of the leading digit of the nonzero [|x|]: 10^k
   <= |x| < 10^(k+1). *)
let leading x =
  let c, q, _ = Exact.decode F.binary64 x in
  let k = ref (Float.to_int (Float.floor (Float.log10 (Float.abs x)))) in
  while Exact.compare_dec ("1", !k + 1) c q <= 0 do
    incr k
  done;
  while Exact.compare_dec ("1", !k) c q > 0 do
    decr k
  done;
  !k

(* [rounds x ~unit (d, e)] checks that [d 10^e] is [|x|] rounded to a multiple
   of [10^unit], to nearest, ties to even. With [d 10^e = m 10^unit], [|x|] lies
   between the midpoints [(10m - 5) 10^(unit-1)] and [(10m + 5) 10^(unit-1)], on
   one only when [m] is even. Callers keep [m] below 2^58. *)
let rounds x ~unit (d, e) =
  let c, q, _ = Exact.decode F.binary64 x in
  satisfies ~claim:"is a multiple of the unit" digits
    (fun (d, e) -> d = "0" || e >= unit)
    (d, e);
  let m = if d = "0" then 0 else int_of_string d * Exact.pow10 (e - unit) in
  let even = m land 1 = 0 in
  let mid k = (string_of_int ((10 * m) + k), unit - 1) in
  (* The signs of each midpoint less [|x|]. *)
  let lo = if m = 0 then -1 else Exact.compare_dec (mid (-5)) c q in
  let hi = Exact.compare_dec (mid 5) c q in
  cover "a tie" (lo = 0 || hi = 0);
  satisfies ~claim:"lies within half a unit of x, a tie to the even one" digits
    (fun _ -> (lo < 0 || (lo = 0 && even)) && (hi > 0 || (hi = 0 && even)))
    (d, e)

(* [is_exact x] checks that a decimal is [|x|]'s exact value. *)
let is_exact x de =
  let c, q, _ = Exact.decode F.binary64 x in
  satisfies ~claim:"is x's exact value" digits
    (fun de -> if x = 0. then de = ("0", 0) else Exact.compare_dec de c q = 0)
    de

(* Inputs *)

let bits lo hi = Gen.map Int64.float_of_bits (Gen.int64_range lo hi)
let signed gen = Gen.(map (fun (x, s) -> if s then -.x else x) (pair gen bool))

(* [k] units in the last place from [x], positive and finite. *)
let ulps x k = Int64.float_of_bits (Int64.add (Int64.bits_of_float x) k)

(* [n 2^-r] with [n] of [p] bits and [r] from 1 to 3: its interval is 2^-r wide
   and its value ends in [5] at the digit after the grid of its shortest
   candidates, so two of them are often equally near. *)
let ties p =
  let open Gen in
  let+ n = int_range (1 lsl (p - 1)) ((1 lsl p) - 1) and+ r = int_range 1 3 in
  Float.ldexp (Float.of_int n) (-r)

let double =
  let open Gen in
  signed
    (frequency
       [
         (4, map Float.abs float);
         (1, bits 1L 0xF_FFFF_FFFF_FFFFL);
         ( 2,
           let+ e = int_range (-1022) 1023 and+ k = int_range (-2) 2 in
           ulps (Float.ldexp 1. e) (Int64.of_int k) );
         ( 2,
           let+ d = int_range 1 999 and+ j = int_range (-323) 305 in
           float_of_string (Printf.sprintf "%de%d" d j) );
         (1, map (fun k -> ulps max_float (Int64.of_int (-k))) (int_range 0 3));
         (2, ties 53);
       ])

let float32 =
  let open Gen in
  signed
    (frequency
       [
         (4, bits 0x3800_0000_0000_0000L 0x47EF_FFFF_E000_0000L);
         ( 1,
           map
             (fun k -> Float.ldexp (Float.of_int k) (-149))
             (int_range 1 0x7F_FFFF) );
         ( 2,
           let+ e = int_range (-126) 127 and+ k = int_range 0 2 in
           let x = Float.ldexp 1. e in
           if k = 0 then x else if k = 1 then next32 x else prev32 x );
         ( 2,
           let+ d = int_range 1 999 and+ j = int_range (-44) 35 in
           single (float_of_string (Printf.sprintf "%de%d" d j)) );
         (2, ties 24);
       ])
  |> Gen.map single

let classes32 x =
  let x = Float.abs x in
  cover "subnormal" (x > 0. && x < 0x1p-126);
  cover "binade bottom" (x >= 0x1p-126 && fst (Float.frexp x) = 0.5)

let classes64 x =
  let x = Float.abs x in
  cover "subnormal" (x > 0. && x < 0x1p-1022);
  cover "binade bottom" (x >= 0x1p-1022 && fst (Float.frexp x) = 0.5);
  cover "largest" (x > 0x1p1023)

(* A format of [p] bits, one bit in a quarter of the draws, and a value [c 2^q]
   of it, at the format's least exponent in a third of them. Every [c] below
   [2^p] with [q] at least that exponent is a value, and [emin] keeps every
   value a double. *)
let format_value =
  let open Gen in
  let* p =
    frequency [ (1, constant 1); (1, int_range 2 8); (2, int_range 9 53) ]
  in
  let* emin = int_range (p - 1075) 200 in
  let qmin = emin - p + 1 and bottom = 1 lsl (p - 1) in
  let+ c =
    frequency
      [
        (2, int_range 0 ((1 lsl p) - 1));
        (1, constant bottom);
        (1, int_range 1 (max 1 (bottom - 1)));
      ]
  and+ q = frequency [ (1, constant qmin); (2, int_range qmin (qmin + 200)) ] in
  ({ F.precision = p; emin }, Float.ldexp (Float.of_int c) q)

let classes_format ((f : F.format), x) =
  let c, q, qmin = Exact.decode f x in
  cover "one bit" (f.precision = 1);
  cover "subnormal" (c > 0 && c < 1 lsl (f.precision - 1));
  cover "binade bottom" (c = 1 lsl (f.precision - 1) && q > qmin)

(* [shorter (d, e)] are the decimals of one digit fewer next to [d 10^e] on
   either side: [m 10^(e+1)] and [(m+1) 10^(e+1)], with [m] the digits of [d]
   but the last. Since [d] ends in a nonzero digit, they lie on either side of
   it, and every decimal of fewer digits than [d] is at or beyond one of them.
   The reals that read as a float form an interval, so if neither reads as [x],
   no decimal shorter than [d] does. *)
let shorter (d, e) =
  let n = String.length d in
  if n < 2 then []
  else
    let m = int_of_string (String.sub d 0 (n - 1)) in
    [ (string_of_int m, e + 1); (string_of_int (m + 1), e + 1) ]

let reads_back f x =
  satisfies ~claim:"reads back as x" digits (Exact.reads f x) (F.shortest f x)

let nothing_shorter f x =
  List.iter
    (satisfies ~claim:"does not read back as x" digits (fun de ->
         not (Exact.reads f x de)))
    (shorter (F.shortest f x))

(* Of the shortest decimals, the nearest. [d 10^e] has [n] digits; the only
   others of [n] digits that can be nearer to [x] are its neighbours on the far
   side of [x]: [(d + 1) 10^e] above, and below [(d - 1) 10^e], or [9 10^(e-1)]
   when [d] is 1. One is nearer when [x] is past the midpoint toward it, [(10d +
   5) 10^(e-1)] above and [(10d - 5) 10^(e-1)] or [95 10^(e-2)] below. At the
   midpoint the two tie, and the even last digit wins. *)
let nearest_law f x =
  let d, e = F.shortest f x in
  let c, q, _ = Exact.decode f x in
  let m = int_of_string d in
  let even = Char.code d.[String.length d - 1] land 1 = 0 in
  let up = ((string_of_int (m + 1), e), (string_of_int ((10 * m) + 5), e - 1))
  and down =
    if m = 1 then (("9", e - 1), ("95", e - 2))
    else ((string_of_int (m - 1), e), (string_of_int ((10 * m) - 5), e - 1))
  in
  (* [o n] compares the midpoint toward the neighbour [n] with [x]. *)
  let o (_, mid) = Exact.compare_dec mid c q in
  let reads_n (n, _) = x <> 0. && Exact.reads f x n in
  cover "a neighbour reads back" (reads_n up || reads_n down);
  cover "tie" (x <> 0. && (o up = 0 || o down = 0));
  let nearer side o =
    satisfies
      ~claim:("is nearer x than the neighbour " ^ side ^ ", or even at a tie")
      digits
      (fun _ -> o > 0 || (o = 0 && even))
      (d, e)
  in
  if reads_n up then nearer "above" (o up);
  if reads_n down then nearer "below" (-o down)

(* A decimal of at most [p] digits, a power of ten from [low] to [high]. *)
let short_decimal ~digits:p ~low ~high =
  let open Gen in
  let+ m = int_range 1 (Exact.pow10 p - 1) and+ j = int_range low high in
  let s = string_of_int m in
  let k = ref (String.length s) in
  while s.[!k - 1] = '0' do
    decr k
  done;
  (String.sub s 0 !k, j + String.length s - !k)

let binary16 = { F.precision = 11; emin = -14 }

(* shortest *)

let test_reads_back64 =
  prop "reads back as x in binary64" ~count:2000 double (fun x ->
      classes64 x;
      reads_back F.binary64 x)

let test_minimal64 =
  prop "nothing shorter reads back in binary64" ~count:2000 double (fun x ->
      classes64 x;
      nothing_shorter F.binary64 x)

let test_reads_back32 =
  prop "reads back as x in binary32" ~count:2000 float32 (fun x ->
      classes32 x;
      reads_back F.binary32 x)

let test_minimal32 =
  prop "nothing shorter reads back in binary32" ~count:2000 float32 (fun x ->
      classes32 x;
      nothing_shorter F.binary32 x)

let test_reads_back_any =
  prop "reads back as x in any format" ~count:300 format_value (fun (f, x) ->
      classes_format (f, x);
      reads_back f x)

let test_minimal_any =
  prop "nothing shorter reads back in any format" ~count:300 format_value
    (fun (f, x) ->
      classes_format (f, x);
      nothing_shorter f x)

let test_nearest64 =
  prop "is the nearest of the shortest in binary64" ~count:2000 double
    (nearest_law F.binary64)

let test_nearest32 =
  prop "is the nearest of the shortest in binary32" ~count:2000 float32
    (nearest_law F.binary32)

(* A decimal of at most 15 digits in the normal range is the shortest of the
   double nearest it; of at most 6 digits, of the float32 nearest it. *)
let test_round_trip64 =
  prop "gives back a decimal of 15 digits in binary64" ~count:1000
    (short_decimal ~digits:15 ~low:(-300) ~high:290)
    (Law.round_trip digits float_exact nearest64 (F.shortest F.binary64))

let test_round_trip32 =
  prop "gives back a decimal of 6 digits in binary32" ~count:1000
    (short_decimal ~digits:6 ~low:(-36) ~high:30)
    (Law.round_trip digits float_exact nearest32 (F.shortest F.binary32))

(* C's DBL_MIN and DBL_MAX and their neighbours; [2^50 + 0.25], whose 17-digit
   neighbours .2 and .3 are equally near, the even one taken; the float32
   nearest 0.1, the .mli's example; and the largest binary16, e5m2 and e4m3fn
   values, which read from the midpoints to the next value their formats would
   have: 65504 from 65488 to 65520, 57344 from 53248 to 61440, 448 from 432 to
   464 with both ends, as its significand is even. *)
let shortest_rows =
  [
    ("zero", F.binary64, 0., ("0", 0));
    ("-0", F.binary64, -0., ("0", 0));
    ("0.1", F.binary64, 0.1, ("1", -1));
    ("5e-324", F.binary64, 5e-324, ("5", -324));
    ("1e23", F.binary64, 1e23, ("1", 23));
    ("2^-1022", F.binary64, 0x1p-1022, ("22250738585072014", -324));
    ( "above 2^-1022",
      F.binary64,
      Float.succ 0x1p-1022,
      ("2225073858507202", -323) );
    ( "below 2^-1022",
      F.binary64,
      Float.pred 0x1p-1022,
      ("2225073858507201", -323) );
    ("max_float", F.binary64, max_float, ("17976931348623157", 292));
    ("2^50 + 0.25", F.binary64, 0x1p50 +. 0.25, ("11258999068426242", -1));
    ("binary32 0.1", F.binary32, single 0.1, ("1", -1));
    ("binary16 65504", binary16, 65504., ("655", 2));
    ("e5m2 57344", { F.precision = 3; emin = -14 }, 57344., ("6", 4));
    ("e4m3fn 448", { F.precision = 4; emin = -6 }, 448., ("45", 1));
  ]

let test_shortest_rows =
  cases "is the stated digits"
    ~id:(fun (id, _, _, _) -> id)
    shortest_rows
    (fun (_, f, x, want) -> equal digits want (F.shortest f x))

let test_precision =
  cases "raises on a precision outside 1 to 53" ~id:string_of_int [ 0; 54; 64 ]
    (fun p ->
      raises_match Exn.invalid_arg (fun () ->
          F.shortest { F.precision = p; emin = -14 } 1.))

let test_not_a_value =
  cases "raises on a value not of the format" ~id:fst
    [
      ("binary32 0.1", (F.binary32, 0.1));
      ("binary16 2^-25", (binary16, 0x1p-25));
    ]
    (fun (_, (f, x)) -> raises_match Exn.invalid_arg (fun () -> F.shortest f x))

(* significant *)

(* [n] up to 17 keeps the rounded digits below 2^58 in [rounds]. Ties come from
   [k 2^-r], [k] odd, whose decimal [k 5^r 10^-r] ends in 5, rounded to one
   digit fewer. *)
let significant_input =
  let open Gen in
  frequency
    [
      (3, pair (int_range 1 17) double);
      ( 1,
        let+ k = int_range 0 (1 lsl 20) and+ r = int_range 2 8 in
        let k = (2 * k) + 1 in
        let scaled = k * Float.to_int (5. ** Float.of_int r) in
        ( String.length (string_of_int scaled) - 1,
          Float.ldexp (Float.of_int k) (-r) ) );
    ]

let test_significant =
  prop "rounds |x| to n significant digits, ties to even" ~count:500
    significant_input (fun (n, x) ->
      assume (x <> 0.);
      rounds x ~unit:(leading x - n + 1) (F.significant n x))

let test_significant_exact =
  prop "is x exactly at x's digit count or more"
    ~examples:[ (max_int, 5e-324); (max_int, 0.005); (767, max_float) ]
    Gen.(pair (of_list [ 767; max_int ]) double)
    (fun (n, x) -> is_exact x (F.significant n x))

(* The .mli's examples; the exact decimal of the double nearest 0.005. *)
let significant_rows =
  [
    (3, 135.5625, ("136", 0));
    (2, 0.125, ("12", -2));
    ( max_int,
      0.005,
      ("5000000000000000104083408558608425664715468883514404296875", -60) );
    (1, 0., ("0", 0));
  ]

let test_significant_rows =
  cases "is the stated digits"
    ~id:(fun (n, x, _) -> Printf.sprintf "%d %h" n x)
    significant_rows
    (fun (n, x, want) -> equal digits want (F.significant n x))

let test_significant_refused =
  test "raises below one digit" (fun () ->
      raises_match Exn.invalid_arg (fun () -> F.significant 0 1.))

(* decimals *)

(* Values from 10^-8 to 10^16 units, so that some round to zero; ties from [k
   2^-r], [k] odd, at [r - 1] decimals, whose scaled value [k 5^(r-1) / 2] ends
   in a half, and from [(2m + 1) 5 10^(-n-1)] at a negative [n]. *)
let decimals_input =
  let open Gen in
  frequency
    [
      ( 3,
        let* n = int_range (-10) 25 in
        let+ d = int_range 1 999_999 and+ j = int_range (-n - 8) (10 - n) in
        (n, float_of_string (Printf.sprintf "%de%d" d j)) );
      ( 1,
        let+ k = int_range 0 (1 lsl 20) and+ r = int_range 1 8 in
        (r - 1, Float.ldexp (Float.of_int ((2 * k) + 1)) (-r)) );
      ( 1,
        let+ m = int_range 0 1_000_000 and+ n = int_range (-6) (-1) in
        let tie = Printf.sprintf "%de%d" (((2 * m) + 1) * 5) (-n - 1) in
        (n, float_of_string tie) );
    ]

let test_decimals =
  prop "rounds |x| to a multiple of 10^-n, ties to even" ~count:500
    Gen.(pair decimals_input bool)
    (fun ((n, x), neg) ->
      let x = if neg then -.x else x in
      let r = F.decimals n x in
      cover "rounds to zero" (r = ("0", 0));
      rounds x ~unit:(-n) r)

let test_decimals_exact =
  prop "is x exactly at 1074 decimals or more"
    ~examples:[ (max_int, 5e-324); (1074, max_float) ]
    Gen.(pair (of_list [ 1074; max_int ]) double)
    (fun (n, x) -> is_exact x (F.decimals n x))

(* The .mli's examples, a negative number that rounds to zero, and the ends of
   [n]. *)
let decimals_rows =
  [
    (2, 0.125, ("12", -2));
    (0, 0.5, ("0", 0));
    (0, 1.5, ("2", 0));
    (-2, 1250., ("12", 2));
    (2, -0.001, ("0", 0));
    (max_int, 1.5, ("15", -1));
    (min_int, 1.5, ("0", 0));
  ]

let test_decimals_rows =
  cases "is the stated digits"
    ~id:(fun (n, x, _) -> Printf.sprintf "%d %h" n x)
    decimals_rows
    (fun (n, x, want) -> equal digits want (F.decimals n x))

(* Every function *)

let test_not_finite =
  cases "raises on NaN and the infinities"
    ~id:(fun (name, x) -> Printf.sprintf "%s %h" name x)
    (List.concat_map
       (fun x -> [ ("shortest", x); ("significant", x); ("decimals", x) ])
       [ Float.nan; Float.infinity; Float.neg_infinity ])
    (fun (name, x) ->
      raises_match Exn.invalid_arg (fun () ->
          match name with
          | "shortest" -> F.shortest F.binary64 x
          | "significant" -> F.significant 3 x
          | _ -> F.decimals 3 x))

let () =
  exit
    (run "munin float text"
       [
         group "every function" [ test_not_finite ];
         group "shortest"
           [
             test_reads_back64;
             test_minimal64;
             test_reads_back32;
             test_minimal32;
             test_reads_back_any;
             test_minimal_any;
             test_nearest64;
             test_nearest32;
             test_round_trip64;
             test_round_trip32;
             test_shortest_rows;
             test_precision;
             test_not_a_value;
           ];
         group "significant"
           [
             test_significant;
             test_significant_exact;
             test_significant_rows;
             test_significant_refused;
           ];
         group "decimals"
           [ test_decimals; test_decimals_exact; test_decimals_rows ];
       ])
