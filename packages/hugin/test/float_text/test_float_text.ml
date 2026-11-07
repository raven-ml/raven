(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The decimal digits hugin writes numbers with. The laws decide whether a
   decimal reads back as a float with [reads], an exact reader of their own.
   [float_of_string] is the C library's [strtod], which Windows' C runtime does
   not promise to round correctly, and the laws must hold on every system. *)

open Windtrap
module F = Hugin_float_text.Float_text

let digits = pair string int
let text (d, e) = Printf.sprintf "%se%d" d e
let single x = Int32.float_of_bits (Int32.bits_of_float x)

(* The float32s next to [x] above and below, for a float32 [x >= 0]. *)
let next32 x = Int32.float_of_bits (Int32.succ (Int32.bits_of_float x))
let prev32 x = Int32.float_of_bits (Int32.pred (Int32.bits_of_float x))

(* Exact reading *)

(* Naturals in limbs of 24 bits, least significant first. [mul a m c] is
   [a m + c] for [m] and [c] below 2^24, so every product stays below 2^48. *)
let mul a m c =
  let carry = ref c in
  let r =
    Array.map
      (fun l ->
        let t = (l * m) + !carry in
        carry := t lsr 24;
        t land 0xFF_FFFF)
      a
  in
  if !carry = 0 then r else Array.append r [| !carry |]

let rec times a ~twos ~tens =
  if twos > 0 then times (mul a (1 lsl min twos 23) 0) ~twos:(twos - 23) ~tens
  else if tens > 0 then times (mul a 10 0) ~twos ~tens:(tens - 1)
  else a

let compare_nat a b =
  let top a =
    let n = ref (Array.length a) in
    while !n > 0 && a.(!n - 1) = 0 do
      decr n
    done;
    !n
  in
  let n = top a and m = top b in
  if n <> m then Int.compare n m
  else
    let i = ref (n - 1) in
    while !i >= 0 && a.(!i) = b.(!i) do
      decr i
    done;
    if !i < 0 then 0 else Int.compare a.(!i) b.(!i)

(* [compare_dec (d, e) n q] compares [d 10^e] with [n 2^q], for [n] below
   2^62, by multiplying both by [10^-e] and [2^-q] where those are whole. *)
let compare_dec (d, e) n q =
  let digit a c = mul a 10 (Char.code c - Char.code '0') in
  let dec = String.fold_left digit [| 0 |] d in
  let bin = [| n land 0xFF_FFFF; (n lsr 24) land 0xFF_FFFF; n lsr 48 |] in
  compare_nat
    (times dec ~twos:(max 0 (-q)) ~tens:(max 0 e))
    (times bin ~twos:(max 0 q) ~tens:(max 0 (-e)))

(* [reads f x de] is whether the decimal [de] reads as [|x|] in [f], to
   nearest, ties to even. [|x| = c 2^q] with [q] at least the subnormals'
   exponent; the reals that read as it lie between the midpoints
   [(2c - 1) 2^(q-1)] and [(2c + 1) 2^(q-1)], or from [(4c - 1) 2^(q-2)] at the
   bottom of a binade above the subnormals, the ends included when [c] is
   even. *)
let decode (f : F.format) x =
  let x = Float.abs x in
  let qmin = f.emin - f.precision + 1 in
  let q =
    if x = 0. then qmin else max (snd (Float.frexp x) - f.precision) qmin
  in
  (Float.to_int (Float.ldexp x (-q)), q, qmin)

let reads (f : F.format) x de =
  let c, q, qmin = decode f x in
  let inside o = o < 0 || (o = 0 && c land 1 = 0) in
  let above =
    if c = 0 then true
    else if c = 1 lsl (f.precision - 1) && q > qmin then
      inside (-compare_dec de ((4 * c) - 1) (q - 2))
    else inside (-compare_dec de ((2 * c) - 1) (q - 1))
  in
  above && inside (compare_dec de ((2 * c) + 1) (q - 1))

(* The value of [f] that [de] reads as, from [guess], the C library's reading,
   which is at most a unit in the last place off. *)
let nearest f ~up ~down de guess =
  let near = [ guess; up guess; down guess ] in
  match List.find_opt (fun x -> reads f x de) near with
  | Some x -> x
  | None -> failf "%s reads as no value next to %h" (text de) guess

let nearest64 de =
  nearest F.binary64 ~up:Float.succ ~down:Float.pred de
    (float_of_string (text de))

let nearest32 de =
  nearest F.binary32 ~up:next32 ~down:prev32 de
    (single (float_of_string (text de)))

(* Inputs *)

let bits lo hi = Gen.map Int64.float_of_bits (Gen.int64_range lo hi)
let signed gen = Gen.(map (fun (x, s) -> if s then -.x else x) (pair gen bool))

(* [k] units in the last place from [x], positive and finite. *)
let ulps x k = Int64.float_of_bits (Int64.add (Int64.bits_of_float x) k)

(* [n 2^-r] with [n] of [p] bits and [r] from 1 to 3: its interval is 2^-r
   wide and its value ends in [5] at the digit after the grid of its shortest
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
           if k = 0 then x else if k = 1 then next32 x else prev32 x
         );
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

(* [shorter (d, e)] are the decimals of one digit fewer next to [d 10^e] on
   either side: [m 10^(e+1)] and [(m+1) 10^(e+1)], with [m] the digits of [d]
   but the last. Since [d] ends in a nonzero digit, they lie on either side of
   it, and every decimal of fewer digits than [d] is at or beyond one of them.
   The reals that read as a float form an interval, so if neither reads as
   [x], no decimal shorter than [d] does. *)
let shorter (d, e) =
  let n = String.length d in
  if n < 2 then []
  else
    let m = int_of_string (String.sub d 0 (n - 1)) in
    [ (string_of_int m, e + 1); (string_of_int (m + 1), e + 1) ]

(* Laws *)

let test_reads_back64 =
  prop "binary64 shortest reads back" ~count:2000 double (fun x ->
      classes64 x;
      let de = F.shortest F.binary64 x in
      equal ~msg:(text de) bool true (reads F.binary64 x de))

let test_minimal64 =
  prop "binary64 nothing shorter reads back" ~count:2000 double (fun x ->
      classes64 x;
      List.iter
        (fun de -> equal ~msg:(text de) bool false (reads F.binary64 x de))
        (shorter (F.shortest F.binary64 x)))

let test_reads_back32 =
  prop "binary32 shortest reads back" ~count:2000 float32 (fun x ->
      classes32 x;
      let de = F.shortest F.binary32 x in
      equal ~msg:(text de) bool true (reads F.binary32 x de))

let test_minimal32 =
  prop "binary32 nothing shorter reads back" ~count:2000 float32 (fun x ->
      classes32 x;
      List.iter
        (fun de -> equal ~msg:(text de) bool false (reads F.binary32 x de))
        (shorter (F.shortest F.binary32 x)))

(* Of the shortest decimals, the nearest. [d 10^e] has [n] digits; the only
   others of [n] digits that can be nearer to [x] are its neighbours on the far
   side of [x]: [(d + 1) 10^e] above, and below [(d - 1) 10^e], or [9 10^(e-1)]
   when [d] is 1. One is nearer when [x] is past the midpoint toward it,
   [(10d + 5) 10^(e-1)] above and [(10d - 5) 10^(e-1)] or [95 10^(e-2)] below.
   At the midpoint the two tie, and the even last digit wins. *)
let nearest_law f x =
  let d, e = F.shortest f x in
  let c, q, _ = decode f x in
  let m = int_of_string d in
  let even = Char.code d.[String.length d - 1] land 1 = 0 in
  let up = ((string_of_int (m + 1), e), (string_of_int ((10 * m) + 5), e - 1))
  and down =
    if m = 1 then (("9", e - 1), ("95", e - 2))
    else ((string_of_int (m - 1), e), (string_of_int ((10 * m) - 5), e - 1))
  in
  (* [o n] compares the midpoint toward the neighbour [n] with [x]. *)
  let o (_, mid) = compare_dec mid c q in
  let reads_n (n, _) = x <> 0. && reads f x n in
  cover "a neighbour reads back" (reads_n up || reads_n down);
  cover "tie" (x <> 0. && (o up = 0 || o down = 0));
  if reads_n up then
    equal ~msg:(text (d, e)) bool true (o up > 0 || (o up = 0 && even));
  if reads_n down then
    equal ~msg:(text (d, e)) bool true (o down < 0 || (o down = 0 && even))

let test_nearest64 =
  prop "binary64 nearest of the shortest" ~count:2000 double
    (nearest_law F.binary64)

let test_nearest32 =
  prop "binary32 nearest of the shortest" ~count:2000 float32
    (nearest_law F.binary32)

(* A decimal of at most 15 digits in the normal range is the shortest of the
   double nearest it; of at most 6 digits, of the float32 nearest it. *)
let short_decimal ~digits:p ~low ~high =
  let open Gen in
  let rec pow10 k = if k = 0 then 1 else 10 * pow10 (k - 1) in
  let+ m = int_range 1 (pow10 p - 1)
  and+ j = int_range low high in
  let s = string_of_int m in
  let k = ref (String.length s) in
  while s.[!k - 1] = '0' do
    decr k
  done;
  (String.sub s 0 !k, j + String.length s - !k)

let test_round_trip64 =
  prop "binary64 short decimal round trip" ~count:1000
    (short_decimal ~digits:15 ~low:(-300) ~high:290)
    (fun de -> equal digits de (F.shortest F.binary64 (nearest64 de)))

let test_round_trip32 =
  prop "binary32 short decimal round trip" ~count:1000
    (short_decimal ~digits:6 ~low:(-36) ~high:30)
    (fun de -> equal digits de (F.shortest F.binary32 (nearest32 de)))

(* Stated cases *)

(* 5^1074, the digits of 2^-1074 = 5^1074 10^-1074, by schoolbook
   multiplication in base 10. *)
let pow5_1074 =
  let a = Array.make 800 0 in
  a.(0) <- 1;
  for _ = 1 to 1074 do
    let carry = ref 0 in
    for i = 0 to Array.length a - 1 do
      let t = (a.(i) * 5) + !carry in
      a.(i) <- t mod 10;
      carry := t / 10
    done
  done;
  let top = ref (Array.length a - 1) in
  while a.(!top) = 0 do
    decr top
  done;
  String.init (!top + 1) (fun i -> Char.chr (48 + a.(!top - i)))

let binary16 = { F.precision = 11; emin = -14 }
let point_one32 = single 0.1

let stated =
  [
    ("shortest zero", (fun () -> F.shortest F.binary64 0.), ("0", 0));
    ("shortest -0", (fun () -> F.shortest F.binary64 (-0.)), ("0", 0));
    ("shortest 0.1", (fun () -> F.shortest F.binary64 0.1), ("1", -1));
    ("shortest 5e-324", (fun () -> F.shortest F.binary64 5e-324), ("5", -324));
    ("shortest 1e23", (fun () -> F.shortest F.binary64 1e23), ("1", 23));
    ( "shortest 2^-1022",
      (fun () -> F.shortest F.binary64 0x1p-1022),
      ("22250738585072014", -324) );
    ( "shortest above 2^-1022",
      (fun () -> F.shortest F.binary64 (Float.succ 0x1p-1022)),
      ("2225073858507202", -323) );
    ( "shortest below 2^-1022",
      (fun () -> F.shortest F.binary64 (Float.pred 0x1p-1022)),
      ("2225073858507201", -323) );
    ( "shortest max_float",
      (fun () -> F.shortest F.binary64 max_float),
      ("17976931348623157", 292) );
    ( "shortest 2^50 + 0.25",
      (fun () -> F.shortest F.binary64 (0x1p50 +. 0.25)),
      ("11258999068426242", -1) );
    ( "shortest binary32 0.1",
      (fun () -> F.shortest F.binary32 point_one32),
      ("1", -1) );
    ( "shortest binary16 65504",
      (fun () -> F.shortest binary16 65504.),
      ("655", 2) );
    ( "significant 3 135.5625",
      (fun () -> F.significant 3 135.5625),
      ("136", 0) );
    ("significant 2 0.125", (fun () -> F.significant 2 0.125), ("12", -2));
    ( "significant max_int 0.005",
      (fun () -> F.significant max_int 0.005),
      ("5000000000000000104083408558608425664715468883514404296875", -60) );
    ( "significant max_int 5e-324",
      (fun () -> F.significant max_int 5e-324),
      (pow5_1074, -1074) );
    ("decimals 2 0.125", (fun () -> F.decimals 2 0.125), ("12", -2));
    ("decimals 0 0.5", (fun () -> F.decimals 0 0.5), ("0", 0));
    ("decimals 0 1.5", (fun () -> F.decimals 0 1.5), ("2", 0));
    ("decimals -2 1250", (fun () -> F.decimals (-2) 1250.), ("12", 2));
    ("decimals 2 -0.001", (fun () -> F.decimals 2 (-0.001)), ("0", 0));
    ("decimals max_int 1.5", (fun () -> F.decimals max_int 1.5), ("15", -1));
    ("decimals min_int 1.5", (fun () -> F.decimals min_int 1.5), ("0", 0));
  ]

let test_stated =
  cases "stated values" ~id:(fun (name, _, _) -> name) stated
    (fun (_, f, want) -> equal digits want (f ()))

(* The exact reader at ties to even and at the overflow threshold,
   1.797693134862315807...e308. *)
let reader_rows =
  [
    ("0.1", 0.1, ("1", -1), true);
    ("2^53 + 1 to 2^53", 0x1p53, ("9007199254740993", 0), true);
    ("2^53 + 1 to 2^53 + 2", 0x1p53 +. 2., ("9007199254740993", 0), false);
    ("2^53 + 3 to 2^53 + 4", 0x1p53 +. 4., ("9007199254740995", 0), true);
    ("2^53 + 3 to 2^53 + 2", 0x1p53 +. 2., ("9007199254740995", 0), false);
    ("below overflow", max_float, ("17976931348623158", 292), true);
    ("past overflow", max_float, ("17976931348623159", 292), false);
  ]

let test_reader =
  cases "exact reader" ~id:(fun (name, _, _, _) -> name) reader_rows
    (fun (_, x, de, want) -> equal bool want (reads F.binary64 x de))

let refused =
  [
    ( "precision 64",
      fun () -> F.shortest { F.precision = 64; emin = -16382 } 0.1 );
    ("precision 0", fun () -> F.shortest { F.precision = 0; emin = 0 } 1.);
    ("binary32 0.1", fun () -> F.shortest F.binary32 0.1);
    ("nan", fun () -> F.shortest F.binary64 Float.nan);
    ("infinity", fun () -> F.significant 3 Float.infinity);
    ("significant 0", fun () -> F.significant 0 1.);
  ]

let test_refused =
  cases "refused" ~id:fst refused (fun (_, f) -> raises_match Exn.invalid_arg f)

let () =
  exit
    (run "hugin float text"
       [
         group "laws"
           [
             test_reads_back64;
             test_minimal64;
             test_reads_back32;
             test_minimal32;
             test_nearest64;
             test_nearest32;
             test_round_trip64;
             test_round_trip32;
           ];
         group "cases" [ test_stated; test_refused; test_reader ];
       ])
