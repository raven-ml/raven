(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The decimal digits brot writes numbers with. The laws decide whether a
   decimal reads back as a float with [reads], an exact reader of their own.
   [float_of_string] is the C library's [strtod], which Windows' C runtime does
   not promise to round correctly, and the laws must hold on every system. *)

open Windtrap
module F = Brot.Float_text

let digits = pair string int
let text (d, e) = Printf.sprintf "%se%d" d e
let single x = Int32.float_of_bits (Int32.bits_of_float x)

(* The float32s next to [x] above and below, for a float32 [x >= 0]. *)
let next32 x = Int32.float_of_bits (Int32.succ (Int32.bits_of_float x))
let prev32 x = Int32.float_of_bits (Int32.pred (Int32.bits_of_float x))

(* Exact reading *)

(* Naturals in limbs of 24 bits, least significant first. [mul a m c] is [a m +
   c] for [m] and [c] below 2^24, so every product stays below 2^48. *)
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

(* [compare_dec (d, e) n q] compares [d 10^e] with [n 2^q], for [n] below 2^62,
   by multiplying both by [10^-e] and [2^-q] where those are whole. *)
let compare_dec (d, e) n q =
  let digit a c = mul a 10 (Char.code c - Char.code '0') in
  let dec = String.fold_left digit [| 0 |] d in
  let bin = [| n land 0xFF_FFFF; (n lsr 24) land 0xFF_FFFF; n lsr 48 |] in
  compare_nat
    (times dec ~twos:(max 0 (-q)) ~tens:(max 0 e))
    (times bin ~twos:(max 0 q) ~tens:(max 0 (-e)))

(* [reads f x de] is whether the decimal [de] reads as [|x|] in [f], to nearest,
   ties to even. [|x| = c 2^q] with [q] at least the subnormals' exponent; the
   reals that read as it lie between the midpoints [(2c - 1) 2^(q-1)] and [(2c +
   1) 2^(q-1)], or from [(4c - 1) 2^(q-2)] at the bottom of a binade above the
   subnormals, the ends included when [c] is even. *)
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

(* Laws *)

let reads_back f x =
  satisfies
    ~msg:(Printf.sprintf "the digits of %h" x)
    ~claim:"a decimal that reads as x" digits (reads f x) (F.shortest f x)

let minimal f x =
  List.iter
    (satisfies ~msg:(Printf.sprintf "shorter than the digits of %h" x)
       ~claim:"a decimal that does not read as x" digits (fun de ->
         not (reads f x de)))
    (shorter (F.shortest f x))

(* Of the shortest decimals, the nearest. [d 10^e] has [n] digits; the only
   others of [n] digits that can be nearer to [x] are its neighbours on the far
   side of [x]: [(d + 1) 10^e] above, and below [(d - 1) 10^e], or [9 10^(e-1)]
   when [d] is 1. One is nearer when [x] is past the midpoint toward it, [(10d +
   5) 10^(e-1)] above and [(10d - 5) 10^(e-1)] or [95 10^(e-2)] below. At the
   midpoint the two tie, and the even last digit wins. *)
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
  let msg = Printf.sprintf "the digits of %h" x in
  if reads_n up then
    satisfies ~msg ~claim:"no farther from x than the decimal above" digits
      (fun _ -> o up > 0 || (o up = 0 && even))
      (d, e);
  if reads_n down then
    satisfies ~msg ~claim:"no farther from x than the decimal below" digits
      (fun _ -> o down < 0 || (o down = 0 && even))
      (d, e)

(* A decimal of at most [p] digits in the normal range. *)
let short_decimal ~digits:p ~low ~high =
  let open Gen in
  let rec pow10 k = if k = 0 then 1 else 10 * pow10 (k - 1) in
  let+ m = int_range 1 (pow10 p - 1) and+ j = int_range low high in
  let s = string_of_int m in
  let k = ref (String.length s) in
  while s.[!k - 1] = '0' do
    decr k
  done;
  (String.sub s 0 !k, j + String.length s - !k)

let reads_back64 =
  prop "binary64 digits read back as the double" ~count:2000 double (fun x ->
      classes64 x;
      reads_back F.binary64 x)

let minimal64 =
  prop "no decimal of fewer digits reads back as the double" ~count:2000 double
    (fun x ->
      classes64 x;
      minimal F.binary64 x)

let nearest_of_shortest64 =
  prop
    "the binary64 digits are the nearest of the shortest, the even one at a tie"
    ~count:2000 double (nearest_law F.binary64)

(* Fifteen digits is the most every decimal keeps through a double. *)
let canonical64 =
  prop "a decimal of at most 15 digits is the digits of its double" ~count:1000
    (short_decimal ~digits:15 ~low:(-300) ~high:290)
    (Law.round_trip digits float_exact nearest64 (F.shortest F.binary64))

let reads_back32 =
  prop "binary32 digits read back as the float32 in single precision"
    ~count:2000 float32 (fun x ->
      classes32 x;
      reads_back F.binary32 x)

let minimal32 =
  prop "no decimal of fewer digits reads back as the float32" ~count:2000
    float32 (fun x ->
      classes32 x;
      minimal F.binary32 x)

let nearest_of_shortest32 =
  prop
    "the binary32 digits are the nearest of the shortest, the even one at a tie"
    ~count:2000 float32 (nearest_law F.binary32)

(* Six digits is the most every decimal keeps through a float32. *)
let canonical32 =
  prop "a decimal of at most 6 digits is the digits of its float32" ~count:1000
    (short_decimal ~digits:6 ~low:(-36) ~high:30)
    (Law.round_trip digits float_exact nearest32 (F.shortest F.binary32))

(* Rows *)

(* Each row's digits are what Python's [repr] writes for the double, or numpy's
   [format_float_scientific(unique=True)] for the float32. *)
let stated =
  [
    ("zero is (\"0\", 0)", F.binary64, 0., ("0", 0));
    ("the sign is not read: -0.", F.binary64, -0., ("0", 0));
    ("repr(0.1)", F.binary64, 0.1, ("1", -1));
    ("repr(5e-324), the least subnormal", F.binary64, 5e-324, ("5", -324));
    ("repr(1e23)", F.binary64, 1e23, ("1", 23));
    ( "repr(2**-1022), the least normal",
      F.binary64,
      0x1p-1022,
      ("22250738585072014", -324) );
    ( "repr of the double above 2**-1022",
      F.binary64,
      Float.succ 0x1p-1022,
      ("2225073858507202", -323) );
    ( "repr of the double below 2**-1022",
      F.binary64,
      Float.pred 0x1p-1022,
      ("2225073858507201", -323) );
    ("repr(max_float)", F.binary64, max_float, ("17976931348623157", 292));
    ( "repr(2**50 + 0.25), a tie to the even digit",
      F.binary64,
      0x1p50 +. 0.25,
      ("11258999068426242", -1) );
    ("float32 0.1", F.binary32, single 0.1, ("1", -1));
    ("float32 max", F.binary32, 0x1.fffffep127, ("34028235", 31));
    ("float32 2**-149, the least subnormal", F.binary32, 0x1p-149, ("1", -45));
  ]

let test_stated =
  cases "a double and a float32 have the digits Python and numpy write"
    ~id:(fun (id, _, _, _) -> id)
    stated
    (fun (_, f, x, want) -> equal digits want (F.shortest f x))

let refused =
  [
    ("precision 0", { F.precision = 0; emin = 0 }, 1.);
    ("precision 54", { F.precision = 54; emin = -1022 }, 1.);
    ("binary32 0.1, which is no float32", F.binary32, 0.1);
    ("NaN", F.binary64, Float.nan);
    ("infinity", F.binary64, Float.infinity);
    ("negative infinity", F.binary64, Float.neg_infinity);
  ]

let test_refused =
  cases
    "raises Invalid_argument for a precision outside 1 to 53 and for a value \
     not of the format"
    ~id:(fun (id, _, _) -> id)
    refused
    (fun (_, f, x) -> raises_match Exn.invalid_arg (fun () -> F.shortest f x))

(* The reader the laws rely on, at ties to even and at the overflow threshold,
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
  cases "reads to nearest, ties to even, up to the overflow threshold"
    ~id:(fun (id, _, _, _) -> id)
    reader_rows
    (fun (_, x, de, want) ->
      satisfies
        ~msg:(Printf.sprintf "read as %h" x)
        ~claim:(if want then "reads as x" else "does not read as x")
        digits
        (fun de -> reads F.binary64 x de = want)
        de)

let () =
  exit
    (run "Float_text"
       [
         group "shortest"
           [
             reads_back64;
             minimal64;
             nearest_of_shortest64;
             canonical64;
             reads_back32;
             minimal32;
             nearest_of_shortest32;
             canonical32;
             test_stated;
             test_refused;
           ];
         group "the exact reader" [ test_reader ];
       ])
