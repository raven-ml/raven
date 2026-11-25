(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The decimal digits hugin writes numbers with. The laws decide where a decimal
   lies against a float, and whether it reads back as it, with an exact reader
   of their own. [float_of_string] is the C library's [strtod], which Windows' C
   runtime does not promise to round correctly, and the laws must hold on every
   system. *)

open Windtrap
module F = Hugin_float_text.Float_text

let digits = pair string int
let text (d, e) = Printf.sprintf "%se%d" d e
let single x = Int32.float_of_bits (Int32.bits_of_float x)

(* The formats the [format] type names besides binary64 and binary32. *)
let binary16 = { F.precision = 11; emin = -14 }
let bfloat16 = { F.precision = 8; emin = -126 }
let e5m2 = { F.precision = 3; emin = -14 }
let e4m3fn = { F.precision = 4; emin = -6 }
let e2m1fn = { F.precision = 2; emin = 0 }

(* Exact reading *)

(* Naturals in limbs of 24 bits, least significant first. For [m] and [c] below
   2^24, [mul a m c] is [a m + c], and every product stays below 2^48. *)
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

(* [decode f x] is [(c, q, qmin)] with [|x| = c 2^q], [q] the exponent of the
   last bit of [f]'s values in the binade of [x] and at least the subnormals'
   [qmin]. [c] is whole when [x] is a value of [f]. *)
let decode (f : F.format) x =
  let x = Float.abs x and qmin = f.emin - f.precision + 1 in
  let q =
    if x = 0. then qmin else max (snd (Float.frexp x) - f.precision) qmin
  in
  (Float.ldexp x (-q), q, qmin)

(* [reads f x de] is whether the decimal [de] reads as [|x|] in [f], to nearest,
   ties to even. The reals that read as [c 2^q] lie between the midpoints to its
   neighbours, [(2c - 1) 2^(q-1)] and [(2c + 1) 2^(q-1)], or from the nearer
   [(4c - 1) 2^(q-2)] at the bottom of a binade above the subnormals, the ends
   included when [c] is even. *)
let reads (f : F.format) x de =
  let c, q, qmin = decode f x in
  let c = Float.to_int c in
  let inside o = o < 0 || (o = 0 && c land 1 = 0) in
  let above =
    if c = 0 then true
    else if c = 1 lsl (f.precision - 1) && q > qmin then
      inside (-compare_dec de ((4 * c) - 1) (q - 2))
    else inside (-compare_dec de ((2 * c) - 1) (q - 1))
  in
  above && inside (compare_dec de ((2 * c) + 1) (q - 1))

(* The values of [f] next to its value [x > 0], above and below. *)
let above f x =
  let c, q, _ = decode f x in
  Float.ldexp (c +. 1.) q

let below (f : F.format) x =
  let c, q, qmin = decode f x in
  if c = Float.ldexp 1. (f.precision - 1) && q > qmin then
    Float.ldexp ((2. *. c) -. 1.) (q - 1)
  else Float.ldexp (c -. 1.) q

(* The value of [f] that [de] reads as. The C library's reading, rounded to [f],
   is at most a unit in the last place off. *)
let nearest f de =
  let c, q, _ = decode f (float_of_string (text de)) in
  let guess = Float.ldexp (Float.round c) q in
  let near = [ guess; above f guess; below f guess ] in
  match List.find_opt (fun x -> reads f x de) near with
  | Some x -> x
  | None -> failf "%s reads as no value next to %h" (text de) guess

(* [lead x] is the exponent of the leading digit of [x > 0], the integer [l]
   with [10^l <= x < 10^(l + 1)]. *)
let lead x =
  let c, q, _ = decode F.binary64 x in
  let c = Float.to_int c in
  let rec fix l =
    if compare_dec ("1", l) c q > 0 then fix (l - 1)
    else if compare_dec ("1", l + 1) c q <= 0 then fix (l + 1)
    else l
  in
  fix (Float.to_int (Float.floor (Float.log10 x)))

(* [pred m] is the digits of [m - 1], for the digits [m] of a positive integer
   of any length. *)
let pred m =
  let b = Bytes.of_string m and i = ref (String.length m - 1) in
  while Bytes.get b !i = '0' do
    Bytes.set b !i '9';
    decr i
  done;
  Bytes.set b !i (Char.chr (Char.code (Bytes.get b !i) - 1));
  Bytes.to_string b

(* [misround g x (d, e)] is what keeps [d 10^e] from being the multiple of
   [10^g] nearest [|x|], the even multiple at a tie, or [None]. Written as the
   multiple [m 10^g], it is the nearest when [|x|] lies between the midpoints to
   its neighbours, [(10m - 5) 10^(g-1)] and [(10m + 5) 10^(g-1)]. *)
let misround g x (d, e) =
  let c, q, _ = decode F.binary64 x in
  let c = Float.to_int c in
  let m = if d = "0" then "0" else d ^ String.make (max 0 (e - g)) '0' in
  let low = if m = "0" then -1 else compare_dec (pred m ^ "5", g - 1) c q in
  let high = compare_dec (m ^ "5", g - 1) c q in
  cover "tie" (low = 0 || high = 0);
  if d <> "0" && e < g then Some "a digit below the grid"
  else if low > 0 then Some "more than half the grid above |x|"
  else if high < 0 then Some "more than half the grid below |x|"
  else if (low = 0 || high = 0) && Char.code m.[String.length m - 1] land 1 = 1
  then Some "odd at a tie"
  else None

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
           if k = 0 then x
           else if k = 1 then above F.binary32 x
           else below F.binary32 x );
         ( 2,
           let+ d = int_range 1 999 and+ j = int_range (-44) 35 in
           float_of_string (Printf.sprintf "%de%d" d j) );
         (2, ties 24);
       ])
  |> Gen.map single

(* A value [c 2^q] of the small format [f]: [c] of [precision] bits, a third of
   the time next to the bottom of a binade, and [q] from the subnormals'
   exponent up to values twice IEEE's largest finite, below 2^(3 - emin). *)
let small (f : F.format) =
  let open Gen in
  let p = f.precision in
  let bottom = 1 lsl (p - 1) in
  signed
    (let+ c =
       frequency
         [
           (2, int_range 0 ((1 lsl p) - 1));
           (1, int_range (bottom - 1) (bottom + 1));
         ]
     and+ q = int_range (f.emin - p + 1) (3 - f.emin - p) in
     Float.ldexp (Float.of_int c) q)

(* The seven formats the [format] type names, with how often the laws of
   [shortest] draw each: binary64 and binary32 about 2150 times each in 7000
   cases. *)
let formats =
  [
    ("binary64", F.binary64, 4, double);
    ("binary32", F.binary32, 4, float32);
    ("binary16", binary16, 1, small binary16);
    ("bfloat16", bfloat16, 1, small bfloat16);
    ("e5m2", e5m2, 1, small e5m2);
    ("e4m3fn", e4m3fn, 1, small e4m3fn);
    ("e2m1fn", e2m1fn, 1, small e2m1fn);
  ]

(* A format and one of its values. *)
let value =
  let draw (name, f, weight, gen) =
    (weight, Gen.map (fun x -> (name, f, x)) gen)
  in
  Gen.with_pp
    (fun ppf (name, _, x) -> Format.fprintf ppf "%s %h" name x)
    (Gen.frequency (List.map draw formats))

(* The classes of values each format must reach, binary64 and binary32 each on
   their own. *)
let classes (name, (f : F.format), x) =
  let x = Float.abs x and least_normal = Float.ldexp 1. f.emin in
  let subnormal = x > 0. && x < least_normal
  and bottom = x >= least_normal && fst (Float.frexp x) = 0.5 in
  List.iter (fun (n, _, _, _) -> cover n (n = name)) formats;
  cover "subnormal" subnormal;
  cover "binade bottom" bottom;
  List.iter
    (fun n ->
      cover (n ^ " subnormal") (name = n && subnormal);
      cover (n ^ " binade bottom") (name = n && bottom))
    [ "binary64"; "binary32" ];
  cover "binary64's top binade" (name = "binary64" && x >= 0x1p1023)

(* [k 2^-r] as [(k, r)], for odd [k] below 2^53 and [r] from 1 to 3. Its exact
   decimal [k 5^r 10^-r] ends in 5, [r] places after the point, so rounded one
   digit sooner it lies halfway. *)
let halfway =
  let open Gen in
  let+ k = int_range 1 ((1 lsl 52) - 1) and+ r = int_range 1 3 in
  ((2 * k) + 1, r)

let halfway_float (k, r) = Float.ldexp (Float.of_int k) (-r)

(* [x] and a number of digits, one fewer than [x]'s exact decimal for a halfway
   [x]. *)
let significant_input =
  let open Gen in
  let tie (k, r) =
    let rec pow5 r = if r = 0 then 1 else 5 * pow5 (r - 1) in
    (halfway_float (k, r), String.length (string_of_int (k * pow5 r)) - 1)
  in
  frequency [ (4, pair double (int_range 1 25)); (1, map tie halfway) ]

(* [x] and a number of decimals, from two digits short of [x]'s leading one to
   20 past it, or one fewer than [x]'s for a halfway [x]. *)
let decimals_input =
  let open Gen in
  let around =
    let* x = double in
    let+ k = int_range (-2) 20 in
    (x, if x = 0. then k else k - lead (Float.abs x))
  in
  let tie (k, r) = (halfway_float (k, r), r - 1) in
  frequency [ (4, around); (1, map tie halfway) ]

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

(* digits *)

let canonical (d, e) =
  let digit c = c >= '0' && c <= '9' in
  (d = "0" && e = 0)
  || d <> "" && String.for_all digit d
     && d.[0] <> '0'
     && d.[String.length d - 1] <> '0'

let test_canonical =
  prop "have no leading or trailing zero" ~count:500
    ~examples:[ (9.96, 2, 1); (0.996, 1, 2); (1e23, 1, -25); (-0., 1, 0) ]
    Gen.(triple double (int_range 1 25) (int_range (-5) 25))
    (fun (x, n, n') ->
      let claim = "digits without leading or trailing zero" in
      satisfies ~claim digits canonical (F.shortest F.binary64 x);
      satisfies ~claim digits canonical (F.significant n x);
      satisfies ~claim digits canonical (F.decimals n' x))

(* [decimals 2 (-0.001)] is the .mli's example. *)
let test_zero =
  cases "are (\"0\", 0) for zero of either sign, and for what rounds to it"
    ~id:(fun (name, _, x) -> Printf.sprintf "%s %g" name x)
    [
      ("shortest", F.shortest F.binary64, 0.);
      ("shortest", F.shortest F.binary64, -0.);
      ("significant 1", F.significant 1, -0.);
      ("decimals 2", F.decimals 2, 0.);
      ("decimals 2", F.decimals 2, -0.001);
    ]
    (fun (_, f, x) -> equal digits ("0", 0) (f x))

let non_finite =
  let functions =
    [
      ("shortest", F.shortest F.binary64);
      ("significant 3", F.significant 3);
      ("decimals 2", F.decimals 2);
    ]
  and values =
    [
      ("nan", Float.nan);
      ("-nan", -.Float.nan);
      ("infinity", Float.infinity);
      ("neg_infinity", Float.neg_infinity);
    ]
  in
  List.concat_map (fun f -> List.map (fun x -> (f, x)) values) functions

let test_non_finite =
  cases "raise Invalid_argument on NaN and the infinities"
    ~id:(fun ((f, _), (x, _)) -> f ^ " " ^ x)
    non_finite
    (fun ((_, f), (_, x)) -> raises_match Exn.invalid_arg (fun () -> f x))

(* format *)

(* Values from IEEE 754-2019 table 3.5 (binary16), Google's bfloat16, and OCP's
   8-bit and MX formats (e5m2, e4m3fn, e2m1fn). Their digits are the fewest that
   read as the value, the nearest of those: 65504 reads from 65488 to 65520,
   ends excluded, where 655e2 is the only decimal of three digits; e4m3fn's 448,
   whose significand 1.110 is even, reads from 432 to 464, ends included, where
   450 is the nearest of 440, 450 and 460. *)
let extremes =
  [
    ("binary32", F.binary32, "largest finite", 0x1.fffffep127, ("34028235", 31));
    ("binary32", F.binary32, "least subnormal", 0x1p-149, ("1", -45));
    ("binary16", binary16, "largest finite", 65504., ("655", 2));
    ("binary16", binary16, "least subnormal", 0x1p-24, ("6", -8));
    ("bfloat16", bfloat16, "largest finite", 0x1.fep127, ("339", 36));
    ("bfloat16", bfloat16, "least subnormal", 0x1p-133, ("9", -41));
    ("e5m2", e5m2, "largest finite", 57344., ("6", 4));
    ("e5m2", e5m2, "least subnormal", 0x1p-16, ("2", -5));
    ("e4m3fn", e4m3fn, "largest finite", 448., ("45", 1));
    ("e4m3fn", e4m3fn, "least subnormal", 0x1p-9, ("2", -3));
    ("e2m1fn", e2m1fn, "largest finite", 6., ("6", 0));
    ("e2m1fn", e2m1fn, "least subnormal", 0.5, ("5", -1));
  ]

let test_extremes =
  cases
    "a named format's largest finite and least subnormal values have their \
     shortest digits"
    ~id:(fun (name, _, which, _, _) -> name ^ " " ^ which)
    extremes
    (fun (_, f, _, x, want) -> equal digits want (F.shortest f x))

(* binary64 and binary32 *)

let test_binary64 =
  test "is { precision = 53; emin = -1022 }" (fun () ->
      equal (pair int int) (53, -1022) (F.binary64.precision, F.binary64.emin))

let test_binary32 =
  test "is { precision = 24; emin = -126 }" (fun () ->
      equal (pair int int) (24, -126) (F.binary32.precision, F.binary32.emin))

(* shortest *)

let test_reads_back =
  prop "reads back as x" ~count:7000 value (fun ((_, f, x) as case) ->
      classes case;
      satisfies ~claim:"reads back as x" digits (reads f x) (F.shortest f x))

let test_minimal =
  prop "is a decimal of the fewest digits that reads back as x" ~count:7000
    value (fun ((_, f, x) as case) ->
      classes case;
      List.iter
        (satisfies ~claim:"does not read back as x" digits (fun de ->
             not (reads f x de)))
        (shorter (F.shortest f x)))

(* Of the shortest decimals, the nearest. [d 10^e] has [n] digits; the only
   others of [n] digits that can be nearer to [x] are its neighbours on the far
   side of [x]: [(d + 1) 10^e] above, and below [(d - 1) 10^e], or [9 10^(e-1)]
   when [d] is 1. One is nearer when [x] is past the midpoint toward it, which
   is [(10d + 5) 10^(e-1)] above, and [(10d - 5) 10^(e-1)] or [95 10^(e-2)]
   below. At the midpoint the two tie, and the even last digit wins. *)
let nearest_law ((name, f, x) as case) =
  classes case;
  let d, e = F.shortest f x in
  let c, q, _ = decode f x in
  let c = Float.to_int c in
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
  let tie = x <> 0. && (o up = 0 || o down = 0) in
  cover "tie" tie;
  cover "binary64 tie" (name = "binary64" && tie);
  cover "binary32 tie" (name = "binary32" && tie);
  let flaw =
    if reads_n up && not (o up > 0 || (o up = 0 && even)) then
      Some "the neighbour above is nearer, or as near and even"
    else if reads_n down && not (o down < 0 || (o down = 0 && even)) then
      Some "the neighbour below is nearer, or as near and even"
    else None
  in
  equal ~msg:(text (d, e)) (option string) None flaw

let test_nearest =
  prop "is the nearest of those, the one with an even last digit at a tie"
    ~count:7000 value nearest_law

(* A decimal of at most [(precision - 1) log10 2] digits in the normal range is
   the shortest of the value nearest it: 15 for binary64, 6 for binary32. *)
let short_decimal =
  let open Gen in
  let rec pow10 k = if k = 0 then 1 else 10 * pow10 (k - 1) in
  let in_format (name, f, digits, low, high) =
    let+ m = int_range 1 (pow10 digits - 1) and+ j = int_range low high in
    let s = string_of_int m in
    let k = ref (String.length s) in
    while s.[!k - 1] = '0' do
      decr k
    done;
    (name, f, (String.sub s 0 !k, j + String.length s - !k))
  in
  with_pp
    (fun ppf (name, _, de) -> Format.fprintf ppf "%s %s" name (text de))
    (one_of
       [
         in_format ("binary64", F.binary64, 15, -300, 290);
         in_format ("binary32", F.binary32, 6, -36, 30);
       ])

let test_round_trip =
  prop "of the value nearest a short decimal is that decimal" ~count:2000
    ~examples:
      [
        ("binary64", F.binary64, ("1", -1));
        ("binary32", F.binary32, ("1", -1));
        ("binary64", F.binary64, ("1", 23));
      ]
    short_decimal
    (fun (name, f, de) ->
      cover "binary64" (name = "binary64");
      cover "binary32" (name = "binary32");
      equal digits de (F.shortest f (nearest f de)))

(* The shortest digits ECMAScript's Number::toString and Ryu give. *)
let test_binary64_edges =
  cases "is the known digits at binary64's subnormal, normal and overflow edges"
    ~id:(fun (n, _, _) -> n)
    [
      ("5e-324", 5e-324, ("5", -324));
      ("2^-1022", 0x1p-1022, ("22250738585072014", -324));
      ("above 2^-1022", Float.succ 0x1p-1022, ("2225073858507202", -323));
      ("below 2^-1022", Float.pred 0x1p-1022, ("2225073858507201", -323));
      ("max_float", max_float, ("17976931348623157", 292));
    ]
    (fun (_, x, want) -> equal digits want (F.shortest F.binary64 x))

(* 2^50 + 0.25 lies halfway between 1125899906842624.2 and 1125899906842624.3,
   both of which read as it. *)
let test_even_tie =
  test "takes the even last digit at a tie, 2^50 + 0.25" (fun () ->
      equal digits ("11258999068426242", -1)
        (F.shortest F.binary64 (0x1p50 +. 0.25)))

(* In [{ precision = 1; emin = 0 }] the values are 0 and the powers of two from
   1, and 4 reads from 3 to 6. *)
let test_precision_one =
  test "takes precision 1" (fun () ->
      equal digits ("4", 0) (F.shortest { F.precision = 1; emin = 0 } 4.))

let test_shortest_refused =
  cases
    "raises Invalid_argument for a precision outside 1 to 53 or a value not of \
     the format"
    ~id:(fun (name, _, _) -> name)
    [
      ("precision 0", { F.precision = 0; emin = 0 }, 1.);
      ("precision 54", { F.precision = 54; emin = -1022 }, 1.);
      ("binary32 0.1", F.binary32, 0.1);
      ("e2m1fn 0.25", e2m1fn, 0.25);
    ]
    (fun (_, f, x) -> raises_match Exn.invalid_arg (fun () -> F.shortest f x))

(* significant *)

let test_significant =
  prop "is |x| rounded to n significant digits, to nearest, ties to even"
    ~count:2000
    ~examples:[ (9.96, 2); (0.5, 3) ]
    significant_input
    (fun (x, n) ->
      let ((d, e) as de) = F.significant n x in
      let leading = if x = 0. then 0 else lead (Float.abs x) in
      cover "a carry adds a digit" (x <> 0. && String.length d - 1 + e > leading);
      equal ~msg:(text de) (option string) None
        (misround (leading - n + 1) x de))

let test_significant_rows =
  cases
    "rounds 135.5625 to 136 at three digits and the tie 0.125 to 0.12 at two"
    ~id:(fun (n, x, _) -> Printf.sprintf "%d %h" n x)
    [ (3, 135.5625, ("136", 0)); (2, 0.125, ("12", -2)) ]
    (fun (n, x, want) -> equal digits want (F.significant n x))

(* [pow5 k] is the digits of 5^k, by schoolbook multiplication in base 10. *)
let pow5 k =
  let a = Array.make k 0 in
  a.(0) <- 1;
  for _ = 1 to k do
    let carry = ref 0 in
    for i = 0 to k - 1 do
      let t = (a.(i) * 5) + !carry in
      a.(i) <- t mod 10;
      carry := t / 10
    done
  done;
  let top = ref (k - 1) in
  while a.(!top) = 0 do
    decr top
  done;
  String.init (!top + 1) (fun i -> Char.chr (48 + a.(!top - i)))

(* The exact decimals of the double nearest 0.005, 0x1.47ae147ae147bp-8, and of
   2^-1074, 5^1074 10^-1074. *)
let test_every_digit =
  cases "at max_int digits is the exact decimal of x"
    ~id:(fun (x, _) -> Printf.sprintf "%h" x)
    [
      ( 0.005,
        lazy ("5000000000000000104083408558608425664715468883514404296875", -60)
      );
      (5e-324, lazy (pow5 1074, -1074));
    ]
    (fun (x, want) -> equal digits (Lazy.force want) (F.significant max_int x))

let test_significant_refused =
  cases "raises Invalid_argument for n below 1" ~id:string_of_int
    [ 0; -1; min_int ] (fun n ->
      raises_match Exn.invalid_arg (fun () -> F.significant n 1.))

(* decimals *)

let test_decimals =
  prop "is |x| rounded to a multiple of 10^-n, to nearest, ties to even"
    ~count:2000
    ~examples:[ (0.996, 2) ]
    decimals_input
    (fun (x, n) ->
      let ((d, e) as de) = F.decimals n x in
      let leading = if x = 0. then 0 else lead (Float.abs x) in
      cover "n negative" (n < 0);
      cover "rounds to zero" (x <> 0. && d = "0");
      cover "a carry adds a digit"
        (d <> "0" && String.length d - 1 + e > leading);
      equal ~msg:(text de) (option string) None (misround (-n) x de))

(* The .mli's examples, and 1.5, which rounds up to the even 2. *)
let test_decimals_ties =
  cases "rounds a tie to the even multiple, for n negative too"
    ~id:(fun (n, x, _) -> Printf.sprintf "%d %h" n x)
    [
      (2, 0.125, ("12", -2));
      (0, 0.5, ("0", 0));
      (0, 1.5, ("2", 0));
      (-2, 1250., ("12", 2));
    ]
    (fun (n, x, want) -> equal digits want (F.decimals n x))

let test_decimals_extremes =
  cases "takes any n, min_int and max_int included"
    ~id:(fun (n, _, _) -> string_of_int n)
    [ (max_int, 1.5, ("15", -1)); (min_int, 1.5, ("0", 0)) ]
    (fun (n, x, want) -> equal digits want (F.decimals n x))

(* The exact reader *)

(* At ties to even and at binary64's overflow threshold,
   1.797693134862315807...e308. *)
let test_reader_reads =
  cases "exact reader reads these decimals as x, to nearest, ties to even"
    ~id:(fun (name, _, _) -> name)
    [
      ("0.1", 0.1, ("1", -1));
      ("2^53 + 1 to 2^53", 0x1p53, ("9007199254740993", 0));
      ("2^53 + 3 to 2^53 + 4", 0x1p53 +. 4., ("9007199254740995", 0));
      ("below overflow to max_float", max_float, ("17976931348623158", 292));
    ]
    (fun (_, x, de) ->
      satisfies ~claim:"reads as x" digits (reads F.binary64 x) de)

let test_reader_refuses =
  cases "exact reader reads these decimals as another value than x"
    ~id:(fun (name, _, _) -> name)
    [
      ("2^53 + 1 to 2^53 + 2", 0x1p53 +. 2., ("9007199254740993", 0));
      ("2^53 + 3 to 2^53 + 2", 0x1p53 +. 2., ("9007199254740995", 0));
      ("past overflow to max_float", max_float, ("17976931348623159", 292));
    ]
    (fun (_, x, de) ->
      satisfies ~claim:"does not read as x" digits
        (fun de -> not (reads F.binary64 x de))
        de)

let () =
  exit
    (run "hugin float text"
       [
         group "digits" [ test_canonical; test_zero; test_non_finite ];
         group "format" [ test_extremes ];
         group "binary64" [ test_binary64 ];
         group "binary32" [ test_binary32 ];
         group "shortest"
           [
             test_reads_back;
             test_minimal;
             test_nearest;
             test_round_trip;
             test_binary64_edges;
             test_even_tie;
             test_precision_one;
             test_shortest_refused;
           ];
         group "significant"
           [
             test_significant;
             test_significant_rows;
             test_every_digit;
             test_significant_refused;
           ];
         group "decimals"
           [ test_decimals; test_decimals_ties; test_decimals_extremes ];
         test_reader_reads;
         test_reader_refuses;
       ])
