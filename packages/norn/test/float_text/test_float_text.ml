(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The decimal digits norn writes numbers with. The laws decide whether a
   decimal reads back as a float with [reads], an exact reader of their own.
   [float_of_string] is the C library's [strtod], which Windows' C runtime does
   not promise to round correctly, and the laws must hold on every system. *)

open Windtrap
module F = Norn_float_text.Float_text

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

(* The narrow formats [format] lists, and a precision of one bit. *)
let narrow =
  [
    { F.precision = 11; emin = -14 };
    { F.precision = 8; emin = -126 };
    { F.precision = 3; emin = -14 };
    { F.precision = 4; emin = -6 };
    { F.precision = 2; emin = 0 };
    { F.precision = 1; emin = 0 };
  ]

(* A value [c 2^q] of a narrow format, [c] of at most its precision's bits and
   [q] from its least exponent up. *)
let narrow_value =
  let open Gen in
  let* (f : F.format) = of_list narrow in
  let qmin = f.emin - f.precision + 1 in
  let+ c = int_range 0 ((1 lsl f.precision) - 1)
  and+ q = int_range qmin (qmin + 40)
  and+ neg = bool in
  let x = Float.ldexp (Float.of_int c) q in
  (f, if neg then -.x else x)

(* A format and one of its values: binary64, binary32 or a narrow one. *)
let value =
  Gen.(
    one_of
      [
        map (fun x -> (F.binary64, x)) double;
        map (fun x -> (F.binary32, x)) float32;
        narrow_value;
      ])
  |> Gen.with_pp (fun ppf ((f : F.format), x) ->
      Format.fprintf ppf "%h in {precision %d; emin %d}" x f.precision f.emin)

let classes (f : F.format) x =
  let x = Float.abs x in
  cover "binary64" (f.precision = 53);
  cover "binary32" (f.precision = 24);
  cover "a narrow format" (f.precision <= 11);
  cover "subnormal" (x > 0. && x < Float.ldexp 1. f.emin);
  cover "binade bottom" (x >= Float.ldexp 1. f.emin && fst (Float.frexp x) = 0.5);
  cover "largest double" (x > 0x1p1023)

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

let reads_as f x de =
  satisfies
    ~claim:(Printf.sprintf "%s reads as %h" (text de) x)
    digits
    (fun de -> reads f x de)
    de

let reads_not_as f x de =
  satisfies
    ~claim:(Printf.sprintf "%s does not read as %h" (text de) x)
    digits
    (fun de -> not (reads f x de))
    de

(* Of the shortest decimals, the nearest. [d 10^e] has [n] digits; the only
   others of [n] digits that can be nearer to [x] are its neighbours on the far
   side of [x]: [(d + 1) 10^e] above, and below [(d - 1) 10^e], or [9 10^(e-1)]
   when [d] is 1. One is nearer when [x] is past the midpoint toward it, [(10d +
   5) 10^(e-1)] above and [(10d - 5) 10^(e-1)] or [95 10^(e-2)] below. At the
   midpoint the two tie, and the even last digit wins. *)
let nearest_law (f, x) =
  classes f x;
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
  let past ~claim ok = satisfies ~claim digits (fun _ -> ok) (d, e) in
  if reads_n up then
    past ~claim:"nearer than the one above" (o up > 0 || (o up = 0 && even));
  if reads_n down then
    past ~claim:"nearer than the one below" (o down < 0 || (o down = 0 && even))

(* A decimal of at most 15 digits in the normal range is the shortest of the
   double nearest it; of at most 6 digits, of the float32 nearest it. *)
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

(* [nearest x de ~down ~up ~even] checks that [|x|] lies between [down] and
   [up], the midpoints from the decimal [de] to the multiples beside it on its
   grid (no [down] below zero), and on a midpoint only if [de] is the even
   multiple: [de] is the multiple nearest [|x|], ties to even. *)
let nearest x de ~down ~up ~even =
  let c, q, _ = decode F.binary64 x in
  let o_up = compare_dec up c q in
  let o_down = match down with None -> -1 | Some d -> compare_dec d c q in
  cover "a tie" (o_up = 0 || o_down = 0);
  satisfies
    ~claim:(Printf.sprintf "the multiple nearest %h, ties to even" x)
    digits
    (fun _ ->
      (o_up > 0 || (o_up = 0 && even)) && (o_down < 0 || (o_down = 0 && even)))
    de

let shortest =
  group "shortest"
    [
      prop "reads back as x" ~count:2000 value (fun (f, x) ->
          classes f x;
          reads_as f x (F.shortest f x));
      prop "has no shorter decimal that reads back" ~count:2000 value
        (fun (f, x) ->
          classes f x;
          List.iter (reads_not_as f x) (shorter (F.shortest f x)));
      prop "is the nearest of the shortest, ties to even" ~count:2000 value
        nearest_law;
      prop "is a short decimal that reads as the nearest double" ~count:1000
        (short_decimal ~digits:15 ~low:(-300) ~high:290) (fun de ->
          equal digits de (F.shortest F.binary64 (nearest64 de)));
      prop "is a short decimal that reads as the nearest float32" ~count:1000
        (short_decimal ~digits:6 ~low:(-36) ~high:30) (fun de ->
          equal digits de (F.shortest F.binary32 (nearest32 de)));
      cases "gives the stated digits"
        ~id:(fun (_, x, _) -> Printf.sprintf "%h" x)
        [
          (F.binary64, 0., ("0", 0));
          (F.binary64, -0., ("0", 0));
          (F.binary64, 0.1, ("1", -1));
          (F.binary64, 5e-324, ("5", -324));
          (F.binary64, 1e23, ("1", 23));
          (F.binary64, 0x1p-1022, ("22250738585072014", -324));
          (F.binary64, Float.succ 0x1p-1022, ("2225073858507202", -323));
          (F.binary64, Float.pred 0x1p-1022, ("2225073858507201", -323));
          (F.binary64, max_float, ("17976931348623157", 292));
          (F.binary64, 0x1p50 +. 0.25, ("11258999068426242", -1));
          (F.binary32, single 0.1, ("1", -1));
          ({ F.precision = 11; emin = -14 }, 65504., ("655", 2));
        ]
        (fun (f, x, de) -> equal digits de (F.shortest f x));
      cases "raises for a precision outside 1 to 53, or x not a value of f"
        ~id:fst
        [
          ("precision 64", ({ F.precision = 64; emin = -16382 }, 0.1));
          ("precision 0", ({ F.precision = 0; emin = 0 }, 1.));
          ("binary32 0.1", (F.binary32, 0.1));
          ("nan", (F.binary64, Float.nan));
        ]
        (fun (_, (f, x)) ->
          raises_match Exn.invalid_arg (fun () -> F.shortest f x));
    ]

(* 5^1074, the digits of 2^-1074 = 5^1074 10^-1074, by schoolbook multiplication
   in base 10. *)
let pow5_1074 () =
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

(* Finite doubles, and multiples of 1/8 whose digits end in 5 at some grid. *)
let finite =
  Gen.(
    frequency
      [
        (3, double);
        (2, map (fun k -> float_of_int k /. 8.) (int_range (-80_000) 80_000));
      ])
  |> Gen.with_pp (fun ppf x -> Format.fprintf ppf "%h" x)

let significant =
  group "significant"
    [
      prop "rounds |x| to n significant digits, ties to even" ~count:2000
        Gen.(pair finite (int_range 1 16))
        (fun (x, n) ->
          let d, e = F.significant n x in
          if x <> 0. then begin
            at_most int ~than:n (String.length d);
            (* [d 10^e] is [m] multiples of [10^k], [k] its [n]-th digit's
               place. Below a power of ten, [d] = 1, the multiple next down is
               on the grid one digit finer, where it is even. *)
            let k = String.length d - n + e in
            let m = int_of_string (d ^ String.make (e - k) '0') in
            let down =
              if d = "1" then (string_of_int ((100 * m) - 5), k - 2)
              else (string_of_int ((10 * m) - 5), k - 1)
            in
            nearest x (d, e) ~down:(Some down)
              ~up:(string_of_int ((10 * m) + 5), k - 1)
              ~even:(d = "1" || m land 1 = 0)
          end);
      cases "gives the stated digits"
        ~id:(fun (n, x, _) -> Printf.sprintf "%d %h" n x)
        [
          (3, 135.5625, ("136", 0));
          (2, 0.125, ("12", -2));
          ( max_int,
            0.005,
            ("5000000000000000104083408558608425664715468883514404296875", -60)
          );
        ]
        (fun (n, x, de) -> equal digits de (F.significant n x));
      test "gives 5e-324 at max_int digits as 5^1074 10^-1074" (fun () ->
          equal digits (pow5_1074 (), -1074) (F.significant max_int 5e-324));
      cases "raises for n below 1, or x not finite" ~id:fst
        [
          ("n 0", (0, 1.));
          ("inf", (3, Float.infinity));
          ("-inf", (3, Float.neg_infinity));
        ]
        (fun (_, (n, x)) ->
          raises_match Exn.invalid_arg (fun () -> F.significant n x));
    ]

let decimals =
  group "decimals"
    [
      prop "rounds |x| to a multiple of 10^-n, ties to even" ~count:2000
        Gen.(
          pair
            (frequency
               [
                 (1, float_range (-1e6) 1e6);
                 ( 1,
                   map (fun k -> float_of_int k /. 8.) (int_range (-8000) 8000)
                 );
               ])
            (int_range (-3) 10))
        (fun (x, n) ->
          (* [d 10^e] is [m] multiples of [10^-n]. *)
          let d, e = F.decimals n x in
          let m =
            if d = "0" then 0 else int_of_string (d ^ String.make (e + n) '0')
          in
          let down = (string_of_int ((10 * m) - 5), -n - 1) in
          nearest x (d, e)
            ~down:(if m = 0 then None else Some down)
            ~up:(string_of_int ((10 * m) + 5), -n - 1)
            ~even:(m land 1 = 0));
      cases "gives the stated digits"
        ~id:(fun (n, x, _) -> Printf.sprintf "%d %h" n x)
        [
          (2, 0.125, ("12", -2));
          (0, 0.5, ("0", 0));
          (0, 1.5, ("2", 0));
          (-2, 1250., ("12", 2));
          (2, -0.001, ("0", 0));
          (max_int, 1.5, ("15", -1));
          (min_int, 1.5, ("0", 0));
        ]
        (fun (n, x, de) -> equal digits de (F.decimals n x));
      cases "raises for x not finite" ~id:(Printf.sprintf "%h")
        [ Float.nan; Float.infinity ] (fun x ->
          raises_match Exn.invalid_arg (fun () -> F.decimals 2 x));
    ]

(* The exact reader the laws above stand on, at ties to even and at the overflow
   threshold, 1.797693134862315807...e308: a reader wrong at a tie could make
   both polarities of the laws agree with a wrong library. *)
let reader =
  cases "the laws' reader rounds ties to even and stops at overflow"
    ~id:(fun (name, _, _, _) -> name)
    [
      ("0.1", 0.1, ("1", -1), true);
      ("2^53 + 1 to 2^53", 0x1p53, ("9007199254740993", 0), true);
      ("2^53 + 1 to 2^53 + 2", 0x1p53 +. 2., ("9007199254740993", 0), false);
      ("2^53 + 3 to 2^53 + 4", 0x1p53 +. 4., ("9007199254740995", 0), true);
      ("2^53 + 3 to 2^53 + 2", 0x1p53 +. 2., ("9007199254740995", 0), false);
      ("below overflow", max_float, ("17976931348623158", 292), true);
      ("past overflow", max_float, ("17976931348623159", 292), false);
    ]
    (fun (_, x, de, reads) ->
      if reads then reads_as F.binary64 x de else reads_not_as F.binary64 x de)

let () =
  exit (run "norn float text" [ shortest; significant; decimals; reader ])
