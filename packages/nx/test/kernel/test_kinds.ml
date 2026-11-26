(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* prog.mli's kinds as nx_kinds.h computes them, called by name through loops
   compiled as nx.cpu's rows: the exact kinds against their meaning, the
   domains' orders, Annex F's special values and quiet NaNs, the complex kinds
   against their bounds, Annex G and their symmetries, then what the header
   states of its own functions. The narrow floats run through each kernel
   library's map. The bounds of the transcendental kinds are test_kernels',
   through Nx_kernel.S.map. *)

open Windtrap
open Exact
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module C = Kernels_cases
module Support = Nx_kernels_support
module K = Nx_kinds_support

let strf = Printf.sprintf

(* f32 values travel as their bits, f64 values as floats *)

let bits32 x = Int32.to_int (Int32.bits_of_float x) land 0xFFFF_FFFF
let f b = Int32.float_of_bits (Int32.of_int b)
let is_nan b = b land 0x7FFF_FFFF > 0x7F80_0000
let bits64 = Int64.bits_of_float
let pp32 ppf b = Format.fprintf ppf "%h (0x%08x)" (f b) b
let pp64 ppf x = Format.fprintf ppf "%h (0x%016Lx)" x (bits64 x)

(* Bit for bit. *)
let exact32 = Testable.make ~pp:pp32 ~equal:( = )
let exact64 = Testable.make ~pp:pp64 ~equal:(fun a b -> bits64 a = bits64 b)

(* Bit for bit, but any NaN for a NaN: an operation's own NaN differs between
   instruction sets. *)
let value32 =
  Testable.make ~pp:pp32 ~equal:(fun a b -> a = b || (is_nan a && is_nan b))

let value64 =
  Testable.make ~pp:pp64 ~equal:(fun a b ->
      bits64 a = bits64 b || (Float.is_nan a && Float.is_nan b))

let inf = Float.infinity
let ninf = Float.neg_infinity
let nan = Float.nan
let pi = Float.pi

(* A kind drawn by its name, and a cover of each name. *)
let kind names = Gen.of_list ~pp:Format.pp_print_string names
let cover_each names k = List.iter (fun n -> cover n (n = k)) names

(* Drawn f32 operands, as bits *)

let gen32 =
  let open Gen in
  let any = map (fun i -> Int32.to_int i land 0xFFFF_FFFF) int32 in
  let special =
    of_list
      [
        0;
        0x8000_0000;
        0x7F80_0000;
        0xFF80_0000;
        0x7FC0_0000;
        0x7FA0_0001;
        0xFFC0_0123;
        0x0000_0001;
        0x8000_0001;
        0x007F_FFFF;
        0x0080_0000;
        0x7F7F_FFFF;
        0xFF7F_FFFF;
        bits32 1.;
        bits32 (-1.);
        bits32 0.5;
        bits32 (-0.5);
        bits32 1.5;
        bits32 2.;
        bits32 (-2.);
        bits32 3.;
        bits32 (-3.);
        bits32 0x1p23;
        bits32 (-0x1p23 -. 1.);
      ]
  in
  let moderate = map bits32 (float_range (-20.) 20.) in
  let integers =
    map (fun i -> bits32 (Float.of_int i)) (int_range (-300) 300)
  in
  let halves =
    map (fun i -> bits32 (Float.of_int i +. 0.5)) (int_range (-300) 300)
  in
  let near_one = map (fun i -> bits32 1. + i) (int_range (-4096) 4096) in
  with_pp
    (fun ppf b -> Format.fprintf ppf "0x%08x" b)
    (frequency
       [
         (4, any);
         (2, special);
         (3, moderate);
         (2, integers);
         (1, halves);
         (2, near_one);
       ])

let cover32 name b =
  cover (name ^ " NaN") (is_nan b);
  cover (name ^ " -0") (b = 0x8000_0000);
  cover (name ^ " infinite") (b land 0x7FFF_FFFF = 0x7F80_0000);
  cover (name ^ " subnormal") (b land 0x7F80_0000 = 0 && b land 0x7F_FFFF <> 0)

(* Drawn f64 operands *)

let gen64 =
  let open Gen in
  let any = map Int64.float_of_bits int64 in
  let special =
    of_list
      [
        0.;
        -0.;
        inf;
        ninf;
        nan;
        Int64.float_of_bits 0x7FF4_0000_0000_0001L;
        Int64.float_of_bits 1L;
        -.Int64.float_of_bits 1L;
        Float.min_float;
        Float.max_float;
        -.Float.max_float;
        1.;
        -1.;
        0.5;
        -0.5;
        1.5;
        2.;
        0x1p52;
        -0x1p52 -. 1.;
      ]
  in
  let moderate = float_range (-1e6) 1e6 in
  let integers = map Float.of_int (int_range (-1000) 1000) in
  let halves = map (fun i -> Float.of_int i +. 0.5) (int_range (-1000) 1000) in
  frequency
    [ (4, any); (2, special); (3, moderate); (2, integers); (1, halves) ]

let cover64 name x =
  cover (name ^ " NaN") (Float.is_nan x);
  cover (name ^ " -0") (bits64 x = bits64 (-0.));
  cover (name ^ " infinite") (Float.abs x = inf);
  cover (name ^ " subnormal") (x <> 0. && Float.abs x < Float.min_float)

let pair32 = Gen.pair gen32 gen32
let pair64 = Gen.pair gen64 gen64

(* Kinds

   A binary32 sum, difference, product, quotient or square root of floats is the
   double one rounded to binary32, since 53 >= 2 * 24 + 2: each reference below
   is OCaml's binary64 arithmetic, rounded to binary32 for f32. *)

let test_arith =
  let kinds = [ "add"; "sub"; "mul" ] in
  let op = function "add" -> ( +. ) | "sub" -> ( -. ) | _ -> ( *. ) in
  prop "add, sub and mul are the exact result rounded once" ~count:60000
    Gen.(triple (kind kinds) pair32 pair64)
    (fun (k, (a, b), (x, y)) ->
      cover_each kinds k;
      cover32 "f32 first" a;
      cover32 "f32 second" b;
      cover64 "f64 first" x;
      cover64 "f64 second" y;
      equal value32 (bits32 (op k (f a) (f b))) (K.f32 k [| a; b |]);
      equal value64 (op k x y) (K.f64 k [| x; y |]))

let test_fdiv =
  prop "fdiv is the IEEE 754 quotient" ~count:20000
    Gen.(pair pair32 pair64)
    (fun ((a, b), (x, y)) ->
      cover32 "f32 divisor" b;
      cover64 "f64 divisor" y;
      equal value32 (bits32 (f a /. f b)) (K.f32 "fdiv" [| a; b |]);
      equal value64 (x /. y) (K.f64 "fdiv" [| x; y |]))

let test_sqrt =
  prop "sqrt is the IEEE 754 square root" ~count:20000
    Gen.(pair gen32 gen64)
    (fun (a, x) ->
      cover32 "f32" a;
      cover64 "f64" x;
      equal value32 (bits32 (Float.sqrt (f a))) (K.f32 "sqrt" [| a |]);
      equal value64 (Float.sqrt x) (K.f64 "sqrt" [| x |]))

let test_recip =
  prop "recip is 1 / x by fdiv" ~count:20000
    Gen.(pair gen32 gen64)
    (fun (a, x) ->
      cover32 "f32" a;
      cover64 "f64" x;
      equal value32 (bits32 (1. /. f a)) (K.f32 "recip" [| a |]);
      equal value64 (1. /. x) (K.f64 "recip" [| x |]))

(* fmod is exact (C99 7.12.10.1), so binary64's of binary32 operands is the
   binary32 remainder. *)
let test_mod =
  prop "mod is fmod, of the dividend's sign" ~count:20000
    Gen.(pair pair32 pair64)
    (fun ((a, b), (x, y)) ->
      cover "f32 divisor a zero" (f b = 0.);
      cover "f64 divisor a zero" (y = 0.);
      equal value32 (bits32 (Float.rem (f a) (f b))) (K.f32 "mod" [| a; b |]);
      equal value64 (Float.rem x y) (K.f64 "mod" [| x; y |]))

let test_rounding =
  let kinds = [ "floor"; "ceil"; "round"; "trunc" ] in
  let op = function
    | "floor" -> Float.floor
    | "ceil" -> Float.ceil
    | "round" -> Float.round
    | _ -> Float.trunc
  in
  let half x = Float.is_finite x && Float.abs (x -. Float.trunc x) = 0.5 in
  prop
    "floor, ceil, round and trunc round toward -inf, toward +inf, half away \
     from zero and toward zero"
    ~count:80000
    Gen.(triple (kind kinds) gen32 gen64)
    (fun (k, a, x) ->
      cover_each kinds k;
      List.iter (fun n -> cover (n ^ " of a half") (n = k && half (f a))) kinds;
      cover32 "f32" a;
      cover64 "f64" x;
      equal value32 (bits32 (op k (f a))) (K.f32 k [| a |]);
      equal value64 (op k x) (K.f64 k [| x |]))

let test_sign =
  let sign x =
    if Float.is_nan x then x
    else if x > 0. then 1.
    else if x < 0. then -1.
    else 0.
  in
  prop "sign is -1, 0 or 1, and NaN for a NaN" ~count:20000
    Gen.(pair gen32 gen64)
    (fun (a, x) ->
      cover32 "f32" a;
      cover64 "f64" x;
      equal value32 (bits32 (sign (f a))) (K.f32 "sign" [| a |]);
      equal value64 (sign x) (K.f64 "sign" [| x |]))

let test_neg_abs =
  let kinds = [ "neg"; "abs" ] in
  let set32 = function
    | "neg" -> fun a -> a lxor 0x8000_0000
    | _ -> fun a -> a land 0x7FFF_FFFF
  in
  let set64 = function
    | "neg" -> fun x -> Int64.(float_of_bits (logxor (bits64 x) min_int))
    | _ -> fun x -> Int64.(float_of_bits (logand (bits64 x) max_int))
  in
  prop "neg and abs set a NaN's sign bit as a number's, its payload kept"
    ~count:40000
    Gen.(triple (kind kinds) gen32 gen64)
    (fun (k, a, x) ->
      cover_each kinds k;
      cover32 "f32" a;
      cover64 "f64" x;
      equal exact32 (set32 k a) (K.f32 k [| a |]);
      equal exact64 (set64 k x) (K.f64 k [| x |]))

(* The ranges' ends are π/2 and π rounded to the type, as the correctly rounded
   results there are. *)
let test_ranges =
  let kinds = [ "asin"; "acos"; "atan"; "atan2" ] in
  let range = function
    | "acos" -> (0., pi)
    | "atan2" -> (-.pi, pi)
    | _ -> (-.pi /. 2., pi /. 2.)
  in
  let unit32 =
    Gen.(
      frequency
        [
          (3, gen32);
          (1, map (fun i -> bits32 1. + i) (int_range (-4096) 4096));
          (1, map (fun i -> bits32 (-1.) + i) (int_range (-4096) 4096));
        ])
  in
  let unit64 =
    Gen.(
      frequency
        [
          (3, gen64);
          ( 2,
            let+ s = bool and+ i = int_range (-4096) 4096 in
            (if s then -1. else 1.) *. (1. +. Float.ldexp (Float.of_int i) (-52))
          );
        ])
  in
  let within name (lo, hi) r =
    if not (Float.is_nan r) then begin
      at_least float_exact ~than:lo r ~msg:name;
      at_most float_exact ~than:hi r ~msg:name
    end
  in
  prop "asin and atan lie in [-π/2, π/2], acos in [0, π], atan2 in [-π, π]"
    ~count:20000
    ~examples:
      [
        ("asin", (bits32 1., 0), (1., 0.));
        ("asin", (bits32 (-1.), 0), (-1., 0.));
        ("acos", (bits32 (-1.), 0), (-1., 0.));
        ("atan", (0x7F80_0000, 0), (inf, 0.));
        ("atan", (0xFF80_0000, 0), (ninf, 0.));
        ("atan2", (0, bits32 (-1.)), (0., -1.));
        ("atan2", (0x8000_0000, bits32 (-1.)), (-0., -1.));
      ]
    Gen.(triple (kind kinds) (pair unit32 gen32) (pair unit64 gen64))
    (fun (k, (a, b), (x, y)) ->
      cover_each kinds k;
      let args32, args64 =
        if k = "atan2" then ([| a; b |], [| x; y |]) else ([| a |], [| x |])
      in
      let lo, hi = range k in
      within "f32" (round_f32 lo, round_f32 hi) (f (K.f32 k args32));
      within "f64" (lo, hi) (K.f64 k args64))

(* The IEEE 754 maximum and minimum: of equal operands, a zero's sign bits
   combined, so that -0 orders below +0. *)
let test_maximum =
  let kinds = [ "maximum"; "minimum" ] in
  let max32 a b =
    if is_nan a || is_nan b then bits32 nan
    else if f a = f b then a land b
    else if f a > f b then a
    else b
  in
  let min32 a b =
    if is_nan a || is_nan b then bits32 nan
    else if f a = f b then a lor b
    else if f a < f b then a
    else b
  in
  let max64 x y =
    if Float.is_nan x || Float.is_nan y then nan
    else if x = y then Int64.float_of_bits (Int64.logand (bits64 x) (bits64 y))
    else if x > y then x
    else y
  in
  let min64 x y =
    if Float.is_nan x || Float.is_nan y then nan
    else if x = y then Int64.float_of_bits (Int64.logor (bits64 x) (bits64 y))
    else if x < y then x
    else y
  in
  let zeros =
    [
      ((0, 0), (0., 0.));
      ((0, 0x8000_0000), (0., -0.));
      ((0x8000_0000, 0), (-0., 0.));
      ((0x8000_0000, 0x8000_0000), (-0., -0.));
    ]
  in
  prop
    "maximum and minimum are IEEE 754's: NaN propagates and -0 orders below +0"
    ~count:40000
    ~examples:
      (List.concat_map
         (fun k -> List.map (fun (a, x) -> (k, a, x)) zeros)
         kinds)
    Gen.(triple (kind kinds) pair32 pair64)
    (fun (k, (a, b), (x, y)) ->
      cover_each kinds k;
      cover32 "f32 first" a;
      cover64 "f64 first" x;
      let r32, r64 = if k = "maximum" then (max32, max64) else (min32, min64) in
      equal value32 (r32 a b) (K.f32 k [| a; b |]);
      equal value64 (r64 x y) (K.f64 k [| x; y |]))

(* Triples whose exact a b + c lies below the least normal 2^emin of a format of
   precision p, as [round] rounds to it: a product near the subnormals with a
   zero or subnormal addend, or a larger one the addend cancels to its rounding
   error. There an fma that rounds twice, or rounds the product before the sum,
   gives other bits or a zero of the other sign. *)
let tiny ~emin ~p round =
  let open Gen in
  let signed =
    let+ neg = bool and+ u = float_range 1. 2. in
    if neg then -.u else u
  in
  let+ ua = signed
  and+ ub = signed
  and+ ea = int_range ((emin / 2) - 20) ((emin / 2) + 20)
  and+ t = int_range (emin - p - 2) (emin + p - 2)
  and+ s = float_range (-1.) 1.
  and+ addend = int_range 0 4 in
  let a = round (Float.ldexp ua ea) and b = round (Float.ldexp ub (t - ea)) in
  let ab = round (a *. b) and sub = round (Float.ldexp s emin) in
  let c =
    match addend with
    | 0 -> 0.
    | 1 -> -0.
    | 2 -> sub
    | 3 -> -.ab
    | _ -> round (sub -. ab)
  in
  (a, b, c)

let tiny32 =
  Gen.map
    (fun (a, b, c) -> (bits32 a, bits32 b, bits32 c))
    (tiny ~emin:(-126) ~p:24 (fun x -> f (bits32 x)))

let tiny64 = tiny ~emin:(-1022) ~p:53 Fun.id

(* The references round the exact value once, in integers: binary32's at its
   precision and least subnormal. *)
let test_fma =
  prop "fma is the product plus the third rounded once" ~count:20000
    Gen.(
      pair
        (frequency [ (1, triple gen32 gen32 gen32); (1, tiny32) ])
        (frequency [ (1, triple gen64 gen64 gen64); (1, tiny64) ]))
    (fun ((a, b, c), (x, y, z)) ->
      let r32 = bits32 (exact_fma ~p:24 ~least:(-149) (f a) (f b) (f c)) in
      let r64 = Exact.fma x y z in
      cover32 "f32 addend" c;
      cover "f32 subnormal result"
        (r32 land 0x7F80_0000 = 0 && r32 land 0x7F_FFFF <> 0);
      cover "f32 signed zero result" (r32 = 0x8000_0000);
      cover "f64 subnormal result" (r64 <> 0. && Float.abs r64 < Float.min_float);
      cover "f64 signed zero result" (r64 = 0. && Float.sign_bit r64);
      equal value32 r32 (K.f32 "fma" [| a; b; c |]);
      equal value64 r64 (K.f64 "fma" [| x; y; z |]))

let test_where =
  prop
    "where selects the second operand where the first is non-zero, the third \
     elsewhere"
    ~count:2000
    Gen.(pair (triple gen32 gen32 gen32) (triple gen64 gen64 gen64))
    (fun ((c, a, b), (z, x, y)) ->
      cover "f32 a zero condition" (f c = 0.);
      cover "f64 a zero condition" (z = 0.);
      equal exact32 (if f c <> 0. then a else b) (K.f32 "where" [| c; a; b |]);
      equal exact64 (if z <> 0. then x else y) (K.f64 "where" [| z; x; y |]))

(* Threefry-2x32-20: Random123's known answers, counter and key low word
   first. *)
let test_threefry =
  let pair lo hi = Int64.(logor (shift_left (of_int hi) 32) (of_int lo)) in
  cases "threefry gives Random123's known answers"
    ~id:(fun (c, k, _) -> strf "counter %016Lx key %016Lx" c k)
    [
      (pair 0 0, pair 0 0, pair 0x6b200159 0x99ba4efe);
      ( pair 0xffffffff 0xffffffff,
        pair 0xffffffff 0xffffffff,
        pair 0x1cb996fc 0xbb002be7 );
      ( pair 0x243f6a88 0x85a308d3,
        pair 0x13198a2e 0x03707344,
        pair 0xc4923a9c 0x483df7a0 );
    ]
    (fun (counter, key, want) -> equal int64 want (K.threefry counter key))

(* Kinds on integers

   Values are int64s holding a type's values: sign-extended from i32,
   zero-extended from u32. Each claim draws the type among the four and the kind
   among its own, 5000 cases for each kind at each type. *)

type ty = { name : string; bits : int; signed : bool }

let types =
  [
    { name = "i32"; bits = 32; signed = true };
    { name = "u32"; bits = 32; signed = false };
    { name = "i64"; bits = 64; signed = true };
    { name = "u64"; bits = 64; signed = false };
  ]

let wrap t x =
  if t.bits = 64 then x
  else if t.signed then Int64.of_int32 (Int64.to_int32 x)
  else Int64.logand x 0xFFFF_FFFFL

let cmp t a b =
  if t.signed then Int64.compare a b else Int64.unsigned_compare a b

let ipow t a e =
  let rec go r b n =
    if n = 0L then r
    else
      let r = if Int64.logand n 1L = 1L then Int64.mul r b else r in
      go r (Int64.mul b b) (Int64.shift_right_logical n 1)
  in
  if t.signed && Int64.compare e 0L < 0 then
    if a = 1L then 1L
    else if a = -1L then if Int64.logand e 1L = 1L then -1L else 1L
    else 0L
  else wrap t (go 1L a (wrap { t with signed = false } e))

(* prog.mli's integer kinds, restated. *)
let int_meaning t kind args =
  let w = wrap t and b2i c = if c then 1L else 0L in
  let minus_one = w (-1L) in
  match (kind, args) with
  | "neg", [ a ] -> w (Int64.neg a)
  | "abs", [ a ] ->
      if t.signed && Int64.compare a 0L < 0 then w (Int64.neg a) else a
  | "sign", [ a ] ->
      if a = 0L then 0L
      else if t.signed && Int64.compare a 0L < 0 then -1L
      else 1L
  | "recip", [ a ] -> if a = 1L || (t.signed && a = -1L) then a else 0L
  | "add", [ a; b ] -> w (Int64.add a b)
  | "sub", [ a; b ] -> w (Int64.sub a b)
  | "mul", [ a; b ] -> w (Int64.mul a b)
  | "fma", [ a; b; c ] -> w (Int64.add (Int64.mul a b) c)
  | "idiv", [ a; b ] ->
      if b = 0L then 0L
      else if t.signed && b = minus_one then w (Int64.neg a)
      else if t.signed then Int64.div a b
      else Int64.unsigned_div a b
  | "mod", [ a; b ] ->
      if b = 0L then a
      else if t.signed && b = minus_one then 0L
      else if t.signed then Int64.rem a b
      else Int64.unsigned_rem a b
  | "pow", [ a; e ] -> ipow t a e
  | "maximum", [ a; b ] -> if cmp t a b < 0 then b else a
  | "minimum", [ a; b ] -> if cmp t b a < 0 then b else a
  | "and", [ a; b ] -> Int64.logand a b
  | "or", [ a; b ] -> Int64.logor a b
  | "xor", [ a; b ] -> w (Int64.logxor a b)
  | "equal", [ a; b ] -> b2i (a = b)
  | "not_equal", [ a; b ] -> b2i (a <> b)
  | "less", [ a; b ] -> b2i (cmp t a b < 0)
  | "less_equal", [ a; b ] -> b2i (cmp t a b <= 0)
  | "where", [ c; a; b ] -> if c <> 0L then a else b
  | _ -> invalid_arg kind

let arity = function
  | "neg" | "abs" | "sign" | "recip" -> 1
  | "fma" | "where" -> 3
  | _ -> 2

let gen_int t =
  let open Gen in
  let extremes =
    List.map (wrap t)
      (if t.bits = 64 then
         [ 0L; 1L; -1L; 2L; -2L; Int64.min_int; Int64.max_int ]
       else [ 0L; 1L; -1L; 2L; -2L; 0x8000_0000L; 0x7FFF_FFFFL ])
  in
  frequency
    [
      (3, map (wrap t) int64);
      (2, of_list extremes);
      (2, map (fun i -> wrap t (Int64.of_int i)) (int_range (-40) 40));
    ]

(* A type and three of its values; a kind reads its arity's first. *)
let int_case =
  let open Gen in
  let* t = of_list ~pp:(fun ppf t -> Format.pp_print_string ppf t.name) types in
  let+ args = array ~size:(constant 3) (gen_int t) in
  (t, args)

(* The kind [k] at [t] of [args] against its meaning. *)
let int_law kinds (k, (t, args)) =
  let args = Array.sub args 0 (arity k) in
  cover_each kinds k;
  cover_each (List.map (fun t -> t.name) types) t.name;
  cover "an extreme"
    (Array.exists (fun a -> a = wrap t Int64.min_int || a = 0L) args);
  equal int64 (int_meaning t k (Array.to_list args)) (K.int t.name k args)

(* The least value by -1, and by zero, at each signed type. *)
let least_by =
  List.concat_map
    (fun t ->
      let least = wrap t Int64.min_int in
      [ (t, [| least; -1L; 0L |]); (t, [| least; 0L; 0L |]) ])
    (List.filter (fun t -> t.signed) types)

let test_int_wrap =
  let kinds = [ "add"; "sub"; "mul"; "fma"; "neg" ] in
  prop "add, sub, mul, fma and neg wrap" ~count:100000
    Gen.(pair (kind kinds) int_case)
    (int_law kinds)

let test_int_abs =
  prop "abs wraps: the least value is its own" ~count:20000
    Gen.(pair (kind [ "abs" ]) int_case)
    (int_law [ "abs" ])

let test_int_sign =
  prop "sign is -1, 0 or 1" ~count:20000
    Gen.(pair (kind [ "sign" ]) int_case)
    (int_law [ "sign" ])

let test_int_recip =
  prop "recip is x at 1 and -1, and 0 elsewhere" ~count:20000
    Gen.(pair (kind [ "recip" ]) int_case)
    (int_law [ "recip" ])

let test_idiv =
  prop
    "idiv truncates toward zero, is 0 by zero, and the least value by -1 is \
     itself"
    ~count:20000
    ~examples:(List.map (fun c -> ("idiv", c)) least_by)
    Gen.(pair (kind [ "idiv" ]) int_case)
    (int_law [ "idiv" ])

let test_mod_int =
  prop
    "mod takes the dividend's sign, is the dividend by zero, and a is b idiv a \
     b + mod a b"
    ~count:20000
    ~examples:(List.map (fun c -> ("mod", c)) least_by)
    Gen.(pair (kind [ "mod" ]) int_case)
    (fun ((_, (t, args)) as case) ->
      int_law [ "mod" ] case;
      let a = args.(0) and b = args.(1) in
      let q = K.int t.name "idiv" [| a; b |] in
      let r = K.int t.name "mod" [| a; b |] in
      equal int64 a (wrap t (Int64.add (Int64.mul b q) r)))

let test_int_pow =
  prop "pow wraps, and by a negative exponent is 0 but of 1 and -1" ~count:20000
    Gen.(pair (kind [ "pow" ]) int_case)
    (fun ((_, (t, args)) as case) ->
      cover "a base of 1 or -1, a negative exponent"
        (t.signed
        && (args.(0) = 1L || args.(0) = -1L)
        && Int64.compare args.(1) 0L < 0);
      int_law [ "pow" ] case)

let test_bitwise =
  let kinds = [ "and"; "or"; "xor" ] in
  prop "and, or and xor are bitwise" ~count:60000
    Gen.(pair (kind kinds) int_case)
    (int_law kinds)

let test_int_order =
  let kinds =
    [ "maximum"; "minimum"; "equal"; "not_equal"; "less"; "less_equal" ]
  in
  prop "maximum, minimum and the comparisons order unsigned types unsigned"
    ~count:120000
    Gen.(pair (kind kinds) int_case)
    (int_law kinds)

let test_int_where =
  prop "where selects the second operand where the first is non-zero"
    ~count:20000
    Gen.(pair (kind [ "where" ]) int_case)
    (int_law [ "where" ])

(* Domains *)

let test_float_compare =
  let kinds = [ "equal"; "not_equal"; "less"; "less_equal" ] in
  let holds = function
    | "equal" -> ( = )
    | "not_equal" -> ( <> )
    | "less" -> ( < )
    | _ -> ( <= )
  in
  let flag p = if p then 1. else 0. in
  let zeros =
    [ ((0, 0x8000_0000), (0., -0.)); ((0x8000_0000, 0), (-0., 0.)) ]
  in
  prop "float comparisons are IEEE 754's: -0 equals +0, a NaN unequal to all"
    ~count:80000
    ~examples:
      (List.concat_map
         (fun k -> List.map (fun (a, x) -> (k, a, x)) zeros)
         kinds)
    Gen.(triple (kind kinds) pair32 pair64)
    (fun (k, (a, b), (x, y)) ->
      cover_each kinds k;
      cover32 "f32 first" a;
      cover64 "f64 first" x;
      equal exact32 (bits32 (flag (holds k (f a) (f b)))) (K.f32 k [| a; b |]);
      equal exact64 (flag (holds k x y)) (K.f64 k [| x; y |]))

(* Complex numbers *)

(* A part of magnitude in [2^-40, 2^40], either sign. *)
let part =
  let open Gen in
  let* m = float_range 1. 2. in
  let* e = int_range (-40) 40 in
  let+ s = bool in
  Float.ldexp (if s then -.m else m) e

let parts = Gen.pair part part
let c64_of (re, im) = [| bits32 (round_f32 re); bits32 (round_f32 im) |]
let pair_of a = (f a.(0), f a.(1))
let c64_call k ops = pair_of (K.c64 k (Array.concat (List.map c64_of ops)))

let c128_call k ops =
  let r = K.c128 k (Array.concat (List.map (fun (a, b) -> [| a; b |]) ops)) in
  (r.(0), r.(1))

let f32_pair (re, im) = (round_f32 re, round_f32 im)
let both_parts = pair value64 value64

(* The order: by the real part, then the imaginary part, -0 below +0 for the
   extremes; a NaN part ranks as NaN. The domain is small enough to check whole:
   every pair of the numbers these six parts make. *)
let small = Gen.of_list [ -1.; -0.; 0.; 1.; inf; nan ]
let small_z = Gen.pair small small
let has_nan (a, b) = Float.is_nan a || Float.is_nan b

let test_complex_compare =
  let kinds = [ "equal"; "not_equal"; "less"; "less_equal" ] in
  let flag p = if p then (1., 0.) else (0., 0.) in
  let meaning k a b =
    let n = has_nan a || has_nan b in
    let eq = fst a = fst b && snd a = snd b in
    let lt = fst a < fst b || (fst a = fst b && snd a < snd b) in
    match k with
    | "equal" -> flag eq
    | "not_equal" -> flag (not eq)
    | "less" -> flag ((not n) && lt)
    | _ -> flag ((not n) && (lt || eq))
  in
  prop "complex comparisons compare the real parts, then the imaginary ones"
    ~count:5184
    Gen.(triple (kind kinds) small_z small_z)
    (fun (k, a, b) ->
      equal both_parts (meaning k a b) (c128_call k [ a; b ]);
      equal both_parts (meaning k a b) (c64_call k [ a; b ]))

let test_complex_maximum =
  let kinds = [ "maximum"; "minimum" ] in
  let cmp (ar, ai) (br, bi) =
    let one x y =
      if x = y then compare (Float.sign_bit y) (Float.sign_bit x)
      else compare x y
    in
    let c = one ar br in
    if c <> 0 then c else one ai bi
  in
  let meaning k a b =
    if has_nan a then a
    else if has_nan b then b
    else if k = "maximum" then if cmp a b >= 0 then a else b
    else if cmp a b <= 0 then a
    else b
  in
  prop
    "complex maximum and minimum follow the order, a NaN operand the first of \
     two"
    ~count:2592
    Gen.(triple (kind kinds) small_z small_z)
    (fun (k, a, b) ->
      equal both_parts (meaning k a b) (c128_call k [ a; b ]);
      equal both_parts (meaning k a b) (c64_call k [ a; b ]))

(* Results *)

(* Annex F's rows, through a kind's loop at one element and over a vector, every
   lane alike. *)
let test_annex_f =
  let lanes (type s) (k : (float, s) Bigarray.kind) kind args =
    let fill v = Bigarray.Array1.init k Bigarray.c_layout 37 (fun _ -> v) in
    let y = fill 0. in
    let z = if Array.length args > 1 then args.(1) else 0. in
    K.run kind (fill args.(0)) (fill z) y;
    List.init 37 (fun i -> y.{i})
  in
  let at64 kind args want =
    equal value64 want (K.f64 kind args);
    equal (list value64)
      (List.init 37 (fun _ -> want))
      (lanes Bigarray.float64 kind args)
  in
  let at32 kind args want =
    equal value64 want (f (K.f32 kind (Array.map bits32 args)));
    equal (list value64)
      (List.init 37 (fun _ -> want))
      (lanes Bigarray.float32 kind args)
  in
  cases
    "Annex F's special values are exact, at one lane and at every lane of a \
     vector"
    ~id:Annex_f.id Annex_f.rows (fun (w, kind, args, want) ->
      match (w : Annex_f.width) with
      | F64 -> at64 kind args want
      | F32 -> at32 kind args want
      | Both ->
          at64 kind args want;
          at32 kind (Array.map round_f32 args) (round_f32 want))

(* The float kinds of one or two operands, as nx_kinds.h names them: first those
   of one operand that no IEEE 754 operation is. *)
let transcendental_kinds =
  [
    "exp";
    "exp2";
    "expm1";
    "log";
    "log2";
    "log1p";
    "sin";
    "cos";
    "tan";
    "asin";
    "acos";
    "atan";
    "sinh";
    "cosh";
    "tanh";
    "erf";
  ]

let unary_kinds =
  transcendental_kinds
  @ [ "neg"; "recip"; "abs"; "sign"; "sqrt"; "floor"; "ceil"; "round"; "trunc" ]

let binary_kinds =
  [ "add"; "sub"; "mul"; "fdiv"; "mod"; "pow"; "atan2"; "maximum"; "minimum" ]

let signalling32 b = is_nan b && b land 0x0040_0000 = 0

let signalling64 x =
  Float.is_nan x && Int64.logand (bits64 x) 0x0008_0000_0000_0000L = 0L

let test_quiet =
  let kinds = unary_kinds @ binary_kinds in
  let quiet32 = Gen.such_that (fun b -> not (signalling32 b)) gen32 in
  let quiet64 = Gen.such_that (fun x -> not (signalling64 x)) gen64 in
  prop "a NaN from operands with no signalling NaN is quiet" ~count:20000
    Gen.(triple (kind kinds) (pair quiet32 quiet32) (pair quiet64 quiet64))
    (fun (k, (a, b), (x, y)) ->
      let two = List.mem k binary_kinds in
      let r32 = K.f32 k (if two then [| a; b |] else [| a |]) in
      let r64 = K.f64 k (if two then [| x; y |] else [| x |]) in
      let nan_in32 = is_nan a || (two && is_nan b) in
      let nan_in64 = Float.is_nan x || (two && Float.is_nan y) in
      cover "f32 NaN from numbers" (is_nan r32 && not nan_in32);
      cover "f32 NaN from a quiet NaN" (is_nan r32 && nan_in32);
      cover "f64 NaN from numbers" (Float.is_nan r64 && not nan_in64);
      cover "f64 NaN from a quiet NaN" (Float.is_nan r64 && nan_in64);
      satisfies ~claim:"not a signalling NaN" exact32
        (fun r -> not (signalling32 r))
        r32;
      satisfies ~claim:"not a signalling NaN" exact64
        (fun r -> not (signalling64 r))
        r64)

(* Complex numbers' arithmetic *)

let gen_z64 = Gen.pair gen64 gen64

(* c64 operands travel as their parts' bits. *)
let gen_b32 =
  Gen.with_pp
    (fun ppf (re, im) -> Format.fprintf ppf "(0x%08x, 0x%08x)" re im)
    (Gen.pair gen32 gen32)

let c64_bits k (re, im) =
  let r = K.c64 k [| re; im |] in
  (r.(0), r.(1))

let both_parts32 = pair value32 value32

let test_complex_parts =
  let kinds = [ "add"; "sub"; "neg" ] in
  prop "add, sub and neg act on each part, neg bit for bit" ~count:300
    Gen.(triple (kind kinds) (pair gen_b32 gen_b32) (pair gen_z64 gen_z64))
    (fun (k, ((ar, ai), (br, bi)), ((xr, xi), (yr, yi))) ->
      cover_each kinds k;
      if k = "neg" then begin
        equal (array exact32)
          [| K.f32 "neg" [| ar |]; K.f32 "neg" [| ai |] |]
          (K.c64 "neg" [| ar; ai |]);
        equal (array exact64)
          [| K.f64 "neg" [| xr |]; K.f64 "neg" [| xi |] |]
          (K.c128 "neg" [| xr; xi |])
      end
      else begin
        equal (array value32)
          [| K.f32 k [| ar; br |]; K.f32 k [| ai; bi |] |]
          (K.c64 k [| ar; ai; br; bi |]);
        equal (array value64)
          [| K.f64 k [| xr; yr |]; K.f64 k [| xi; yi |] |]
          (K.c128 k [| xr; xi; yr; yi |])
      end)

(* Bounds against the exact product and quotient, as double-doubles, where parts
   lie in [2^-40, 2^40]. *)
let test_complex_mul =
  prop "mul is within 2u" ~count:20000
    Gen.(pair parts parts)
    (fun (a, b) ->
      at_most float_exact ~than:2. ~msg:"c128"
        (normwise 0x1p-53 (c128_call "mul" [ a; b ]) (exact_mul a b));
      let a = f32_pair a and b = f32_pair b in
      at_most float_exact ~than:2. ~msg:"c64"
        (normwise 0x1p-24 (c64_call "mul" [ a; b ]) (exact_mul a b)))

(* A result with a NaN part, or a reference with one, is past every complex
   bound. *)
let test_normwise_nan =
  let want = exact_mul (1.5, -0.25) (0.75, 2.) in
  cases "a NaN part is past every normwise bound"
    ~id:(fun (g, _) -> strf "%h %h" (fst g) (snd g))
    [
      ((nan, nan), want);
      ((nan, 1.625), want);
      ((1.625, nan), want);
      ((1.625, 2.8125), ((nan, 0.), (2.8125, 0.)));
    ]
    (fun (g, want) -> equal float_exact inf (normwise 0x1p-53 g want))

let test_complex_fdiv =
  prop "fdiv is within 4u" ~count:20000
    Gen.(pair parts parts)
    (fun (a, b) ->
      at_most float_exact ~than:4. ~msg:"c128"
        (normwise 0x1p-53 (c128_call "fdiv" [ a; b ]) (exact_div a b));
      let a = f32_pair a and b = f32_pair b in
      at_most float_exact ~than:4. ~msg:"c64"
        (normwise 0x1p-24 (c64_call "fdiv" [ a; b ]) (exact_div a b)))

(* Dividends whose parts are subnormal over divisors whose larger part is
   within 8 times the least normal: the quotients are normal, within 4u. The
   exact quotient is taken of both operands scaled by 2^600, the same
   quotient, so that no product of the reference underflows. *)
let test_complex_subnormal =
  let scaled (re, im) = (Float.ldexp re 600, Float.ldexp im 600) in
  let check name u least call round =
    prop (strf "fdiv %s of subnormal dividends within 4u" name) ~count:2000
      Gen.(
        let signed m =
          let+ v = float_range 0. m and+ s = bool in
          if s then -.v else v
        in
        let* a = pair (signed least) (signed least) in
        let* p = float_range least (8. *. least)
        and+ q = signed (8. *. least) in
        let+ swap = bool in
        (a, if swap then (q, p) else (p, q)))
      (fun (a, (br, bi)) ->
        cover "the real part the larger" (Float.abs br >= Float.abs bi);
        cover "the imaginary part the larger" (Float.abs br < Float.abs bi);
        let a = round a and b = round (br, bi) in
        let z = exact_div (scaled a) (scaled b) in
        at_most float_exact ~than:4. (normwise u (call a b) z))
  in
  [
    check "c64" 0x1p-24 0x1p-126 (fun a b -> c64_call "fdiv" [ a; b ]) f32_pair;
    check "c128" 0x1p-53 Float.min_float
      (fun a b -> c128_call "fdiv" [ a; b ])
      Fun.id;
  ]

(* Parts whose squares overflow, over quotients of normal parts: Smith's
   algorithm divides the divisor's smaller part by its larger, so no
   intermediate overflows. At c128 the exact quotient is taken of both operands
   scaled by 2^-600, so that no product of the reference overflows. *)
let test_complex_large =
  let large lo hi =
    Gen.(
      let+ m = float_range 1. 1.5 and+ e = int_range lo hi and+ s = bool in
      Float.ldexp (if s then -.m else m) e)
  in
  let operands lo hi =
    Gen.(
      pair (pair (large lo hi) (large lo hi)) (pair (large lo hi) (large lo hi)))
  in
  let scaled (re, im) = (Float.ldexp re (-600), Float.ldexp im (-600)) in
  let real_larger (re, im) = Float.abs re >= Float.abs im in
  prop "fdiv is within 4u of parts whose squares overflow" ~count:2000
    Gen.(pair (operands 960 1022) (operands 100 126))
    (fun ((a, b), (c, d)) ->
      cover "c128's divisor real part the larger" (real_larger b);
      cover "c128's divisor imaginary part the larger" (not (real_larger b));
      cover "c64's divisor real part the larger" (real_larger d);
      cover "c64's divisor imaginary part the larger" (not (real_larger d));
      at_most float_exact ~than:4. ~msg:"c128"
        (normwise 0x1p-53
           (c128_call "fdiv" [ a; b ])
           (exact_div (scaled a) (scaled b)));
      let c = f32_pair c and d = f32_pair d in
      at_most float_exact ~than:4. ~msg:"c64"
        (normwise 0x1p-24 (c64_call "fdiv" [ c; d ]) (exact_div c d)))

(* A quotient whose imaginary part is 0 while its real part overflows. *)
let test_complex_smith =
  test "fdiv keeps a zero part beside an infinite one" (fun () ->
      let want = (ninf, 0.) in
      equal both_parts want
        (c128_call "fdiv" [ (0., 0x1p-50); (0., -.Float.ldexp 1. (-1074)) ]);
      equal both_parts want (c64_call "fdiv" [ (0., 0x1p-20); (0., -0x1p-149) ]))

let test_complex_recip =
  prop "recip x is fdiv (1 + 0i) x" ~count:20000
    Gen.(pair gen_z64 gen_b32)
    (fun (z, (re, im)) ->
      equal both_parts
        (c128_call "fdiv" [ (1., 0.); z ])
        (c128_call "recip" [ z ]);
      equal (array value32)
        (K.c64 "fdiv" [| bits32 1.; 0; re; im |])
        (K.c64 "recip" [| re; im |]))

(* A NaN part in an operand of Mul, Fdiv or Recip makes both parts NaN. *)
let test_complex_nan =
  let nan_parts =
    satisfies ~claim:"both parts NaN" both_parts (fun (re, im) ->
        Float.is_nan re && Float.is_nan im)
  in
  prop "a NaN part makes both parts of mul, fdiv and recip NaN"
    Gen.(triple parts parts (int_range 0 3))
    (fun ((ar, ai), (br, bi), at) ->
      let ops = [| ar; ai; br; bi |] in
      ops.(at) <- nan;
      let a = (ops.(0), ops.(1)) and b = (ops.(2), ops.(3)) in
      cover "the NaN in the divisor" (at >= 2);
      nan_parts (c128_call "mul" [ a; b ]);
      nan_parts (c128_call "fdiv" [ a; b ]);
      nan_parts (c64_call "mul" [ a; b ]);
      nan_parts (c64_call "fdiv" [ a; b ]);
      if at >= 2 then begin
        nan_parts (c128_call "recip" [ b ]);
        nan_parts (c64_call "recip" [ b ])
      end)

(* The complex transcendental kinds *)

let complex_kinds =
  [
    "exp";
    "exp2";
    "log";
    "log2";
    "sqrt";
    "sign";
    "sin";
    "cos";
    "tan";
    "asin";
    "acos";
    "atan";
    "sinh";
    "cosh";
    "tanh";
    "asinh";
    "acosh";
    "atanh";
  ]

(* |z' - z| / |z| of the moduli alone, in units of [u], infinite where a part of
   [z'] is NaN, as [normwise] reads it, so that no bound passes it. *)
let modulus_error u (re, im)
    (((zr, _), (zi, _)) : (float * float) * (float * float)) =
  if Float.is_nan re || Float.is_nan im then Float.infinity
  else
    let m = Float.hypot zr zi in
    Float.abs (Float.hypot re im -. m) /. m /. u

(* C99 Annex G's special values, G.6.1 to G.6.4, at c128 and c64. Where it
   leaves a sign unspecified, the value is nx_kinds.h's choice. A NaN part is
   any NaN. *)

let special_values =
  [
    ("exp", (0., 0.), (1., 0.)); ("exp", (-0., 0.), (1., 0.));
    ("exp", (0., -0.), (1., -0.)); ("exp", (2., inf), (nan, nan));
    ("exp", (2., nan), (nan, nan)); ("exp", (inf, 0.), (inf, 0.));
    ("exp", (ninf, 1.), (0., 0.)); ("exp", (ninf, 3.), (-0., 0.));
    ("exp", (inf, 1.), (inf, inf)); ("exp", (inf, 3.), (ninf, inf));
    ("exp", (ninf, inf), (0., 0.)); ("exp", (ninf, ninf), (0., -0.));
    ("exp", (inf, inf), (inf, nan)); ("exp", (ninf, nan), (0., 0.));
    ("exp", (inf, nan), (inf, nan)); ("exp", (nan, 0.), (nan, 0.));
    ("exp", (nan, 1.), (nan, nan)); ("exp", (nan, nan), (nan, nan));
    ("exp2", (0., 0.), (1., 0.)); ("exp2", (ninf, 1.), (0., 0.));
    ("exp2", (inf, 1.), (inf, inf)); ("exp2", (nan, 0.), (nan, 0.));
    ("exp2", (2., inf), (nan, nan)); ("exp2", (3., 0.), (8., 0.));
    ("log", (-0., 0.), (ninf, pi)); ("log", (0., 0.), (ninf, 0.));
    ("log", (0., -0.), (ninf, -0.)); ("log", (2., inf), (inf, pi /. 2.));
    ("log", (2., nan), (nan, nan)); ("log", (ninf, 1.), (inf, pi));
    ("log", (inf, 1.), (inf, 0.)); ("log", (ninf, inf), (inf, 3. *. pi /. 4.));
    ("log", (inf, inf), (inf, pi /. 4.)); ("log", (inf, nan), (inf, nan));
    ("log", (ninf, nan), (inf, nan)); ("log", (nan, 1.), (nan, nan));
    ("log", (nan, inf), (inf, nan)); ("log", (nan, nan), (nan, nan));
    ("log2", (0., 0.), (ninf, 0.)); ("log2", (inf, 1.), (inf, 0.));
    ("log2", (nan, inf), (inf, nan)); ("log2", (8., 0.), (3., 0.));
    ("sqrt", (0., 0.), (0., 0.)); ("sqrt", (-0., 0.), (0., 0.));
    ("sqrt", (2., inf), (inf, inf)); ("sqrt", (nan, inf), (inf, inf));
    ("sqrt", (2., nan), (nan, nan)); ("sqrt", (ninf, 1.), (0., inf));
    ("sqrt", (inf, 1.), (inf, 0.)); ("sqrt", (ninf, nan), (nan, inf));
    ("sqrt", (inf, nan), (inf, nan)); ("sqrt", (nan, 1.), (nan, nan));
    ("sqrt", (nan, nan), (nan, nan));
    ("sinh", (0., 0.), (0., 0.)); ("sinh", (0., inf), (0., nan));
    ("sinh", (0., nan), (0., nan)); ("sinh", (2., inf), (nan, nan));
    ("sinh", (2., nan), (nan, nan)); ("sinh", (inf, 0.), (inf, 0.));
    ("sinh", (inf, 1.), (inf, inf)); ("sinh", (inf, 3.), (ninf, inf));
    ("sinh", (inf, inf), (inf, nan)); ("sinh", (inf, nan), (inf, nan));
    ("sinh", (nan, 0.), (nan, 0.)); ("sinh", (nan, 1.), (nan, nan));
    ("sinh", (nan, nan), (nan, nan));
    ("cosh", (0., 0.), (1., 0.)); ("cosh", (0., inf), (nan, 0.));
    ("cosh", (0., nan), (nan, 0.)); ("cosh", (2., inf), (nan, nan));
    ("cosh", (2., nan), (nan, nan)); ("cosh", (inf, 0.), (inf, 0.));
    ("cosh", (inf, 1.), (inf, inf)); ("cosh", (inf, 3.), (ninf, inf));
    ("cosh", (inf, inf), (inf, nan)); ("cosh", (inf, nan), (inf, nan));
    ("cosh", (nan, 0.), (nan, 0.)); ("cosh", (nan, 1.), (nan, nan));
    ("cosh", (nan, nan), (nan, nan));
    ("tanh", (0., 0.), (0., 0.)); ("tanh", (2., inf), (nan, nan));
    ("tanh", (2., nan), (nan, nan)); ("tanh", (inf, 1.), (1., 0.));
    ("tanh", (inf, 2.), (1., -0.)); ("tanh", (inf, inf), (1., 0.));
    ("tanh", (inf, nan), (1., 0.)); ("tanh", (nan, 0.), (nan, 0.));
    ("tanh", (nan, 1.), (nan, nan)); ("tanh", (nan, nan), (nan, nan));
    ("acos", (0., 0.), (pi /. 2., -0.)); ("acos", (-0., 0.), (pi /. 2., -0.));
    ("acos", (0., nan), (pi /. 2., nan)); ("acos", (2., inf), (pi /. 2., ninf));
    ("acos", (2., nan), (nan, nan)); ("acos", (ninf, 1.), (pi, ninf));
    ("acos", (inf, 1.), (0., ninf));
    ("acos", (ninf, inf), (3. *. pi /. 4., ninf));
    ("acos", (inf, inf), (pi /. 4., ninf)); ("acos", (inf, nan), (nan, ninf));
    ("acos", (nan, 1.), (nan, nan)); ("acos", (nan, inf), (nan, ninf));
    ("acos", (nan, nan), (nan, nan));
    ("asin", (0., 0.), (0., 0.)); ("asin", (-0., 0.), (-0., 0.));
    ("asin", (0., nan), (0., nan)); ("asin", (2., nan), (nan, nan));
    ("asin", (inf, 1.), (pi /. 2., inf)); ("asin", (1., inf), (0., inf));
    ("asin", (inf, inf), (pi /. 4., inf)); ("asin", (nan, inf), (nan, inf));
    ("asin", (inf, nan), (nan, inf)); ("asin", (nan, 1.), (nan, nan));
    ("asin", (nan, 0.), (nan, nan)); ("asin", (nan, nan), (nan, nan));
    ("atan", (0., 0.), (0., 0.)); ("atan", (nan, 0.), (nan, 0.));
    ("atan", (inf, 1.), (pi /. 2., 0.)); ("atan", (1., inf), (pi /. 2., 0.));
    ("atan", (0., 1.), (0., inf)); ("atan", (inf, nan), (pi /. 2., 0.));
    ("atan", (nan, inf), (nan, 0.)); ("atan", (2., nan), (nan, nan));
    ("atan", (nan, 2.), (nan, nan)); ("atan", (nan, nan), (nan, nan));
    ("sin", (0., 0.), (0., 0.)); ("sin", (inf, 0.), (nan, 0.));
    ("cos", (0., 0.), (1., -0.)); ("cos", (inf, 0.), (nan, -0.));
    ("tan", (0., 0.), (0., 0.)); ("tan", (0., inf), (0., 1.));
    ("sign", (inf, 2.), (1., 0.));
    ("sign", (inf, ninf), (sqrt 0.5, -.sqrt 0.5));
    ("sign", (0., -0.), (0., -0.)); ("sign", (nan, 1.), (nan, nan));
    ("sign", (2., nan), (nan, nan));
    ("asinh", (0., 0.), (0., 0.)); ("asinh", (1., inf), (inf, pi /. 2.));
    ("asinh", (1., nan), (nan, nan)); ("asinh", (inf, 1.), (inf, 0.));
    ("asinh", (inf, inf), (inf, pi /. 4.)); ("asinh", (inf, nan), (inf, nan));
    ("asinh", (nan, 0.), (nan, 0.)); ("asinh", (nan, 1.), (nan, nan));
    ("asinh", (nan, inf), (inf, nan)); ("asinh", (nan, nan), (nan, nan));
    ("acosh", (0., 0.), (0., pi /. 2.)); ("acosh", (-0., 0.), (0., pi /. 2.));
    ("acosh", (2., inf), (inf, pi /. 2.)); ("acosh", (2., nan), (nan, nan));
    ("acosh", (ninf, 1.), (inf, pi)); ("acosh", (inf, 1.), (inf, 0.));
    ("acosh", (ninf, inf), (inf, 3. *. pi /. 4.));
    ("acosh", (inf, inf), (inf, pi /. 4.)); ("acosh", (inf, nan), (inf, nan));
    ("acosh", (ninf, nan), (inf, nan)); ("acosh", (nan, 1.), (nan, nan));
    ("acosh", (nan, inf), (inf, nan)); ("acosh", (nan, nan), (nan, nan));
    ("atanh", (0., 0.), (0., 0.)); ("atanh", (0., nan), (0., nan));
    ("atanh", (1., 0.), (inf, 0.)); ("atanh", (2., inf), (0., pi /. 2.));
    ("atanh", (2., nan), (nan, nan)); ("atanh", (inf, 1.), (0., pi /. 2.));
    ("atanh", (inf, inf), (0., pi /. 2.)); ("atanh", (inf, nan), (0., nan));
    ("atanh", (nan, 1.), (nan, nan)); ("atanh", (nan, inf), (0., pi /. 2.));
    ("atanh", (nan, nan), (nan, nan));
  ]

let special_id (k, (x, y), _) = strf "%s %h %h" k x y

(* A row at c128, and at c64 with its parts rounded to binary32. *)
let special (kind, z, (re, im)) =
  equal both_parts (re, im) (c128_call kind [ z ]);
  equal both_parts (round_f32 re, round_f32 im) (c64_call kind [ f32_pair z ])

let test_complex_special =
  cases "the transcendental kinds give Annex G's special values" ~id:special_id
    special_values special

(* C99 G.6 defines csin, ccos and ctan by the hyperbolic kinds: sin z = -i sinh
   (iz), cos z = cosh (iz), tan z = -i tanh (iz). At z = -i w each gives its
   hyperbolic kind's special value at w, turned by -i for sin and tan. *)
let rotated =
  let minus_i (re, im) = (im, Float.neg re) in
  List.filter_map
    (fun (k, w, r) ->
      match k with
      | "sinh" -> Some ("sin", minus_i w, minus_i r)
      | "cosh" -> Some ("cos", minus_i w, r)
      | "tanh" -> Some ("tan", minus_i w, minus_i r)
      | _ -> None)
    special_values

let test_complex_rotated =
  cases "sin, cos and tan give Annex G's values of sinh, cosh and tanh at iz"
    ~id:special_id rotated special

(* A signalling NaN part beside an infinity or a part past ASIN_BIG: the
   values of a quiet NaN of its sign. A compiler may fold asin's and acos's
   test of both parts against ASIN_BIG into a max, which a signalling NaN
   turns to NaN. *)

let snan = Int64.float_of_bits 0x7FF0_0000_0000_0001L
let nsnan = Int64.float_of_bits 0xFFF0_0000_0000_0001L
let far = 0x1p40

let signalling_values =
  [
    ("asin", (inf, snan), (nan, inf)); ("asin", (ninf, snan), (nan, inf));
    ("asin", (inf, nsnan), (nan, ninf)); ("asin", (snan, inf), (nan, inf));
    ("asin", (snan, ninf), (nan, ninf)); ("asin", (nsnan, inf), (nan, inf));
    ("asin", (far, snan), (nan, nan)); ("asin", (snan, far), (nan, nan));
    ("acos", (inf, snan), (nan, ninf)); ("acos", (ninf, snan), (nan, ninf));
    ("acos", (inf, nsnan), (nan, inf)); ("acos", (snan, inf), (nan, ninf));
    ("acos", (snan, ninf), (nan, inf)); ("acos", (far, snan), (nan, nan));
    ("acos", (snan, far), (nan, nan));
    ("asinh", (inf, snan), (inf, nan)); ("asinh", (ninf, snan), (ninf, nan));
    ("asinh", (inf, nsnan), (inf, nan)); ("asinh", (snan, inf), (inf, nan));
    ("asinh", (nsnan, inf), (ninf, nan)); ("asinh", (snan, ninf), (inf, nan));
    ("asinh", (far, snan), (nan, nan)); ("asinh", (snan, far), (nan, nan));
    ("acosh", (inf, snan), (inf, nan)); ("acosh", (ninf, snan), (inf, nan));
    ("acosh", (inf, nsnan), (inf, nan)); ("acosh", (snan, inf), (inf, nan));
    ("acosh", (snan, ninf), (inf, nan)); ("acosh", (far, snan), (nan, nan));
    ("acosh", (snan, far), (nan, nan));
  ]

(* A part's binary32 bits, a signalling NaN kept signalling. *)
let signalling_bits x =
  if bits64 x = bits64 snan then 0x7F80_0001
  else if bits64 x = bits64 nsnan then 0xFF80_0001
  else bits32 (round_f32 x)

let test_complex_signalling =
  cases "a signalling NaN part gives a quiet NaN's values"
    ~id:(fun (k, (x, y), _) ->
      strf "%s 0x%016Lx 0x%016Lx" k (bits64 x) (bits64 y))
    signalling_values
    (fun (kind, ((x, y) as z), (re, im)) ->
      equal both_parts (re, im) (c128_call kind [ z ]);
      equal both_parts
        (round_f32 re, round_f32 im)
        (pair_of (K.c64 kind [| signalling_bits x; signalling_bits y |])))

(* Values at the cuts as double-doubles, from mpmath at 300 bits: ln (2 + √3),
   π/2, π and ln 3 / 2. *)
let ln_2_sqrt3 = (0x1.5124271980435p+0, -0x1.9065ed0bf9dcbp-54)
let pio2 = (0x1.921fb54442d18p+0, 0x1.1a62633145c07p-54)
let pi_dd = (0x1.921fb54442d18p+1, 0x1.1a62633145c07p-53)
let half_ln3 = (0x1.193ea7aad030bp-1, -0x1.a256f99caabebp-55)
let zero_dd = (0., 0.)
let two_dd = (2., 0.)

(* C99 7.3's principal values either side of each cut, continuous onto the side
   of the zero part's sign (G.6): asin (x ± 0i) = π/2 ± i acosh x and acos x =
   π/2 - asin x for x > 1, asinh z = -i asin (iz), acosh (-x ± 0i) = acosh x ±
   iπ, atanh (x ± 0i) = ½ ln ((x + 1) / (x - 1)) ± iπ/2, and sqrt (-x ± 0i) = ±
   i √x. Each within its kind's bound. *)
let cuts =
  [
    ("asin", (2., 0.), (pio2, ln_2_sqrt3), 5.);
    ("asin", (2., -0.), (pio2, dd_neg ln_2_sqrt3), 5.);
    ("acos", (2., 0.), (zero_dd, dd_neg ln_2_sqrt3), 5.);
    ("acos", (2., -0.), (zero_dd, ln_2_sqrt3), 5.);
    ("asinh", (0., 2.), (ln_2_sqrt3, pio2), 5.);
    ("asinh", (-0., 2.), (dd_neg ln_2_sqrt3, pio2), 5.);
    ("acosh", (-2., 0.), (ln_2_sqrt3, pi_dd), 5.);
    ("acosh", (-2., -0.), (ln_2_sqrt3, dd_neg pi_dd), 5.);
    ("atanh", (2., 0.), (half_ln3, pio2), 5.);
    ("atanh", (2., -0.), (half_ln3, dd_neg pio2), 5.);
    ("sqrt", (-4., 0.), (zero_dd, two_dd), 3.);
    ("sqrt", (-4., -0.), (zero_dd, dd_neg two_dd), 3.);
  ]

let test_complex_cuts =
  cases "each branch cut takes its side from the zero part's sign"
    ~id:(fun (k, (x, y), _, _) -> strf "%s %h %h" k x y)
    cuts
    (fun (kind, z, want, bound) ->
      at_most float_exact ~than:bound ~msg:"c128"
        (normwise 0x1p-53 (c128_call kind [ z ]) want);
      at_most float_exact ~than:bound ~msg:"c64"
        (normwise 0x1p-24 (c64_call kind [ z ]) want))

let test_complex_acosh =
  let not_negative = satisfies ~claim:"a real part not negative" in
  prop "acosh's real part is not negative" ~count:1000
    Gen.(pair gen_z64 gen_b32)
    (fun (z, w) ->
      not_negative value64
        (fun re -> Float.is_nan re || re >= 0.)
        (fst (c128_call "acosh" [ z ]));
      not_negative exact32
        (fun re -> is_nan re || f re >= 0.)
        (fst (c64_bits "acosh" w)))

(* On the real axis, where the float kind is defined, the real part is the float
   kind's value. Zeros are left out: the .mli states it at a non-zero real
   part. *)
let test_complex_real_axis =
  let nonzero64 = Gen.such_that (fun x -> x <> 0.) in
  let nonzero32 = Gen.such_that (fun b -> f b <> 0.) in
  let unit64 =
    let ends = Gen.of_list [ 1.; -1.; 0x1p-1074 ] in
    Gen.(frequency [ (3, float_range (-1.) 1.); (1, ends) ])
  in
  let real =
    [
      ("exp", gen64, gen32); ("exp2", gen64, gen32); ("sinh", gen64, gen32);
      ("cosh", gen64, gen32); ("tanh", gen64, gen32); ("sin", gen64, gen32);
      ("cos", gen64, gen32); ("tan", gen64, gen32); ("atan", gen64, gen32);
      ( "sqrt",
        Gen.map Float.abs gen64,
        Gen.map (fun b -> b land 0x7FFF_FFFF) gen32 );
      ("asin", unit64, Gen.map bits32 unit64);
      ("acos", unit64, Gen.map bits32 unit64);
    ]
  in
  let kinds = List.map (fun (k, _, _) -> k) real in
  prop "on the real axis the real part is the float kind's" ~count:1200
    Gen.(
      let* k = kind kinds in
      let _, g64, g32 = List.find (fun (n, _, _) -> n = k) real in
      let+ x = nonzero64 g64 and+ b = nonzero32 g32 in
      (k, x, b))
    (fun (k, x, b) ->
      cover_each kinds k;
      equal value64 (K.f64 k [| x |]) (fst (c128_call k [ (x, 0.) ]));
      equal value64 (f (K.f32 k [| b |])) (fst (c64_call k [ (f b, 0.) ])))

let test_complex_log_real =
  List.map
    (fun k ->
      prop (strf "%s c128 at x + 0i is the float kind's of |x|" k) gen64
        (fun x ->
          equal value64
            (K.f64 k [| Float.abs x |])
            (fst (c128_call k [ (x, 0.) ]))))
    [ "log"; "log2" ]

let test_complex_pow_real =
  let positive64 = Gen.such_that (fun x -> x > 0.) (Gen.map Float.abs gen64) in
  let positive32 =
    Gen.such_that
      (fun b -> f b > 0.)
      (Gen.map (fun b -> b land 0x7FFF_FFFF) gen32)
  in
  prop "pow of a positive base and a real exponent is the float kind's"
    Gen.(pair (pair positive64 gen64) (pair positive32 gen32))
    (fun ((x, w), (a, b)) ->
      equal value64
        (K.f64 "pow" [| x; w |])
        (fst (c128_call "pow" [ (x, 0.); (w, 0.) ]));
      equal value64
        (f (K.f32 "pow" [| a; b |]))
        (fst (c64_call "pow" [ (f a, 0.); (f b, 0.) ])))

(* exp2 at every finite y: |2^(x + iy)| = 2^x for y drawn over the whole range
   of the part's format. *)
let test_complex_exp2_modulus =
  let anywhere ~emin ~emax =
    let open Gen in
    let* e = int_range emin emax in
    let* m = float_range 1. 1.99 in
    let+ s = bool in
    Float.ldexp (if s then -.m else m) e
  in
  let check name u call round ~emin ~emax ~p =
    prop (strf "|exp2 (x + iy)| is 2^x %s" name)
      Gen.(pair (float_range (-60.) 60.) (anywhere ~emin ~emax))
      (fun (x, y) ->
        let x, y = round (x, y) in
        cover "|y| at least 2^p" (Float.abs y >= Float.ldexp 1. p);
        let want = Float.pow 2. x in
        let got = call "exp2" [ (x, y) ] in
        at_most float_exact ~than:4.
          (modulus_error u got ((want, 0.), (0., 0.))))
  in
  [
    check "c128" 0x1p-53 c128_call Fun.id ~emin:(-1074) ~emax:1023 ~p:53;
    check "c64" 0x1p-24 c64_call f32_pair ~emin:(-149) ~emax:127 ~p:24;
  ]

(* The phase y ln 2 of exp2 (iy), where it takes y ln 2's every bit: cos (y ln
   2) and sin (y ln 2) from mpmath, at the c128 or the c64 y. *)
let test_complex_exp2_phase =
  cases "exp2's phase far up the imaginary axis is within 4u"
    ~id:(fun (w, y, _) -> strf "%s %h" w y)
    [
      ("c128", 0x1p30, (0x1.66e39f669c166p-1, -0x1.6d2939368da98p-1));
      ( "c128",
        0x1.5eb851eb851ecp+74,
        (0x1.59f6745a8633cp-1, 0x1.796e36e47cb4ap-1) );
      ("c128", 1e300, (0x1.b80528d30f790p-1, -0x1.05c67494c0ab7p-1));
      ("c128", Float.max_float, (0x1.cedaf85d0796ep-1, -0x1.b5c31ba408c70p-2));
      ("c64", 0x1p30, (0x1.66e39f669c166p-1, -0x1.6d2939368da98p-1));
      ("c64", 0x1.5eb852p+74, (-0x1.f0744683c33c2p-1, -0x1.f4d5bd9a1b741p-3));
      ("c64", 0x1.fffffep+127, (0x1.8a07e4d1b0d09p-1, -0x1.46ee442fa86e9p-1));
    ]
    (fun (w, y, (c, s)) ->
      let u, call =
        if w = "c128" then (0x1p-53, c128_call) else (0x1p-24, c64_call)
      in
      at_most float_exact ~than:4.
        (normwise u (call "exp2" [ (0., y) ]) ((c, 0.), (s, 0.))))

(* pow at the values its .mli states: 1 where the exponent is 0, any base
   included; 0 at a zero base where the exponent's real part is positive;
   the sides of the negative real axis by the sign of a zero imaginary
   part. *)
let test_complex_pow_special =
  let check call round =
    List.iter
      (fun z -> equal both_parts (1., 0.) (call "pow" [ round z; (0., 0.) ]))
      [ (nan, nan); (0., 0.); (inf, 0.); (2., 3.); (ninf, nan) ];
    equal both_parts (1., 0.) (call "pow" [ round (2., 3.); (-0., 0.) ]);
    List.iter
      (fun w ->
        let re, im = call "pow" [ (0., 0.); round w ] in
        equal (pair float_exact float_exact) (0., 0.) (Float.abs re, Float.abs im))
      [ (2., 0.); (0.5, 0.); (1., 1.); (3., -2.) ]
  in
  [
    test "pow at 0 and 1, c128" (fun () -> check c128_call Fun.id);
    test "pow at 0 and 1, c64" (fun () -> check c64_call f32_pair);
    test "pow on either side of the negative real axis" (fun () ->
        let bound = 3. +. (3. *. 0.5 *. Float.hypot (Float.log 4.) pi) in
        let side u call im want =
          let got = call "pow" [ (-4., im); (0.5, 0.) ] in
          at_most float_exact ~than:bound (normwise u got ((0., 0.), (want, 0.)))
        in
        side 0x1p-53 c128_call 0. 2.;
        side 0x1p-53 c128_call (-0.) (-2.);
        side 0x1p-24 c64_call 0. 2.;
        side 0x1p-24 c64_call (-0.) (-2.));
  ]

(* log |z| near the unit circle keeps its digits: the real part of log at
   points whose modulus is 1 within a few ulps is within 4u of itself. *)
let test_complex_log_circle =
  test "log's real part near the unit circle" (fun () ->
      let check u call z want =
        let re, _ = call "log" [ z ] in
        at_most float_exact ~than:4.
          (normwise u (re, 0.) ((want, 0.), (0., 0.)))
          ~msg:(strf "log %h %h: %h" (fst z) (snd z) re)
      in
      check 0x1p-53 c128_call (-0.46684475937821884, -0.8843392848002926)
        0x1.34c78a964b468p-60;
      check 0x1p-53 c128_call (0.6, 0.8) 0x1.999999999999ap-56;
      check 0x1p-53 c128_call (0.8, 0.6) 0x1.999999999999ap-56;
      check 0x1p-53 c128_call (-0.28, 0.96) (-0x1.eb851eb851eb8p-56);
      check 0x1p-24 c64_call (0x1.333334p-1, 0x1.99999ap-1) 0x1.9999995c28f5dp-26;
      check 0x1p-24 c64_call (-0x1.1eb852p-2, 0x1.eb851ep-1)
        (-0x1.5c28f5f65fd8bp-26))

(* Identities within the bounds the kinds compose to, at parts in [2^-40, 2^40]:
   exp (log z) = z within (5 + 4|log z|)u, the exponential carrying log's
   absolute error into a relative one; sqrt z * sqrt z = z within 8u, twice
   sqrt's bound and mul's. *)
let dd_of (re, im) = ((re, 0.), (im, 0.))
let modulus (re, im) = Float.hypot re im

let test_complex_exp_log =
  prop "exp (log z) is z within (5 + 4 |log z|)u" parts (fun z ->
      let l = c128_call "log" [ z ] in
      at_most float_exact
        ~than:(5. +. (4. *. modulus l))
        ~msg:"c128"
        (normwise 0x1p-53 (c128_call "exp" [ l ]) (dd_of z));
      let z = f32_pair z in
      let l = c64_call "log" [ z ] in
      at_most float_exact
        ~than:(5. +. (4. *. modulus l))
        ~msg:"c64"
        (normwise 0x1p-24 (c64_call "exp" [ l ]) (dd_of z)))

let test_complex_sqrt_square =
  prop "sqrt z * sqrt z is z within 8u" parts (fun z ->
      let s = c128_call "sqrt" [ z ] in
      at_most float_exact ~than:8. ~msg:"c128"
        (normwise 0x1p-53 (c128_call "mul" [ s; s ]) (dd_of z));
      let z = f32_pair z in
      let s = c64_call "sqrt" [ z ] in
      at_most float_exact ~than:8. ~msg:"c64"
        (normwise 0x1p-24 (c64_call "mul" [ s; s ]) (dd_of z)))

(* Symmetries over drawn parts, NaNs, infinities, zeros and subnormals among
   them, bit for bit, any NaN for a NaN result part. c64 operands are conjugated
   and negated on their bits: the symmetries count a NaN operand part's sign
   bit, which a conversion between binary32 and binary64 may drop, as RISC-V's
   gives the canonical NaN. *)

let conj (re, im) = (re, Float.neg im)
let negz (re, im) = (Float.neg re, Float.neg im)
let sign32 = 0x8000_0000
let conj32 (re, im) = (re, im lxor sign32)
let negz32 (re, im) = (re lxor sign32, im lxor sign32)

let test_complex_conj =
  prop "f (conj z) is conj (f z)" ~count:1800
    Gen.(triple (kind complex_kinds) gen_z64 gen_b32)
    (fun (k, z, w) ->
      cover_each complex_kinds k;
      equal both_parts (conj (c128_call k [ z ])) (c128_call k [ conj z ]);
      equal both_parts32 (conj32 (c64_bits k w)) (c64_bits k (conj32 w)))

let test_complex_parity =
  let odd =
    [ "sinh"; "tanh"; "sin"; "tan"; "asin"; "atan"; "asinh"; "atanh"; "sign" ]
  in
  let kinds = odd @ [ "cosh"; "cos" ] in
  prop "odd kinds give -f z at -z, even ones f z" ~count:1100
    Gen.(triple (kind kinds) gen_z64 gen_b32)
    (fun (k, z, w) ->
      cover_each kinds k;
      let is_odd = List.mem k odd in
      let side v = if is_odd then negz v else v in
      let side32 v = if is_odd then negz32 v else v in
      equal both_parts (side (c128_call k [ z ])) (c128_call k [ negz z ]);
      equal both_parts32 (side32 (c64_bits k w)) (c64_bits k (negz32 w)))

(* nx_kinds.h *)

(* The f64 kinds that gen/kinds.py's goldens hold. *)
let golden_kinds = transcendental_kinds @ [ "atan2"; "pow" ]

(* Whether [y] is within one ulp of the exact value, [r] that value correctly
   rounded and [lo] its remainder in ulps of [r]: [r] itself, or [r]'s neighbour
   on [lo]'s side, counting floats; [r] alone where [lo] is 0, an exact value,
   no finite rounding or a NaN. *)
let within_one_ulp y (r, lo) =
  let side = if lo > 0. then 1L else if lo < 0. then -1L else 0L in
  if Float.is_nan r then Float.is_nan y
  else if not (Float.is_finite r) then bits64 y = bits64 r
  else
    let d = Int64.sub (rank64 y) (rank64 r) in
    d = 0L || d = side

(* Golden rows, a result next to their [r], and whether {!within_one_ulp} takes
   it: [r] alone where the value is exact, and [r]'s neighbour on the exact
   value's side alone where [r]'s neighbours are the least subnormal apart. *)
let one_ulp_rows =
  [
    ("pow", [| 0x1p-1074; 1. |], 0x1p-1073, false);
    ("sin", [| 0. |], 0x1p-1074, false);
    ("sin", [| 0x1p-1074 |], 0., true);
    ("sin", [| 0x1p-1074 |], 0x1p-1073, false);
    ("atan", [| 0x1p-1074 |], 0., true);
    ("atan", [| 0x1p-1074 |], 0x1p-1073, false);
    ( "pow",
      [| 0x1.31ab48c1294bbp+6; -0x1.46b41332042bfp+7 |],
      0x1.125fbee250730p-1022,
      true );
    ( "pow",
      [| 0x1.31ab48c1294bbp+6; -0x1.46b41332042bfp+7 |],
      0x1.125fbee25072ep-1022,
      false );
  ]

let test_one_ulp =
  cases
    "the one-ulp check takes r alone at an exact value, else r's neighbour on \
     the exact value's side"
    ~id:(fun (k, args, y, _) ->
      strf "%s %s: %h" k
        (String.concat " " (Array.to_list (Array.map (strf "%h") args)))
        y)
    one_ulp_rows
    (fun (k, args, y, takes) ->
      let n = Array.length args in
      let same r = Array.for_all2 (fun a b -> bits64 a = bits64 b) args r in
      let row = List.find (fun r -> same (Array.sub r 0 n)) (golden k) in
      equal bool takes
        (within_one_ulp y (row.(n), row.(n + 1)))
        ~msg:(strf "r %h, remainder %h" row.(n) row.(n + 1)))

let test_goldens =
  test
    "every f64 kind is within one ulp of the exact value at the goldens' points"
    (fun () ->
      List.iter
        (fun k ->
          List.iter
            (fun row ->
              let n = Array.length row - 2 in
              let args = Array.sub row 0 n in
              let y = K.f64 k args in
              if not (within_one_ulp y (row.(n), row.(n + 1))) then
                failf "%s %s: %h, the exact value %h + %h" k
                  (String.concat " "
                     (Array.to_list (Array.map (strf "%h") args)))
                  y row.(n)
                  row.(n + 1))
            (golden k))
        golden_kinds)

(* exp2 is exact at integers whose power is a float, log2 at powers of two,
   subnormal ones included. *)
let test_exact_points =
  test "exp2 is exact at integers whose power is a float, log2 at powers of two"
    (fun () ->
      for k = -149 to 127 do
        let p = Float.ldexp 1. k in
        equal exact32 (bits32 p) (K.f32 "exp2" [| bits32 (Float.of_int k) |]);
        equal exact32 (bits32 (Float.of_int k)) (K.f32 "log2" [| bits32 p |])
      done;
      for k = -1074 to 1023 do
        let p = Float.ldexp 1. k in
        equal value64 p (K.f64 "exp2" [| Float.of_int k |]);
        equal value64 (Float.of_int k) (K.f64 "log2" [| p |])
      done)

(* Digests of the transcendental kinds over a fixed point set

   What every library computing a kind through nx_kinds.h gives: each f32 and
   f64 transcendental kind of one or two operands over Annex F's operands and,
   in every binade of either sign, its first and last pattern and six drawn
   significands; a kind of two operands at every pair of Annex F's operands and
   at each binade point paired with another. The digest is MD5 over the results'
   bits, every NaN as one NaN. The exact kinds' bits are their meaning's, which
   the laws above check. *)

let splitmix s =
  let open Int64 in
  let s = add s 0x9E3779B97F4A7C15L in
  let s = mul (logxor s (shift_right_logical s 30)) 0xBF58476D1CE4E5B9L in
  let s = mul (logxor s (shift_right_logical s 27)) 0x94D049BB133111EBL in
  logxor s (shift_right_logical s 31)

let annex_f_operands =
  [ 0.; -0.; inf; ninf; nan; 1.; -1.; 0.5; 2.; -3.; 0x1p-149; 0x1p-1074 ]

(* The NaN every NaN result is digested as: the quiet NaN of the sign bit
   clear and no payload, 0x7FC00000 as binary32. *)
let one_nan = Int64.float_of_bits 0x7FF8_0000_0000_0000L

(* The points of a format of [width] bits with [fraction] fraction bits,
   as [of_bits] reads their patterns. *)
let points ~width ~fraction ~of_bits =
  let open Int64 in
  let binades = (1 lsl (width - 1 - fraction)) - 1 and draws = ref 1L in
  let draw () =
    draws := splitmix !draws;
    logand !draws (sub (shift_left 1L fraction) 1L)
  in
  let binade sign e =
    let top =
      logor
        (shift_left (of_int sign) (width - 1))
        (shift_left (of_int e) fraction)
    in
    List.map
      (fun frac -> of_bits (logor top frac))
      (0L :: sub (shift_left 1L fraction) 1L :: List.init 6 (fun _ -> draw ()))
  in
  annex_f_operands
  @ List.concat_map
      (fun sign -> List.concat_map (binade sign) (List.init binades Fun.id))
      [ 0; 1 ]

let digest_kinds = (transcendental_kinds, [ "pow"; "atan2" ])

let test_digests =
  test "the transcendental kinds' bits are the same on every target" (fun () ->
      let pairs ps =
        let n = List.length ps and a = Array.of_list ps in
        let specials = List.length annex_f_operands in
        let firsts = ref [] and seconds = ref [] in
        for i = 0 to (specials * specials) - 1 do
          firsts := a.(i mod specials) :: !firsts;
          seconds := a.(i / specials) :: !seconds
        done;
        for i = specials to n - 1 do
          firsts := a.(i) :: !firsts;
          seconds := a.(((i * 7919) mod (n - specials)) + specials) :: !seconds
        done;
        (List.rev !firsts, List.rev !seconds)
      in
      let digest (type s) (k : (float, s) Bigarray.kind) ~bits name xs zs =
        let ba l =
          Bigarray.Array1.of_array k Bigarray.c_layout (Array.of_list l)
        in
        let x = ba xs and z = ba zs and y = ba xs in
        K.run name x z y;
        let b = Buffer.create (8 * Bigarray.Array1.dim y) in
        for i = 0 to Bigarray.Array1.dim y - 1 do
          bits b (if Float.is_nan y.{i} then one_nan else y.{i})
        done;
        String.sub (Digest.to_hex (Digest.string (Buffer.contents b))) 0 16
      in
      let rows (type s) ty (k : (float, s) Bigarray.kind) ~bits ps =
        let unaries, binaries = digest_kinds in
        let xs, zs = pairs ps in
        List.map
          (fun name -> strf "%s %s %s" name ty (digest k ~bits name ps ps))
          unaries
        @ List.map
            (fun name -> strf "%s %s %s" name ty (digest k ~bits name xs zs))
            binaries
      in
      let f32 =
        rows "f32" Bigarray.float32
          ~bits:(fun b v -> Buffer.add_int32_le b (Int32.bits_of_float v))
          (points ~width:32 ~fraction:23 ~of_bits:(fun p ->
               Int32.float_of_bits (Int64.to_int32 p)))
      in
      let f64 =
        rows "f64" Bigarray.float64
          ~bits:(fun b v -> Buffer.add_int64_le b (bits64 v))
          (points ~width:64 ~fraction:52 ~of_bits:Int64.float_of_bits)
      in
      expect (String.concat "\n" (f32 @ f64)) @@ __POS_OF__ {|
        exp f32 d148aa0cfe798ea4
        exp2 f32 06b24c17b64f3f45
        expm1 f32 d122ce2e9e735110
        log f32 133e7ba9c8eebf3b
        log2 f32 3c6f7234b66688a4
        log1p f32 eb7ada460c663d78
        sin f32 00e1cfad28177c0c
        cos f32 e5551b89e045b9f2
        tan f32 fa822cde5b371ce9
        asin f32 9727fcb8f7ace84b
        acos f32 ca63f77d95f5b984
        atan f32 1746b586621af9b2
        sinh f32 9df7ed4ea00d14cd
        cosh f32 4dd8dfbd5619947f
        tanh f32 0f924977a511b4e7
        erf f32 fedbb1616eae7321
        pow f32 f84c9d0db0dcfe2a
        atan2 f32 486465bc2d1c5df8
        exp f64 b351bb921af7acbe
        exp2 f64 981055566a4c7a0d
        expm1 f64 3485bc4d87872344
        log f64 fa070786ccd5f17d
        log2 f64 d2efd3c3ba9f06e8
        log1p f64 274c1dde447e2bb2
        sin f64 3009f282a78b4b8f
        cos f64 0f2ab675630a27e4
        tan f64 2d18b8071332e6ce
        asin f64 ea43be89c29c264e
        acos f64 2cf25916c19bf431
        atan f64 3952e282059eaa93
        sinh f64 506a5ec38e14841f
        cosh f64 2d25fdc9f7313606
        tanh f64 f311ad69c41b5b98
        erf f64 86bf2943e85c9366
        pow f64 de2d29a86133409e
        atan2 f64 28f45baa0494ac7e
        |})

(* The kinds with a near function over a vector of 1100 lanes, past the loops'
   block of 1024: each lane is the kind's own bits, whether its run of lanes
   computed the near function or the kind. The near function's test, from
   nx_kinds.h: for sin, cos and tan a finite |x| at or past the reduction's
   switch, 2^15 at f32 and 2^28 at f64; for exp an |x| past 86 at f32 and 708 at
   f64, or a NaN. *)
let near_kinds = [ "sin"; "cos"; "tan"; "exp" ]
let near_lanes = 1100

let switch k ~f32 =
  match (k, f32) with
  | "exp", true -> 86.
  | "exp", false -> 708.
  | _, true -> 0x1p15
  | _, false -> 0x1p28

let past k ~f32 x =
  if k = "exp" then Float.is_nan x || Float.abs x > switch k ~f32
  else Float.is_finite x && Float.abs x >= switch k ~f32

(* [near_lanes] lanes from [seed], each past the test with the chance [share],
   at magnitudes within a factor of 4 of the switch on either side or far below
   it; where [share] is not 0, the first two lanes are the switch and the float
   below it. *)
let near_operands (type s) (ba : (float, s) Bigarray.kind) k ~f32 ~share seed =
  let st = Random.State.make [| seed |] in
  let s = switch k ~f32 in
  let round x = if f32 then round_f32 x else x in
  let lane _ =
    let sign = if Random.State.bool st then 1. else -1. in
    let scale lo hi = Float.pow 2. (lo +. Random.State.float st (hi -. lo)) in
    if Random.State.float st 1. < share then round (sign *. s *. scale 0. 2.)
    else if Random.State.bool st then round (sign *. s *. scale (-2.) 0.)
    else round (sign *. scale (-30.) 0.)
  in
  let x = Bigarray.Array1.init ba Bigarray.c_layout near_lanes lane in
  if share > 0. then begin
    x.{0} <- s;
    x.{1} <- (if f32 then f (bits32 s - 1) else Float.pred s)
  end;
  x

(* Whether the block of lanes [lo] to [hi - 1] of [x] holds lanes either side of
   [k]'s test. *)
let mixed k ~f32 x lo hi =
  let past = List.init (hi - lo) (fun i -> past k ~f32 x.{lo + i}) in
  List.mem true past && List.mem false past

let test_near =
  prop
    "sin, cos, tan and exp over a vector give the kind's bits at every lane, \
     either side of the near function's test"
    Gen.(triple (kind near_kinds) (of_list [ 0.; 0.5 ]) int)
    (fun (k, share, seed) ->
      cover_each near_kinds k;
      let x = near_operands Bigarray.float32 k ~f32:true ~share seed in
      let x64 = near_operands Bigarray.float64 k ~f32:false ~share seed in
      cover "no lane past the test"
        (List.for_all
           (fun i -> not (past k ~f32:true x.{i}))
           (List.init near_lanes Fun.id));
      cover "a block of lanes either side of the test"
        (mixed k ~f32:true x 0 1024 && mixed k ~f32:false x64 0 1024);
      let y =
        Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout near_lanes
      in
      K.run k x x y;
      for i = 0 to near_lanes - 1 do
        equal value32
          ~msg:(strf "f32 lane %d, %h" i x.{i})
          (K.kind32 k (bits32 x.{i}))
          (bits32 y.{i})
      done;
      let y =
        Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout near_lanes
      in
      K.run k x64 x64 y;
      for i = 0 to near_lanes - 1 do
        equal value64
          ~msg:(strf "f64 lane %d, %h" i x64.{i})
          (K.kind64 k x64.{i})
          y.{i}
      done)

(* The narrow floats

   A float narrower than float32 computes in f32 and rounds once on the store,
   through each kernel library's map. At the zeros of either sign, where IEEE
   754 fixes the sign of a sum, a difference, a maximum and a minimum, each
   narrow format holds the f32 kind's result stored; each transcendental kind at
   every code of each format, and pow and atan2 at every pair of the formats of
   a byte or less and at every code of the 16-bit ones against a drawn partner,
   hold the reference of the map laws, which computes the f32 kind and stores
   once. *)

type narrow = Narrow : (float, 's) D.t -> narrow

let pp_narrow ppf (Narrow dt) = Format.pp_print_string ppf (D.name dt)

let narrow_floats =
  D.
    [
      Narrow Float16;
      Narrow Bfloat16;
      Narrow Float8_e4m3fn;
      Narrow Float8_e5m2;
      Narrow Float4_e2m1fn;
    ]

let answer = Testable.make ~pp:Nx_array_support.pp_answer ~equal:( = )
let signed_zeros = [ (0., 0.); (0., -0.); (-0., 0.); (-0., -0.) ]

let zero_rows =
  List.concat_map
    (fun dt ->
      List.concat_map
        (fun (name, k) -> List.map (fun z -> (dt, name, k, z)) signed_zeros)
        Nx_kernel.Prog.
          [
            ("add", Add);
            ("sub", Sub);
            ("maximum", Maximum);
            ("minimum", Minimum);
          ])
    narrow_floats

let narrow_zero (b : Support.backend) (Narrow dt, name, k, (a, c)) =
  if not (b.computes (K2 (Binary k)) (D.Any dt)) then
    skip ~reason:(strf "%s states no %s at %s" b.name name (D.name dt)) ();
  let module K' = (val b.kernels) in
  let want = D.of_float dt (f (K.f32 name [| bits32 a; bits32 c |])) in
  let one v = Support.on b (A.of_array dt [| 1 |] [| v |]) in
  let dst = A.create b.device dt [| 1 |] in
  equal answer A.Done (Support.map2 (module K') (Binary k) ~dst (one a) (one c));
  equal value64 want (A.get (Support.host dst) [| 0 |])

(* Every code of the narrow float [dt], as [A.bitcast] reads the unsigned
   integers of its width. *)
let every_code (type s) (dt : (float, s) D.t) =
  let n = 1 lsl D.bits dt in
  let codes = Array.init n Fun.id in
  let (A.Any u) =
    match D.bits dt with
    | 4 -> A.Any (A.of_array D.Uint4 [| n |] codes)
    | 8 -> A.Any (A.of_array D.Uint8 [| n |] codes)
    | _ -> A.Any (A.of_array D.Uint16 [| n |] codes)
  in
  Option.get (A.bitcast dt u)

let transcendentals =
  List.filter (fun (_, name) -> List.mem name transcendental_kinds) C.unaries

let narrow_transcendentals b (Narrow dt) =
  C.law_map1 ~kinds:transcendentals b (C.Case (every_code dt))

let pow_atan2 = Nx_kernel.Prog.[ Binary Pow; Binary Atan2 ]

let narrow_pairs b (Narrow dt) =
  let x = every_code dt in
  let n = L.numel (A.layout x) in
  let code i = A.get x [| i |] in
  let firsts = Array.init (n * n) (fun i -> code (i / n)) in
  let seconds = Array.init (n * n) (fun i -> code (i mod n)) in
  C.law_map2 ~kinds:pow_atan2 b
    (C.Pair (A.of_array dt [| n * n |] firsts, A.of_array dt [| n * n |] seconds))

(* [x]'s partners for pow in [dt], from [seed]: for a finite [x] off 1, [y] with
   [y log2 |x|] within the format's exponents, give or take 2, an integer for a
   negative [x]; a drawn code elsewhere. *)
let pow_partners (type s) (dt : (float, s) D.t) (x : (float, s) A.t) seed =
  let st = Random.State.make [| seed |] in
  let format = D.float_format dt in
  let lo = Float.log2 format.min_normal -. Float.of_int format.fraction_bits in
  let hi = Float.log2 format.max_finite in
  let n = L.numel (A.layout x) in
  let codes = every_code dt in
  let partner i =
    let v = A.get x [| i |] in
    let l = Float.log2 (Float.abs v) in
    if (not (Float.is_finite l)) || l = 0. then
      A.get codes [| Random.State.int st (L.numel (A.layout codes)) |]
    else
      let y = (lo -. 2. +. Random.State.float st (hi -. lo +. 4.)) /. l in
      if v < 0. then Float.round y else y
  in
  A.of_array dt [| n |] (Array.init n partner)

let narrow_partners b seed =
  List.iter
    (fun (Narrow dt) ->
      let x = every_code dt in
      let n = L.numel (A.layout x) in
      C.law_map2 ~kinds:pow_atan2 b (C.Pair (x, C.seeded dt [| n |] seed));
      C.law_map2 ~kinds:[ Binary Pow ] b (C.Pair (x, pow_partners dt x seed)))
    D.[ Narrow Float16; Narrow Bfloat16 ]

(* The narrow floats' laws on [d]'s backend, a group named for it. *)
let narrow (d : Support.declared) =
  let run f x =
    let b = Support.opened d in
    b.around (fun () -> f b x)
  in
  group
    (d.label ^ " narrow floats compute in f32 and round once")
    [
      cases "signed zeros at the narrow floats"
        ~id:(fun (Narrow dt, name, _, (a, c)) ->
          strf "%s %s %h %h" (D.name dt) name a c)
        zero_rows (run narrow_zero);
      prop "transcendental kinds at every code"
        (Gen.of_list ~pp:pp_narrow narrow_floats)
        (run narrow_transcendentals);
      prop "pow and atan2 at every pair of the 8- and 4-bit formats"
        (Gen.of_list ~pp:pp_narrow
           D.[ Narrow Float8_e4m3fn; Narrow Float8_e5m2; Narrow Float4_e2m1fn ])
        (run narrow_pairs);
      prop ~count:3
        "pow and atan2 at every float16 and bfloat16 code against a drawn \
         partner"
        Gen.nat (run narrow_partners);
    ]

let () =
  exit
  @@ run "nx_kernel.kinds"
       ([
          group "kinds"
            [
              test_arith;
              test_fdiv;
              test_sqrt;
              test_recip;
              test_mod;
              test_rounding;
              test_sign;
              test_neg_abs;
              test_ranges;
              test_maximum;
              test_fma;
              test_where;
              test_threefry;
            ];
          group "kinds on integers"
            [
              test_int_wrap;
              test_int_abs;
              test_int_sign;
              test_int_recip;
              test_idiv;
              test_mod_int;
              test_int_pow;
              test_bitwise;
              test_int_order;
              test_int_where;
            ];
          group "domains"
            [ test_float_compare; test_complex_compare; test_complex_maximum ];
          group "results" [ test_annex_f; test_quiet ];
          group "complex numbers"
            ([
               test_complex_parts;
               test_complex_mul;
               test_normwise_nan;
               test_complex_fdiv;
             ]
            @ test_complex_subnormal
            @ [
                test_complex_large;
                test_complex_smith;
                test_complex_recip;
                test_complex_nan;
                test_complex_special;
                test_complex_rotated;
                test_complex_signalling;
                test_complex_cuts;
                test_complex_acosh;
                test_complex_real_axis;
              ]
            @ test_complex_log_real @ [ test_complex_pow_real ]
            @ test_complex_exp2_modulus
            @ [ test_complex_exp2_phase ]
            @ test_complex_pow_special
            @ [
                test_complex_log_circle;
                test_complex_exp_log;
                test_complex_sqrt_square;
                test_complex_conj;
                test_complex_parity;
              ]);
          group "nx_kinds.h"
            [
              test_goldens;
              test_one_ulp;
              test_exact_points;
              test_digests;
              test_near;
            ];
        ]
       @ List.map narrow
           (Support.declare Support.cpu
           :: List.filter_map Fun.id
                [ Support.declared_cuda; Support.declared_amd ]))
