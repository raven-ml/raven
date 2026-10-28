(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx_kinds.h's functions, called through loops compiled as nx.cpu's rows:
   the exact kinds against their meaning, with NaN, -0, infinities,
   subnormals and integer extremes drawn, and the narrow floats' signed
   zeros through nx.cpu; the float kinds at C99 Annex F's special values and
   exp2 and log2 at their exact points; threefry's known answers; the complex
   kinds at C99 Annex G's special values, at signalling NaNs and by their
   symmetries. Their bounds are test_kernels', through Nx_kernel.S.map. *)

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

(* Bit for bit. *)
let exact32 = Testable.make ~pp:pp32 ~equal:( = )

(* Bit for bit, but any NaN for a NaN: an operation's own NaN differs between
   instruction sets. *)
let value32 =
  Testable.make ~pp:pp32 ~equal:(fun a b -> a = b || (is_nan a && is_nan b))

let value64 =
  Testable.make
    ~pp:(fun ppf x -> Format.fprintf ppf "%h (0x%016Lx)" x (bits64 x))
    ~equal:(fun a b ->
      Int64.equal (bits64 a) (bits64 b) || (Float.is_nan a && Float.is_nan b))

(* exp2 is exact at integers whose power is a float, log2 at powers of two,
   subnormal ones included. *)
let test_exact_points =
  test "exp2 and log2 at integers and powers of two" (fun () ->
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

(* Drawn f32 operands *)

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
  frequency
    [
      (4, any);
      (2, special);
      (3, moderate);
      (2, integers);
      (1, halves);
      (2, near_one);
    ]

let cover32 name b =
  cover (name ^ " NaN") (is_nan b);
  cover (name ^ " -0") (b = 0x8000_0000);
  cover (name ^ " infinite") (b land 0x7FFF_FFFF = 0x7F80_0000);
  cover (name ^ " subnormal") (b land 0x7F80_0000 = 0 && b land 0x7F_FFFF <> 0)

(* f32 exact kinds against their meaning, any NaN for a NaN

   A binary32 sum, difference, product, quotient or square root of floats is the
   double one rounded to binary32, since 53 >= 2 * 24 + 2. *)

let round32 = bits32
let nan32 = bits32 Float.nan

(* The binary32 fma: a b + c is exact as s + e by TwoSum on the double product;
   rounded to odd in binary64, then to binary32, it rounds once. *)
let fma32 a b c =
  let x = f a and y = f b and z = f c in
  if not (Float.is_finite x && Float.is_finite y && Float.is_finite z) then
    round32 (Float.fma x y z)
  else
    let p = Sys.opaque_identity (x *. y) in
    let s = p +. z in
    let bb = s -. p in
    let e = p -. (s -. bb) +. (z -. bb) in
    if e = 0. then round32 s
    else
      let t =
        if e > 0. = (s > 0.) then s
        else Float.copy_sign (Float.pred (Float.abs s)) s
      in
      round32 (Int64.float_of_bits (Int64.logor (bits64 t) 1L))

let binary32 =
  [
    ("add", fun a b -> round32 (f a +. f b));
    ("sub", fun a b -> round32 (f a -. f b));
    ("mul", fun a b -> round32 (f a *. f b));
    ("fdiv", fun a b -> round32 (f a /. f b));
    ("mod", fun a b -> round32 (Float.rem (f a) (f b)));
    (* IEEE 754-2019: -0 below +0 *)
    ( "maximum",
      fun a b ->
        if is_nan a || is_nan b then nan32
        else if f a = f b then a land b
        else if f a > f b then a
        else b );
    ( "minimum",
      fun a b ->
        if is_nan a || is_nan b then nan32
        else if f a = f b then a lor b
        else if f a < f b then a
        else b );
    ("equal", fun a b -> round32 (if f a = f b then 1. else 0.));
    ("not_equal", fun a b -> round32 (if f a <> f b then 1. else 0.));
    ("less", fun a b -> round32 (if f a < f b then 1. else 0.));
    ("less_equal", fun a b -> round32 (if f a <= f b then 1. else 0.));
  ]

let test_binary32 =
  List.map
    (fun (kind, meaning) ->
      prop (strf "%s f32" kind) ~count:20000
        Gen.(pair gen32 gen32)
        (fun (a, b) ->
          cover32 "first" a;
          cover32 "second" b;
          let t =
            match kind with
            | "equal" | "not_equal" | "less" | "less_equal" -> exact32
            | _ -> value32
          in
          equal t (meaning a b) (K.f32 kind [| a; b |])))
    binary32

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

(* fma rounds once through the kinds' loops, compiled as nx.cpu's rows. *)
let test_fma32 =
  prop "fma f32 rounds once" ~count:20000
    Gen.(frequency [ (1, triple gen32 gen32 gen32); (1, tiny32) ])
    (fun (a, b, c) ->
      let r = fma32 a b c in
      cover32 "addend" c;
      cover "subnormal result"
        (r land 0x7F80_0000 = 0 && r land 0x7F_FFFF <> 0);
      cover "signed zero result" (r = 0x8000_0000);
      equal value32 r (K.f32 "fma" [| a; b; c |]))

let test_where32 =
  prop "where f32"
    Gen.(triple gen32 gen32 gen32)
    (fun (c, a, b) ->
      equal exact32 (if f c <> 0. then a else b) (K.f32 "where" [| c; a; b |]))

(* Unary kinds; neg and abs keep a NaN's payload. *)
let unary32 =
  let float_op op a = round32 (op (f a)) in
  [
    ("neg", (fun a -> a lxor 0x8000_0000), true);
    ("abs", (fun a -> a land 0x7FFF_FFFF), true);
    ( "sign",
      float_op (fun x ->
          if Float.is_nan x then x
          else if x > 0. then 1.
          else if x < 0. then -1.
          else 0.),
      false );
    ("sqrt", float_op Float.sqrt, false);
    ("recip", float_op (fun x -> 1. /. x), false);
    ("floor", float_op Float.floor, false);
    ("ceil", float_op Float.ceil, false);
    ("round", float_op Float.round, false);
    ("trunc", float_op Float.trunc, false);
  ]

let test_unary32 =
  List.map
    (fun (kind, meaning, keeps) ->
      prop (strf "%s f32" kind) ~count:20000 gen32 (fun a ->
          cover32 "operand" a;
          equal
            (if keeps then exact32 else value32)
            (meaning a) (K.f32 kind [| a |])))
    unary32

(* f64 exact kinds, against OCaml's binary64 arithmetic, any NaN for a NaN *)

let gen64 =
  let open Gen in
  let any = map Int64.float_of_bits int64 in
  let special =
    of_list
      [
        0.;
        -0.;
        Float.infinity;
        Float.neg_infinity;
        Float.nan;
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

let test_f64_exact =
  let binary =
    [
      ("add", ( +. ));
      ("sub", ( -. ));
      ("mul", ( *. ));
      ("fdiv", ( /. ));
      ( "maximum",
        fun x y ->
          if Float.is_nan x || Float.is_nan y then Float.nan
          else if x = y then
            Int64.float_of_bits (Int64.logand (bits64 x) (bits64 y))
          else if x > y then x
          else y );
      ( "minimum",
        fun x y ->
          if Float.is_nan x || Float.is_nan y then Float.nan
          else if x = y then
            Int64.float_of_bits (Int64.logor (bits64 x) (bits64 y))
          else if x < y then x
          else y );
    ]
  in
  List.map
    (fun (kind, op) ->
      prop (strf "%s f64" kind) ~count:20000
        Gen.(pair gen64 gen64)
        (fun (a, b) ->
          cover "NaN" (Float.is_nan a || Float.is_nan b);
          equal value64 (op a b) (K.f64 kind [| a; b |])))
    binary
  @ [
      prop "fma f64 rounds once" ~count:20000
        Gen.(
          frequency
            [
              (1, triple gen64 gen64 gen64);
              (1, tiny ~emin:(-1022) ~p:53 Fun.id);
            ])
        (fun (a, b, c) ->
          let r = Float.fma a b c in
          cover "subnormal result" (r <> 0. && Float.abs r < Float.min_float);
          cover "signed zero result" (r = 0. && Float.sign_bit r);
          equal value64 r (K.f64 "fma" [| a; b; c |]));
      prop "mod f64" ~count:20000
        Gen.(pair gen64 gen64)
        (fun (a, b) -> equal value64 (Float.rem a b) (K.f64 "mod" [| a; b |]));
    ]
  @ List.map
      (fun (kind, op) ->
        prop (strf "%s f64" kind) ~count:20000 gen64 (fun a ->
            cover "NaN" (Float.is_nan a);
            equal value64 (op a) (K.f64 kind [| a |])))
      [
        ("sqrt", Float.sqrt);
        ("floor", Float.floor);
        ("ceil", Float.ceil);
        ("round", Float.round);
        ("trunc", Float.trunc);
        ("recip", fun x -> 1. /. x);
      ]

(* The narrow floats

   A float narrower than float32 computes in f32 and rounds once on the
   store, through each kernel library's map. At the zeros of either sign,
   where IEEE 754 fixes the sign of a sum, a difference, a maximum and a
   minimum, each narrow format holds the f32 kind's result stored; each
   transcendental kind at every code of each format, and pow and atan2 at
   every pair of the formats of a byte or less and at every code of the
   16-bit ones against a drawn partner, hold the reference of the map
   laws, which computes the f32 kind and stores once. *)

type narrow = Narrow : (float, 's) D.t -> narrow

let narrow_floats =
  D.
    [
      Narrow Float16; Narrow Bfloat16; Narrow Float8_e4m3fn;
      Narrow Float8_e5m2; Narrow Float4_e2m1fn;
    ]

let signed_zeros = [ (0., 0.); (0., -0.); (-0., 0.); (-0., -0.) ]

let test_narrow_zeros =
  let kinds =
    Nx_kernel.Prog.
      [ ("add", Add); ("sub", Sub); ("maximum", Maximum); ("minimum", Minimum) ]
  in
  let rows =
    List.concat_map
      (fun (b : Support.backend) ->
        List.concat_map
          (fun dt ->
            List.concat_map
              (fun (name, k) ->
                List.map (fun z -> (b, dt, name, k, z)) signed_zeros)
              kinds)
          narrow_floats)
      Support.backends
  in
  cases
    ~name:(fun ((b : Support.backend), Narrow dt, name, _, (a, c)) ->
      strf "%s %s %s %h %h" b.name (D.name dt) name a c)
    "signed zeros at the narrow floats" rows
    (fun (b, Narrow dt, name, k, (a, c)) ->
      let module K' = (val b.kernels) in
      let want = D.of_float dt (f (K.f32 name [| bits32 a; bits32 c |])) in
      let one v = C.on b (A.of_array dt [| 1 |] [| v |]) in
      let dst = A.create b.device dt [| 1 |] in
      equal C.answer A.Done
        (Support.map2 (module K') (Binary k) ~dst (one a) (one c));
      equal value64 want (A.get (C.host dst) [| 0 |]))

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
  let names =
    [ "exp"; "exp2"; "expm1"; "log"; "log2"; "log1p"; "sin"; "cos"; "tan";
      "asin"; "acos"; "atan"; "sinh"; "cosh"; "tanh"; "erf" ]
  in
  List.filter (fun (_, name) -> List.mem name names) C.unaries

(* Every code of each format is one operand, so the law draws nothing. *)
let test_narrow_transcendentals =
  prop ~count:1 "transcendental kinds at every code of the narrow floats"
    Gen.unit (fun () ->
      List.iter
        (fun (b : Support.backend) ->
          List.iter
            (fun (Narrow dt) ->
              C.law_map1 ~kinds:transcendentals b (C.Case (every_code dt)))
            narrow_floats)
        Support.backends)

(* [x]'s partners for pow in [dt], from [seed]: for a finite [x] off 1,
   [y] with [y log2 |x|] within the format's exponents, give or take 2, an
   integer for a negative [x]; a drawn code elsewhere. *)
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

let test_narrow_pairs =
  let binaries = Nx_kernel.Prog.[ Binary Pow; Binary Atan2 ] in
  let rows =
    List.concat_map
      (fun b -> List.map (fun dt -> (b, dt)) narrow_floats)
      Support.backends
  in
  prop ~count:3 "pow and atan2 at the narrow floats' pairs" Gen.nat (fun seed ->
      List.iter
        (fun ((b : Support.backend), Narrow dt) ->
          let x = every_code dt in
          let n = L.numel (A.layout x) in
          if n <= 256 then begin
            let codes i = A.get x [| i |] in
            let firsts = A.of_array dt [| n * n |] (Array.init (n * n) (fun i -> codes (i / n))) in
            let seconds = A.of_array dt [| n * n |] (Array.init (n * n) (fun i -> codes (i mod n))) in
            C.law_map2 ~kinds:binaries b (C.Pair (firsts, seconds))
          end
          else begin
            C.law_map2 ~kinds:binaries b (C.Pair (x, C.seeded dt [| n |] seed));
            C.law_map2 ~kinds:[ Binary Pow ] b (C.Pair (x, pow_partners dt x seed))
          end)
        rows)

(* Integer kinds

   Values are int64s holding a type's values: sign-extended from i32,
   zero-extended from u32. Arithmetic is modular; idiv truncates and is 0 by 0;
   mod takes the dividend's sign and is the dividend by 0; the least value by -1
   wraps; a negative power is 0 but of 1 and -1. *)

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

let int_kinds =
  [
    ("neg", 1);
    ("abs", 1);
    ("sign", 1);
    ("recip", 1);
    ("add", 2);
    ("sub", 2);
    ("mul", 2);
    ("fma", 3);
    ("idiv", 2);
    ("mod", 2);
    ("pow", 2);
    ("maximum", 2);
    ("minimum", 2);
    ("and", 2);
    ("or", 2);
    ("xor", 2);
    ("equal", 2);
    ("not_equal", 2);
    ("less", 2);
    ("less_equal", 2);
    ("where", 3);
  ]

let test_ints =
  List.concat_map
    (fun t ->
      List.map
        (fun (kind, arity) ->
          prop (strf "%s %s" kind t.name) ~count:5000
            Gen.(array ~size:(constant arity) (gen_int t))
            (fun args ->
              let l = Array.to_list args in
              cover "an extreme"
                (List.exists (fun a -> a = wrap t Int64.min_int || a = 0L) l);
              equal int64 (int_meaning t kind l) (K.int t.name kind args)))
        int_kinds)
    types

(* Threefry-2x32-20: Random123's known answers, counter and key low word
   first. *)
let test_threefry =
  let pair lo hi = Int64.(logor (shift_left (of_int hi) 32) (of_int lo)) in
  cases
    ~name:(fun (c, k, _) -> strf "counter %016Lx key %016Lx" c k)
    "threefry"
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

(* Complex

   c64 and c128 against their meaning: Add, Sub and Neg are their parts' f32
   and f64 kinds, any NaN for a NaN but Neg's; Mul is within 2u, and Fdiv and Recip within
   4u, of the exact result, normwise, where parts lie in [2^-40, 2^40], u the
   part's unit roundoff; a NaN part in an operand of Mul, Fdiv or Recip makes
   both parts NaN; the order is the real part's, then the imaginary part's. *)


(* A part of magnitude in [2^-40, 2^40], either sign. *)
let part =
  let open Gen in
  let* m = float_range 1. 2. in
  let* e = int_range (-40) 40 in
  let+ s = bool in
  Float.ldexp (if s then -.m else m) e

let parts = Gen.(pair part part)

let c64_of (re, im) = [| bits32 (round_f32 re); bits32 (round_f32 im) |]
let pair_of a = (f a.(0), f a.(1))

let c64_call k ops = pair_of (K.c64 k (Array.concat (List.map c64_of ops)))

let c128_call k ops =
  let r = K.c128 k (Array.concat (List.map (fun (a, b) -> [| a; b |]) ops)) in
  (r.(0), r.(1))

let f32_pair (re, im) = (round_f32 re, round_f32 im)

(* Mul, Fdiv and Recip within their bounds, at c64 and c128. *)
let test_complex_bounds =
  let check name bound u call round exact =
    prop (strf "%s within %gu" name bound) ~count:20000
      Gen.(pair parts parts)
      (fun (a, b) ->
        let a = round a and b = round b in
        let z = exact a b in
        at_most float_exact ~than:bound (normwise u (call a b) z))
  in
  [
    check "mul c64" 2. 0x1p-24
      (fun a b -> c64_call "mul" [ a; b ])
      f32_pair exact_mul;
    check "mul c128" 2. 0x1p-53
      (fun a b -> c128_call "mul" [ a; b ])
      Fun.id exact_mul;
    check "fdiv c64" 4. 0x1p-24
      (fun a b -> c64_call "fdiv" [ a; b ])
      f32_pair exact_div;
    check "fdiv c128" 4. 0x1p-53
      (fun a b -> c128_call "fdiv" [ a; b ])
      Fun.id exact_div;
    check "recip c64" 4. 0x1p-24
      (fun _ b -> c64_call "recip" [ b ])
      f32_pair
      (fun _ b -> exact_div (1., 0.) b);
    check "recip c128" 4. 0x1p-53
      (fun _ b -> c128_call "recip" [ b ])
      Fun.id
      (fun _ b -> exact_div (1., 0.) b);
  ]

(* Add, Sub and Neg are their parts'; Neg keeps a NaN's payload. *)
let test_complex_parts =
  let two32 k (ar, ai, br, bi) =
    equal (array value32)
      [| K.f32 k [| ar; br |]; K.f32 k [| ai; bi |] |]
      (K.c64 k [| ar; ai; br; bi |])
  in
  let two64 k (ar, ai, br, bi) =
    equal (array value64)
      [| K.f64 k [| ar; br |]; K.f64 k [| ai; bi |] |]
      (K.c128 k [| ar; ai; br; bi |])
  in
  let gen32s = Gen.(pair (pair gen32 gen32) (pair gen32 gen32)) in
  let gen64s = Gen.(pair (pair gen64 gen64) (pair gen64 gen64)) in
  let flat ((a, b), (c, d)) = (a, b, c, d) in
  List.concat_map
    (fun k ->
      [
        prop (strf "%s c64" k) gen32s (fun x -> two32 k (flat x));
        prop (strf "%s c128" k) gen64s (fun x -> two64 k (flat x));
      ])
    [ "add"; "sub" ]
  @ [
      prop "neg c64" Gen.(pair gen32 gen32) (fun (a, b) ->
          equal (array exact32)
            [| K.f32 "neg" [| a |]; K.f32 "neg" [| b |] |]
            (K.c64 "neg" [| a; b |]));
      prop "neg c128" Gen.(pair gen64 gen64) (fun (a, b) ->
          equal (array value64)
            [| K.f64 "neg" [| a |]; K.f64 "neg" [| b |] |]
            (K.c128 "neg" [| a; b |]));
    ]

(* A NaN part in an operand of Mul, Fdiv or Recip makes both parts NaN. *)
let test_complex_nan =
  prop "a NaN part makes both parts of mul, fdiv and recip NaN"
    Gen.(triple parts parts (int_range 0 3))
    (fun ((ar, ai), (br, bi), at) ->
      let ops = [| ar; ai; br; bi |] in
      ops.(at) <- Float.nan;
      let a = (ops.(0), ops.(1)) and b = (ops.(2), ops.(3)) in
      let both (re, im) = Float.is_nan re && Float.is_nan im in
      equal bool true (both (c128_call "mul" [ a; b ]));
      equal bool true (both (c128_call "fdiv" [ a; b ]));
      equal bool true (both (c64_call "mul" [ a; b ]));
      equal bool true (both (c64_call "fdiv" [ a; b ]));
      if at >= 2 then begin
        equal bool true (both (c128_call "recip" [ b ]));
        equal bool true (both (c64_call "recip" [ b ]))
      end)

(* A quotient whose imaginary part is 0 while its real part overflows. *)
let test_complex_smith =
  test "fdiv keeps a zero part beside an infinite one" (fun () ->
      let want = (Float.neg_infinity, 0.) in
      let pp ppf (a, b) = Format.fprintf ppf "%h + %hi" a b in
      let t = Testable.make ~pp ~equal:( = ) in
      equal t want
        (c128_call "fdiv" [ (0., 0x1p-50); (0., -.Float.ldexp 1. (-1074)) ]);
      equal t want (c64_call "fdiv" [ (0., 0x1p-20); (0., -0x1p-149) ]))

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

(* The order: by the real part, then the imaginary part, -0 below +0 for the
   extremes; a NaN part ranks as NaN. *)
let test_complex_order =
  let small = Gen.of_list [ -1.; -0.; 0.; 1.; Float.infinity; Float.nan ] in
  let z = Gen.pair small small in
  let nan (a, b) = Float.is_nan a || Float.is_nan b in
  let cmp (ar, ai) (br, bi) =
    let one x y =
      if x = y then compare (Float.sign_bit y) (Float.sign_bit x)
      else compare x y
    in
    let c = one ar br in
    if c <> 0 then c else one ai bi
  in
  let meaning k a b =
    let n = nan a || nan b in
    let flag p = if p then (1., 0.) else (0., 0.) in
    match k with
    | "equal" -> flag (fst a = fst b && snd a = snd b)
    | "not_equal" -> flag (not (fst a = fst b && snd a = snd b))
    | "less" -> flag ((not n) && (fst a < fst b || (fst a = fst b && snd a < snd b)))
    | "less_equal" ->
        flag ((not n) && (fst a < fst b || (fst a = fst b && snd a <= snd b)))
    | "maximum" -> if nan a then a else if nan b then b else if cmp a b >= 0 then a else b
    | _ -> if nan a then a else if nan b then b else if cmp a b <= 0 then a else b
  in
  (* Each part's bits, any NaN for a NaN: the values travel through f32. *)
  let t = pair value64 value64 in
  List.map
    (fun k ->
      prop (strf "%s by real, then imaginary part" k) Gen.(pair z z)
        (fun (a, b) ->
          cover "NaN" (nan a || nan b);
          cover "equal reals" (fst a = fst b);
          equal t (meaning k a b) (c128_call k [ a; b ]);
          equal t (meaning k a b) (c64_call k [ a; b ])))
    [ "equal"; "not_equal"; "less"; "less_equal"; "maximum"; "minimum" ]

(* The complex transcendental kinds *)

let complex_kinds =
  [
    "exp"; "exp2"; "log"; "log2"; "sqrt"; "sign"; "sin"; "cos"; "tan"; "asin";
    "acos"; "atan"; "sinh"; "cosh"; "tanh"; "asinh"; "acosh"; "atanh";
  ]

(* |z' - z| / |z| of the moduli alone, in units of [u]. *)
let modulus_error u (re, im) (((zr, _), (zi, _)) : (float * float) * (float * float)) =
  let m = Float.hypot zr zi in
  Float.abs (Float.hypot re im -. m) /. m /. u

(* C99 Annex G's special values, G.6.1 to G.6.4, at c128 and c64. Where it
   leaves a sign unspecified, the value is nx_kinds.h's choice. A NaN part is
   any NaN. *)

let inf = Float.infinity
let ninf = Float.neg_infinity
let nan = Float.nan
let pi = Float.pi

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
    ("sqrt", (nan, nan), (nan, nan)); ("sqrt", (-4., 0.), (0., 2.));
    ("sqrt", (-4., -0.), (0., -2.));
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
    ("sign", (2., nan), (nan, nan)); ("sign", (-3., 4.), (-0.6, 0.8));
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

let test_complex_special =
  cases
    ~name:(fun (k, (x, y), _) -> strf "%s %h %h" k x y)
    "Annex G's special values" special_values
    (fun (kind, z, (re, im)) ->
      let t = pair value64 value64 in
      equal t (re, im) (c128_call kind [ z ]);
      equal t (round_f32 re, round_f32 im) (c64_call kind [ f32_pair z ]))

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
let signalling32 x =
  if Int64.equal (bits64 x) (bits64 snan) then 0x7F80_0001
  else if Int64.equal (bits64 x) (bits64 nsnan) then 0xFF80_0001
  else bits32 (round_f32 x)

let test_complex_signalling =
  cases
    ~name:(fun (k, (x, y), _) ->
      strf "%s 0x%016Lx 0x%016Lx" k (bits64 x) (bits64 y))
    "signalling NaN parts" signalling_values
    (fun (kind, ((x, y) as z), (re, im)) ->
      let t = pair value64 value64 in
      equal t (re, im) (c128_call kind [ z ]);
      equal t (round_f32 re, round_f32 im)
        (pair_of (K.c64 kind [| signalling32 x; signalling32 y |])))

(* Laws over drawn parts, NaNs, infinities, zeros and subnormals among
   them. Each holds bit for bit, any NaN for a NaN. *)

let gen_z64 = Gen.pair gen64 gen64
let gen_z32 = Gen.map (fun (a, b) -> (f a, f b)) (Gen.pair gen32 gen32)
let conj (re, im) = (re, Float.neg im)
let negz (re, im) = (Float.neg re, Float.neg im)
let both_parts = pair value64 value64

(* For each kind, f(conj z) = conj f(z), C99's G.6's symmetry. *)
let test_complex_conj =
  List.concat_map
    (fun k ->
      [
        prop (strf "%s c128 of conj z is its conj" k) gen_z64 (fun z ->
            equal both_parts
              (conj (c128_call k [ z ]))
              (c128_call k [ conj z ]));
        prop (strf "%s c64 of conj z is its conj" k) gen_z32 (fun z ->
            equal both_parts (conj (c64_call k [ z ])) (c64_call k [ conj z ]));
      ])
    complex_kinds

(* The odd kinds give -f(z) at -z, the even ones f(z). *)
let test_complex_parity =
  let odd =
    [ "sinh"; "tanh"; "sin"; "tan"; "asin"; "atan"; "asinh"; "atanh"; "sign" ]
  in
  let even = [ "cosh"; "cos" ] in
  List.concat_map
    (fun k ->
      let side v = if List.mem k odd then negz v else v in
      [
        prop (strf "%s c128 at -z" k) gen_z64 (fun z ->
            equal both_parts
              (side (c128_call k [ z ]))
              (c128_call k [ negz z ]));
        prop (strf "%s c64 at -z" k) gen_z32 (fun z ->
            equal both_parts (side (c64_call k [ z ])) (c64_call k [ negz z ]));
      ])
    (odd @ even)

(* On the real axis, where the float kind is defined, the real part is the
   float kind's value. *)
let test_complex_real_axis =
  let unit64 =
    let ends = Gen.of_list [ 0.; -0.; 1.; -1.; 0x1p-1074 ] in
    Gen.(frequency [ (3, float_range (-1.) 1.); (1, ends) ])
  in
  let unit32 = Gen.map bits32 unit64 in
  let positive64 = Gen.map Float.abs gen64 in
  let positive32 = Gen.map (fun b -> b land 0x7FFF_FFFF) gen32 in
  let real =
    [
      ("exp", gen64, gen32); ("exp2", gen64, gen32); ("sinh", gen64, gen32);
      ("cosh", gen64, gen32); ("tanh", gen64, gen32); ("sin", gen64, gen32);
      ("cos", gen64, gen32); ("tan", gen64, gen32); ("atan", gen64, gen32);
      ("sqrt", positive64, positive32); ("asin", unit64, unit32);
      ("acos", unit64, unit32);
    ]
  in
  List.concat_map
    (fun (k, g64, g32) ->
      [
        prop (strf "%s c128 at x + 0i is the float kind's" k) g64 (fun x ->
            equal value64 (K.f64 k [| x |]) (fst (c128_call k [ (x, 0.) ])));
        prop (strf "%s c64 at x + 0i is the float kind's" k) g32 (fun x ->
            equal value64
              (f (K.f32 k [| x |]))
              (fst (c64_call k [ (f x, 0.) ])));
      ])
    real
  @ List.map
      (fun k ->
        prop (strf "%s c128 at x + 0i is the float kind's of |x|" k) gen64
          (fun x ->
            equal value64
              (K.f64 k [| Float.abs x |])
              (fst (c128_call k [ (x, 0.) ]))))
      [ "log"; "log2" ]
  @ [
      prop "pow c128 of positive reals is the float kind's"
        (Gen.pair gen64 gen64)
        (fun (x, w) ->
          assume (x > 0.);
          equal value64 (K.f64 "pow" [| x; w |])
            (fst (c128_call "pow" [ (x, 0.); (w, 0.) ])));
    ]

(* exp2 at every finite y: |2^(x + iy)| = 2^x for y drawn over the whole
   range of the part's format, and its phase y ln2 within exp2's bound at
   y up to the largest finite part, where it takes y ln2's every bit:
   values from mpmath. *)
let test_complex_exp2_far =
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
  let phase u call y (c, s) =
    let got = call "exp2" [ (0., y) ] in
    at_most float_exact ~than:4. (normwise u got ((c, 0.), (s, 0.)))
      ~msg:(strf "exp2 %h i: got %h %h" y (fst got) (snd got))
  in
  [
    check "c128" 0x1p-53 c128_call Fun.id ~emin:(-1074) ~emax:1023 ~p:53;
    check "c64" 0x1p-24 c64_call f32_pair ~emin:(-149) ~emax:127 ~p:24;
    test "exp2's phase far up the imaginary axis" (fun () ->
        let c128 = phase 0x1p-53 c128_call and c64 = phase 0x1p-24 c64_call in
        c128 0x1p30 (0x1.66e39f669c166p-1, -0x1.6d2939368da98p-1);
        c128 0x1.5eb851eb851ecp+74 (0x1.59f6745a8633cp-1, 0x1.796e36e47cb4ap-1);
        c128 1e300 (0x1.b80528d30f790p-1, -0x1.05c67494c0ab7p-1);
        c128 Float.max_float (0x1.cedaf85d0796ep-1, -0x1.b5c31ba408c70p-2);
        c64 0x1p30 (0x1.66e39f669c166p-1, -0x1.6d2939368da98p-1);
        c64 0x1.5eb852p+74 (-0x1.f0744683c33c2p-1, -0x1.f4d5bd9a1b741p-3);
        c64 0x1.fffffep+127 (0x1.8a07e4d1b0d09p-1, -0x1.46ee442fa86e9p-1));
  ]

(* pow at the values its .mli states: 1 where the exponent is 0, any base
   included; 0 at a zero base where the exponent's real part is positive;
   the sides of the negative real axis by the sign of a zero imaginary
   part. *)
let test_complex_pow_special =
  let one = pair value64 value64 in
  let check call round =
    List.iter
      (fun z -> equal one (1., 0.) (call "pow" [ round z; (0., 0.) ]))
      [ (nan, nan); (0., 0.); (inf, 0.); (2., 3.); (ninf, nan) ];
    equal one (1., 0.) (call "pow" [ round (2., 3.); (-0., 0.) ]);
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
          (Float.abs (re -. want) /. Float.abs want /. u)
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

(* Identities within the bounds the kinds' compose to, at parts in
   [2^-40, 2^40]: exp (log z) = z within (5 + 4|log z|)u, the exponential
   carrying log's absolute error into a relative one; sqrt z * sqrt z = z
   within 8u, twice sqrt's bound and mul's. *)
let test_complex_identities =
  let modulus (re, im) = Float.hypot re im in
  let dd z = let re, im = z in ((re, 0.), (im, 0.)) in
  let check name u call round =
    [
      prop (strf "exp (log z) = z %s" name) parts (fun z ->
          let z = round z in
          let l = call "log" [ z ] in
          let bound = 5. +. (4. *. modulus l) in
          at_most float_exact ~than:bound
            (normwise u (call "exp" [ l ]) (dd z)));
      prop (strf "sqrt z * sqrt z = z %s" name) parts (fun z ->
          let z = round z in
          let s = call "sqrt" [ z ] in
          at_most float_exact ~than:8.
            (normwise u (call "mul" [ s; s ]) (dd z)));
    ]
  in
  check "c128" 0x1p-53 c128_call Fun.id @ check "c64" 0x1p-24 c64_call f32_pair

(* C99 Annex F's special values of the float kinds, F.10, and results the
   exact value rounds to an infinity or a zero, where a rank's distance
   bounds nothing: an infinity is one rank past the largest float. Bit for
   bit, any NaN for a NaN, through a kind's loop at one element and over a
   vector, every lane alike. A row of [Both] widths holds the
   f64 result, exact or correctly rounded, whose rounding to binary32 is the
   f32 one; a row of one width holds that width's operands and result. *)

type width = Both | F64 | F32

let annex_f =
  let pio2 = 0x1.921fb54442d18p0 and pio4 = 0x1.921fb54442d18p-1 in
  let tpio4 = 0x1.2d97c7f3321d2p1 in
  let rows w k = List.map (fun (args, r) -> (w, k, args, r)) in
  let one ?(w = Both) k rs =
    rows w k (List.map (fun (x, r) -> ([| x |], r)) rs)
  in
  let two ?(w = Both) k rs =
    rows w k (List.map (fun (x, y, r) -> ([| x; y |], r)) rs)
  in
  let exp_like k =
    one k [ (0., 1.); (-0., 1.); (ninf, 0.); (inf, inf); (nan, nan) ]
  in
  let log_like k =
    one k
      [
        (0., ninf); (-0., ninf); (1., 0.); (-1., nan); (ninf, nan); (inf, inf);
        (nan, nan);
      ]
  in
  let odd_periodic k =
    one k [ (0., 0.); (-0., -0.); (inf, nan); (ninf, nan); (nan, nan) ]
  in
  List.concat
    [
      exp_like "exp";
      one "exp" [ (710., inf); (-1000., 0.) ];
      one ~w:F32 "exp" [ (89., inf); (-110., 0.) ];
      exp_like "exp2";
      one "exp2" [ (1024., inf); (-1100., 0.); (-2000., 0.) ];
      one ~w:F32 "exp2" [ (128., inf); (-151., 0.) ];
      one "expm1"
        [ (0., 0.); (-0., -0.); (ninf, -1.); (inf, inf); (nan, nan);
          (710., inf) ];
      one ~w:F32 "expm1" [ (89., inf) ];
      log_like "log";
      log_like "log2";
      one "log1p"
        [
          (0., 0.); (-0., -0.); (-1., ninf); (-2., nan); (ninf, nan);
          (inf, inf); (nan, nan);
        ];
      odd_periodic "sin";
      odd_periodic "tan";
      one "cos" [ (0., 1.); (-0., 1.); (inf, nan); (ninf, nan); (nan, nan) ];
      one "asin" [ (0., 0.); (-0., -0.); (2., nan); (-2., nan); (nan, nan) ];
      one "acos" [ (1., 0.); (2., nan); (-2., nan); (nan, nan) ];
      one "atan"
        [ (0., 0.); (-0., -0.); (inf, pio2); (ninf, -.pio2); (nan, nan) ];
      one "sinh"
        [ (0., 0.); (-0., -0.); (inf, inf); (ninf, ninf); (nan, nan);
          (711., inf); (-711., ninf) ];
      one ~w:F32 "sinh" [ (90., inf); (-90., ninf) ];
      one "cosh"
        [ (0., 1.); (-0., 1.); (inf, inf); (ninf, inf); (nan, nan);
          (711., inf); (-711., inf) ];
      one ~w:F32 "cosh" [ (90., inf); (-90., inf) ];
      one "tanh" [ (0., 0.); (-0., -0.); (inf, 1.); (ninf, -1.); (nan, nan) ];
      one "erf" [ (0., 0.); (-0., -0.); (inf, 1.); (ninf, -1.); (nan, nan) ];
      (* atan2 y x *)
      two "atan2"
        [
          (0., -0., pi); (-0., -0., -.pi); (0., 0., 0.); (-0., 0., -0.);
          (0., -1., pi); (-0., -1., -.pi); (0., 1., 0.); (-0., 1., -0.);
          (-1., 0., -.pio2); (-1., -0., -.pio2); (1., 0., pio2);
          (1., -0., pio2); (1., ninf, pi); (-1., ninf, -.pi); (1., inf, 0.);
          (-1., inf, -0.); (inf, 1., pio2); (ninf, 1., -.pio2);
          (inf, ninf, tpio4); (ninf, ninf, -.tpio4); (inf, inf, pio4);
          (ninf, inf, -.pio4); (nan, 1., nan); (1., nan, nan);
        ];
      two ~w:F64 "atan2" [ (-0x1p-1074, 1e300, -0.); (0x1p-1074, -1e300, pi) ];
      two ~w:F32 "atan2"
        [ (-0x1p-149, 0x1p127, -0.); (0x1p-149, -0x1p127, round_f32 pi) ];
      two "pow"
        [
          (-0., -3., ninf); (0., -3., inf); (0., ninf, inf); (-0., ninf, inf);
          (0., -2., inf); (-0., -2., inf); (-0., -0.5, inf); (-0., 3., -0.);
          (0., 3., 0.); (-0., 2., 0.); (-0., 0.5, 0.); (-1., inf, 1.);
          (-1., ninf, 1.); (1., nan, 1.); (1., ninf, 1.); (nan, 0., 1.);
          (nan, -0., 1.); (inf, 0., 1.); (-2., 0.5, nan); (0.5, ninf, inf);
          (-0.5, ninf, inf); (2., ninf, 0.); (-2., ninf, 0.); (0.5, inf, 0.);
          (2., inf, inf); (ninf, -3., -0.); (ninf, -2., 0.); (ninf, -0.5, 0.);
          (ninf, 3., ninf); (ninf, 2., inf); (ninf, 0.5, inf); (inf, -1., 0.);
          (inf, 0.5, inf); (nan, 1., nan); (2., nan, nan);
          (-0x1p-60, 3., -0x1p-180); (-0x1p-600, 3., -0.);
          (2., 1024., inf); (-2., 1025., ninf); (2., -1100., 0.);
          (-2., -1101., -0.); (0.5, 1100., 0.); (-0.5, 1101., -0.);
        ];
      (* Odd integers end where the format's integers grow even. *)
      two ~w:F64 "pow"
        [
          (-0., 0x1.fffffffffffffp52, -0.); (-0., 0x1p53, 0.);
          (ninf, -0x1.fffffffffffffp52, -0.); (ninf, -0x1p53, 0.);
          (ninf, 0x1.fffffffffffffp52, ninf); (ninf, 0x1p53, inf);
          (-1., 0x1.fffffffffffffp52, -1.); (-1., 0x1p53, 1.);
        ];
      two ~w:F32 "pow"
        [
          (-0., 0x1.fffffep23, -0.); (-0., 0x1p24, 0.);
          (ninf, -0x1.fffffep23, -0.); (ninf, -0x1p24, 0.);
          (ninf, 0x1.fffffep23, ninf); (ninf, 0x1p24, inf);
          (-1., 0x1.fffffep23, -1.); (-1., 0x1p24, 1.);
          (2., 128., inf); (-2., 129., ninf); (2., -151., 0.);
          (-2., -151., -0.);
        ];
    ]

let test_annex_f =
  let name (w, k, args, _) =
    strf "%s %s %s"
      (match w with Both -> "f64 and f32" | F64 -> "f64" | F32 -> "f32")
      k
      (String.concat " " (Array.to_list (Array.map (strf "%h") args)))
  in
  let lanes (type s) (k : (float, s) Bigarray.kind) kind args =
    let fill v = Bigarray.Array1.init k Bigarray.c_layout 37 (fun _ -> v) in
    let y = fill 0. in
    let z = if Array.length args > 1 then args.(1) else 0. in
    K.run kind (fill args.(0)) (fill z) y;
    List.init 37 (fun i -> y.{i})
  in
  let at64 kind args want =
    equal value64 want (K.f64 kind args);
    equal (list value64) (List.init 37 (fun _ -> want))
      (lanes Bigarray.float64 kind args)
  in
  let at32 kind args want =
    equal value64 want (f (K.f32 kind (Array.map bits32 args)));
    equal (list value64) (List.init 37 (fun _ -> want))
      (lanes Bigarray.float32 kind args)
  in
  cases ~name "Annex F's special values" annex_f (fun (w, kind, args, want) ->
      match w with
      | F64 -> at64 kind args want
      | F32 -> at32 kind args want
      | Both ->
          at64 kind args want;
          at32 kind (Array.map round_f32 args) (round_f32 want))

(* Digests of the float kinds over a fixed point set

   What every library computing a kind through nx_kinds.h gives: each f32
   and f64 kind of one or two operands over Annex F's operands and, in every
   binade of either sign, its first and last pattern and six drawn
   significands; a kind of two operands at every pair of Annex F's operands
   and at each binade point paired with another. The digest is MD5 over the
   results' bits, every NaN as one NaN. *)

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
    let top = logor (shift_left (of_int sign) (width - 1)) (shift_left (of_int e) fraction) in
    List.map
      (fun frac -> of_bits (logor top frac))
      (0L :: sub (shift_left 1L fraction) 1L :: List.init 6 (fun _ -> draw ()))
  in
  annex_f_operands
  @ List.concat_map (fun sign -> List.concat_map (binade sign) (List.init binades Fun.id)) [ 0; 1 ]

let digest_kinds =
  ( [ "exp"; "exp2"; "expm1"; "log"; "log2"; "log1p"; "sin"; "cos"; "tan";
      "asin"; "acos"; "atan"; "sinh"; "cosh"; "tanh"; "erf"; "neg"; "recip";
      "abs"; "sign"; "sqrt"; "floor"; "ceil"; "round"; "trunc" ],
    [ "add"; "sub"; "mul"; "fdiv"; "mod"; "pow"; "atan2"; "maximum";
      "minimum"; "equal"; "not_equal"; "less"; "less_equal" ] )

let test_digests =
  test "digests of every float kind, the same on every target" (fun () ->
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
        let ba l = Bigarray.Array1.of_array k Bigarray.c_layout (Array.of_list l) in
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
        List.map (fun name -> strf "%s %s %s" name ty (digest k ~bits name ps ps)) unaries
        @ List.map (fun name -> strf "%s %s %s" name ty (digest k ~bits name xs zs)) binaries
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
        neg f32 4cd8b9f26caee951
        recip f32 c893237dcd6689a7
        abs f32 0c6bc9e2df075c81
        sign f32 793e46243ee50087
        sqrt f32 d70441c0b52ee39f
        floor f32 36aab323384d7621
        ceil f32 31b039f5a050d158
        round f32 6f14dbaf407b10e4
        trunc f32 9fbeccc1f44c1ca4
        add f32 66a8f8de796eb6b1
        sub f32 ab671dcc371bfd85
        mul f32 735351c38d3f9707
        fdiv f32 8f96d3ecbb1e95e5
        mod f32 18cd8a08dbcdfa2b
        pow f32 f84c9d0db0dcfe2a
        atan2 f32 486465bc2d1c5df8
        maximum f32 2bdcf68d860d9fb7
        minimum f32 6b882342775a9223
        equal f32 6a8a7670224e59ea
        not_equal f32 4227f0a35da257f7
        less f32 4f324d5e38c1da8e
        less_equal f32 f0602177e93e3205
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
        neg f64 58bf1dce50ce5715
        recip f64 12be1d5649e0e1a3
        abs f64 34647d3ef73113e1
        sign f64 2d8a13979ca8f72e
        sqrt f64 89507a0b4bd4871b
        floor f64 89cbfa641d5fb8ba
        ceil f64 681f0cc69ffa1a43
        round f64 b897f9686c38667f
        trunc f64 211778a8aab717b0
        add f64 23b9e8138c933dea
        sub f64 fb1334bcd9891671
        mul f64 ca54f9e5f2153e9e
        fdiv f64 bd7fbfa5e638ae7a
        mod f64 e2883cecda254de3
        pow f64 de2d29a86133409e
        atan2 f64 28f45baa0494ac7e
        maximum f64 25694c6f0f9f0fcb
        minimum f64 f25e39f1e49785f2
        equal f64 7f673d784fb01b37
        not_equal f64 4df55ea11a62355c
        less f64 ba9c4bbd090e7e09
        less_equal f64 48f19170c3ce04c4
        |})

let () =
  exit
  @@ run "nx_array.kinds"
       [
         group ~timeout:10. "exact kinds give IEEE results"
           (test_binary32
           @ [ test_fma32; test_where32 ]
           @ test_unary32 @ test_f64_exact
           @ [ test_narrow_zeros; test_exact_points ]);
         group ~timeout:60. "narrow floats compute in f32 and round once"
           [ test_narrow_transcendentals; test_narrow_pairs ];
         group ~timeout:10. "integer kinds are modular" test_ints;
         group ~timeout:10. "threefry's known answers" [ test_threefry ];
         group ~timeout:10. "Annex F's special values" [ test_annex_f ];
         group ~timeout:10. "the header's bits on every target" [ test_digests ];
         group ~timeout:10. "complex arithmetic within its bounds"
           (test_complex_bounds @ test_complex_subnormal @ test_complex_parts
           @ test_complex_order
           @ [ test_complex_nan; test_complex_smith ]);
         group ~timeout:10. "complex kinds at their special values"
           [ test_complex_special; test_complex_signalling ];
         group ~timeout:10. "complex kinds by their symmetries"
           (test_complex_exp2_far @ test_complex_pow_special
           @ [ test_complex_log_circle ]
           @ test_complex_conj @ test_complex_parity @ test_complex_real_axis
           @ test_complex_identities);
       ]
