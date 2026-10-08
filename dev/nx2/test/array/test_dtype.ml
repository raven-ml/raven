(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype

let strf = Printf.sprintf
let pp_hex ppf x = Format.fprintf ppf "%h" x
let any_name (D.Any dt) = D.name dt

(* A witness of [dt]'s values that prints them as [dt] does and compares floats
   bit for bit. *)
let value : type v s. (v, s) D.t -> v testable =
 fun dt ->
  match D.kind dt with
  | D.Float ->
      Testable.make ~pp:(D.pp_value dt) ~equal:(Testable.equal float_exact)
  | D.Complex ->
      Testable.make ~pp:(D.pp_value dt) ~equal:(fun (a : Complex.t) b ->
          Testable.equal float_exact a.re b.re
          && Testable.equal float_exact a.im b.im)
  | D.Boolean -> bool
  | D.Signed | D.Unsigned -> Testable.make ~pp:(D.pp_value dt) ~equal:( = )

(* The table, as the interface states it *)

type kind = Float | Complex | Signed | Unsigned | Boolean

let kind_name = function
  | Float -> "Float"
  | Complex -> "Complex"
  | Signed -> "Signed"
  | Unsigned -> "Unsigned"
  | Boolean -> "Boolean"

let kind_w =
  Testable.make
    ~pp:(fun ppf k -> Format.pp_print_string ppf (kind_name k))
    ~equal:( = )

let kind_of : type v s. (v, s) D.t -> kind =
 fun dt ->
  match D.kind dt with
  | D.Float -> Float
  | D.Complex -> Complex
  | D.Signed -> Signed
  | D.Unsigned -> Unsigned
  | D.Boolean -> Boolean

(* Each dtype's name, width and kind. *)
let table =
  [
    (D.Any D.Float64, "float64", 64, Float);
    (D.Any D.Float32, "float32", 32, Float);
    (D.Any D.Float16, "float16", 16, Float);
    (D.Any D.Bfloat16, "bfloat16", 16, Float);
    (D.Any D.Float8_e4m3fn, "float8_e4m3fn", 8, Float);
    (D.Any D.Float8_e5m2, "float8_e5m2", 8, Float);
    (D.Any D.Float4_e2m1fn, "float4_e2m1fn", 4, Float);
    (D.Any D.Int64, "int64", 64, Signed);
    (D.Any D.Uint64, "uint64", 64, Unsigned);
    (D.Any D.Int32, "int32", 32, Signed);
    (D.Any D.Uint32, "uint32", 32, Unsigned);
    (D.Any D.Int16, "int16", 16, Signed);
    (D.Any D.Uint16, "uint16", 16, Unsigned);
    (D.Any D.Int8, "int8", 8, Signed);
    (D.Any D.Uint8, "uint8", 8, Unsigned);
    (D.Any D.Int4, "int4", 4, Signed);
    (D.Any D.Uint4, "uint4", 4, Unsigned);
    (D.Any D.Complex128, "complex128", 128, Complex);
    (D.Any D.Complex64, "complex64", 64, Complex);
    (D.Any D.Bool, "bool", 8, Boolean);
    (D.Any D.Bit, "bit", 1, Boolean);
  ]

let row_name (_, name, _, _) = name

let test_all () =
  equal
    (slist string String.compare)
    (List.map row_name table) (List.map any_name D.all)

let test_codes () =
  List.iteri
    (fun i (D.Any dt) -> equal ~msg:(D.name dt) int i (D.code dt))
    D.all

let test_row (D.Any dt, name, bits, kind) =
  equal string name (D.name dt);
  equal string name (Format.asprintf "%a" D.pp dt);
  equal int bits (D.bits dt);
  equal kind_w kind (kind_of dt)

(* [is k] for each kind. *)
let is_kind =
  [
    (Float, fun (D.Any dt) -> D.is D.Float dt);
    (Complex, fun (D.Any dt) -> D.is D.Complex dt);
    (Signed, fun (D.Any dt) -> D.is D.Signed dt);
    (Unsigned, fun (D.Any dt) -> D.is D.Unsigned dt);
    (Boolean, fun (D.Any dt) -> D.is D.Boolean dt);
  ]

let test_is () =
  List.iter
    (fun (k, is) ->
      List.iter
        (fun ((D.Any dt as a), _, _, kind) ->
          equal
            ~msg:(strf "is %s %s" (kind_name k) (D.name dt))
            bool (kind = k) (is a))
        table)
    is_kind

(* The index of each kind in nx_dtype.h's enum nx_kind. *)
let kind_index = function
  | Float -> 0
  | Complex -> 1
  | Signed -> 2
  | Unsigned -> 3
  | Boolean -> 4

let test_header (D.Any dt, name, bits, kind) =
  equal
    (option (triple string int int))
    (Some (name, bits, kind_index kind))
    (Nx_array_support.row (D.code dt))

let test_no_row_past_the_last () =
  equal (option pass) None (Nx_array_support.row (List.length table))

let names_refused =
  [
    "";
    "Float32";
    "FLOAT32";
    " float32";
    "float32 ";
    "float";
    "float8_e4m3";
    "float4_e2m1";
    "float8_e4m3fnuz";
    "float8_e5m2fnuz";
    "float8_e8m0fnu";
    "complex32";
    "uint1";
  ]

let test_of_name (D.Any dt, name, _, _) =
  match D.of_name name with
  | Some (D.Any found) -> equal ~msg:"code" int (D.code dt) (D.code found)
  | None -> failf "of_name %S is None" name

let test_of_other_name s =
  equal (option string) None (Option.map any_name (D.of_name s))

let test_equal () =
  List.iter
    (fun (D.Any a) ->
      List.iter
        (fun (D.Any b) ->
          let same = D.name a = D.name b in
          let msg = strf "%s, %s" (D.name a) (D.name b) in
          equal ~msg bool same (D.equal a b);
          equal ~msg bool same (Option.is_some (D.equal_witness a b)))
        D.all)
    D.all

(* Bytes *)

(* ⌈n·b/8⌉ with n = 8q + r, or [None] past [max_int]. *)
let bytes_reference b n =
  let q = n / 8 and r = n mod 8 in
  let tail = ((r * b) + 7) / 8 in
  if q > (max_int - tail) / b then None else Some ((q * b) + tail)

(* Counts about [dt]'s bounds: small ones, the greatest count whose bytes fit
   and its neighbours, and the extremes. *)
let counts (D.Any dt) =
  let last = max_int / max 1 (D.bits dt / 8) in
  Gen.frequency
    [
      (4, Gen.int_range (-2) 40);
      (3, Gen.of_list ~pp:Format.pp_print_int [ last - 1; last; last + 1 ]);
      (1, Gen.of_list ~pp:Format.pp_print_int [ max_int; min_int; -1 ]);
    ]

let law_bytes (D.Any dt) =
  prop (D.name dt) (counts (D.Any dt)) (fun n ->
      let b = D.bits dt in
      cover "a negative count" (n < 0);
      if b < 8 then cover "rounds a partial byte up" (n > 0 && n * b mod 8 <> 0);
      if b > 8 then cover "overflows" (n > 0 && bytes_reference b n = None);
      match bytes_reference b n with
      | Some v when n >= 0 -> equal int v (D.bytes dt n)
      | _ -> raises_match Exn.invalid_arg (fun () -> D.bytes dt n))

(* Values: identities and limits *)

(* A dtype, its zero and one, and its least and greatest values; [None] where
   the limits raise. *)
type values = Values : ('v, 's) D.t * 'v * 'v * ('v * 'v) option -> values

let c re = { Complex.re; im = 0. }
let inf = Float.infinity
let nan = Float.nan

let values =
  [
    Values (D.Float64, 0., 1., Some (-.inf, inf));
    Values (D.Float32, 0., 1., Some (-.inf, inf));
    Values (D.Float16, 0., 1., Some (-.inf, inf));
    Values (D.Bfloat16, 0., 1., Some (-.inf, inf));
    Values (D.Float8_e4m3fn, 0., 1., Some (-448., 448.));
    Values (D.Float8_e5m2, 0., 1., Some (-.inf, inf));
    Values (D.Float4_e2m1fn, 0., 1., Some (-6., 6.));
    Values (D.Int64, 0L, 1L, Some (Int64.min_int, Int64.max_int));
    Values (D.Uint64, 0L, 1L, Some (0L, -1L));
    Values (D.Int32, 0l, 1l, Some (Int32.min_int, Int32.max_int));
    Values (D.Uint32, 0l, 1l, Some (0l, -1l));
    Values (D.Int16, 0, 1, Some (-32768, 32767));
    Values (D.Uint16, 0, 1, Some (0, 65535));
    Values (D.Int8, 0, 1, Some (-128, 127));
    Values (D.Uint8, 0, 1, Some (0, 255));
    Values (D.Int4, 0, 1, Some (-8, 7));
    Values (D.Uint4, 0, 1, Some (0, 15));
    Values (D.Complex128, c 0., c 1., None);
    Values (D.Complex64, c 0., c 1., None);
    Values (D.Bool, false, true, Some (false, true));
    Values (D.Bit, false, true, Some (false, true));
  ]

let values_name (Values (dt, _, _, _)) = D.name dt

let test_identities (Values (dt, zero, one, _)) =
  equal ~msg:"zero" (value dt) zero (D.zero dt);
  equal ~msg:"one" (value dt) one (D.one dt)

let test_limits (Values (dt, _, _, limits)) =
  match limits with
  | Some (lo, hi) ->
      equal ~msg:"min_value" (value dt) lo (D.min_value dt);
      equal ~msg:"max_value" (value dt) hi (D.max_value dt)
  | None ->
      raises_match Exn.invalid_arg (fun () -> D.min_value dt);
      raises_match Exn.invalid_arg (fun () -> D.max_value dt)

(* Float formats, by their definitions *)

(* What an all-ones exponent field encodes: an infinity with a zero fraction and
   NaN otherwise ([Ieee]), NaN with an all-ones fraction and a finite value
   otherwise ([Nan_only]), or finite values ([Finite]). *)
type top = Ieee | Nan_only | Finite
type format = { exp : int; frac : int; bias : int; top : top }
type float_dtype = F : (float, 's) D.t * format -> float_dtype

let formats =
  [
    F (D.Float64, { exp = 11; frac = 52; bias = 1023; top = Ieee });
    F (D.Float32, { exp = 8; frac = 23; bias = 127; top = Ieee });
    F (D.Float16, { exp = 5; frac = 10; bias = 15; top = Ieee });
    F (D.Bfloat16, { exp = 8; frac = 7; bias = 127; top = Ieee });
    F (D.Float8_e4m3fn, { exp = 4; frac = 3; bias = 7; top = Nan_only });
    F (D.Float8_e5m2, { exp = 5; frac = 2; bias = 15; top = Ieee });
    F (D.Float4_e2m1fn, { exp = 2; frac = 1; bias = 1; top = Finite });
  ]

let format_name (F (dt, _)) = D.name dt

(* The value of the sign-less code [c] of [f]. *)
let decode f c =
  let e = c lsr f.frac and m = c land ((1 lsl f.frac) - 1) in
  let all_ones = e = (1 lsl f.exp) - 1 in
  match f.top with
  | Ieee when all_ones -> if m = 0 then inf else nan
  | Nan_only when all_ones && m = (1 lsl f.frac) - 1 -> nan
  | _ ->
      if e = 0 then Float.ldexp (Float.of_int m) (1 - f.bias - f.frac)
      else
        Float.ldexp (Float.of_int (m lor (1 lsl f.frac))) (e - f.bias - f.frac)

(* The code of [f]'s largest finite value. *)
let last f =
  match f.top with
  | Ieee -> (((1 lsl f.exp) - 1) lsl f.frac) - 1
  | Nan_only -> (1 lsl (f.exp + f.frac)) - 2
  | Finite -> (1 lsl (f.exp + f.frac)) - 1

(* The value of code [c] as if the exponent were unbounded: past [last f], the
   values the format would have next. *)
let unbounded f c = decode { f with top = Finite } c

(* The epsilon, least normal and largest finite value of [f]; binary64's codes
   do not fit in an [int]. *)
let facts (F (dt, f)) =
  match dt with
  | D.Float64 -> (Float.epsilon, Float.min_float, Float.max_float)
  | _ -> (Float.ldexp 1. (-f.frac), decode f (1 lsl f.frac), decode f (last f))

let test_float_format (F (dt, f) as fd) =
  let ff = D.float_format dt in
  let epsilon, min_normal, max_finite = facts fd in
  equal int f.exp ff.exponent_bits;
  equal int f.frac ff.fraction_bits;
  equal bool (f.top = Ieee) ff.infinities;
  equal bool (f.top <> Finite) ff.nans;
  equal float_exact epsilon ff.epsilon;
  equal float_exact min_normal ff.min_normal;
  equal float_exact max_finite ff.max_finite

(* Stores *)

(* Stores of the rule's special values and of values a user names. Each row is a
   dtype, an input and the value the store holds. *)
type store = Store : ('v, 's) D.t * float * 'v -> store

let store_name (Store (dt, x, _)) =
  if Float.is_nan x then strf "%s nan %Lx" (D.name dt) (Int64.bits_of_float x)
  else strf "%s %h" (D.name dt) x

let test_store (Store (dt, x, v)) = equal (value dt) v (D.of_float dt x)
let snan = Int64.float_of_bits 0x7FF0000000000001L

let conversion_table =
  [
    (* NaN *)
    Store (D.Float64, nan, nan);
    Store (D.Float32, nan, nan);
    Store (D.Float16, nan, nan);
    Store (D.Bfloat16, nan, nan);
    Store (D.Float8_e5m2, nan, nan);
    Store (D.Float8_e4m3fn, nan, nan);
    Store (D.Float4_e2m1fn, nan, 0.);
    Store (D.Float4_e2m1fn, -.nan, 0.);
    (* A signalling NaN, whose top bits alone read as an infinity. *)
    Store (D.Float32, snan, nan);
    Store (D.Float16, snan, nan);
    Store (D.Bfloat16, snan, nan);
    Store (D.Float8_e5m2, snan, nan);
    Store (D.Float8_e4m3fn, snan, nan);
    Store (D.Float4_e2m1fn, snan, 0.);
    (* Infinities *)
    Store (D.Float64, inf, inf);
    Store (D.Float32, -.inf, -.inf);
    Store (D.Float16, inf, inf);
    Store (D.Bfloat16, -.inf, -.inf);
    Store (D.Float8_e5m2, inf, 57344.);
    Store (D.Float8_e5m2, -.inf, -57344.);
    Store (D.Float8_e4m3fn, inf, 448.);
    Store (D.Float8_e4m3fn, -.inf, -448.);
    Store (D.Float4_e2m1fn, inf, 6.);
    Store (D.Float4_e2m1fn, -.inf, -6.);
    (* Overflow is judged after rounding. *)
    Store (D.Float16, 65519., 65504.);
    Store (D.Float16, 65520., inf);
    Store (D.Float16, -65520., -.inf);
    Store (D.Float32, 0x1.ffffffp127, inf);
    Store (D.Float8_e5m2, 61440., 57344.);
    Store (D.Float8_e5m2, -1e300, -57344.);
    Store (D.Float8_e4m3fn, 464., 448.);
    Store (D.Float8_e4m3fn, 1e300, 448.);
    Store (D.Float4_e2m1fn, 7., 6.);
    Store (D.Float4_e2m1fn, -1e300, -6.);
    (* Nearest, once from the double. *)
    Store (D.Float16, 0.1, 0x1.998p-4);
    Store (D.Float32, 0.1, 0x1.99999ap-4);
    Store (D.Float16, 1. +. 0x1p-11 +. 0x1p-40, 1. +. 0x1p-10);
    (* Integers truncate toward zero, saturate, and store NaN as 0. *)
    Store (D.Int8, 300., 127);
    Store (D.Int8, -300., -128);
    Store (D.Int8, -2.7, -2);
    Store (D.Int32, 3e9, Int32.max_int);
    Store (D.Int64, 1e19, Int64.max_int);
    Store (D.Uint8, nan, 0);
    Store (D.Uint8, -1., 0);
    Store (D.Uint32, nan, 0l);
    Store (D.Uint32, 4294967295., -1l);
    Store (D.Uint64, nan, 0L);
    Store (D.Uint64, 1e20, -1L);
    Store (D.Int4, 8., 7);
    Store (D.Uint4, 16., 15);
    (* Complex numbers and booleans *)
    Store (D.Complex128, 0.1, c 0.1);
    Store (D.Complex64, 0.1, c 0x1.99999ap-4);
    Store (D.Bool, -0., false);
    Store (D.Bool, nan, true);
    Store (D.Bit, -2., true);
  ]

(* Inputs around the neighbouring values [a] and [b] of [f], codes [c] and [c +
   1], with the values they store: [a], [b], their midpoint, which rounds to the
   even code, and a double either side of it. Past the largest finite value, [b]
   is the next value of an unbounded exponent, which overflows. *)
let around f ~overflow c =
  let a = decode f c and b = unbounded f (c + 1) in
  let stored v = if c + 1 > last f && v = b then overflow else v in
  let mid = (a +. b) /. 2. in
  let even = if c land 1 = 0 then a else stored b in
  [
    (a, a);
    (b, stored b);
    (mid, even);
    (Float.pred mid, a);
    (Float.succ mid, stored b);
  ]

(* Stores of [x] and [-x] into [dt] that differ from [v] and [-v]. *)
let misstores dt inputs =
  List.concat_map
    (fun (x, v) ->
      List.filter_map
        (fun (x, v) ->
          let got = D.of_float dt x in
          if Testable.equal float_exact v got then None
          else Some (strf "%h stored %h, expected %h" x got v))
        [ (x, v); (-.x, -.v) ])
    inputs

let overflow dt f = if D.bits dt <= 8 then decode f (last f) else inf

let extremes dt f =
  let over = overflow dt f in
  [
    (inf, over);
    (Float.max_float, over);
    (unbounded f (last f + 1) *. 2., over);
    (0x1p-1074, 0.);
    (Float.min_float, 0.);
    (0., 0.);
  ]

(* Every finite value of a format of 16 bits or less, every tie between two,
   their neighbours, overflow and underflow. *)
let test_every_tie (F (dt, f)) =
  let overflow = overflow dt f in
  let wrong = ref (misstores dt (extremes dt f)) in
  for c = 0 to last f do
    wrong := misstores dt (around f ~overflow c) @ !wrong
  done;
  let n = List.length !wrong in
  equal ~msg:(strf "%d wrong stores" n) (list string) []
    (List.filteri (fun i _ -> i < 8) !wrong)

(* A NaN keeps its sign through a store, into an element and back. *)
let test_nan_sign (F (dt, _)) =
  List.iter
    (fun x ->
      let msg = if Float.sign_bit x then "-nan" else "nan" in
      equal ~msg bool (Float.sign_bit x) (Float.sign_bit (D.of_float dt x));
      let a = A.of_array dt [| 1 |] [| x |] in
      equal ~msg bool (Float.sign_bit x) (Float.sign_bit (A.get a [| 0 |])))
    [ nan; -.nan ]

(* A float16 store of a NaN sets its quiet bit and keeps its sign and the top
   ten bits of its payload, as a run, one element at a time, and into an array:
   the bits of each double and of its float16. *)
let nan_payloads =
  [
    (0x7FF0040000000000L, 0x7E01);
    (0xFFF0040000000000L, 0xFE01);
    (0x7FF8080000000000L, 0x7E02);
    (0x7FF3FF0000000000L, 0x7EFF);
    (0x7FF0000000000001L, 0x7E00);
  ]

let test_nan_payloads () =
  let xs = List.map (fun (x, _) -> Int64.float_of_bits x) nan_payloads in
  let want = List.map snd nan_payloads in
  let bits a = A.to_array (Option.get (A.bitcast D.Uint16 a)) in
  let n = List.length xs in
  let run = A.of_array D.Float16 [| n |] (Array.of_list xs) in
  equal ~msg:"of_array" (array int) (Array.of_list want) (bits run);
  let one = A.create Rig.host D.Float16 [| n |] in
  List.iteri (fun i x -> A.set one [| i |] x) xs;
  equal ~msg:"set" (array int) (Array.of_list want) (bits one)

(* Floats compared bit for bit, NaNs by their sign. *)
let signed_float =
  Testable.make ~pp:pp_hex ~equal:(fun a b ->
      if Float.is_nan a || Float.is_nan b then
        Float.is_nan a && Float.is_nan b && Float.sign_bit a = Float.sign_bit b
      else Testable.equal float_exact a b)

(* Every code of a format of 16 bits or less reads as its definition's value,
   one element at a time and as a run. *)
let test_every_code (F (dt, f)) =
  let width = 1 + f.exp + f.frac in
  let n = 1 lsl width in
  let codes = Array.init n Fun.id in
  let patterns =
    match width with
    | 4 -> A.Any (A.of_array D.Uint4 [| n |] codes)
    | 8 -> A.Any (A.of_array D.Uint8 [| n |] codes)
    | _ -> A.Any (A.of_array D.Uint16 [| n |] codes)
  in
  let (A.Any p) = patterns in
  let a = Option.get (A.bitcast dt p) in
  let value c =
    let v = decode f (c land ((n / 2) - 1)) in
    if c >= n / 2 then -.v else v
  in
  let expected = Array.map value codes in
  equal ~msg:"run" (array signed_float) expected (A.to_array a);
  equal ~msg:"elements" (array signed_float) expected
    (Array.map (fun c -> A.get a [| c |]) codes)

(* Wider formats: ties around codes drawn across the range. *)
let law_ties (F (dt, f)) =
  let codes =
    let top = last f and normal = 1 lsl f.frac in
    Gen.frequency
      [
        (6, Gen.int_range 0 top);
        ( 1,
          Gen.of_list ~pp:Format.pp_print_int
            [ 0; 1; normal - 1; normal; top - 1; top ] );
      ]
  in
  prop (D.name dt) codes (fun c ->
      cover "a subnormal" (c < 1 lsl f.frac);
      cover "past the largest" (c = last f);
      equal (list string) []
        (misstores dt (around f ~overflow:(overflow dt f) c @ extremes dt f)))

(* The exact reference: [x] rounded to nearest, ties to even, at [f]'s precision
   with no upper exponent bound, then the store rule's overflow, infinities and
   NaN. Scaling by a power of two is exact, and so are the floor and the
   difference. *)
let nearest f x =
  if x = 0. || not (Float.is_finite x) then x
  else
    let _, e = Float.frexp x in
    let q = Int.max (e - 1) (1 - f.bias) - f.frac in
    let s = Float.ldexp x (-q) in
    let fl = Float.floor s in
    let d = s -. fl in
    let r =
      if d > 0.5 || (d = 0.5 && Float.rem fl 2. <> 0.) then fl +. 1. else fl
    in
    Float.copy_sign (Float.ldexp r q) x

let exact dt f x =
  let max = decode f (last f) and saturates = D.bits dt <= 8 in
  if Float.is_nan x then if f.top = Finite then 0. else nan
  else
    let r = nearest f x in
    if Float.abs r > max then Float.copy_sign (if saturates then max else inf) x
    else r

(* Every float32 whose low half is about a float16 or bfloat16 rounding bit
   (every float8 rounding bit lies in the high half), the float64 neighbours of
   those whose low half is the bit alone, where rounding through float32 first
   would land on a tie, and the specials. *)
let sweep =
  lazy
    (let of_halves lows =
       List.concat_map
         (fun hi ->
           List.map
             (fun lo -> Int32.float_of_bits (Int32.of_int ((hi lsl 16) lor lo)))
             lows)
         (List.init 0x10000 Fun.id)
     in
     let f32 =
       of_halves
         [
           0; 1; 0x7FFF; 0x8000; 0x8001; 0xFFFF; 0x0FFF; 0x1000; 0x1001; 0x3000;
         ]
     in
     let near =
       of_halves [ 0x8000; 0x1000; 0x3000 ]
       |> List.filter Float.is_finite
       |> List.concat_map (fun x -> [ Float.pred x; Float.succ x ])
     in
     Array.of_list
       ([ snan; nan; -.nan; inf; -.inf; 0.; -0.; 1e39; -1e300; 5e-324 ]
       @ f32 @ near))

(* One store and a run of stores of every sweep value agree with the exact
   reference. *)
let test_sweep (F (dt, f)) =
  let xs = Lazy.force sweep in
  let want = Array.map (exact dt f) xs in
  let wrong got =
    let bad = ref [] in
    Array.iteri
      (fun i x ->
        if not (Testable.equal float_exact want.(i) got.(i)) then
          bad := strf "%h stored %h, expected %h" x got.(i) want.(i) :: !bad)
      xs;
    List.filteri (fun i _ -> i < 8) (List.rev !bad)
  in
  equal ~msg:"of_float" (list string) [] (wrong (Array.map (D.of_float dt) xs));
  equal ~msg:"of_array" (list string) []
    (wrong (A.to_array (A.of_array dt [| Array.length xs |] xs)))

let law_float64 =
  prop "float64 stores the double" (Gen.with_pp pp_hex Gen.any_float) (fun x ->
      equal float_exact x (D.of_float D.Float64 x))

(* From 64-bit integers *)

(* The bit length of [m], read unsigned. *)
let bit_length m =
  let rec go n m =
    if m = 0L then n else go (n + 1) (Int64.shift_right_logical m 1)
  in
  go 0 m

(* [v], read signed or unsigned, rounded to nearest, ties to even, at [p]
   significant bits, by integer arithmetic: the exact value, which a double
   holds. *)
let round_int ~signed p v =
  let negative = signed && Int64.compare v 0L < 0 in
  let m = if negative then Int64.neg v else v in
  let n = bit_length m in
  let r =
    if n <= p then Int64.to_float m
    else
      let s = n - p in
      let q = Int64.shift_right_logical m s in
      let rest = Int64.logand m (Int64.pred (Int64.shift_left 1L s)) in
      let c = Int64.unsigned_compare rest (Int64.shift_left 1L (s - 1)) in
      let up = c > 0 || (c = 0 && Int64.logand q 1L = 1L) in
      Float.ldexp (Int64.to_float (if up then Int64.succ q else q)) s
  in
  if negative then -.r else r

(* The value of the code [c] of [f], with its sign bit. *)
let signed_decode f c =
  let w = f.exp + f.frac in
  let v = decode f (c land ((1 lsl w) - 1)) in
  if (c lsr w) land 1 = 1 then -.v else v

(* Integers at and about every tie of a precision of 2 to 53 bits below 2^64,
   and the extremes. *)
let integers =
  lazy
    (let ties =
       List.concat_map
         (fun k ->
           List.concat_map
             (fun p ->
               if k <= p then []
               else
                 List.concat_map
                   (fun j ->
                     let t =
                       Int64.add (Int64.shift_left 1L k)
                         (Int64.shift_left (Int64.of_int j) (k - p))
                     in
                     [ Int64.pred t; t; Int64.succ t ])
                   [ 1; 3 ])
             [ 2; 3; 4; 8; 11; 24; 53 ])
         (List.init 64 Fun.id)
     in
     [ 0L; 1L; -1L; Int64.max_int; Int64.min_int ] @ ties)

(* A store of an int64 or uint64 into a narrow float rounds the integer once:
   converting it to a double first rounds past 2^53 and can land on a tie. *)
let test_integers (F (dt, f)) =
  let p = f.frac + 1 in
  let wrong signed store =
    List.filter_map
      (fun v ->
        let want = exact dt f (round_int ~signed p v) in
        let got = signed_decode f (store (D.code dt) v) in
        if Testable.equal float_exact want got then None
        else
          Some
            (strf "%s %s stored %h, expected %h"
               (if signed then "int64" else "uint64")
               (if signed then Int64.to_string v else Printf.sprintf "%Lu" v)
               got want))
      (Lazy.force integers)
  in
  let first8 l = List.filteri (fun i _ -> i < 8) l in
  equal ~msg:"int64" (list string) []
    (first8 (wrong true Nx_array_support.of_int64));
  equal ~msg:"uint64" (list string) []
    (first8 (wrong false Nx_array_support.of_uint64))

(* Integers *)

(* Doubles across [lo, hi], their neighbours and the extremes. *)
let near lo hi =
  Gen.with_pp pp_hex
    (Gen.frequency
       [
         (4, Gen.float_range ((2. *. lo) -. 2.) ((2. *. hi) +. 2.));
         ( 2,
           Gen.of_list
             [
               lo;
               hi;
               Float.pred lo;
               Float.succ hi;
               Float.pred hi;
               lo -. 1.;
               hi +. 1.;
               -0.5;
               -0.;
               0.5;
             ] );
         (1, Gen.any_float);
       ])

(* [x] truncated toward zero and clamped to [lo, hi]; NaN is 0. *)
let clamp lo hi x =
  if Float.is_nan x then 0. else Float.min hi (Float.max lo (Float.trunc x))

let law_small (type s) (dt : (int, s) D.t) lo hi =
  prop (D.name dt) (near lo hi) (fun x ->
      cover "saturates" (x > hi || x < lo);
      equal (value dt) (Float.to_int (clamp lo hi x)) (D.of_float dt x))

let law_int32 =
  let lo = -0x1p31 and hi = 0x1p31 -. 1. in
  prop "int32" (near lo hi) (fun x ->
      cover "saturates" (x > hi || x < lo);
      equal (value D.Int32)
        (Int32.of_float (clamp lo hi x))
        (D.of_float D.Int32 x))

let law_uint32 =
  let hi = 0x1p32 -. 1. in
  prop "uint32" (near 0. hi) (fun x ->
      cover "saturates" (x > hi || x < 0.);
      equal (value D.Uint32)
        (Int64.to_int32 (Int64.of_float (clamp 0. hi x)))
        (D.of_float D.Uint32 x))

let law_int64 =
  prop "int64" (near (-0x1p63) 0x1p63) (fun x ->
      let expected =
        if Float.is_nan x then 0L
        else if x >= 0x1p63 then Int64.max_int
        else if x < -0x1p63 then Int64.min_int
        else Int64.of_float x
      in
      cover "saturates" (x >= 0x1p63 || x < -0x1p63);
      equal (value D.Int64) expected (D.of_float D.Int64 x))

let law_uint64 =
  prop "uint64" (near 0. 0x1p64) (fun x ->
      let expected =
        if Float.is_nan x || x < 1. then 0L
        else if x >= 0x1p64 then -1L
        else if x >= 0x1p63 then
          Int64.add (Int64.of_float (x -. 0x1p63)) Int64.min_int
        else Int64.of_float x
      in
      cover "past int64" (x >= 0x1p63 && x < 0x1p64);
      cover "saturates" (x >= 0x1p64 || x <= -1.);
      equal (value D.Uint64) expected (D.of_float D.Uint64 x))

let integer_laws =
  [
    law_small D.Int4 (-8.) 7.;
    law_small D.Uint4 0. 15.;
    law_small D.Int8 (-128.) 127.;
    law_small D.Uint8 0. 255.;
    law_small D.Int16 (-32768.) 32767.;
    law_small D.Uint16 0. 65535.;
    law_int32;
    law_uint32;
    law_int64;
    law_uint64;
  ]

(* Complex numbers and booleans *)

(* Either zero. *)
let a_zero = Testable.make ~pp:pp_hex ~equal:(fun a b -> a = 0. && b = 0.)

let law_complex (type s) (dt : (Complex.t, s) D.t) (component : float -> float)
    =
  prop (D.name dt) (Gen.with_pp pp_hex Gen.any_float) (fun x ->
      let v = D.of_float dt x in
      equal float_exact (component x) v.re;
      equal ~msg:"imaginary part" a_zero 0. v.im)

let law_boolean (type s) (dt : (bool, s) D.t) =
  prop (D.name dt) ~examples:[ -0.; 0.; nan ] (Gen.with_pp pp_hex Gen.any_float)
    (fun x ->
      cover "NaN" (Float.is_nan x);
      cover "-0" (1. /. x = Float.neg_infinity);
      equal bool (x <> 0.) (D.of_float dt x))

(* Printing *)

type printed = Printed : ('v, 's) D.t * 'v * string -> printed

let printed_name (Printed (dt, _, s)) = strf "%s %s" (D.name dt) s

let test_printed (Printed (dt, v, s)) =
  equal string s (Format.asprintf "%a" (D.pp_value dt) v)

let printed =
  [
    Printed (D.Uint32, -1l, "4294967295");
    Printed (D.Uint64, -1L, "18446744073709551615");
    Printed (D.Int32, -1l, "-1");
    Printed (D.Uint16, 65535, "65535");
    Printed (D.Int4, -8, "-8");
    Printed (D.Float32, 0x1.99999ap-4, "0.1");
    Printed (D.Float16, 0x1.998p-4, "0.1");
    Printed (D.Float64, nan, "nan");
    Printed (D.Float16, inf, "inf");
    Printed (D.Float8_e5m2, -.inf, "-inf");
    Printed (D.Complex64, { Complex.re = 1.; im = 2. }, "1+2i");
    Printed (D.Complex128, { Complex.re = 1.; im = -2. }, "1-2i");
    Printed (D.Float32, -.nan, "nan");
    Printed (D.Float16, 65504., "65500");
    Printed (D.Float8_e4m3fn, -448., "-450");
    Printed (D.Float16, 0x1p-6, "0.01563");
    Printed (D.Float8_e4m3fn, 0.125, "0.13");
    Printed (D.Bfloat16, 0x1p+97, "1.59e+29");
    Printed (D.Float64, 0x1p-1017, "7.120236347223045e-307");
    Printed (D.Complex64, { Complex.re = 0.5; im = -2. }, "0.5-2i");
    Printed (D.Bool, false, "false");
    Printed (D.Bit, true, "true");
  ]

let law_text gen (text : 'v -> string) dt =
  prop (D.name dt) gen (fun v ->
      equal string (text v) (Format.asprintf "%a" (D.pp_value dt) v))

let law_int (type s) (dt : (int, s) D.t) =
  law_text (Gen.int_range (D.min_value dt) (D.max_value dt)) string_of_int dt

let printing_integers =
  [
    law_int D.Int4;
    law_int D.Uint4;
    law_int D.Int8;
    law_int D.Uint8;
    law_int D.Int16;
    law_int D.Uint16;
    law_text Gen.int32 Int32.to_string D.Int32;
    law_text Gen.int32 (strf "%lu") D.Uint32;
    law_text Gen.int64 Int64.to_string D.Int64;
    law_text Gen.int64 (strf "%Lu") D.Uint64;
  ]

(* The significant digits of a decimal text and the power of ten of its last
   one, as ("125", -4) for "-0.0125"; [None] for a zero. *)
let significand s =
  let s = String.lowercase_ascii s in
  let s = if s.[0] = '-' then String.sub s 1 (String.length s - 1) else s in
  let mantissa, exp =
    match String.split_on_char 'e' s with
    | [ m ] -> (m, 0)
    | [ m; e ] ->
        let e =
          if e.[0] = '+' then String.sub e 1 (String.length e - 1) else e
        in
        (m, int_of_string e)
    | _ -> failf "%S is not a decimal" s
  in
  let whole, frac =
    match String.split_on_char '.' mantissa with
    | [ w ] -> (w, "")
    | [ w; f ] -> (w, f)
    | _ -> failf "%S is not a decimal" s
  in
  let digits = whole ^ frac and exp = exp - String.length frac in
  let rec drop_leading d =
    if d <> "" && d.[0] = '0' then
      drop_leading (String.sub d 1 (String.length d - 1))
    else d
  in
  let rec drop_trailing d e =
    let n = String.length d in
    if n > 0 && d.[n - 1] = '0' then
      drop_trailing (String.sub d 0 (n - 1)) (e + 1)
    else (d, e)
  in
  match drop_trailing (drop_leading digits) exp with
  | "", _ -> None
  | d, e -> Some (d, e)

(* The two decimals of one significant digit fewer than [s] that bracket it. If
   none of them reads back as a value, no shorter decimal does: the decimals
   that read back as a value form an interval around it. *)
let shorter s =
  match significand s with
  | None -> []
  | Some (d, _) when String.length d = 1 -> []
  | Some (d, e) ->
      let q = int_of_string (String.sub d 0 (String.length d - 1)) in
      let sign = if s.[0] = '-' then "-" else "" in
      [ strf "%s%de%d" sign q (e + 1); strf "%s%de%d" sign (q + 1) (e + 1) ]

(* Whether the decimal [t] reads back as [v] in [f]: [t] rounds to [v] at [f]'s
   precision with the exponent unbounded, so a decimal that saturates to the
   largest finite value does not read back as it. *)
let reads_back f v t =
  Testable.equal float_exact v (nearest f (float_of_string t))

let check_printed (F (dt, f)) v =
  let s = Format.asprintf "%a" (D.pp_value dt) v in
  let reads_back = reads_back f v in
  let special =
    if Float.is_nan v then Some "nan"
    else if v = inf then Some "inf"
    else if v = -.inf then Some "-inf"
    else None
  in
  match special with
  | Some t -> if s = t then None else Some (strf "%h printed %s" v s)
  | None when not (reads_back s) -> Some (strf "%h printed %s" v s)
  | None -> (
      match List.filter reads_back (shorter s) with
      | t :: _ -> Some (strf "%h printed %s, but %s reads back" v s t)
      | [] ->
          (* Of the decimals of as many digits that read back, the nearest:
             printf rounds [v] to them correctly, ties to even. *)
          let digits =
            Option.fold ~none:1
              ~some:(fun (d, _) -> String.length d)
              (significand s)
          in
          let nearest = strf "%.*e" (digits - 1) v in
          if reads_back nearest && significand nearest <> significand s then
            Some
              (strf "%h printed %s, but the nearer %s reads back" v s nearest)
          else None)

(* Every value of a format of 16 bits or less. *)
let test_every_value_printed (F (dt, f)) =
  let wrong = ref [] in
  for c = 0 to (1 lsl (f.exp + f.frac)) - 1 do
    List.iter
      (fun v ->
        match check_printed (F (dt, f)) v with
        | Some e -> wrong := e :: !wrong
        | None -> ())
      [ decode f c; -.decode f c ]
  done;
  let n = List.length !wrong in
  equal
    ~msg:(strf "%d values misprinted" n)
    (list string) []
    (List.filteri (fun i _ -> i < 8) !wrong)

let law_printed (F (dt, f)) =
  let emin = 2 - (1 lsl (f.exp - 1)) - f.frac
  and emax = (1 lsl (f.exp - 1)) - 1 in
  let powers =
    List.init (emax - emin + 1) (fun e -> Float.ldexp 1. (emin + e))
  in
  prop (D.name dt) ~examples:powers (Gen.with_pp pp_hex Gen.any_float) (fun x ->
      equal (option string) None (check_printed (F (dt, f)) (D.of_float dt x)))

(* The suite *)

let narrow = List.filter (fun (F (dt, _)) -> D.bits dt <= 16) formats
let wide = List.filter (fun (F (dt, _)) -> D.bits dt = 32) formats

let tests =
  [
    group "table"
      [
        test "all holds every dtype once" test_all;
        test "a dtype's code is its index in all" test_codes;
        cases ~name:row_name "name, pp, bits and kind state each format" table
          test_row;
        test "is k holds of the dtypes of kind k" test_is;
        cases ~name:row_name "nx_dtype.h states each row" table test_header;
        test "nx_dtype.h has no row past the last code"
          test_no_row_past_the_last;
        test "equal and equal_witness hold of a dtype and itself only"
          test_equal;
      ];
    group "names"
      [
        cases ~name:row_name "of_name finds each name" table test_of_name;
        cases ~name:(strf "%S") "of_name refuses other names" names_refused
          test_of_other_name;
      ];
    group "bytes is the bytes n elements fill" (List.map law_bytes D.all);
    group "values"
      [
        cases ~name:values_name "zero and one are 0 and 1" values
          test_identities;
        cases ~name:values_name "min_value and max_value are the range" values
          test_limits;
        cases ~name:format_name "float_format states each format" formats
          test_float_format;
      ];
    group "stores"
      [
        cases ~name:store_name "the conversion table" conversion_table
          test_store;
        cases ~name:format_name
          "every value, tie and neighbour rounds once to even" narrow
          test_every_tie;
        group "ties round once to even" (List.map law_ties wide);
        cases ~name:format_name
          "every float32 about a rounding bit and its float64 neighbours store \
           as the exact reference says"
          (List.filter (fun (F (dt, _)) -> D.bits dt < 64) formats)
          test_sweep;
        cases ~name:format_name "a NaN keeps its sign"
          (List.filter (fun (F (_, f)) -> f.top <> Finite) formats)
          test_nan_sign;
        test
          "a float16 NaN is quieted, keeping its sign and the top of its \
           payload"
          test_nan_payloads;
        cases ~name:format_name "every code reads as its definition's value"
          narrow test_every_code;
        law_float64;
        cases ~name:format_name
          "an int64 or uint64 stores as the exact reference rounds it once"
          narrow test_integers;
        group "integers truncate toward zero and saturate" integer_laws;
        group "complex numbers store a real part in their component's format"
          [
            law_complex D.Complex128 Fun.id;
            law_complex D.Complex64 (D.of_float D.Float32);
          ];
        group "booleans store x <> 0." [ law_boolean D.Bool; law_boolean D.Bit ];
      ];
    group "printing"
      [
        cases ~name:printed_name "values print in their dtype's reading" printed
          test_printed;
        group "integers print in their dtype's reading" printing_integers;
        cases ~name:format_name
          "every value prints as the shortest decimal that reads back, the \
           nearest of those"
          narrow test_every_value_printed;
        group
          "a wider float prints as the shortest decimal that reads back, the \
           nearest of those"
          (List.map law_printed
             (List.filter (fun (F (dt, _)) -> D.bits dt >= 32) formats));
      ];
  ]

let () = exit (run "nx_array.dtype" tests)
