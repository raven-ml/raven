(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module D = Nx_array.Dtype

let strf = Printf.sprintf
let any_name (D.Any dt) = D.name dt
let any = Testable.make ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) ~equal:( = )

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

let float_dtypes = List.filter (fun (D.Any dt) -> D.is D.Float dt) D.all

(* The table *)

let kind_index : type v s. (v, s) D.t -> int =
 fun dt ->
  match D.kind dt with
  | D.Float -> 0
  | D.Complex -> 1
  | D.Signed -> 2
  | D.Unsigned -> 3
  | D.Boolean -> 4

let test_codes () =
  List.iteri
    (fun i (D.Any dt) -> equal ~msg:(D.name dt) int i (D.code dt))
    D.all;
  equal (option pass) None (Nx_array_support.row (List.length D.all))

let test_header (D.Any dt) =
  let row = Nx_array_support.row (D.code dt) in
  equal
    (option (triple string int int))
    (Some (D.name dt, D.bits dt, kind_index dt))
    row

let test_bits () =
  let bits (D.Any dt) = (D.name dt, D.bits dt) in
  equal
    (list (pair string int))
    [
      ("float64", 64);
      ("float32", 32);
      ("float16", 16);
      ("bfloat16", 16);
      ("float8_e4m3", 8);
      ("float8_e5m2", 8);
      ("float4_e2m1", 4);
      ("int64", 64);
      ("uint64", 64);
      ("int32", 32);
      ("uint32", 32);
      ("int16", 16);
      ("uint16", 16);
      ("int8", 8);
      ("uint8", 8);
      ("int4", 4);
      ("uint4", 4);
      ("complex128", 128);
      ("complex64", 64);
      ("bool", 8);
      ("bit", 1);
    ]
    (List.map bits D.all)

let test_bytes () =
  equal int 0 (D.bytes D.Bit 0);
  equal int 1 (D.bytes D.Bit 1);
  equal int 1 (D.bytes D.Bit 8);
  equal int 2 (D.bytes D.Bit 9);
  equal int 2 (D.bytes D.Int4 3);
  equal int 2 (D.bytes D.Uint4 4);
  equal int 3 (D.bytes D.Float4_e2m1 5);
  equal int ((max_int / 8) + 1) (D.bytes D.Bit max_int);
  equal int ((max_int / 2) + 1) (D.bytes D.Int4 max_int);
  equal int 48 (D.bytes D.Complex128 3);
  equal int (max_int / 8 * 8) (D.bytes D.Float64 (max_int / 8));
  raises_match Exn.invalid_arg (fun () -> D.bytes D.Float64 ((max_int / 8) + 1));
  raises_match Exn.invalid_arg (fun () ->
      D.bytes D.Complex128 ((max_int / 16) + 1));
  raises_match Exn.invalid_arg (fun () -> D.bytes D.Uint8 (-1));
  raises_match Exn.invalid_arg (fun () -> D.bytes D.Bit (-1))

let test_names () =
  List.iter
    (fun (D.Any dt as a) -> equal (option any) (Some a) (D.of_name (D.name dt)))
    D.all;
  equal (option any) None (D.of_name "Float32");
  equal (option any) None (D.of_name "float8_e4m3fnuz");
  equal (option any) None (D.of_name "")

let test_equal () =
  List.iter
    (fun (D.Any a) ->
      List.iter
        (fun (D.Any b) ->
          let same = D.code a = D.code b in
          let msg = strf "%s %s" (D.name a) (D.name b) in
          equal ~msg bool same (D.equal a b);
          equal ~msg bool same (Option.is_some (D.equal_witness a b)))
        D.all)
    D.all

let test_kinds () =
  let kinds k =
    List.map any_name (List.filter (fun (D.Any dt) -> D.is k dt) D.all)
  in
  equal (list string)
    [
      "float64";
      "float32";
      "float16";
      "bfloat16";
      "float8_e4m3";
      "float8_e5m2";
      "float4_e2m1";
    ]
    (kinds D.Float);
  equal (list string) [ "complex128"; "complex64" ] (kinds D.Complex);
  equal (list string)
    [ "int64"; "int32"; "int16"; "int8"; "int4" ]
    (kinds D.Signed);
  equal (list string)
    [ "uint64"; "uint32"; "uint16"; "uint8"; "uint4" ]
    (kinds D.Unsigned);
  equal (list string) [ "bool"; "bit" ] (kinds D.Boolean)

(* Float formats *)

let format_row (D.Any dt) =
  match D.kind dt with
  | D.Float ->
      let f = D.float_format dt in
      Some
        ( D.name dt,
          (f.exponent_bits, f.mantissa_bits, f.infinities, f.nans),
          (f.epsilon, f.min_normal, f.max_finite) )
  | _ -> None

let test_formats () =
  equal
    (list
       (triple string (quad int int bool bool)
          (triple float_exact float_exact float_exact)))
    [
      ("float64", (11, 52, true, true), (0x1p-52, 0x1p-1022, Float.max_float));
      ("float32", (8, 23, true, true), (0x1p-23, 0x1p-126, 0x1.fffffep127));
      ("float16", (5, 10, true, true), (0x1p-10, 0x1p-14, 65504.));
      ("bfloat16", (8, 7, true, true), (0x1p-7, 0x1p-126, 0x1.fep127));
      ("float8_e4m3", (4, 3, false, true), (0x1p-3, 0x1p-6, 448.));
      ("float8_e5m2", (5, 2, true, true), (0x1p-2, 0x1p-14, 57344.));
      ("float4_e2m1", (2, 1, false, false), (0x1p-1, 1., 6.));
    ]
    (List.filter_map format_row D.all)

(* Values *)

let test_identities () =
  let check : type v s. (v, s) D.t -> unit =
   fun dt ->
    equal ~msg:(D.name dt) (value dt) (D.of_float dt 0.) (D.zero dt);
    equal ~msg:(D.name dt) (value dt) (D.of_float dt 1.) (D.one dt)
  in
  List.iter (fun (D.Any dt) -> check dt) D.all

let test_limits () =
  let ( => ) dt (lo, hi) =
    equal ~msg:(D.name dt) (value dt) lo (D.min_value dt);
    equal ~msg:(D.name dt) (value dt) hi (D.max_value dt)
  in
  D.Float64 => (Float.neg_infinity, Float.infinity);
  D.Float16 => (Float.neg_infinity, Float.infinity);
  D.Float8_e5m2 => (Float.neg_infinity, Float.infinity);
  D.Float8_e4m3 => (-448., 448.);
  D.Float4_e2m1 => (-6., 6.);
  D.Int64 => (Int64.min_int, Int64.max_int);
  D.Uint64 => (0L, -1L);
  D.Int32 => (Int32.min_int, Int32.max_int);
  D.Uint32 => (0l, -1l);
  D.Int16 => (-32768, 32767);
  D.Uint16 => (0, 65535);
  D.Int8 => (-128, 127);
  D.Uint8 => (0, 255);
  D.Int4 => (-8, 7);
  D.Uint4 => (0, 15);
  D.Bool => (false, true);
  D.Bit => (false, true);
  raises_match (Exn.invalid_arg ~substring:"complex128") (fun () ->
      D.min_value D.Complex128);
  raises_match (Exn.invalid_arg ~substring:"complex64") (fun () ->
      D.max_value D.Complex64)

(* Stores, by the conversion rule. Each row is a dtype, an input and the value
   the store holds. *)

type store = Store : ('v, 's) D.t * float * 'v -> store

let pp_store ppf (Store (dt, x, _)) = Format.fprintf ppf "%s %h" (D.name dt) x
let stores_name s = Format.asprintf "%a" pp_store s
let check_store (Store (dt, x, v)) = equal (value dt) v (D.of_float dt x)
let nan = Float.nan
let inf = Float.infinity

(* Specials: NaN, ±0 and ±inf in every float format. *)
let specials =
  [
    Store (D.Float64, nan, nan);
    Store (D.Float64, -0., -0.);
    Store (D.Float64, -.inf, -.inf);
    Store (D.Float32, nan, nan);
    Store (D.Float32, -0., -0.);
    Store (D.Float32, inf, inf);
    Store (D.Float32, -.inf, -.inf);
    Store (D.Float16, nan, nan);
    Store (D.Float16, -0., -0.);
    Store (D.Float16, inf, inf);
    Store (D.Float16, -.inf, -.inf);
    Store (D.Bfloat16, nan, nan);
    Store (D.Bfloat16, -0., -0.);
    Store (D.Bfloat16, inf, inf);
    Store (D.Bfloat16, -.inf, -.inf);
    Store (D.Float8_e5m2, nan, nan);
    Store (D.Float8_e5m2, -0., -0.);
    Store (D.Float8_e5m2, inf, inf);
    Store (D.Float8_e5m2, -.inf, -.inf);
    Store (D.Float8_e4m3, nan, nan);
    Store (D.Float8_e4m3, -0., -0.);
    Store (D.Float8_e4m3, inf, nan);
    Store (D.Float8_e4m3, -.inf, nan);
    Store (D.Float4_e2m1, nan, 0.);
    Store (D.Float4_e2m1, -0., -0.);
    Store (D.Float4_e2m1, inf, 6.);
    Store (D.Float4_e2m1, -.inf, -6.);
  ]

(* Past the largest finite value, and its neighbours. *)
let overflows =
  [
    Store (D.Float32, 0x1.fffffep127, 0x1.fffffep127);
    Store (D.Float32, 0x1.ffffffp127, inf) (* the tie rounds to even: up *);
    Store (D.Float32, 0x1.fffffefffffffp127, 0x1.fffffep127);
    Store (D.Float32, -1e300, -.inf);
    Store (D.Float16, 65504., 65504.);
    Store (D.Float16, 65519.99, 65504.);
    Store (D.Float16, 65520., inf);
    Store (D.Float16, -1e10, -.inf);
    Store (D.Bfloat16, 0x1.fep127, 0x1.fep127);
    Store (D.Bfloat16, 0x1.ffp127, inf);
    Store (D.Float8_e5m2, 57344., 57344.);
    Store (D.Float8_e5m2, 61440., 57344.);
    Store (D.Float8_e5m2, -1e300, -57344.);
    Store (D.Float8_e4m3, 448., 448.);
    Store (D.Float8_e4m3, 464., 448.);
    Store (D.Float8_e4m3, 1e300, 448.);
    Store (D.Float8_e4m3, -500., -448.);
    Store (D.Float4_e2m1, 6., 6.);
    Store (D.Float4_e2m1, 7., 6.);
    Store (D.Float4_e2m1, -1e300, -6.);
  ]

(* Ties round to even, once from the double: a value just past a tie rounds away
   from it, where rounding through float32 first would land on the tie and round
   to even. *)
let ties =
  [
    Store (D.Float32, 1. +. 0x1p-24, 1.);
    Store (D.Float32, 1. +. 0x3p-24, 1. +. 0x1p-22);
    Store (D.Float32, 1. +. 0x1p-24 +. 0x1p-50, 1. +. 0x1p-23);
    Store (D.Float16, 1. +. 0x1p-11, 1.);
    Store (D.Float16, 1. +. 0x3p-11, 1. +. 0x1p-9);
    Store (D.Float16, 1. +. 0x1p-11 +. 0x1p-40, 1. +. 0x1p-10);
    Store (D.Float16, 1. +. 0x1p-11 -. 0x1p-40, 1.);
    Store (D.Bfloat16, 1. +. 0x1p-8, 1.);
    Store (D.Bfloat16, 1. +. 0x1p-8 +. 0x1p-40, 1. +. 0x1p-7);
    Store (D.Float8_e4m3, 1. +. 0x1p-4, 1.);
    Store (D.Float8_e4m3, 1. +. 0x3p-4, 1.25);
    Store (D.Float8_e4m3, 1. +. 0x1p-4 +. 0x1p-40, 1.125);
    Store (D.Float8_e5m2, 1. +. 0x1p-3, 1.);
    Store (D.Float8_e5m2, 1. +. 0x1p-3 +. 0x1p-40, 1.25);
    Store (D.Float4_e2m1, 1.25, 1.);
    Store (D.Float4_e2m1, 1.75, 2.);
    Store (D.Float4_e2m1, 2.5, 2.);
    Store (D.Float4_e2m1, 3.5, 4.);
    Store (D.Float4_e2m1, 5., 4.);
    Store (D.Float4_e2m1, 5. +. 0x1p-40, 6.);
    Store (D.Float4_e2m1, -0.75, -1.);
    Store (D.Float16, 0.1, 0x1.998p-4);
    Store (D.Float32, 0.1, 0x1.99999ap-4);
  ]

(* Below the least normal: subnormals, and zeros of the value's sign. *)
let underflows =
  [
    Store (D.Float32, 0x1p-149, 0x1p-149);
    Store (D.Float32, 0x1p-150, 0.);
    Store (D.Float32, 0x1.000001p-150, 0x1p-149);
    Store (D.Float32, -0x1p-151, -0.);
    Store (D.Float16, 0x1p-24, 0x1p-24);
    Store (D.Float16, 0x1p-25, 0.);
    Store (D.Float16, 0x1p-25 +. 0x1p-60, 0x1p-24);
    Store (D.Float16, 0x3p-25, 0x1p-23);
    Store (D.Float16, -1e-300, -0.);
    Store (D.Bfloat16, 0x1p-133, 0x1p-133);
    Store (D.Bfloat16, 0x1p-134, 0.);
    Store (D.Bfloat16, 1e-300, 0.);
    Store (D.Float8_e4m3, 0x1p-9, 0x1p-9);
    Store (D.Float8_e4m3, 0x1p-10, 0.);
    Store (D.Float8_e4m3, 0x3p-10, 0x1p-8);
    Store (D.Float8_e5m2, 0x1p-16, 0x1p-16);
    Store (D.Float8_e5m2, 0x1p-17, 0.);
    Store (D.Float8_e5m2, -0x1p-17, -0.);
    Store (D.Float4_e2m1, 0.5, 0.5);
    Store (D.Float4_e2m1, 0.25, 0.);
    Store (D.Float4_e2m1, 0.25 +. 0x1p-40, 0.5);
    Store (D.Float4_e2m1, 0.75, 1.);
    Store (D.Float4_e2m1, -0.1, -0.);
  ]

(* Integers truncate toward zero, saturate and store NaN as 0. *)
let integers =
  [
    Store (D.Int8, 300., 127);
    Store (D.Int8, -300., -128);
    Store (D.Int8, 127.9, 127);
    Store (D.Int8, -128.9, -128);
    Store (D.Int8, -2.7, -2);
    Store (D.Int8, nan, 0);
    Store (D.Int8, inf, 127);
    Store (D.Int8, -.inf, -128);
    Store (D.Uint8, nan, 0);
    Store (D.Uint8, -1., 0);
    Store (D.Uint8, -0.5, 0);
    Store (D.Uint8, 255.5, 255);
    Store (D.Uint8, 256., 255);
    Store (D.Uint8, inf, 255);
    Store (D.Int4, 7.99, 7);
    Store (D.Int4, 8., 7);
    Store (D.Int4, -9., -8);
    Store (D.Uint4, 16., 15);
    Store (D.Uint4, nan, 0);
    Store (D.Int16, 40000., 32767);
    Store (D.Int16, -40000., -32768);
    Store (D.Uint16, 70000., 65535);
    Store (D.Uint16, -3., 0);
    Store (D.Int32, 3e9, Int32.max_int);
    Store (D.Int32, -3e9, Int32.min_int);
    Store (D.Int32, 2147483647.5, Int32.max_int);
    Store (D.Int32, nan, 0l);
    Store (D.Uint32, 4294967295., -1l);
    Store (D.Uint32, 5e9, -1l);
    Store (D.Uint32, 4294967294.9, -2l);
    Store (D.Uint32, -1., 0l);
    Store (D.Uint32, nan, 0l);
    Store (D.Int64, 0x1p63, Int64.max_int);
    Store (D.Int64, -0x1p63, Int64.min_int);
    Store (D.Int64, -0x1p64, Int64.min_int);
    Store (D.Int64, 0x1.fffffffffffffp62, 0x7ffffffffffffc00L);
    Store (D.Int64, nan, 0L);
    Store (D.Uint64, 0x1p64, -1L);
    Store (D.Uint64, 0x1p63, Int64.min_int);
    Store (D.Uint64, 0x1.fffffffffffffp63, 0xfffffffffffff800L);
    Store (D.Uint64, -1., 0L);
    Store (D.Uint64, nan, 0L);
    Store (D.Uint64, inf, -1L);
  ]

let others =
  [
    Store (D.Complex128, 0.1, { Complex.re = 0.1; im = 0. });
    Store (D.Complex64, 0.1, { Complex.re = 0x1.99999ap-4; im = 0. });
    Store (D.Complex64, 1e300, { Complex.re = inf; im = 0. });
    Store (D.Bool, 0., false);
    Store (D.Bool, -0., false);
    Store (D.Bool, 0.5, true);
    Store (D.Bool, nan, true);
    Store (D.Bit, 0., false);
    Store (D.Bit, -2., true);
  ]

(* A store into a float format, from its facts alone: scale [x] to the grid of
   its binade, round half to even, then overflow by the format's rule: the
   formats of a byte or less saturate. *)
let reference (f : D.float_format) x =
  let saturates = f.exponent_bits + f.mantissa_bits < 8 in
  if Float.is_nan x then if f.nans then Float.nan else 0.
  else if Float.abs x = Float.infinity then
    if f.infinities then x
    else if f.nans then Float.nan
    else Float.copy_sign f.max_finite x
  else
    let a = Float.abs x in
    let binade v = snd (Float.frexp v) - 1 in
    let e = binade (Float.max a f.min_normal) in
    let q = Float.ldexp 1. (e - f.mantissa_bits) in
    let r = a /. q in
    let n = Float.floor r in
    let d = r -. n in
    let n =
      if d > 0.5 || (d = 0.5 && Float.rem n 2. = 1.) then n +. 1. else n
    in
    let v = n *. q in
    let v =
      if v <= f.max_finite then v
      else if saturates then f.max_finite
      else Float.infinity
    in
    Float.copy_sign v x

(* Doubles near a format's grid, from below its subnormals to past its largest
   value: points of the grid, ties between them, and a double's step or two
   either side of each. A draw says whether it is a tie. *)
let near_grid (f : D.float_format) =
  let open Gen in
  let binade v = snd (Float.frexp v) - 1 in
  let emin = binade f.min_normal and emax = binade f.max_finite in
  let lo = emin - f.mantissa_bits - 2 and hi = emax + 1 in
  (* Wide formats span hundreds of binades: draw the ends often. *)
  let+ e =
    frequency
      [
        (2, int_range lo hi);
        (1, int_range lo (emin + 1));
        (1, int_range (emax - 1) hi);
      ]
  and+ k = int_range 0 (1 lsl (f.mantissa_bits + 2))
  and+ half = bool
  and+ nudge = int_range (-2) 2
  and+ neg = bool in
  let x = Float.of_int ((2 * k) + Bool.to_int half) in
  let x = Float.ldexp x (e - f.mantissa_bits - 1) in
  let x = x +. Float.ldexp (Float.of_int nudge) (binade x - 52) in
  ((if neg then -.x else x), half && nudge = 0)

let hex ppf (x, _) = Format.fprintf ppf "%h" x

(* Float64 is left out: its stores keep the double, and a double has no ties of
   its own grid. *)
let law_store (D.Any dt) =
  match D.kind dt with
  | D.Float when D.bits dt < 64 ->
      let f = D.float_format dt in
      [
        prop (D.name dt)
          (Gen.with_pp hex (near_grid f))
          (fun (x, tie) ->
            let a = Float.abs x in
            cover "a tie" tie;
            cover "past the largest" (a > f.max_finite);
            cover "subnormal" (a > 0. && a < f.min_normal);
            equal float_exact (reference f x) (D.of_float dt x));
      ]
  | _ -> []

(* Integers: truncate, then clamp to the range the limits state. *)
let law_saturate (D.Any dt) =
  let check : type v s. (v, s) D.t -> (v -> float) -> float -> float -> test =
   fun dt to_float lo hi ->
    let gen =
      Gen.with_pp
        (fun ppf x -> Format.fprintf ppf "%h" x)
        Gen.(
          frequency
            [
              (4, float_range ((2. *. lo) -. 2.) ((2. *. hi) +. 2.));
              ( 1,
                map Float.of_int
                  (int_range (int_of_float lo - 2) (int_of_float lo + 2)) );
              (1, any_float);
            ])
    in
    prop (D.name dt) gen (fun x ->
        let expected =
          if Float.is_nan x then 0.
          else Float.min hi (Float.max lo (Float.trunc x)) +. 0.
        in
        cover "saturates" (x > hi || x < lo);
        equal float_exact expected (to_float (D.of_float dt x)))
  in
  let u32 v = Int64.to_float (Int64.logand (Int64.of_int32 v) 0xFFFFFFFFL) in
  match dt with
  | D.Int4 -> [ check dt Float.of_int (-8.) 7. ]
  | D.Uint4 -> [ check dt Float.of_int 0. 15. ]
  | D.Int8 -> [ check dt Float.of_int (-128.) 127. ]
  | D.Uint8 -> [ check dt Float.of_int 0. 255. ]
  | D.Int16 -> [ check dt Float.of_int (-32768.) 32767. ]
  | D.Uint16 -> [ check dt Float.of_int 0. 65535. ]
  | D.Int32 -> [ check dt Int32.to_float (-2147483648.) 2147483647. ]
  | D.Uint32 -> [ check dt u32 0. 4294967295. ]
  | _ -> []

(* Printing *)

let test_printer () =
  let text dt v = Format.asprintf "%a" (D.pp_value dt) v in
  equal string "4294967295" (text D.Uint32 (-1l));
  equal string "18446744073709551615" (text D.Uint64 (-1L));
  equal string "-1" (text D.Int32 (-1l));
  equal string "65535" (text D.Uint16 65535);
  equal string "0.1" (text D.Float32 (D.of_float D.Float32 0.1));
  equal string "0.1" (text D.Float16 (D.of_float D.Float16 0.1));
  equal string "0.1" (text D.Float64 0.1);
  equal string "-0" (text D.Float64 (-0.));
  equal string "nan" (text D.Float32 Float.nan);
  equal string "-inf" (text D.Float16 Float.neg_infinity);
  equal string "448" (text D.Float8_e4m3 448.);
  equal string "1.5" (text D.Float4_e2m1 1.5);
  equal string "1+2i" (text D.Complex64 { Complex.re = 1.; im = 2. });
  equal string "0.1-0.5i" (text D.Complex128 { Complex.re = 0.1; im = -0.5 });
  equal string "true" (text D.Bit true)

(* A float prints as a decimal that reads back as itself in its format. *)
let law_printed (D.Any dt) =
  match D.kind dt with
  | D.Float ->
      [
        prop (D.name dt)
          (Gen.with_pp hex (near_grid (D.float_format dt)))
          (fun (x, _) ->
            let v = D.of_float dt x in
            let s = Format.asprintf "%a" (D.pp_value dt) v in
            equal float_exact v (D.of_float dt (float_of_string s)));
      ]
  | _ -> []

let tests =
  [
    group "table"
      [
        test "codes count from zero in the order of all" test_codes;
        cases ~name:any_name "nx_dtype.h states each row" D.all test_header;
        test "bits are each format's width" test_bits;
        test "bytes rounds sub-byte runs up and refuses overflow" test_bytes;
        test "of_name finds every name and nothing else" test_names;
        test "equal and equal_witness agree with codes" test_equal;
        test "is partitions the dtypes into kinds" test_kinds;
        test "float_format states each format's facts" test_formats;
      ];
    group "values"
      [
        test "zero and one are stores of 0. and 1." test_identities;
        test "min_value and max_value are the range" test_limits;
      ];
    group "stores"
      [
        cases ~name:stores_name "NaN, zeros and infinities" specials check_store;
        cases ~name:stores_name "past the largest" overflows check_store;
        cases ~name:stores_name "ties round once to even" ties check_store;
        cases ~name:stores_name "below the least normal" underflows check_store;
        cases ~name:stores_name "integers saturate" integers check_store;
        cases ~name:stores_name "complex and booleans" others check_store;
        group "float formats round as their facts say"
          (List.concat_map law_store float_dtypes);
        group "integers truncate and saturate"
          (List.concat_map law_saturate D.all);
      ];
    group "printing"
      [
        test "values print in their dtype's reading" test_printer;
        group "a float reads back from its text"
          (List.concat_map law_printed float_dtypes);
      ];
  ]

let () = exit (run "nx_array.dtype" tests)
