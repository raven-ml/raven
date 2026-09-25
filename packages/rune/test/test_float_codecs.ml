(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx and tolk each encode floats into bfloat16, float16 and the two float8
   formats: nx when it stores a value or casts a tensor, tolk when it folds a
   constant into a compiled graph. Eager and compiled code agree only where the
   two encodings do, so both are checked here against an exact reference.

   The rule is one for every format: round once to nearest even from the
   source's width, and past the largest finite value give the format's infinity,
   or NaN in float8_e4m3, which has none. The one exception is nx's float16 from
   a float64, which rounds to float32 first, as OCaml's float16 bigarrays do.

   The inputs are every float32 bit pattern whose low half is one of a few
   values around each format's rounding bit, the float64 neighbours of some of
   those, and random float64 values between 2^-40 and 2^24. *)

open Windtrap
module B = Nx_buffer
module D = Tolk_uop.Dtype

(* Reference *)

(* [mant] stored fraction bits, [emin] the exponent of the least normal value,
   [max] the largest finite value. *)
type format = {
  name : string;
  mant : int;
  emin : int;
  max : float;
  infinities : bool;
}

let bf16 =
  {
    name = "bfloat16";
    mant = 7;
    emin = -126;
    max = 0x1.fep127;
    infinities = true;
  }

let f16 =
  { name = "float16"; mant = 10; emin = -14; max = 65504.; infinities = true }

let e4m3 =
  { name = "float8_e4m3"; mant = 3; emin = -6; max = 448.; infinities = false }

let e5m2 =
  {
    name = "float8_e5m2";
    mant = 2;
    emin = -14;
    max = 57344.;
    infinities = true;
  }

(* [x] rounded to nearest, ties to even, at [f]'s precision with no upper
   exponent bound. Scaling by a power of two is exact here, and so are the floor
   and the difference. *)
let nearest f x =
  if x = 0. || not (Float.is_finite x) then x
  else
    let _, e = Float.frexp x in
    let q = Int.max (e - 1) f.emin - f.mant in
    let s = Float.ldexp x (-q) in
    let fl = Float.floor s in
    let d = s -. fl in
    let r =
      if d > 0.5 || (d = 0.5 && Float.rem fl 2. <> 0.) then fl +. 1. else fl
    in
    Float.copy_sign (Float.ldexp r q) x

let reference f x =
  let r = nearest f x in
  if Float.is_nan r || Float.abs r <= f.max then r
  else if f.infinities then Float.copy_sign Float.infinity x
  else Float.nan

let to_f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* Codecs *)

let nx_value kind x =
  let b = B.create kind 1 in
  B.set b 0 x;
  B.get b 0

let nx_code kind x =
  let b = B.create kind 1 in
  B.set b 0 x;
  B.get (B.reinterpret Nx_dtype.uint8 b) 0

let nx_decode kind code =
  let b = B.create kind 1 in
  B.set (B.reinterpret Nx_dtype.uint8 b) 0 code;
  B.get b 0

let tolk_fp8 dt x = D.fp8_to_float dt (D.float_to_fp8 dt x)

(* Inputs *)

let f32_inputs lows =
  let xs = ref [] in
  for hi = 0xFFFF downto 0 do
    List.iter
      (fun lo ->
        xs := Int32.float_of_bits (Int32.of_int ((hi lsl 16) lor lo)) :: !xs)
      lows
  done;
  Array.of_list !xs

(* The low halves: zero, the bfloat16 rounding bit, the float16 rounding bit,
   each alone and with a sticky bit on either side. Every float8 rounding bit
   lies in the high half. *)
let f32_sweep =
  f32_inputs
    [ 0; 1; 0x7FFF; 0x8000; 0x8001; 0xFFFF; 0x0FFF; 0x1000; 0x1001; 0x3000 ]

let random_f64 n =
  let s = ref 0x2545F4914F6CDD1DL in
  Array.init n (fun _ ->
      s := Int64.add (Int64.mul !s 6364136223846793005L) 1442695040888963407L;
      let b = !s in
      let m =
        Int64.float_of_bits
          (Int64.logor 0x3FF0000000000000L (Int64.logand b 0xFFFFFFFFFFFFFL))
      in
      let e = Int64.to_int (Int64.shift_right_logical b 58) - 40 in
      Float.ldexp (if Int64.compare b 0L < 0 then -.m else m) e)

let specials =
  [|
    Float.nan;
    -.Float.nan;
    Float.infinity;
    Float.neg_infinity;
    0.;
    -0.;
    1e39;
    -1e39;
    1e-50;
    -1e-50;
    5e-324;
    Float.max_float;
    -.Float.max_float;
    0x1.fffffep127;
    0x1.ffffffp127;
    0x1p-149;
    0x1p-150;
    Float.succ 0x1p-150;
  |]

(* The two float64 neighbours of each float32 whose low half is zero or holds
   the bfloat16 or float16 rounding bit alone. Where that float32 is a tie at
   the target precision, rounding to float32 first moves them onto it. *)
let f64_sweep =
  let near =
    Array.to_list (f32_inputs [ 0; 0x8000; 0x1000; 0x3000 ])
    |> List.filter Float.is_finite
    |> List.concat_map (fun x -> [ Float.pred x; Float.succ x ])
  in
  Array.concat [ specials; Array.of_list near; random_f64 50_000 ]

(* Checks *)

let same a b =
  Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
  || (Float.is_nan a && Float.is_nan b)

let agree ~msg inputs want got =
  let bad =
    Array.to_list inputs
    |> List.filter_map (fun x ->
        let w = want x and g = got x in
        if same w g then None
        else Some (Printf.sprintf "%h: want %h, got %h" x w g))
  in
  equal ~msg (list string) [] (List.filteri (fun i _ -> i < 8) bad)

let check_float8 f kind dt inputs () =
  agree ~msg:(f.name ^ " nx") inputs (reference f) (nx_value kind);
  agree ~msg:(f.name ^ " tolk") inputs (reference f) (tolk_fp8 dt)

let check_bf16 inputs () =
  agree ~msg:"bfloat16 nx" inputs (reference bf16) (nx_value Nx_dtype.bfloat16);
  agree ~msg:"bfloat16 tolk" inputs (reference bf16) D.float_to_bf16

let check_f16 inputs ~wide () =
  let exact = reference f16 in
  let nx = if wide then fun x -> exact (to_f32 x) else exact in
  agree ~msg:"float16 nx" inputs nx (nx_value Nx_dtype.float16);
  agree ~msg:"float16 tolk" inputs exact D.float_to_fp16

(* nx's casts encode through its kernels, not through element stores. *)
let check_casts inputs ~src () =
  let t = Nx.cast src (Nx.create Nx.float64 [| Array.length inputs |] inputs) in
  let cast f dt want =
    let got = Nx.to_array (Nx.cast Nx.float64 (Nx.cast dt t)) in
    let src = Nx.to_array (Nx.cast Nx.float64 t) in
    let bad = ref [] in
    Array.iteri
      (fun i x ->
        let w = want x in
        if (not (same w got.(i))) && List.length !bad < 8 then
          bad := Printf.sprintf "%h: want %h, got %h" x w got.(i) :: !bad)
      src;
    equal ~msg:(f.name ^ " cast") (list string) [] (List.rev !bad)
  in
  cast bf16 Nx.bfloat16 (reference bf16);
  cast e4m3 Nx.float8_e4m3 (reference e4m3);
  cast e5m2 Nx.float8_e5m2 (reference e5m2)

(* Every code decodes to the same value on both sides, and every code but NaN
   encodes back to itself. *)
let check_codes kind dt ~is_nan () =
  for code = 0 to 255 do
    let msg = Printf.sprintf "code 0x%02x" code in
    let v = nx_decode kind code in
    equal ~msg:(msg ^ " decodes alike") bool true
      (same v (D.fp8_to_float dt code));
    equal ~msg:(msg ^ " is NaN") bool (is_nan code) (Float.is_nan v);
    if is_nan code then (
      equal
        ~msg:(msg ^ " nx re-encodes a NaN")
        bool true
        (is_nan (nx_code kind v));
      equal
        ~msg:(msg ^ " tolk re-encodes a NaN")
        bool true
        (is_nan (D.float_to_fp8 dt v)))
    else (
      equal ~msg:(msg ^ " nx re-encodes") int code (nx_code kind v);
      equal ~msg:(msg ^ " tolk re-encodes") int code (D.float_to_fp8 dt v))
  done

let () =
  let e4m3_nan c = c land 0x7F = 0x7F and e5m2_nan c = c land 0x7F > 0x7C in
  run "float codecs"
    [
      group "float8 codes"
        [
          test "e4m3"
            (check_codes Nx_dtype.float8_e4m3 D.fp8e4m3 ~is_nan:e4m3_nan);
          test "e5m2"
            (check_codes Nx_dtype.float8_e5m2 D.fp8e5m2 ~is_nan:e5m2_nan);
        ];
      group "from float32"
        [
          test "e4m3"
            (check_float8 e4m3 Nx_dtype.float8_e4m3 D.fp8e4m3 f32_sweep);
          test "e5m2"
            (check_float8 e5m2 Nx_dtype.float8_e5m2 D.fp8e5m2 f32_sweep);
          test "bfloat16" (check_bf16 f32_sweep);
          test "float16" (check_f16 f32_sweep ~wide:false);
          test "nx casts" (check_casts f32_sweep ~src:Nx.float32);
        ];
      group "from float64"
        [
          test "e4m3"
            (check_float8 e4m3 Nx_dtype.float8_e4m3 D.fp8e4m3 f64_sweep);
          test "e5m2"
            (check_float8 e5m2 Nx_dtype.float8_e5m2 D.fp8e5m2 f64_sweep);
          test "bfloat16" (check_bf16 f64_sweep);
          test "float16" (check_f16 f64_sweep ~wide:true);
          test "nx casts" (check_casts f64_sweep ~src:Nx.float64);
        ];
    ]
