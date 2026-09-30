(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx encodes floats into float16, bfloat16 and the float8 formats in three
   places: when it stores a value, when it casts a tensor, and in
   [Nx_dtype.Scalar.encode], with which compiled graphs fold constants. Eager
   and compiled code agree where these do, so each is checked here against an
   exact reference.

   The rule is one for every format: round once to nearest even from the
   source's width. Past the largest finite value, float16 and bfloat16 give
   their infinity, and the float8 formats saturate to the largest finite value
   of the sign. An infinity stays one where the format has infinities and is NaN
   where it has none. The fnuz formats have no negative zero.

   The inputs are every float32 bit pattern whose low half is one of a few
   values around each format's rounding bit, the float64 neighbours of some of
   those, and random float64 values between 2^-40 and 2^24. *)

open Windtrap
module S = Nx_dtype.Scalar
module E = Nx_array.Elements

(* Reference *)

(* [mant] stored fraction bits, [emin] the exponent of the least normal value,
   [max] the largest finite value, [infinities] whether the format has them,
   [saturates] whether finite values past [max] clamp to it. *)
type format = {
  scalar : S.t;
  mant : int;
  emin : int;
  max : float;
  infinities : bool;
  saturates : bool;
  fnuz : bool;
}

let format ?(infinities = true) ?(saturates = true) ?(fnuz = false) scalar ~mant
    ~emin ~max =
  { scalar; mant; emin; max; infinities; saturates; fnuz }

let f16 = format S.Float16 ~saturates:false ~mant:10 ~emin:(-14) ~max:65504.

let bf16 =
  format S.BFloat16 ~saturates:false ~mant:7 ~emin:(-126) ~max:0x1.fep127

let e4m3 = format S.Float8_e4m3 ~infinities:false ~mant:3 ~emin:(-6) ~max:448.
let e5m2 = format S.Float8_e5m2 ~mant:2 ~emin:(-14) ~max:57344.

let e4m3fnuz =
  format S.Float8_e4m3fnuz ~infinities:false ~fnuz:true ~mant:3 ~emin:(-7)
    ~max:240.

let e5m2fnuz =
  format S.Float8_e5m2fnuz ~infinities:false ~fnuz:true ~mant:2 ~emin:(-15)
    ~max:57344.

let name f = S.to_string f.scalar

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
  if Float.is_nan r then r
  else if not (Float.is_finite x) then if f.infinities then x else Float.nan
  else if Float.abs r > f.max then
    Float.copy_sign (if f.saturates then f.max else Float.infinity) x
  else if f.fnuz && r = 0. then 0.
  else r

(* Codecs *)

let codec f x = S.decode f.scalar (S.encode f.scalar x)
let byte b = Nx_device.Buffer.bigarray Bigarray.int8_unsigned b

let nx_value dtype x =
  let b = Nx_array.Elements.create dtype 1 in
  E.set dtype b 0 x;
  E.get dtype b 0

let nx_code dtype x =
  let b = Nx_array.Elements.create dtype 1 in
  E.set dtype b 0 x;
  (byte b).{0}

let nx_decode dtype code =
  let b = Nx_array.Elements.create dtype 1 in
  (byte b).{0} <- code;
  E.get dtype b 0

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

let check_codec f inputs () =
  agree ~msg:(name f ^ " encode") inputs (reference f) (codec f)

let check_store f dtype inputs () =
  agree ~msg:(name f ^ " store") inputs (reference f) (nx_value dtype)

(* nx's casts encode through its kernels, not through element stores. *)
let check_casts inputs ~src () =
  let t = Nx.cast src (Nx.create Nx.float64 [| Array.length inputs |] inputs) in
  let values = Nx.to_array (Nx.cast Nx.float64 t) in
  let cast f dt =
    let got = Nx.to_array (Nx.cast Nx.float64 (Nx.cast dt t)) in
    let bad = ref [] in
    Array.iteri
      (fun i x ->
        let w = reference f x in
        if (not (same w got.(i))) && List.length !bad < 8 then
          bad := Printf.sprintf "%h: want %h, got %h" x w got.(i) :: !bad)
      values;
    equal ~msg:(name f ^ " cast") (list string) [] (List.rev !bad)
  in
  cast f16 Nx.float16;
  cast bf16 Nx.bfloat16;
  cast e4m3 Nx.float8_e4m3;
  cast e5m2 Nx.float8_e5m2

(* A 64-bit integer next to a bfloat16 midpoint past 2^53: [m * 2^s] and [(m +
   1) * 2^s] are consecutive bfloat16 values, and converting the integer to a
   double first rounds it onto their midpoint. Each input comes with the
   bfloat16 value it rounds to once. *)
let int_ties exponents =
  List.concat_map
    (fun e ->
      List.concat_map
        (fun m ->
          let s = e - 7 in
          let mid =
            Int64.add
              (Int64.shift_left (Int64.of_int m) s)
              (Int64.shift_left 1L (s - 1))
          in
          let value m = Float.ldexp (Float.of_int m) s in
          let even = if m land 1 = 0 then m else m + 1 in
          [
            (Int64.pred mid, value m);
            (mid, value even);
            (Int64.succ mid, value (m + 1));
          ])
        [ 128; 129; 200; 254 ])
    exponents

let check_int_casts src ties () =
  let inputs = Array.of_list (List.map fst ties) in
  let t = Nx.create src [| Array.length inputs |] inputs in
  let got = Nx.to_array (Nx.cast Nx.float64 (Nx.cast Nx.bfloat16 t)) in
  let bad =
    List.filteri (fun i (_, w) -> not (same w got.(i))) ties
    |> List.map (fun (x, w) -> Printf.sprintf "%Lu: want %h" x w)
  in
  equal ~msg:"bfloat16 cast" (list string) [] bad

(* Every code decodes to its value, and every code but NaN encodes back to
   itself. A NaN encodes back to a NaN of its sign, in the formats whose NaNs
   have one. *)
let check_codes f ~is_nan () =
  for code = 0 to 255 do
    let msg = Printf.sprintf "%s code 0x%02x" (name f) code in
    let v = S.decode f.scalar code in
    equal ~msg:(msg ^ " is NaN") bool (is_nan code) (Float.is_nan v);
    if is_nan code then
      let back = S.encode f.scalar v in
      equal
        ~msg:(msg ^ " re-encodes a NaN of its sign")
        bool true
        (is_nan back && (f.fnuz || back land 0x80 = code land 0x80))
    else equal ~msg:(msg ^ " re-encodes") int code (S.encode f.scalar v)
  done

(* An element's store and load go through the same codec. *)
let check_buffer_codes dtype f () =
  for code = 0 to 255 do
    let msg = Printf.sprintf "%s code 0x%02x" (name f) code in
    let v = S.decode f.scalar code in
    equal ~msg:(msg ^ " loads alike") bool true (same v (nx_decode dtype code));
    if not (Float.is_nan v) then
      equal ~msg:(msg ^ " stores alike") int code (nx_code dtype v)
  done

let check_formats () =
  let raises f =
    match f () with _ -> false | exception Invalid_argument _ -> true
  in
  is_true ~msg:"encode int8" (raises (fun () -> S.encode S.Int8 1.));
  is_true ~msg:"encode float32" (raises (fun () -> S.encode S.Float32 1.));
  is_true ~msg:"decode past the format"
    (raises (fun () -> S.decode S.Float8_e4m3 256));
  is_true ~msg:"decode a negative code"
    (raises (fun () -> S.decode S.BFloat16 (-1)));
  equal ~msg:"bfloat16 bits" int 0x3F80 (S.encode S.BFloat16 1.);
  equal ~msg:"float16 bits" int 0x3C00 (S.encode S.Float16 1.);
  equal ~msg:"fnuz has one NaN" int 0x80
    (S.encode S.Float8_e4m3fnuz (-.Float.nan))

let () =
  let e4m3_nan c = c land 0x7F = 0x7F
  and e5m2_nan c = c land 0x7F > 0x7C
  and fnuz_nan c = c = 0x80 in
  let formats = [ f16; bf16; e4m3; e5m2; e4m3fnuz; e5m2fnuz ] in
  let sweep inputs =
    List.map (fun f -> test (name f) (check_codec f inputs)) formats
    @ [
        test "float16 store" (check_store f16 Nx_dtype.float16 inputs);
        test "bfloat16 store" (check_store bf16 Nx_dtype.bfloat16 inputs);
        test "e4m3 store" (check_store e4m3 Nx_dtype.float8_e4m3 inputs);
        test "e5m2 store" (check_store e5m2 Nx_dtype.float8_e5m2 inputs);
      ]
  in
  exit
    (run "float codecs"
       [
         group "formats" [ test "domain and bits" check_formats ];
         group "float8 codes"
           [
             test "e4m3" (check_codes e4m3 ~is_nan:e4m3_nan);
             test "e5m2" (check_codes e5m2 ~is_nan:e5m2_nan);
             test "e4m3fnuz" (check_codes e4m3fnuz ~is_nan:fnuz_nan);
             test "e5m2fnuz" (check_codes e5m2fnuz ~is_nan:fnuz_nan);
             test "e4m3 buffers" (check_buffer_codes Nx_dtype.float8_e4m3 e4m3);
             test "e5m2 buffers" (check_buffer_codes Nx_dtype.float8_e5m2 e5m2);
           ];
         group "from float32"
           (sweep f32_sweep
           @ [ test "nx casts" (check_casts f32_sweep ~src:Nx.float32) ]);
         group "from float64"
           (sweep f64_sweep
           @ [ test "nx casts" (check_casts f64_sweep ~src:Nx.float64) ]);
         group "from 64-bit integers"
           [
             test "int64 casts round once past 2^53"
               (check_int_casts Nx.int64 (int_ties [ 54; 55; 60; 62 ]));
             test "uint64 casts round once past 2^53"
               (check_int_casts Nx.uint64 (int_ties [ 54; 62; 63 ]));
           ];
       ])
