(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module V = Mynd

(* Dtypes *)

type code = C : string * (_, _) Nx.dtype -> code
type flt = F : string * (float, _) Nx.dtype -> flt

let small_codes =
  [
    C ("uint4", Nx.uint4);
    C ("int4", Nx.int4);
    C ("uint8", Nx.uint8);
    C ("int8", Nx.int8);
    C ("uint16", Nx.uint16);
    C ("int16", Nx.int16);
  ]

let wide_codes =
  [
    C ("uint32", Nx.uint32);
    C ("int32", Nx.int32);
    C ("uint64", Nx.uint64);
    C ("int64", Nx.int64);
  ]

let floats =
  [
    F ("float4_e2m1fn", Nx.float4_e2m1fn);
    F ("float8_e5m2", Nx.float8_e5m2);
    F ("float8_e4m3fn", Nx.float8_e4m3fn);
    F ("bfloat16", Nx.bfloat16);
    F ("float16", Nx.float16);
    F ("float32", Nx.float32);
    F ("float64", Nx.float64);
  ]

let code_name (C (n, _)) = n
let float_name (F (n, _)) = n
let signed dt = Nx.Dtype.is Nx.Dtype.Signed dt

(* [s] value bits and the largest code [m = 2ˢ - 1]. *)
let value_bits dt = if signed dt then Nx.Dtype.bits dt - 1 else Nx.Dtype.bits dt
let largest dt = (1 lsl value_bits dt) - 1

(* Every code of a dtype of up to 16 bits, as a host array, and as ints. *)
let all_codes dt =
  let lo = if signed dt then -largest dt - 1 else 0 in
  let ks = Array.init (largest dt - lo + 1) (fun i -> lo + i) in
  ( ks,
    Nx.cast dt
      (Nx.create Nx.int64 [| Array.length ks |] (Array.map Int64.of_int ks)) )

let ints x = Array.map Int64.to_int (Nx.to_array (Nx.cast Nx.int64 x))
let values x = Nx.to_array (Nx.cast Nx.float64 x)

(* [nearest ~p ~emin n d] is [n / d], [0 <= n <= d < 2¹⁶], rounded to the
   nearest float of [p <= 24] significant bits whose least normal exponent is
   [emin], ties to even: the exact quotient rounded once. *)
let nearest ~p ~emin n d =
  if n = 0 then 0.
  else
    let e = ref 0 in
    while n lsl - !e < d do
      decr e
    done;
    let e = Int.max !e emin in
    let num = n lsl (p - 1 - e) in
    let q = num / d and r = num mod d in
    let q = if 2 * r > d || (2 * r = d && q land 1 = 1) then q + 1 else q in
    Float.ldexp (Float.of_int q) (e - p + 1)

let rounded_once (type s) (dt : (float, s) Nx.dtype) n d =
  if Nx.Dtype.equal dt Nx.float64 then Float.of_int n /. Float.of_int d
  else
    let f = Nx.Dtype.float_format dt in
    let _, e = Float.frexp f.min_normal in
    nearest ~p:(f.fraction_bits + 1) ~emin:(e - 1) n d

(* The fraction [k / m] of a code, as a sign and a magnitude in [[0, m]]. *)
let fraction dt k =
  let m = largest dt in
  let k = Int.max k (-m) in
  (k < 0, Int.abs k)

(* Integer to float *)

let to_float =
  let rows =
    List.concat_map (fun c -> List.map (fun f -> (c, f)) floats) small_codes
  in
  cases "a code of up to 16 bits is k / m rounded once"
    ~id:(fun (c, f) -> code_name c ^ " to " ^ float_name f)
    rows
    (fun (C (_, cdt), F (_, fdt)) ->
      let ks, x = all_codes cdt in
      let want =
        Array.map
          (fun k ->
            let neg, a = fraction cdt k in
            let v = rounded_once fdt a (largest cdt) in
            if neg then -.v else v)
          ks
      in
      equal (array float_exact) want (values (V.convert fdt x)))

(* 32-bit codes divide exactly at float64, so into float64 they round once. *)
let wide_to_float64 =
  prop "a 32-bit code into float64 is k / m rounded once"
    Gen.(pair bool int32)
    (fun (unsigned, k) ->
      let want =
        if unsigned then
          Int64.to_float (Int64.logand (Int64.of_int32 k) 0xffff_ffffL)
          /. 4294967295.
        else Float.max (Int32.to_float k) (-2147483647.) /. 2147483647.
      in
      let x =
        if unsigned then
          V.convert Nx.float64 (Nx.create Nx.uint32 [| 1 |] [| k |])
        else V.convert Nx.float64 (Nx.create Nx.int32 [| 1 |] [| k |])
      in
      equal float_exact want (Nx.item [ 0 ] x))

(* Float to integer *)

let to_code =
  cases "floats clamp, scale and round to even"
    ~id:(fun (n, _) -> n)
    [
      ( "uint8",
        fun () ->
          ints
            (V.convert Nx.uint8
               (Nx.create Nx.float32 [| 8 |]
                  [|
                    -0.5; 0.; 0.5; 1.; 1.5; Float.nan; infinity; neg_infinity;
                  |]))
          |> equal (array int) [| 0; 0; 128; 255; 255; 0; 255; 0 |] );
      ( "int8",
        fun () ->
          ints
            (V.convert Nx.int8
               (Nx.create Nx.float32 [| 7 |]
                  [| -2.; -1.; -0.5; 0.; 0.5; 1.; Float.nan |]))
          |> equal (array int) [| -127; -127; -64; 0; 64; 127; 0 |] );
      ( "uint16 from float16",
        fun () ->
          ints
            (V.convert Nx.uint16
               (Nx.create Nx.float16 [| 3 |] [| 0.5; 1.; 0.25 |]))
          |> equal (array int) [| 32768; 65535; 16384 |] );
      ( "uint4 from float4",
        fun () ->
          ints
            (V.convert Nx.uint4
               (Nx.create Nx.float4_e2m1fn [| 3 |] [| 0.5; 1.; 0. |]))
          |> equal (array int) [| 8; 15; 0 |] );
      ( "int16",
        fun () ->
          ints
            (V.convert Nx.int16
               (Nx.create Nx.float64 [| 3 |] [| -0.5; 0.5; -1. |]))
          |> equal (array int) [| -16384; 16384; -32767 |] );
      ( "uint32 from float32",
        fun () ->
          Nx.to_array
            (V.convert Nx.uint32
               (Nx.create Nx.float32 [| 3 |] [| 1.; 0.5; 0. |]))
          |> equal (array int32) [| -1l; 0x8000_0000l; 0l |] );
      ( "int64 at its ends",
        fun () ->
          Nx.to_array
            (V.convert Nx.int64
               (Nx.create Nx.float64 [| 3 |] [| 1.; -1.; 0. |]))
          |> equal (array int64)
               [| Int64.max_int; Int64.neg Int64.max_int; 0L |] );
      ( "uint64 at its ends",
        fun () ->
          Nx.to_array
            (V.convert Nx.uint64
               (Nx.create Nx.float64 [| 3 |] [| 1.; 0.; -1. |]))
          |> equal (array int64) [| -1L; 0L; 0L |] );
    ]
    (fun (_, f) -> f ())

(* The code is the nearest to the product in float32, which for an 8-bit or
   16-bit code is within [m 2⁻²⁴] of the exact product. *)
let nearest_code =
  prop "a float32 lands on the code nearest x m" ~count:300
    Gen.(pair (float_range (-1.5) 1.5) bool)
    (fun (x, wide) ->
      let x = Int32.float_of_bits (Int32.bits_of_float x) in
      let src = Nx.create Nx.float32 [| 1 |] [| x |] in
      let m, k =
        if wide then (65535., (ints (V.convert Nx.uint16 src)).(0))
        else (255., (ints (V.convert Nx.uint8 src)).(0))
      in
      let exact = Float.min 1. (Float.max 0. x) *. m in
      less float_exact
        ~than:(0.5 +. (m *. 0x1p-24))
        (Float.abs (Float.of_int k -. exact)))

(* Round trips *)

let round_trips =
  [
    ("uint4", [ "float8_e4m3fn"; "bfloat16"; "float16"; "float32"; "float64" ]);
    ("int4", [ "float8_e4m3fn"; "bfloat16"; "float16"; "float32"; "float64" ]);
    ("uint8", [ "bfloat16"; "float16"; "float32"; "float64" ]);
    ("int8", [ "bfloat16"; "float16"; "float32"; "float64" ]);
    ("uint16", [ "float32"; "float64" ]);
    ("int16", [ "float32"; "float64" ]);
  ]

let round_trip =
  let rows =
    List.concat_map
      (fun c ->
        List.filter_map
          (fun f ->
            if List.mem (float_name f) (List.assoc (code_name c) round_trips)
            then Some (c, f)
            else None)
          floats)
      small_codes
  in
  cases "codes come back through the floats the doc names"
    ~id:(fun (c, f) -> code_name c ^ " through " ^ float_name f)
    rows
    (fun (C (_, cdt), F (_, fdt)) ->
      let ks, x = all_codes cdt in
      let want = Array.map (fun k -> Int.max k (-largest cdt)) ks in
      equal (array int) want (ints (V.convert cdt (V.convert fdt x))))

let wide_round_trip =
  prop "32-bit codes come back through float64" Gen.int32 (fun k ->
      let u = Nx.create Nx.uint32 [| 1 |] [| k |] in
      let s = Nx.create Nx.int32 [| 1 |] [| k |] in
      equal int32 k
        (Nx.item [ 0 ] (V.convert Nx.uint32 (V.convert Nx.float64 u)));
      equal int32
        (if k = Int32.min_int then Int32.succ k else k)
        (Nx.item [ 0 ] (V.convert Nx.int32 (V.convert Nx.float64 s))))

(* Integer to integer *)

(* [recode src dst k] is the code of [dst] nearest [k]'s fraction, exactly: [⌊(2
   a n + m) / 2m⌋] for the magnitude [a]. *)
let recode src dst k =
  let neg, a = fraction src k in
  let m = largest src and n = largest dst in
  let c = ((2 * a * n) + m) / (2 * m) in
  if not neg then c else if signed dst then -c else 0

let small_pairs =
  let rows =
    List.concat_map
      (fun c -> List.map (fun c' -> (c, c')) small_codes)
      small_codes
  in
  cases "codes of up to 16 bits become the nearest code"
    ~id:(fun (c, c') -> code_name c ^ " to " ^ code_name c')
    rows
    (fun (C (_, src), C (_, dst)) ->
      let ks, x = all_codes src in
      let want =
        if Nx.Dtype.equal src dst then ks else Array.map (recode src dst) ks
      in
      equal (array int) want (ints (V.convert dst x)))

(* Wider pairs against Python's integers, from golden/convert.npz
   (contrib/mynd/test/gen/convert.py) *)
let golden = lazy (Nx_io.load_npz "golden/convert.npz")

let entry key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack Nx.int64 p
  | None -> failf "golden/convert.npz has no %s" key

let wide_pairs =
  let all = small_codes @ wide_codes in
  let rows =
    List.concat_map
      (fun c ->
        List.filter_map
          (fun c' ->
            if List.mem c small_codes && List.mem c' small_codes then None
            else if code_name c = code_name c' then None
            else Some (c, c'))
          all)
      all
  in
  cases "wider codes become the nearest code"
    ~id:(fun (c, c') -> code_name c ^ " to " ^ code_name c')
    rows
    (fun (C (src_name, src), C (dst_name, dst)) ->
      let key = src_name ^ "." ^ dst_name in
      let x = Nx.cast src (entry (key ^ ".x")) in
      equal (array int64)
        (Nx.to_array (entry (key ^ ".y")))
        (Nx.to_array (Nx.cast Nx.int64 (V.convert dst x))))

(* Others *)

let identity =
  test "the same dtype is x itself" (fun () ->
      let x = Nx.create Nx.int8 [| 2 |] [| -128; 5 |] in
      equal (array int) [| -128; 5 |] (ints (V.convert Nx.int8 x)))

let float_to_float =
  test "float to float is a cast with its derivative" (fun () ->
      let x = Nx.create Nx.float64 [| 3 |] [| 0.1; -2.; 1e40 |] in
      equal (array float_exact)
        (Nx.to_array (Nx.cast Nx.float32 x))
        (Nx.to_array (V.convert Nx.float32 x));
      let g =
        Rune.grad'
          (fun x -> Nx.sum (V.convert Nx.float32 x |> Nx.cast Nx.float64))
          x
      in
      equal (array float_exact) [| 1.; 1.; 1. |] (Nx.to_array g))

let compiled =
  test "compiled is eager" (fun () ->
      let _, x = all_codes Nx.uint8 in
      let f = Rune.jit' (V.convert Nx.float32) x in
      equal (array float_exact) (values (V.convert Nx.float32 x)) (values f);
      let y = Nx.linspace Nx.float32 (-0.25) 1.25 301 in
      equal (array int)
        (ints (V.convert Nx.uint8 y))
        (ints (Rune.jit' (V.convert Nx.uint8) y)))

let refusals =
  group "misuse"
    [
      test "booleans hold no fraction" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.convert") (fun () ->
              V.convert Nx.float32 (Nx.create Nx.bool [| 1 |] [| true |])));
      test "complex numbers hold no fraction" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.convert") (fun () ->
              V.convert Nx.complex64 (Nx.zeros Nx.float32 [| 1 |])));
    ]

(* Transfer functions *)

let css_lin v =
  let a = Float.abs v in
  if a <= 0.04045 then v /. 12.92
  else Float.copy_sign (Float.pow ((a +. 0.055) /. 1.055) 2.4) v

let css_gam v =
  let a = Float.abs v in
  if a > 0.0031308 then
    Float.copy_sign ((1.055 *. Float.pow a (1. /. 2.4)) -. 0.055) v
  else 12.92 *. v

let transfer_points =
  [
    -1.5;
    -0.5;
    -0.04045;
    -0.01;
    -0.;
    0.;
    0.0031308;
    0.003;
    0.04045;
    0.04046;
    0.2;
    0.5;
    1.;
    2.;
  ]

let transfer_values =
  test "the transfer functions are CSS Color 4's" (fun () ->
      let x =
        Nx.create Nx.float64
          [| List.length transfer_points |]
          (Array.of_list transfer_points)
      in
      equal
        (array (float 1e-15))
        (Array.map css_lin (Nx.to_array x))
        (Nx.to_array (V.srgb_to_linear x));
      equal
        (array (float 1e-15))
        (Array.map css_gam (Nx.to_array x))
        (Nx.to_array (V.linear_to_srgb x)))

let transfer_inverse =
  prop "linear_to_srgb inverts srgb_to_linear" (Gen.float_range (-2.) 2.)
    (fun v ->
      let x = Nx.create Nx.float64 [| 1 |] [| v |] in
      equal (float 1e-14) v
        (Nx.item [ 0 ] (V.linear_to_srgb (V.srgb_to_linear x))))

(* The derivative against a central difference, at points off the joins. *)
let transfer_derivative =
  prop "the transfer functions' derivatives" ~count:200
    ~examples:[ 0.; -0.; 1.; -1. ] (Gen.float_range (-1.5) 1.5) (fun v ->
      let d = 1e-6 in
      assume (Float.abs (Float.abs v -. 0.04045) > 2. *. d);
      assume (Float.abs (Float.abs v -. 0.0031308) > 2. *. d);
      let x = Nx.create Nx.float64 [| 1 |] [| v |] in
      List.iter
        (fun (name, f, g) ->
          let fd = (g (v +. d) -. g (v -. d)) /. (2. *. d) in
          let got = Nx.item [ 0 ] (Rune.grad' (fun x -> Nx.sum (f x)) x) in
          equal ~msg:name (float_rel ~rel:1e-5 ~abs:1e-6) fd got)
        [
          ("srgb_to_linear", V.srgb_to_linear, css_lin);
          ("linear_to_srgb", V.linear_to_srgb, css_gam);
        ])

(* Alpha *)

let alpha =
  test "premultiply and unpremultiply" (fun () ->
      let x =
        Nx.create Nx.float32 [| 3; 4 |]
          [| 0.5; 0.25; 1.; 0.5; 0.2; 0.4; 0.6; 0.; 1.; 1.; 1.; 1. |]
      in
      equal (array float_exact)
        [| 0.25; 0.125; 0.5; 0.5; 0.; 0.; 0.; 0.; 1.; 1.; 1.; 1. |]
        (values (V.premultiply x));
      equal (array float_exact)
        [| 1.; 0.5; 2.; 0.5; 0.; 0.; 0.; 0.; 1.; 1.; 1.; 1. |]
        (values (V.unpremultiply x)))

let alpha_inverse =
  prop "unpremultiply inverts premultiply where alpha is not 0"
    Gen.(pair (float_range 0. 1.) (float_range 0.01 1.))
    (fun (c, a) ->
      let x = Nx.create Nx.float64 [| 2 |] [| c; a |] in
      equal
        (array (float 1e-15))
        [| c; a |]
        (Nx.to_array (V.unpremultiply (V.premultiply x))))

let alpha_zero =
  test "unpremultiply's derivative at zero alpha is 0" (fun () ->
      let x = Nx.create Nx.float64 [| 2 |] [| 0.3; 0. |] in
      let g = Rune.grad' (fun x -> Nx.sum (V.unpremultiply x)) x in
      equal (array float_exact) [| 0.; 1. |] (Nx.to_array g))

let () =
  exit
    (run "mynd.values"
       [
         to_float;
         wide_to_float64;
         to_code;
         nearest_code;
         round_trip;
         wide_round_trip;
         small_pairs;
         wide_pairs;
         identity;
         float_to_float;
         compiled;
         refusals;
         transfer_values;
         transfer_inverse;
         transfer_derivative;
         alpha;
         alpha_inverse;
         alpha_zero;
       ])
