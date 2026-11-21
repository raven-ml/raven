(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

type t = { dtype : Nx.Dtype.any; m : int; k : int; n : int; seed : int }

let pp ppf c =
  let (Nx.Dtype.Any dt) = c.dtype in
  Format.fprintf ppf "%a x [%d; %d] by [%d; %d], seed %d" Nx.Dtype.pp dt c.m
    c.k c.n c.k c.seed

let scale_bytes = [| 0; 1; 2; 126; 127; 128; 252; 253; 254; 255 |]

let cases =
  let dtypes = Nx.Dtype.[ Any Nx.float32; Any Nx.bfloat16; Any Nx.float16 ] in
  let shapes =
    [ (16, 64, 32); (32, 96, 48); (1, 64, 10); (7, 32, 17); (17, 96, 23) ]
  in
  List.concat_map
    (fun dtype ->
      List.mapi (fun i (m, k, n) -> { dtype; m; k; n; seed = i }) shapes)
    dtypes

(* The weight: every block of row [r] has the scale byte
   [scale_bytes.(r mod 10)], and block [b] holds code [(j + r + b) mod 16] at
   value [j], each code twice. A row of byte 253 or 254, where the larger
   codes overflow, holds the positive codes alone, [(j + r + b) mod 8], each
   four times, so that its infinities stay of one sign. *)
let weight ~n ~k =
  let blocks = k / 32 in
  let byte r = scale_bytes.(r mod Array.length scale_bytes) in
  let code r j =
    let m = if byte r = 253 || byte r = 254 then 8 else 16 in
    (j + r + (j / 32)) mod m
  in
  let codes =
    Array.init (n * k / 2) (fun i ->
        let r = i / (k / 2) and j = 2 * (i mod (k / 2)) in
        code r j lor (code r (j + 1) lsl 4))
  in
  let scales = Array.init (n * blocks) (fun i -> byte (i / blocks)) in
  Nx_quant.mxfp4
    ~scales:(Nx.create Nx.uint8 [| n; blocks |] scales)
    (Nx.create Nx.uint8 [| n; k / 2 |] codes)

(* [γ(k, 2u)] at float32, {!Nx.contract}'s bound. *)
let gamma k =
  let kv = float_of_int k *. 0x1p-23 in
  kv /. (1. -. kv)

(* [y], a product of [x] by [w], against [exact] and [mass], [Σ|x w|], each
   [float64] arrays of [m n]: the label of each output out of its bound among
   those [kept] holds, at most eight. *)
let broken (type b) ~label ~kept (dt : (float, b) Nx.dtype) ~k exact mass y =
  let round v =
    let v = Nx.create Nx.float64 [| Array.length v |] v in
    Nx.to_array (Nx.cast Nx.float64 (Nx.cast dt v))
  in
  let tiny = float_of_int k *. 0x1p-149 in
  let e = Array.map (fun s -> (gamma (k + 1) *. s) +. tiny) mass in
  let finite z = if Float.is_finite z then z else 0. in
  let lo = round (Array.mapi (fun i z -> finite z -. e.(i)) exact) in
  let hi = round (Array.mapi (fun i z -> finite z +. e.(i)) exact) in
  let out = ref [] in
  Array.iteri
    (fun i z ->
      let g = y.(i) in
      let ok =
        if Float.is_nan z then Float.is_nan g
        else if Float.is_finite z then lo.(i) <= g && g <= hi.(i)
        else g = z
      in
      if (not ok) && kept i && List.length !out < 8 then
        out :=
          Printf.sprintf "%s [%d]: exactly %h, got %h, within [%h, %h]" label i
            z g lo.(i) hi.(i)
          :: !out)
    exact;
  List.rev !out

(* Whether output [i], of [n] columns, has no value below float32's least
   normal: its weight row's scale byte is 126 or more. A row of byte 0, 1 or 2
   scales its codes to at most [6 2{^-125}], so their products with [x]'s
   elements, at most [2{^-8}], are [0] or below the least normal. A row of a
   larger byte scales its codes to multiples of [2{^-2}], and [x]'s elements,
   float32 draws of multiples of [2{^-24}] times [2{^-8}], are multiples of
   [2{^-32}] in each dtype, so each product and partial sum is [0] or at
   least [2{^-34}]. *)
let normal ~n i = scale_bytes.(i mod n mod Array.length scale_bytes) >= 126

let law ?(flushes = false) (module S : Nx.Devices) c =
  let (Nx.Dtype.Any dt) = c.dtype in
  match Nx.Dtype.kind dt with
  | Float ->
      let w = weight ~n:c.n ~k:c.k in
      let x =
        let key = Nx.Rng.key c.seed in
        let u = Nx.Rng.uniform ~key Nx.float32 [| c.m; c.k |] in
        Nx.cast dt (Nx.mul_s u 0x1p-8)
      in
      let wide v = Nx.cast Nx.float64 v in
      let wf = Nx.matrix_transpose (wide (Nx_quant.dequant Nx.float32 w)) in
      let exact = Nx.to_array (Nx.matmul (wide x) wf) in
      let mass = Nx.to_array (Nx.matmul (Nx.abs (wide x)) (Nx.abs wf)) in
      let infinite z = Float.abs z = Float.infinity in
      equal ~msg:"an output is infinite" bool true
        (Array.exists infinite exact);
      let ws = Nx_quant.place S.on w in
      let before = Rune.evaluations () in
      let y =
        Rune.jit Nx.Ptree.(tensor @-> returns tensor) (Nx_quant.apply ws)
          (Nx.place S.on x)
      in
      equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
      let values y = Nx.to_array (wide (Nx.place Nx.Host.on y)) in
      let check ~kept label y =
        equal ~msg:label (list string) []
          (broken ~label ~kept dt ~k:c.k exact mass (values y))
      in
      check ~kept:(Fun.const true) "eager" (Nx_quant.apply w x);
      check ~kept:(fun i -> (not flushes) || normal ~n:c.n i) "compiled" y
  | _ -> invalid_arg "Mxfp4_products.law: not a float dtype"

(* Decodes

   Each e2m1 code at each scale byte, through a compiled product of bfloat16
   rows, and through a compiled dequant: the subnormal scales 0 and 1, and 254
   and 255, where codes of 2 or more overflow and every code is NaN, among
   them. *)

type decode = { tokens : int; experts : bool; row : int }

let pp_decode ppf d =
  Format.fprintf ppf "%d tokens%s, rows of 2^%d" d.tokens
    (if d.experts then ", through experts" else "")
    d.row

let decodes =
  List.concat_map
    (fun experts ->
      List.concat_map
        (fun tokens -> List.map (fun row -> { tokens; experts; row }) [ -8; 8 ])
        [ 1; 64 ])
    [ false; true ]

let decode_bytes = Array.init 256 Fun.id
let decode_k = 256

(* e2m1's values, by code: sign, then magnitude. *)
let e2m1 c =
  let m = [| 0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6. |].(c land 7) in
  if c land 8 = 0 then m else -.m

(* The value of code [c] at scale byte [s], at float32: [2^(s - 127)] times
   the code, infinite past float32's range, NaN at 255. *)
let decoded_value s c =
  if s = 255 then Float.nan
  else
    Int32.float_of_bits (Int32.bits_of_float (Float.ldexp (e2m1 c) (s - 127)))

(* Row [16 b + c] holds code [c] at each of its [decode_k] values, at scale
   byte [decode_bytes.(b)]; each row of [x] holds [2^row] at each, so an
   output is [decode_k] equal terms that sum to the row's value times
   [2^(row + 8)] exactly, [+0] for both zeros. *)
let decode_law ?(flushes = false) (module S : Nx.Devices) d =
  let n = 16 * Array.length decode_bytes and k = decode_k in
  let byte r = decode_bytes.(r / 16) in
  let codes =
    Array.init (n * k / 2) (fun i ->
        let c = i / (k / 2) mod 16 in
        c lor (c lsl 4))
  in
  let scales = Array.init (n * k / 32) (fun i -> byte (i / (k / 32))) in
  let lead = if d.experts then [| 2 |] else [||] in
  let twice a = if d.experts then Array.append a a else a in
  let part last a =
    Nx.create Nx.uint8 (Array.append lead [| n; last |]) (twice a)
  in
  let w = Nx_quant.mxfp4 ~scales:(part (k / 32) scales) (part (k / 2) codes) in
  let ws = Nx_quant.place S.on w in
  let x =
    Nx.place S.on (Nx.full Nx.bfloat16 [| d.tokens; k |] (Float.ldexp 1. d.row))
  in
  let y =
    if d.experts then
      let ids =
        Nx.place S.on
          (Nx.init Nx.int64 [| d.tokens |] (fun i ->
               Int64.of_int (i.(0) mod 2)))
      in
      Rune.jit
        Nx.Ptree.(tensor @-> tensor @-> returns tensor)
        (fun ids x -> Nx_quant.apply_experts ~ids ws x)
        ids x
    else Rune.jit Nx.Ptree.(tensor @-> returns tensor) (Nx_quant.apply ws) x
  in
  let show v = if Float.is_nan v then "nan" else Printf.sprintf "%h" v in
  let value r = decoded_value (byte r) (r mod 16) in
  (* On a target that flushes, the rows whose value and products are not
     subnormal. *)
  let normal v = v = 0. || not (Float.abs v < 0x1p-126) in
  let kept r =
    (not flushes) || (normal (value r) && normal (Float.ldexp (value r) d.row))
  in
  let f32 v = Int32.float_of_bits (Int32.bits_of_float v) in
  let expected =
    Array.init (d.tokens * n) (fun i ->
        let r = i mod n in
        if kept r then show (f32 (Float.ldexp (value r) (d.row + 8)) +. 0.)
        else "-")
  in
  let got =
    Array.mapi
      (fun i v -> if kept (i mod n) then show v else "-")
      (Nx.to_array (Nx.cast Nx.float64 (Nx.place Nx.Host.on y)))
  in
  equal (array string) expected got

(* Row [16 b + c] of one block holds code [c] at scale byte [b]: its first
   value, dequantized, is the code's value at that byte. *)
let dequant_law ?(flushes = false) (module S : Nx.Devices) () =
  let n = 16 * 256 in
  let codes =
    Array.init (n * 16) (fun i ->
        let c = i / 16 mod 16 in
        c lor (c lsl 4))
  in
  let scales = Array.init n (fun r -> r / 16) in
  let on a last = Nx.place S.on (Nx.create Nx.uint8 [| n; last |] a) in
  let y =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> returns tensor)
      (fun scales codes ->
        Nx.slice [ A; I 0 ]
          (Nx_quant.dequant Nx.float32 (Nx_quant.mxfp4 ~scales codes)))
      (on scales 1) (on codes 16)
  in
  let show v = if Float.is_nan v then "nan" else Printf.sprintf "%h" v in
  (* On a target that flushes, the values that are not subnormal. *)
  let kept v = (not flushes) || not (v <> 0. && Float.abs v < 0x1p-126) in
  let expected =
    Array.init n (fun r ->
        let v = decoded_value (r / 16) (r mod 16) in
        if kept v then show v else "-")
  in
  let got =
    Array.mapi
      (fun r v ->
        if kept (decoded_value (r / 16) (r mod 16)) then show v else "-")
      (Nx.to_array (Nx.place Nx.Host.on y))
  in
  equal (array string) expected got
