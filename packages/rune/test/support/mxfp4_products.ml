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
   [float64] arrays of [m n]: the label of each output out of its bound, at
   most eight. *)
let broken (type b) ~label (dt : (float, b) Nx.dtype) ~k exact mass y =
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
      if (not ok) && List.length !out < 8 then
        out :=
          Printf.sprintf "%s [%d]: exactly %h, got %h, within [%h, %h]" label i
            z g lo.(i) hi.(i)
          :: !out)
    exact;
  List.rev !out

let law (module S : Nx.Devices) c =
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
      let check label y =
        equal ~msg:label (list string) []
          (broken ~label dt ~k:c.k exact mass (values y))
      in
      check "eager" (Nx_quant.apply w x);
      check "compiled" y
  | _ -> invalid_arg "Mxfp4_products.law: not a float dtype"
