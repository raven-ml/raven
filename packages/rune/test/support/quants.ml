(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type format = Q8_0 | Q6_K | Q4_K | Mxfp4 | Mxfp4_blocks

let formats = [ Q8_0; Q6_K; Q4_K; Mxfp4; Mxfp4_blocks ]

let name = function
  | Q8_0 -> "q8_0"
  | Q6_K -> "q6_k"
  | Q4_K -> "q4_k"
  | Mxfp4 -> "mxfp4"
  | Mxfp4_blocks -> "mxfp4_blocks"

let block_values = function Q8_0 | Mxfp4 | Mxfp4_blocks -> 32 | _ -> 256

(* The bytes of a block, and the offsets of its float16 fields. *)
let block_bytes = function
  | Q8_0 -> 34
  | Q6_K -> 210
  | Q4_K -> 144
  | Mxfp4 -> 16
  | Mxfp4_blocks -> 17

let halves = function
  | Q8_0 -> [ 0 ]
  | Q6_K -> [ 208 ]
  | Q4_K -> [ 0; 2 ]
  | Mxfp4 | Mxfp4_blocks -> []

let weight format ~rows ~blocks =
  let size = block_bytes format in
  let count = rows * blocks in
  let b = Array.init (count * size) (fun i -> ((i * 37) + (i / 7)) mod 256) in
  for j = 0 to count - 1 do
    let at o = (j * size) + o in
    List.iter
      (fun o -> b.(at (o + 1)) <- (0x18 + (j mod 20)) lor (j mod 2 * 0x80))
      (halves format);
    if format = Mxfp4_blocks then b.(at 0) <- 118 + (j mod 19)
  done;
  let u8 last a = Nx.create Nx.uint8 [| rows; last |] a in
  match format with
  | Q8_0 -> Nx_quant.q8_0 (u8 (blocks * size) b)
  | Q6_K -> Nx_quant.q6_k (u8 (blocks * size) b)
  | Q4_K -> Nx_quant.q4_k (u8 (blocks * size) b)
  | Mxfp4_blocks -> Nx_quant.mxfp4_blocks (u8 (blocks * size) b)
  | Mxfp4 ->
      let scales = Array.init count (fun j -> 118 + (j mod 19)) in
      Nx_quant.mxfp4 ~scales:(u8 blocks scales) (u8 (blocks * size) b)

(* [γ(n, v)] for a unit roundoff [v]. *)
let gamma n v =
  let n = float_of_int n in
  n *. v /. (1. -. (n *. v))

let reference (type b) (x : (float, b, Nx.host) Nx.t) w =
  let dt = Nx.dtype x in
  let k = (Nx.shape x).(Nx.ndim x - 1) in
  let x64 = Nx.cast Nx.float64 x
  and v64 = Nx.matrix_transpose (Nx_quant.dequant Nx.float64 w) in
  let exact = Nx.matmul x64 v64 in
  let mass = Nx.matmul (Nx.abs x64) (Nx.abs v64) in
  let u = match dt with Nx.Dtype.Float64 -> 0x1p-53 | _ -> 0x1p-24 in
  (* The result's bound, and the reference's own: Nx.contract's bound for
     its float64 matmul. *)
  let sum =
    Nx.mul_s mass (gamma (k + 1) (2. *. u) +. gamma (k + 1) 0x1p-52)
  in
  let side =
    match dt with
    | Nx.Dtype.Float32 | Nx.Dtype.Float64 -> sum
    | _ ->
        let format = Nx_array.Dtype.float_format dt in
        Nx.add sum
          (Nx.add_s
             (Nx.mul_s (Nx.add (Nx.abs exact) sum) (format.epsilon /. 2.))
             (format.min_normal *. format.epsilon))
  in
  (exact, side)
