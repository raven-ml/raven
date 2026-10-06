(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let eps (type b) (dtype : (float, b) Nx.dtype) =
  match dtype with
  | Nx.Float64 -> epsilon_float
  | Nx.Float32 -> Float.ldexp 1. (-23)
  | Nx.Float16 -> Float.ldexp 1. (-10)
  | Nx.BFloat16 -> Float.ldexp 1. (-7)
  | Nx.Float8_e4m3 -> Float.ldexp 1. (-3)
  | Nx.Float8_e5m2 -> Float.ldexp 1. (-2)

let constant dtype a = Nx.create dtype [| Array.length a |] a

type map = { f : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t }

let on_float (type a c) fn m (x : (a, c) Nx.t) : (a, c) Nx.t =
  match Nx.dtype x with
  | Nx.Float64 -> m.f x
  | Nx.Float32 -> m.f x
  | Nx.Float16 -> m.f x
  | Nx.BFloat16 -> m.f x
  | Nx.Float8_e4m3 -> m.f x
  | Nx.Float8_e5m2 -> m.f x
  | dtype ->
      invalid_arg
        (Printf.sprintf "%s: a %s leaf; the leaves must be float tensors" fn
           (Nx_dtype.to_string dtype))

let shape s =
  "[" ^ String.concat "," (Array.to_list (Array.map string_of_int s)) ^ "]"

let check_increasing fn what x =
  let n = Nx.dim 0 x in
  if n > 1 then
    let lo = Nx.shrink [| (0, n - 1) |] x and hi = Nx.shrink [| (1, n) |] x in
    Nx.check
      Nx.Ptree.(pair tensor tensor)
      (Nx.less lo hi) (lo, hi)
      (fun i (lo, hi) ->
        Invalid_argument
          (Printf.sprintf
             "%s: %s are not strictly increasing at [%d]: %g after %g" fn what
             (i.(0) + 1)
             (Nx.item [] hi) (Nx.item [] lo)))
