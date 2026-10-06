(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let float32_max = Int32.float_of_bits 0x7f7fffffl
let bfloat16_max = Int32.float_of_bits 0x7f7f0000l

(* [limits dt] is [(tiny, huge, eps)]. *)
let limits (type b) (dt : (float, b) Nx.dtype) =
  match dt with
  | Nx_dtype.Float64 -> (Float.min_float, Float.max_float, Float.epsilon)
  | Nx_dtype.Float32 ->
      (Float.ldexp 1. (-126), float32_max, Float.ldexp 1. (-23))
  | Nx_dtype.Float16 -> (Float.ldexp 1. (-14), 65504., Float.ldexp 1. (-10))
  | Nx_dtype.BFloat16 ->
      (Float.ldexp 1. (-126), bfloat16_max, Float.ldexp 1. (-7))
  | Nx_dtype.Float8_e4m3 -> (Float.ldexp 1. (-6), 448., Float.ldexp 1. (-3))
  | Nx_dtype.Float8_e5m2 -> (Float.ldexp 1. (-14), 57344., Float.ldexp 1. (-2))

let tiny dt =
  let t, _, _ = limits dt in
  t

let huge dt =
  let _, h, _ = limits dt in
  h

let eps dt =
  let _, _, e = limits dt in
  e
