(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop
module U = Uop
module D = Dtype
module T = Tensor

let dtype_of_fill = function
  | T.Sint _ -> D.Val.default_int
  | T.Sfloat _ -> D.Val.default_float
  | T.Sbool _ -> D.Val.bool

let broadcast_scalar dt fill shape =
  let v = T.of_uop (U.const (T.scalar_const dt fill)) in
  Movement.expand (Movement.reshape v (List.map (fun _ -> 1) shape)) shape

let full ?dtype shape fill =
  let dt = match dtype with Some d -> d | None -> dtype_of_fill fill in
  broadcast_scalar dt fill shape

let zeros ?dtype shape = full ?dtype shape (T.Sfloat 0.0)
let ones ?dtype shape = full ?dtype shape (T.Sfloat 1.0)
let const_like t fill = broadcast_scalar (T.val_dtype t) fill (T.shape t)

let full_like ?dtype t fill =
  let dt = match dtype with Some d -> d | None -> T.val_dtype t in
  full ~dtype:dt (T.shape t) fill

let zeros_like ?dtype t = full_like ?dtype t (T.Sint 0)
let ones_like ?dtype t = full_like ?dtype t (T.Sint 1)
