(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop
module U = Uop
module D = Dtype
module T = Tensor

let cast t dt =
  if D.equal (T.dtype t) dt then t else T.of_uop (U.cast ~src:(T.uop t) ~dtype:dt)

let bitcast t dt =
  if D.equal (T.dtype t) dt then t
  else if D.itemsize (T.dtype t) <> D.itemsize dt then
    invalid_arg "Dtype_ops.bitcast: element sizes differ"
  else T.of_uop (U.bitcast ~src:(T.uop t) ~dtype:dt)

let is_floating_point t = D.is_float (T.dtype t)
let element_size t = D.itemsize (T.dtype t)
let float t = cast t D.float32
let half t = cast t D.float16
let int t = cast t D.int32
let bool t = cast t D.bool
let double t = cast t D.float64
let long t = cast t D.int64
