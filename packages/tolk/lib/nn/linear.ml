(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_frontend

type t = { weight : Tensor.t; bias : Tensor.t option }

let create ?(bias = true) in_features out_features =
  let bound = 1. /. Float.sqrt (float_of_int in_features) in
  (* The weight draws from the random stream before the bias. *)
  let weight =
    Rand.uniform ~low:(-.bound) ~high:bound [ out_features; in_features ]
  in
  let bias =
    if bias then Some (Rand.uniform ~low:(-.bound) ~high:bound [ out_features ])
    else None
  in
  { weight; bias }

let apply l x =
  let x = Op.dot x (Movement.transpose l.weight) in
  match l.bias with None -> x | Some b -> Elementwise.add x b
