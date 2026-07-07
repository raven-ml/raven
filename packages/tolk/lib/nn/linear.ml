(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_frontend

type t = { weight : Tensor.t; bias : Tensor.t option }

let create ?(bias = true) in_features out_features =
  {
    weight = Creation.zeros [ out_features; in_features ];
    bias = (if bias then Some (Creation.zeros [ out_features ]) else None);
  }

let apply l x =
  let x = Op.dot x (Movement.transpose l.weight) in
  match l.bias with None -> x | Some b -> Elementwise.add x b
