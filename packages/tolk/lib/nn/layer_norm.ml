(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_frontend

type t = {
  weight : Tensor.t option;
  bias : Tensor.t option;
  eps : float;
  dim : int;
}

let create ?(eps = 1e-5) ?(elementwise_affine = true) dim =
  {
    weight = (if elementwise_affine then Some (Creation.ones [ dim ]) else None);
    bias = (if elementwise_affine then Some (Creation.zeros [ dim ]) else None);
    eps;
    dim;
  }

let apply ln x =
  let shape = Tensor.shape x in
  if List.nth shape (List.length shape - 1) <> ln.dim then
    invalid_arg
      (Printf.sprintf "Layer_norm.apply: last axis of input must be %d" ln.dim);
  let x = Op.layernorm ~eps:ln.eps x in
  match (ln.weight, ln.bias) with
  | Some w, Some b -> Elementwise.add (Elementwise.mul x w) b
  | _ -> x
