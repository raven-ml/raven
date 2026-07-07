(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_frontend

type t = { weight : Tensor.t }

let create vocab_size embed_size =
  { weight = Creation.zeros [ vocab_size; embed_size ] }

let apply e idx =
  if not (Tolk_uop.Dtype.is_int (Tensor.dtype idx)) then
    invalid_arg "Embedding.apply: index must be an integer tensor";
  let vocab_size = List.hd (Tensor.shape e.weight) in
  let arange = Op.arange vocab_size in
  let one_hot =
    Movement.unsqueeze (Elementwise.eq arange (Movement.unsqueeze idx (-1))) (-1)
  in
  Reduce.sum ~axis:[ -2 ]
    ~dtype:(Tensor.val_dtype e.weight)
    (Elementwise.where one_hot e.weight (Tensor.i 0))
