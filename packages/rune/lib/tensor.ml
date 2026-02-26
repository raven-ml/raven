(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module F = Nx_core.Make_frontend (Nx_rune)
include F

(* ───── Default Context Wrappers ───── *)

let context = Lazy.from_fun Nx_rune.create_context
let create dtype shape arr = create (Lazy.force context) dtype shape arr
let init dtype shape f = init (Lazy.force context) dtype shape f
let empty dtype shape = empty (Lazy.force context) dtype shape
let full dtype shape value = full (Lazy.force context) dtype shape value
let ones dtype shape = ones (Lazy.force context) dtype shape
let zeros dtype shape = zeros (Lazy.force context) dtype shape
let scalar dtype v = scalar (Lazy.force context) dtype v
let eye ?m ?k dtype n = eye (Lazy.force context) ?m ?k dtype n
let identity dtype n = identity (Lazy.force context) dtype n

let arange dtype start stop step =
  arange (Lazy.force context) dtype start stop step

let arange_f dtype start stop step =
  arange_f (Lazy.force context) dtype start stop step

let linspace dtype ?endpoint start stop num =
  linspace (Lazy.force context) dtype ?endpoint start stop num

let logspace dtype ?endpoint ?base start stop num =
  logspace (Lazy.force context) dtype ?endpoint ?base start stop num

let geomspace dtype ?endpoint start stop num =
  geomspace (Lazy.force context) dtype ?endpoint start stop num

let of_bigarray ba = of_bigarray (Lazy.force context) ba
let of_buffer ba = of_buffer (Lazy.force context) ba
let to_bigarray = to_bigarray
let to_buffer = to_buffer
let rand dtype shape = rand (Lazy.force context) dtype shape
let randn dtype shape = randn (Lazy.force context) dtype shape

let randint dtype ?high shape low =
  randint (Lazy.force context) dtype ?high shape low

let bernoulli ~p shape = F.bernoulli (Lazy.force context) ~p shape
let permutation n = F.permutation (Lazy.force context) n
let shuffle x = F.shuffle (Lazy.force context) x

let categorical ?axis ?shape logits =
  F.categorical (Lazy.force context) ?axis ?shape logits

let truncated_normal dtype ~lower ~upper shape =
  F.truncated_normal (Lazy.force context) dtype ~lower ~upper shape

module Rng = Nx_core.Rng

(* ───── FFT ───── *)

let fftfreq ?d n = fftfreq (Lazy.force context) ?d n
let rfftfreq ?d n = rfftfreq (Lazy.force context) ?d n
