(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module F = Nx_core.Make_frontend (Nx_effect)
include F

let context = Lazy.from_fun Nx_effect.create_context

module Rng = struct
  include Nx_core.Rng
end

(* Re-export extended type aliases *)
type bfloat16_t = (float, Nx_buffer.bfloat16_elt) t
type bool_t = (bool, Nx_buffer.bool_elt) t
type int4_t = (int, Nx_buffer.int4_signed_elt) t
type uint4_t = (int, Nx_buffer.int4_unsigned_elt) t
type float8_e4m3_t = (float, Nx_buffer.float8_e4m3_elt) t
type float8_e5m2_t = (float, Nx_buffer.float8_e5m2_elt) t

(* Re-export extended dtype value constructors *)
let bfloat16 = Nx_core.Dtype.bfloat16
let bool = Nx_core.Dtype.bool
let int4 = Nx_core.Dtype.int4
let uint4 = Nx_core.Dtype.uint4
let float8_e4m3 = Nx_core.Dtype.float8_e4m3
let float8_e5m2 = Nx_core.Dtype.float8_e5m2

(* ───── Overriding Functions With Default Context ───── *)

let create dtype shape arr = F.create (Lazy.force context) dtype shape arr
let init dtype shape f = F.init (Lazy.force context) dtype shape f
let empty dtype shape = F.empty (Lazy.force context) dtype shape
let full dtype shape value = F.full (Lazy.force context) dtype shape value
let ones dtype shape = F.ones (Lazy.force context) dtype shape
let zeros dtype shape = F.zeros (Lazy.force context) dtype shape
let scalar dtype v = F.scalar (Lazy.force context) dtype v
let eye ?m ?k dtype n = F.eye (Lazy.force context) ?m ?k dtype n
let identity dtype n = F.identity (Lazy.force context) dtype n

let arange dtype start stop step =
  F.arange (Lazy.force context) dtype start stop step

let arange_f dtype start stop step =
  F.arange_f (Lazy.force context) dtype start stop step

let linspace dtype ?endpoint start stop num =
  F.linspace (Lazy.force context) dtype ?endpoint start stop num

let logspace dtype ?endpoint ?base start stop num =
  F.logspace (Lazy.force context) dtype ?endpoint ?base start stop num

let geomspace dtype ?endpoint start stop num =
  F.geomspace (Lazy.force context) dtype ?endpoint start stop num

let of_bigarray ba = F.of_bigarray (Lazy.force context) ba
let of_buffer ba ~shape = F.of_buffer (Lazy.force context) ~shape ba
let to_bigarray = F.to_bigarray
let to_buffer = F.to_buffer
let rand dtype shape = F.rand (Lazy.force context) dtype shape
let randn dtype shape = F.randn (Lazy.force context) dtype shape

let randint dtype ?high shape low =
  F.randint (Lazy.force context) dtype ?high shape low

let bernoulli ~p shape = F.bernoulli (Lazy.force context) ~p shape
let permutation n = F.permutation (Lazy.force context) n
let shuffle x = F.shuffle (Lazy.force context) x

let categorical ?axis ?shape logits =
  F.categorical (Lazy.force context) ?axis ?shape logits

let truncated_normal dtype ~lower ~upper shape =
  F.truncated_normal (Lazy.force context) dtype ~lower ~upper shape

(* ───── FFT ───── *)

let fftfreq ?d n = F.fftfreq (Lazy.force context) ?d n
let rfftfreq ?d n = F.rfftfreq (Lazy.force context) ?d n
