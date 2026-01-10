(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module F = Nx_core.Make_frontend (Nx_c)
include F

let context = Lazy.from_fun Nx_c.create_context

module Rng = struct
  include F.Rng

  let uniform ~key dtype shape =
    F.Rng.uniform (Lazy.force context) ~key dtype shape

  let normal ~key dtype shape =
    F.Rng.normal (Lazy.force context) ~key dtype shape

  let randint dtype ~key ?high shape low =
    F.Rng.randint (Lazy.force context) dtype ~key ?high shape low

  let bernoulli ~key ~p shape =
    F.Rng.bernoulli (Lazy.force context) ~key ~p shape

  let permutation ~key n = F.Rng.permutation (Lazy.force context) ~key n
  let shuffle ~key x = F.Rng.shuffle (Lazy.force context) ~key x

  let categorical ~key ?axis ?shape logits =
    F.Rng.categorical (Lazy.force context) ~key ?axis ?shape logits

  let truncated_normal ~key dtype ~(lower : float) ~(upper : float) shape =
    F.Rng.truncated_normal (Lazy.force context) ~key dtype ~lower ~upper shape
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
let of_buffer ba = F.of_buffer (Lazy.force context) ba
let to_bigarray = F.to_bigarray
let to_buffer = F.to_buffer
let rand dtype ~key shape = F.rand (Lazy.force context) dtype ~key shape
let randn dtype ~key shape = F.randn (Lazy.force context) dtype ~key shape

let randint dtype ~key ?high shape low =
  F.randint (Lazy.force context) dtype ~key ?high shape low

let dropout ~key ~rate x = F.dropout ~key ~rate x

(* ───── FFT ───── *)

let fftfreq ?d n = F.fftfreq (Lazy.force context) ?d n
let rfftfreq ?d n = F.rfftfreq (Lazy.force context) ?d n
