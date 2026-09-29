(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module F = Nx_core.Make_frontend (Nx_effect)
include F

exception Linalg_error = Nx_core.Backend_intf.Linalg_error

let context = Nx_effect.Placement.host

module Device = Nx_effect.Device

module Placement = Nx_effect.Placement

let place = Nx_effect.place
let placement = Nx_effect.placement

module Ptree = Ptree

type packed = Nx_effect.packed = P : ('a, 'b) t -> packed

let unpack (type a b) (dt : (a, b) dtype) (P x) : (a, b) t =
  match Nx_dtype.equal_witness (dtype x) dt with
  | Some Type.Equal -> x
  | None ->
      invalid_arg
        (Printf.sprintf "unpack: expected dtype %s, got %s"
           (Nx_dtype.to_string dt)
           (Nx_dtype.to_string (dtype x)))

module Rng = struct
  include F.Rng

  let key seed = F.Rng.key context seed
  let next_key () = F.Rng.next_key context

  let fold_in_axis k =
    F.Rng.fold_in_tensor k (Nx_effect.axis_index (Nx_effect.context k))

  type t = key

  let ptree = Ptree.iso of_tensor Fun.id Ptree.tensor
end

(* Re-export extended type aliases *)
type bfloat16_t = (float, Nx_dtype.bfloat16_elt) t
type bool_t = (bool, Nx_dtype.bool_elt) t
type int4_t = (int, Nx_dtype.int4_elt) t
type uint4_t = (int, Nx_dtype.uint4_elt) t
type float8_e4m3_t = (float, Nx_dtype.float8_e4m3_elt) t
type float8_e5m2_t = (float, Nx_dtype.float8_e5m2_elt) t

(* Re-export extended dtype value constructors *)
let bfloat16 = Nx_dtype.bfloat16
let bool = Nx_dtype.bool
let int4 = Nx_dtype.int4
let uint4 = Nx_dtype.uint4
let float8_e4m3 = Nx_dtype.float8_e4m3
let float8_e5m2 = Nx_dtype.float8_e5m2

(* ───── Overriding Functions With Default Context ───── *)

let create dtype shape arr = F.create context dtype shape arr
let init dtype shape f = F.init context dtype shape f
let full dtype shape value = F.full context dtype shape value
let ones dtype shape = F.ones context dtype shape
let zeros dtype shape = F.zeros context dtype shape
let scalar dtype v = F.scalar context dtype v
let eye ?m ?k dtype n = F.eye context ?m ?k dtype n

(* A value made like a placed one lives where that one does, split as it is. *)
let full_like x v =
  match x with
  | Nx_effect.Placed r -> Nx_effect.full r.r_placement (dtype x) (shape x) v
  | Nx_effect.Host _ | Nx_effect.Traced _ -> F.full_like x v

let zeros_like x = full_like x (Nx_dtype.zero (dtype x))
let ones_like x = full_like x (Nx_dtype.one (dtype x))
let fill v x = full_like x v

let arange dtype start stop step =
  F.arange context dtype start stop step

let arange_f dtype start stop step =
  F.arange_f context dtype start stop step

let linspace dtype ?endpoint start stop num =
  F.linspace context dtype ?endpoint start stop num

let logspace dtype ?endpoint ?base start stop num =
  F.logspace context dtype ?endpoint ?base start stop num

let geomspace dtype ?endpoint start stop num =
  F.geomspace context dtype ?endpoint start stop num

let of_bigarray ba = F.of_bigarray context ba
let to_bigarray = F.to_bigarray
let rand dtype shape = F.rand context dtype shape
let randn dtype shape = F.randn context dtype shape
let randint ?low ~high shape = F.randint context ?low ~high shape
let bernoulli p = F.bernoulli context p
let permutation n = F.permutation context n
let shuffle x = F.shuffle context x
let categorical ?axis logits = F.categorical context ?axis logits

let truncated_normal lower upper =
  F.truncated_normal context lower upper

(* ───── FFT ───── *)

let fftfreq dtype ?d n = F.fftfreq context dtype ?d n
let rfftfreq dtype ?d n = F.rfftfreq context dtype ?d n
let hann dt n = F.hann context dt n
