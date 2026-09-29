(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

include Frontend

exception Linalg_error = Nx_array.Backend_intf.Linalg_error

let context = Nx_effect.Placement.host

module Device = Nx_effect.Device
module Backend = Nx_effect.Backend

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
  include Frontend.Rng

  let key seed = Frontend.Rng.key context seed
  let next_key () = Frontend.Rng.next_key context

  let fold_in_axis k =
    Frontend.Rng.fold_in_tensor k (Nx_effect.axis_index (Nx_effect.context k))

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

let create dtype shape arr = Frontend.create context dtype shape arr
let init dtype shape f = Frontend.init context dtype shape f
let full dtype shape value = Frontend.full context dtype shape value
let ones dtype shape = Frontend.ones context dtype shape
let zeros dtype shape = Frontend.zeros context dtype shape
let scalar dtype v = Frontend.scalar context dtype v
let eye ?m ?k dtype n = Frontend.eye context ?m ?k dtype n

(* A value made like a placed one lives where that one does, split as it is. *)
let full_like x v =
  match x with
  | Nx_effect.Placed r -> Nx_effect.full r.r_placement (dtype x) (shape x) v
  | Nx_effect.Host _ | Nx_effect.Traced _ -> Frontend.full_like x v

let zeros_like x = full_like x (Nx_dtype.zero (dtype x))
let ones_like x = full_like x (Nx_dtype.one (dtype x))
let fill v x = full_like x v

let arange dtype start stop step =
  Frontend.arange context dtype start stop step

let arange_f dtype start stop step =
  Frontend.arange_f context dtype start stop step

let linspace dtype ?endpoint start stop num =
  Frontend.linspace context dtype ?endpoint start stop num

let logspace dtype ?endpoint ?base start stop num =
  Frontend.logspace context dtype ?endpoint ?base start stop num

let geomspace dtype ?endpoint start stop num =
  Frontend.geomspace context dtype ?endpoint start stop num

let of_bigarray ba = Frontend.of_bigarray context ba
let to_bigarray = Frontend.to_bigarray
let rand dtype shape = Frontend.rand context dtype shape
let randn dtype shape = Frontend.randn context dtype shape
let randint ?low ~high shape = Frontend.randint context ?low ~high shape
let bernoulli p = Frontend.bernoulli context p
let permutation n = Frontend.permutation context n
let shuffle x = Frontend.shuffle context x
let categorical ?axis logits = Frontend.categorical context ?axis logits

let truncated_normal lower upper =
  Frontend.truncated_normal context lower upper

(* ───── FFT ───── *)

let fftfreq dtype ?d n = Frontend.fftfreq context dtype ?d n
let rfftfreq dtype ?d n = Frontend.rfftfreq context dtype ?d n
let hann dt n = Frontend.hann context dt n
