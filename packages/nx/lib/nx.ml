(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

include Frontend

(* A caller owns the array it is given. *)
let shape x = Array.copy (shape x)

exception Linalg_error = Nx_backend.Linalg_error

let context = Placement.host

module Device = Device
module Placement = Placement

let place = Entry.place
let placement = Value.placement
let of_buffer = Value.of_buffer
let to_buffer = Entry.to_buffer
let shards = Value.shards

let of_shards p dtype view buffers =
  Value.of_shards "Nx.of_shards" p dtype view buffers

module Ptree = Ptree

type packed = Value.packed = P : ('a, 'b) t -> packed

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

(* A value made like a placed or traced one lives where that one does, split as
   it is. *)
let full_like x v =
  match x with
  | (Value.Placed _ | Value.Traced _)
    when not (Placement.on_disk (Value.placement x)) ->
      Entry.full (Value.placement x) (dtype x) (shape x) v
  | Value.Host _ | Value.Placed _ | Value.Traced _ -> Frontend.full_like x v

let zeros_like x = full_like x (Nx_dtype.zero (dtype x))
let ones_like x = full_like x (Nx_dtype.one (dtype x))
let fill v x = full_like x v
let arange dtype start stop step = Frontend.arange context dtype start stop step

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
let truncated_normal lower upper = Frontend.truncated_normal context lower upper

(* ───── FFT ───── *)

let fftfreq dtype ?d n = Frontend.fftfreq context dtype ?d n
let rfftfreq dtype ?d n = Frontend.rfftfreq context dtype ?d n
let hann dt n = Frontend.hann context dt n

(* ───── Special functions ───── *)

let erfc = Special.erfc
let ndtr = Special.ndtr
let log_ndtr = Special.log_ndtr
let ndtri = Special.ndtri
let lgamma = Special.lgamma
let digamma = Special.digamma
let lbeta = Special.lbeta

(* For transformations and file formats *)

module Op = struct
  include Op

  let eval = Intercept.eval
  let placement = Route.placement

  type interpreter = Intercept.interpreter = {
    run : 'r. 'r t -> 'r;
    claims : 'r. 'r t -> bool;
  }

  let intercept = Intercept.intercept
  let intercepted = Intercept.intercepted
end

module Repr = struct
  type ('a, 'b) node = ('a, 'b) Value.node = ..

  module Storage = struct
    type t = Value.cell

    let v p buffers = Value.shard_storage "Nx.Repr.Storage.v" p buffers

    let buffers (s : t) =
      match Value.Cell.state s with
      | Live bs -> bs
      | Consumed k -> invalid_arg (Value.why_consumed k)

    let placement (s : t) = s.placement
    let borrow = Value.Cell.borrow
    let release = Value.Cell.release
    let upgrade = Value.Cell.upgrade
    let consume s ~path = Value.Cell.consume s { path }
    let finish = Value.Cell.finish
    let pin = Value.Cell.pin
    let unpin = Value.Cell.unpin

    let live s =
      match Value.Cell.state s with Live _ -> true | Consumed _ -> false

    let pins (s : t) = Atomic.get s.bound
  end

  module Placed = struct
    type ('a, 'b) t = ('a, 'b) Value.resident

    let v p dtype view (s : Storage.t) =
      Value.placed_value "Nx.Repr.Placed.v" p dtype view s

    let id (x : ('a, 'b) t) = x.r_id
    let view (x : ('a, 'b) t) = x.r_view
    let storage (x : ('a, 'b) t) = x.r_cell
  end

  module Traced = struct
    type ('a, 'b) t = ('a, 'b) Value.traced

    let v ~context ?view p dtype shape node =
      Value.traced ?view context p dtype shape node

    let id (x : ('a, 'b) t) = x.t_id
    let node (x : ('a, 'b) t) = x.t_node
  end

  type ('a, 'b) t = ('a, 'b) Value.t =
    | Host : ('a, 'b) Nx_array.t -> ('a, 'b) t
    | Placed : ('a, 'b) Placed.t -> ('a, 'b) t
    | Traced : ('a, 'b) Traced.t -> ('a, 'b) t

  let v x = x

  let host (a : ('a, 'b) Nx_array.t) =
    Value.host_value "Nx.Repr.host" a.dtype a.view a.buffer

  let context = Value.context
  let view = Value.view
end
