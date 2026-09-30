(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

include Frontend

(* A caller owns the array it is given. *)
let shape x = Array.copy (shape x)

exception Linalg_error = Nx_backend.Linalg_error

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

(* A value made like a placed one lives where that one does, split as it is. *)
let full_like x v =
  match x with
  | Nx_effect.Placed { r_placement = p; _ } when not (Nx_effect.on_disk p) ->
      Nx_effect.full p (dtype x) (shape x) v
  | Nx_effect.Host _ | Nx_effect.Placed _ | Nx_effect.Traced _ ->
      Frontend.full_like x v

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

(* For transformations and file formats *)

module Op = struct
  include Nx_effect.Op

  type conversion = Nx_effect.conversion = Cast | Bitcast

  let eval = Nx_effect.eval
  let placement = Nx_effect.result_placement

  type interpreter = Nx_effect.interpreter = { run : 'r. 'r t -> 'r }

  let intercept = Nx_effect.intercept
  let intercepted = Nx_effect.intercepted
end

module Repr = struct
  type ('a, 'b) node = ('a, 'b) Nx_effect.node = ..

  (* Whether [v] reaches only elements [0] to [n - 1]. *)
  let within v n =
    Nx_array.View.numel v = 0
    ||
    let lo = ref (Nx_array.View.offset v)
    and hi = ref (Nx_array.View.offset v) in
    Array.iteri
      (fun i d ->
        let s = (Nx_array.View.strides v).(i) * (d - 1) in
        if s < 0 then lo := !lo + s else hi := !hi + s)
      (Nx_array.View.shape v);
    !lo >= 0 && !hi < n

  module Storage = struct
    type t = Nx_effect.cell

    let v p buffers =
      let ds = Placement.devices p in
      if List.compare_lengths ds buffers <> 0 then
        invalid_arg "Nx.Repr.Storage.v: one buffer per device of the placement";
      let length = Nx_device.Buffer.length (List.hd buffers) in
      List.iter2
        (fun d b ->
          if Nx_device.Buffer.length b <> length then
            invalid_arg "Nx.Repr.Storage.v: buffers of different lengths";
          if Nx_device.Buffer.device b != Nx_effect.runtime_of d then
            invalid_arg
              (Printf.sprintf "Nx.Repr.Storage.v: a buffer for %s is on %s"
                 (Device.name d)
                 (Nx_device.name (Nx_device.Buffer.device b))))
        ds buffers;
      Nx_effect.cell ~placement:p ~length (Nx_effect.Runtime buffers)

    let buffers (s : t) =
      match Nx_effect.Cell.state s with
      | Nx_effect.Live (Nx_effect.Runtime bs) -> bs
      | Live _ ->
          invalid_arg "Nx.Repr.Storage.buffers: storage in memory of its own"
      | Consumed k -> invalid_arg (Nx_effect.why_consumed k)

    let placement (s : t) = s.placement
    let borrow = Nx_effect.Cell.borrow
    let release = Nx_effect.Cell.release
    let upgrade = Nx_effect.Cell.upgrade
    let consume s ~path = Nx_effect.Cell.consume s { path }
    let finish = Nx_effect.Cell.finish
    let pin = Nx_effect.Cell.pin
    let unpin = Nx_effect.Cell.unpin

    let live s =
      match Nx_effect.Cell.state s with Live _ -> true | Consumed _ -> false

    let pins (s : t) = Atomic.get s.bound
  end

  module Placed = struct
    type ('a, 'b) t = ('a, 'b) Nx_effect.resident

    let v p dtype view (s : Storage.t) =
      if not (within view s.length) then
        invalid_arg "Nx.Repr.Placed.v: the view reaches outside the storage";
      Nx_effect.placed p dtype view s

    let id (x : ('a, 'b) t) = x.r_id
    let view (x : ('a, 'b) t) = x.r_view
    let storage (x : ('a, 'b) t) = x.r_cell
  end

  module Traced = struct
    type ('a, 'b) t = ('a, 'b) Nx_effect.traced

    let v ~context p dtype shape node =
      Nx_effect.traced context p dtype shape node

    let id (x : ('a, 'b) t) = x.t_id
    let node (x : ('a, 'b) t) = x.t_node
  end

  type ('a, 'b) t = ('a, 'b) Nx_effect.t =
    | Host : ('a, 'b) Nx_array.t -> ('a, 'b) t
    | Placed : ('a, 'b) Placed.t -> ('a, 'b) t
    | Traced : ('a, 'b) Traced.t -> ('a, 'b) t

  let v x = x

  let host (a : ('a, 'b) Nx_array.t) =
    Nx_effect.check_host "Nx.Repr.host" a.dtype a.buffer;
    if not (within a.view (Nx_device.Buffer.length a.buffer)) then
      invalid_arg "Nx.Repr.host: the view reaches outside the buffer";
    Nx_effect.Host a

  let context = Nx_effect.context
end
