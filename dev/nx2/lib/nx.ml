(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('v, 's, 'd) t = ('v, 's, 'd) Value.t
type ('v, 's) dtype = ('v, 's) Nx_array.Dtype.t

let shape = Prim.shape
let dtype = Prim.dtype

module Dtype = Nx_array.Dtype

let float64 = Dtype.Float64
let float32 = Dtype.Float32
let float16 = Dtype.Float16
let bfloat16 = Dtype.Bfloat16
let float8_e4m3fn = Dtype.Float8_e4m3fn
let float8_e5m2 = Dtype.Float8_e5m2
let float4_e2m1fn = Dtype.Float4_e2m1fn
let int64 = Dtype.Int64
let uint64 = Dtype.Uint64
let int32 = Dtype.Int32
let uint32 = Dtype.Uint32
let int16 = Dtype.Int16
let uint16 = Dtype.Uint16
let int8 = Dtype.Int8
let uint8 = Dtype.Uint8
let int4 = Dtype.Int4
let uint4 = Dtype.Uint4
let complex128 = Dtype.Complex128
let complex64 = Dtype.Complex64
let bool = Dtype.Bool
let bit = Dtype.Bit

type host = Devices.host
type 'd devices = 'd Devices.t

let host = Devices.host
let rigs s = Array.init (Devices.count s) (Devices.rig s)

module Mesh = struct
  type 'd t = 'd Devices.mesh

  let v s axes = Devices.mesh_v ~by:"Nx.Mesh.v" s axes
end

module Placement = struct
  type 'd t = 'd Devices.placement

  let host = Devices.one Devices.host 0
  let on = Devices.on
  let split ~axis s = Devices.split ~by:"Nx.Placement.split" ~axis s
  let mesh m cuts = Devices.mesh ~by:"Nx.Placement.mesh" m cuts
  let devices = Devices.set
  let equal = Devices.equal
  let pp = Devices.pp_placement
end

module type Devices = sig
  type d

  val v : d devices
  val on : d Placement.t
  val split : axis:int -> d Placement.t
end

let devices ?kernels ds : (module Devices) =
  let v = Devices.mint ~by:"Nx.devices" ?kernels ds in
  (module struct
    type d

    let v = v
    let on = Devices.on v
    let split ~axis = Devices.split ~by:"Nx.Placement.split" ~axis v
  end)

let place p x = Exec.run ~by:"Nx.place" (Value.Place (p, x))
let placement = Prim.placement

module Repr = struct
  let of_array s a = Repr.of_array ~by:"Nx.Repr.of_array" s a
  let array = Repr.array
  let of_shards p arrays = Repr.of_shards ~by:"Nx.Repr.of_shards" p arrays
  let shards = Repr.shards
end

(* Constants and arithmetic *)

module D = Nx_array.Dtype
module P = Nx_kernel.Prog

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

let bits ~by dt v =
  match P.bits dt v with
  | b -> b
  | exception Invalid_argument e -> invalid_argf "%s: %s" by e

let fill ~by dt shape v =
  let prog = Prim.program (Const (D.Any dt, bits ~by dt v)) [||] in
  let x, () =
    Exec.run ~by (Value.Map { shape; prog; outs = Value.[ dt ]; loads = [||] })
  in
  x

let zeros dt shape = fill ~by:"Nx.zeros" dt shape (D.zero dt)
let scalar dt v = fill ~by:"Nx.scalar" dt [||] v

(* Zeros of [x]'s dtype and shape filled where [x] lies: [x]'s elements are
   never read. *)
let zeros_like x =
  let by = "Nx.zeros_like" in
  let dt = dtype x in
  let z = fill ~by dt (shape x) (D.zero dt) in
  if Prim.is_constant x then z
  else Exec.run ~by (Value.Place (Prim.placement x, z))

(* The shape [s] and [s'] broadcast to: aligned at their last axes, each extent
   equal or [1]. *)
let broadcast_shape ~by s s' =
  let r = max (Array.length s) (Array.length s') in
  let at s i =
    let k = i - (r - Array.length s) in
    if k < 0 then 1 else s.(k)
  in
  Array.init r (fun i ->
      let a = at s i and b = at s' i in
      if a = b || b = 1 then a
      else if a = 1 then b
      else
        invalid_argf "%s: shapes %a and %a do not broadcast" by pp_shape s
          pp_shape s')

let broadcast ~by s x =
  if shape x = s then x else Exec.run ~by (Value.Move (Broadcast s, x))

let rec dims_equal a b i =
  i = Prim.rank a || (Prim.dim a i = Prim.dim b i && dims_equal a b (i + 1))

(* Whether [a] and [b] have one shape, allocating nothing. *)
let same_shape a b = Prim.rank a = Prim.rank b && dims_equal a b 0

let binary ~by k a b =
  if same_shape a b then Exec.apply2 ~by (Binary k) (dtype a) a b
  else
    let s = broadcast_shape ~by (shape a) (shape b) in
    Exec.apply2 ~by (Binary k) (dtype a) (broadcast ~by s a) (broadcast ~by s b)

let add a b = binary ~by:"Nx.add" Add a b
let mul a b = binary ~by:"Nx.mul" Mul a b

let less a b =
  let by = "Nx.less" in
  if same_shape a b then Exec.apply2 ~by (Compare Less) D.Bool a b
  else
    let s = broadcast_shape ~by (shape a) (shape b) in
    Exec.apply2 ~by (Compare Less) D.Bool (broadcast ~by s a)
      (broadcast ~by s b)

let where c x y =
  let by = "Nx.where" in
  if same_shape c x && same_shape x y then Exec.apply3 ~by Where c x y
  else
    let s =
      broadcast_shape ~by (broadcast_shape ~by (shape c) (shape x)) (shape y)
    in
    Exec.apply3 ~by Where (broadcast ~by s c) (broadcast ~by s x)
      (broadcast ~by s y)

let cast (type v s w r d) (dt : (w, r) D.t) (x : (v, s, d) t) : (w, r, d) t =
  match D.equal_witness (dtype x) dt with
  | Some Type.Equal -> x
  | None -> Exec.apply1 ~by:"Nx.cast" Cast dt x

let reshape s x =
  Exec.run ~by:"Nx.reshape" (Value.Move (Reshape (Array.copy s), x))

let copy x = Exec.run ~by:"Nx.copy" (Value.Copy x)
