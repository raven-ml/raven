(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk_next
module View = Nx_array.View
module Repr = Nx.Repr
module Placement = Nx.Placement
module Device = Nx.Device

exception Jit_error of string

let jit_error fmt = Printf.ksprintf (fun s -> raise (Jit_error s)) fmt

type (_, _) Repr.node += Uop : Ops.t -> ('a, 'b) Repr.node

(* Dtypes *)

let dtype : type a b. (a, b) Nx_dtype.t -> Dtype.t option = function
  | Float16 -> Some Float16
  | Float32 -> Some Float32
  | Float64 -> Some Float64
  | BFloat16 -> Some Bfloat16
  | Float8_e4m3 -> Some Fp8e4m3
  | Float8_e5m2 -> Some Fp8e5m2
  | Int8 -> Some Int8
  | UInt8 -> Some Uint8
  | Int16 -> Some Int16
  | UInt16 -> Some Uint16
  | Int32 -> Some Int32
  | UInt32 -> Some Uint32
  | Int64 -> Some Int64
  | UInt64 -> Some Uint64
  | Bool -> Some Bool
  | Int4 | UInt4 | Complex64 | Complex128 -> None

(* [const dt v] is the element [v] of [dt] as a constant. *)
let const : type a b. (a, b) Nx_dtype.t -> a -> Dtype.const =
 fun dt v ->
  match dt with
  | Float16 -> `Float v
  | Float32 -> `Float v
  | Float64 -> `Float v
  | BFloat16 -> `Float v
  | Float8_e4m3 -> `Float v
  | Float8_e5m2 -> `Float v
  | Int4 -> `Int (Z.of_int v)
  | UInt4 -> `Int (Z.of_int v)
  | Int8 -> `Int (Z.of_int v)
  | UInt8 -> `Int (Z.of_int v)
  | Int16 -> `Int (Z.of_int v)
  | UInt16 -> `Int (Z.of_int v)
  | Int32 -> `Int (Z.of_int32 v)
  | UInt32 -> `Int (Z.of_int32_unsigned v)
  | Int64 -> `Int (Z.of_int64 v)
  | UInt64 -> `Int (Z.of_int64_unsigned v)
  | Bool -> `Bool v
  | Complex64 | Complex128 -> invalid_arg "a complex constant"

(* Views over storage

   A view reaches the elements of its storage by strides from an offset. Over a
   flat node of those elements the same view is movements: a broadcast axis is
   an expand, a negative stride a flip, and the stepped axes, by decreasing
   stride, are rows of a reshape cut by a shrink when each stride nests in the
   one before it. Overlapping windows are built as [pool] builds them: the
   storage repeated end to end and read back in rows one element longer, so that
   each row starts one element further on. *)

let ceil_div a b = (a + b - 1) / b
let ints l = List.map (fun n -> Ops.Int n) l

let shrink_axis u axis lo hi =
  Ops.shrink u
    (List.mapi
       (fun i _ -> if i = axis then Some (Ops.Int lo, Ops.Int hi) else None)
       (Ops.shape u))

(* The stepped axes of a view, by decreasing magnitude of stride. *)
let stepped shape strides =
  List.init (Array.length shape) Fun.id
  |> List.filter (fun d -> shape.(d) > 1 && strides.(d) <> 0)
  |> List.stable_sort (fun a b ->
      Int.compare (Int.abs strides.(b)) (Int.abs strides.(a)))

let rec nests shape strides = function
  | a :: (b :: _ as rest) ->
      let sa = Int.abs strides.(a) and sb = Int.abs strides.(b) in
      sa mod sb = 0 && sa >= sb * shape.(b) && nests shape strides rest
  | [ _ ] | [] -> true

(* [window flat start n] is the [n] elements of the flat node [flat] from
   [start], padded where [flat] ends first: those elements are never read. *)
let window flat start n =
  let size =
    match Ops.shape flat with [ Ops.Int size ] -> size | _ -> assert false
  in
  let flat =
    if start + n <= size then flat
    else Ops.pad flat [ Some (Ops.Int 0, Ops.Int (start + n - size)) ]
  in
  shrink_axis flat 0 start (start + n)

(* The stepped [axes] as rows of reshapes cut by shrinks. *)
let nested flat start shape strides axes =
  let magnitude a = Int.abs strides.(a) in
  let rec split = function
    | a :: (b :: _ as rest) -> (magnitude a / magnitude b) :: split rest
    | [ a ] -> [ magnitude a ]
    | [] -> []
  in
  let a0 = List.hd axes in
  let rows = window flat start (shape.(a0) * magnitude a0) in
  let t = Ops.reshape rows (ints (shape.(a0) :: split axes)) in
  let t =
    Ops.shrink t
      (List.map (fun a -> Some (Ops.Int 0, Ops.Int shape.(a))) axes
      @ [ Some (Ops.Int 0, Ops.Int 1) ])
  in
  Ops.reshape t (ints (List.map (fun a -> shape.(a)) axes))

(* The stepped [axes] from the innermost out. Before an axis of size [n] and
   stride [s], [u] holds, at [p] and inner indices [i], the element [p] plus the
   inner offset of [i]. Rows of [u] one element longer than [u]'s first axis
   start one element further on, and each [s]th element of a row is the new
   axis. *)
let overlapping flat start shape strides axes =
  let extent =
    List.fold_left
      (fun e a -> e + ((shape.(a) - 1) * Int.abs strides.(a)))
      1 axes
  in
  let step (u, m) a =
    let n = shape.(a) and s = Int.abs strides.(a) in
    let inner = List.tl (Ops.shape u) in
    let m' = m - ((n - 1) * s) and w = ((n - 1) * s) + 1 in
    let k = ceil_div (m' * (m + 1)) m in
    let u = Ops.reshape u (Ops.Int 1 :: Ops.shape u) in
    let u = Ops.expand u (Ops.Int k :: List.tl (Ops.shape u)) in
    let u = Ops.reshape u (Ops.Int (k * m) :: inner) in
    let u = shrink_axis u 0 0 (m' * (m + 1)) in
    let u = Ops.reshape u (Ops.Int m' :: Ops.Int (m + 1) :: inner) in
    let u = shrink_axis u 1 0 w in
    let u =
      Ops.pad u
        (None
        :: Some (Ops.Int 0, Ops.Int ((n * s) - w))
        :: List.map (fun _ -> None) inner)
    in
    let u = Ops.reshape u (Ops.Int m' :: Ops.Int n :: Ops.Int s :: inner) in
    let u = shrink_axis u 2 0 1 in
    (Ops.reshape u (Ops.Int m' :: Ops.Int n :: inner), m')
  in
  let u, _ =
    List.fold_left step (window flat start extent, extent) (List.rev axes)
  in
  Ops.reshape u (List.tl (Ops.shape u))

(* Whether [strides] lay [shape] out in C order: the stride of each axis of more
   than one element is the number of elements after it. *)
let c_order shape strides =
  let rec go d after =
    d < 0
    || ((shape.(d) = 1 || strides.(d) = after) && go (d - 1) (after * shape.(d)))
  in
  go (Array.length shape - 1) 1

(* [strided flat ~offset shape strides] is the view of [shape] and [strides]
   from element [offset] of the flat node [flat]. *)
let strided flat ~offset shape strides =
  let r = Array.length shape in
  if c_order shape strides then
    Ops.reshape
      (window flat offset (Array.fold_left ( * ) 1 shape))
      (ints (Array.to_list shape))
  else
    let start =
      Array.fold_left ( + ) offset
        (Array.init r (fun d ->
             if strides.(d) < 0 then (shape.(d) - 1) * strides.(d) else 0))
    in
    let axes = stepped shape strides in
    let t =
      match axes with
      | [] -> window flat start 1
      | _ when nests shape strides axes -> nested flat start shape strides axes
      | _ -> overlapping flat start shape strides axes
    in
    let position a = Option.get (List.find_index (( = ) a) axes) in
    let t =
      if axes = [] then t
      else Ops.permute t (List.map position (List.sort Int.compare axes))
    in
    let base =
      List.init r (fun d -> if strides.(d) = 0 then 1 else shape.(d))
    in
    let t = Ops.reshape t (ints base) in
    let flipped =
      List.filter
        (fun d -> strides.(d) < 0 && shape.(d) > 1)
        (List.init r Fun.id)
    in
    let t = if flipped = [] then t else Ops.flip t flipped in
    Ops.expand t (ints (Array.to_list shape))

(* Traces *)

(* The storage of a value, compared by identity: a host buffer, or the storage
   every view of a placed value shares. *)
type storage = Host_buffer of Nx_device.Buffer.t | Placed of Repr.Storage.t

let same_storage s0 s1 =
  match (s0, s1) with
  | Host_buffer b0, Host_buffer b1 -> b0 == b1
  | Placed c0, Placed c1 -> c0 == c1
  | Host_buffer _, Placed _ | Placed _, Host_buffer _ -> false

type capture = {
  storage : storage;
  view : View.t;
  at : Placement.t;
  node : Ops.t;
}

type scope = {
  renderer : Device.t -> Renderer.t;
  mutable dtypes : (Device.t * Dtype.t list) list;
  mutable names : (string * Device.t) list;
  mutable captures : capture list;
  mutable bound : (Ops.t * Nx_device.Buffer.t list) list;
}

let scope ~renderer =
  { renderer; dtypes = []; names = []; captures = []; bound = [] }

let devices s = List.rev s.names
let captures s = List.rev s.bound

(* The name that [s]'s nodes give [d]. *)
let name s d =
  let n = Device.name d in
  (match List.assoc_opt n s.names with
  | Some d' when Device.equal d d' -> ()
  | Some _ ->
      invalid_arg
        (Printf.sprintf "two devices are named %s in one compiled function" n)
  | None -> s.names <- (n, d) :: s.names);
  n

let supports s d dt =
  let dts =
    match List.find_opt (fun (d', _) -> Device.equal d d') s.dtypes with
    | Some (_, dts) -> dts
    | None ->
        let dts = Renderer.supported_dtypes (s.renderer d) in
        s.dtypes <- (d, dts) :: s.dtypes;
        dts
  in
  List.exists (Dtype.equal dt) dts

(* [check s what p dt] is the counterpart of [dt] if every device of [p]
   computes it. *)
let check : type a b.
    scope -> string -> Placement.t -> (a, b) Nx_dtype.t -> Dtype.t =
 fun s what p dt ->
  let refuse where =
    jit_error "cannot compile %s: %s is not supported%s" what
      (Nx_dtype.to_string dt) where
  in
  match dtype dt with
  | None -> refuse ""
  | Some tdt ->
      List.iter
        (fun d ->
          if not (supports s d tdt) then refuse (" on " ^ Device.name d))
        (Placement.devices p);
      tdt

(* Layouts

   tolk places a value on one device, as a copy on each of several, or as equal
   slices of one axis over several, in order. The layout of a value of [shape]
   at [p] is read from the window each device of [p] holds. *)

type layout = One | Copies | Split of int

let layout what p shape =
  let windows = List.map (Placement.window p shape) (Placement.devices p) in
  let n = List.length windows in
  let whole = Array.map (fun extent -> (0, extent)) shape in
  let slices axis =
    let tile = shape.(axis) / n in
    List.for_all2
      (fun k w ->
        Array.for_all2 ( = ) w
          (Array.mapi
             (fun a whole ->
               if a = axis then (k * tile, (k + 1) * tile) else whole)
             whole))
      (List.init n Fun.id) windows
  in
  if n = 1 then One
  else if List.for_all (( = ) whole) windows then Copies
  else
    match List.find_opt slices (List.init (Array.length shape) Fun.id) with
    | Some axis -> Split axis
    | None ->
        jit_error "cannot compile %s at %s: it is not one axis over its devices"
          what
          (Format.asprintf "%a" Placement.pp p)

let device_of s p =
  match Placement.devices p with
  | [ d ] -> Ops.Single (name s d)
  | ds -> Ops.Multi (List.map (name s) ds)

(* Whether values of [shape] at [p] and [q] lie alike on the same devices,
   whatever computes on them. *)
let same_layout p q shape =
  let dp = Placement.devices p in
  List.equal Device.equal dp (Placement.devices q)
  && List.for_all
       (fun d -> Placement.window p shape d = Placement.window q shape d)
       dp

let same_devices p q =
  List.equal Device.equal (Placement.devices p) (Placement.devices q)

let on_disk x =
  match Placement.devices (Nx.placement x) with
  | [ d ] -> Device.equal d (Device.of_runtime Nx_device.disk)
  | _ -> false

(* Where an operation reads [x]: where it lies, or the host for a value on the
   disk. *)
let home x = if on_disk x then Placement.host else Nx.placement x

(* Traced values *)

let shape_of u =
  Array.of_list
    (List.map
       (function Ops.Int n -> n | Ops.Sym _ -> assert false)
       (Ops.shape u))

(* A value made beside one at [p] is a full copy on each of its devices. *)
let context p =
  if Placement.equal p Placement.host then Placement.host
  else Placement.replicated ~backend:(Placement.backend p) (Placement.devices p)

let traced p dt u = Repr.Traced.v ~context:(context p) p dt (shape_of u) (Uop u)

let uop : type a b. (a, b) Nx.t -> Ops.t =
 fun x ->
  match Repr.v x with
  | Repr.Traced t -> (
      match Repr.Traced.node t with
      | Uop u -> u
      | _ -> invalid_arg "a value traced by another transformation")
  | Repr.Host _ | Repr.Placed _ ->
      invalid_arg "not a value traced by a compiled function"

(* Storage

   A value over storage, a capture or a parameter, is a node of the run of its
   storage that its view reaches, viewed by movements. The run starts at the
   element at or below the first one reached whose offset is a multiple of 16
   bytes, since kernels load up to 16 bytes at a time from where a buffer
   starts. *)

let alignment = 16

(* The storage of a value that is not traced, one buffer per device of its
   placement, and its view. *)
let storage : type a b.
    (a, b) Nx.t -> storage * Nx_device.Buffer.t list * View.t =
 fun x ->
  match Repr.v x with
  | Repr.Host a -> (Host_buffer a.buffer, [ a.buffer ], a.view)
  | Repr.Placed r ->
      let st = Repr.Placed.storage r in
      let holders = Placement.devices (Repr.Storage.placement st) in
      let buffers = Repr.Storage.buffers st in
      let buffer_on d =
        List.nth buffers (Option.get (List.find_index (Device.equal d) holders))
      in
      ( Placed st,
        List.map buffer_on (Placement.devices (Nx.placement x)),
        Repr.Placed.view r )
  | Repr.Traced _ -> invalid_arg "a traced value has no storage"

(* [viewed u p shape v start] is the view [v] of a value of [shape] at [p], over
   the flat node [u] of its storage from element [start]: each device's view,
   reassembled when [p] splits the value. *)
let viewed what u p shape v start =
  let local =
    strided u ~offset:(View.offset v - start) (View.shape v) (View.strides v)
  in
  match layout what p shape with
  | Split axis -> Ops.unshard local [ axis ]
  | One | Copies -> local

(* The run of elements [v] reaches from its aligned start, as [(start,
   span)]. *)
let run tdt v =
  let lo, hi = View.extent v in
  let per = Int.max 1 (alignment / Dtype.itemsize tdt) in
  let start = lo - (lo mod per) in
  (start, hi - start)

let held s what p tdt shape bufs v =
  let start, span = run tdt v in
  let buffer = Ops.new_buffer (device_of s p) span tdt in
  let isz = Dtype.itemsize tdt in
  let view b =
    Nx_device.Buffer.view b ~offset:(start * isz) (Nx_device.Buffer.dtype b)
      span
  in
  s.bound <- (buffer, List.map view bufs) :: s.bound;
  viewed what buffer p shape v start

let param s ~slot x =
  let what = "an argument" in
  let p = Nx.placement x in
  let tdt = check s what p (Nx.dtype x) in
  let _, _, v = storage x in
  let shape = Nx.shape x in
  let u =
    if View.numel v = 0 then
      Ops.expand
        (Ops.const ~dtype:tdt (`Int Z.zero))
        (ints (Array.to_list shape))
    else
      let start, span = run tdt v in
      viewed what
        (Ops.param ~shape:[ Ops.Int span ] ~device:(device_of s p) slot tdt)
        p shape v start
  in
  traced p (Nx.dtype x) u

(* Captures

   A value the traced function closes over is bound once, at the placement of
   the operation that meets it. One that reaches a single element is that
   element, a constant: read from its storage when the host owns it, and once
   from its device otherwise. Any other is storage the program holds, at that
   placement, where [Nx.place] puts it first if it lies elsewhere. *)

(* The constant [c] broadcast to [shape]. *)
let broadcast c shape =
  let shape = Array.to_list shape in
  Ops.expand (Ops.reshape c (ints (List.map (fun _ -> 1) shape))) (ints shape)

let same_view v0 v1 =
  View.offset v0 = View.offset v1
  && View.shape v0 = View.shape v1
  && View.strides v0 = View.strides v1

(* [bind s what p x] is the node of [x], a value at [p] that is not traced. *)
let bind : type a b. scope -> string -> Placement.t -> (a, b) Nx.t -> Ops.t =
 fun s what p x ->
  let dt = Nx.dtype x and shape = Nx.shape x in
  let tdt = check s what p dt in
  let key, bufs, v = storage x in
  let one =
    View.numel v > 0
    &&
    let lo, hi = View.extent v in
    hi - lo = 1
  in
  let owned =
    match key with
    | Host_buffer b -> not (Nx_device.Buffer.is_borrowed b)
    | Placed _ -> true
  in
  if View.numel v = 0 then broadcast (Ops.const ~dtype:tdt (`Int Z.zero)) shape
  else if one && owned then
    let first = Nx.item (List.map (fun _ -> 0) (Array.to_list shape)) x in
    broadcast (Ops.const ~dtype:tdt (const dt first)) shape
  else
    let same c =
      same_storage c.storage key && same_view c.view v && Placement.equal c.at p
    in
    match List.find_opt same s.captures with
    | Some c -> c.node
    | None ->
        let node = held s what p tdt shape bufs v in
        s.captures <- { storage = key; view = v; at = p; node } :: s.captures;
        node

(* [capture s what p x] is the node of [x], a value that is not traced, at [p]:
   where it lies if that is [p], placed there first otherwise. *)
let capture s what p x =
  if (not (on_disk x)) && same_layout (Nx.placement x) p (Nx.shape x) then
    bind s what p x
  else bind s what p (Nx.place p x)

(* [node s what p x] is the node of the operand [x] of an operation at [p]. A
   traced value on other devices is copied there, as nx places a host operand of
   an operation on a device. *)
let node s what p x =
  match Repr.v x with
  | Repr.Traced _ ->
      let u = uop x in
      if same_devices (Nx.placement x) p then u
      else Ops.copy_to_device u (device_of s p)
  | Repr.Host _ | Repr.Placed _ -> capture s what p x

(* Movements *)

let move u : Nx.Op.move -> Ops.t = function
  | Reshape shape -> Ops.reshape u (ints (Array.to_list shape))
  | Expand shape -> Ops.expand u (ints (Array.to_list shape))
  | Permute order -> Ops.permute u (Array.to_list order)
  | Shrink limits ->
      Ops.shrink u
        (Array.to_list
           (Array.map (fun (lo, hi) -> Some (Ops.Int lo, Ops.Int hi)) limits))
  | Flip dims ->
      Ops.flip u
        (List.filter (fun a -> dims.(a)) (List.init (Array.length dims) Fun.id))
  | Window { axis; size; step } ->
      (* The windows along [axis] take its place, and their elements are a new
         last axis. *)
      let r = Array.length (shape_of u) in
      let last = List.filter (( <> ) axis) (List.init r Fun.id) @ [ axis ] in
      let windows = Ops.pool ~stride:[ step ] (Ops.permute u last) [ size ] in
      Ops.permute windows
        (List.init r (fun a ->
             if a = axis then r - 1 else if a < axis then a else a - 1)
        @ [ r ])

(* [place s what p q x] is the traced [x], at [p], at [q]: the same node where
   [q] lays it out alike, copied to [q]'s devices, or split over them. *)
let place s what p q x =
  let u = uop x and shape = Nx.shape x in
  if same_layout p q shape then u
  else
    match layout what q shape with
    | Split axis -> (
        match device_of s q with
        | Ops.Multi names -> Ops.shard ~axis u names
        | Ops.Single _ as d -> Ops.copy_to_device u d)
    | One | Copies -> Ops.copy_to_device u (device_of s q)

(* Operations *)

let op : type r. scope -> r Nx.Op.t -> r =
 fun s o ->
  let what = Nx.Op.name o in
  let p = Nx.Op.placement o in
  let ret dt u =
    ignore (check s what p dt);
    traced p dt u
  in
  let refuse () = jit_error "cannot compile %s" what in
  match[@warning "@4@8"] o with
  | Unary (k, x) -> ret (Nx.dtype x) (Lower_arith.unary k (node s what p x))
  | Binary (k, x, y) ->
      ret (Nx.dtype x)
        (Lower_arith.binary k (node s what p x) (node s what p y))
  | Compare (k, x, y) ->
      ret Nx_dtype.bool
        (Lower_arith.compare k (node s what p x) (node s what p y))
  | Where (c, x, y) ->
      ret (Nx.dtype x)
        (Ops.where (node s what p c) (node s what p x) (node s what p y))
  | Convert (Cast, dt, x) ->
      ret dt (Lower_arith.cast (check s what p dt) (node s what p x))
  | Convert (Bitcast, dt, x) ->
      ret dt (Lower_arith.bitcast (check s what p dt) (node s what p x))
  | Threefry (key, counter) ->
      let k = node s what p key in
      if not (Ops.op_in_backward_slice_with_self k [ Op.Param ]) then
        jit_error
          "a random draw from a key that does not depend on the function's \
           arguments would repeat on every call; pass the key as an argument";
      ret Nx_dtype.int32 (Lower_arith.threefry k (node s what p counter))
  | Reduce (k, axes, x) ->
      ret (Nx.dtype x)
        (Lower_reduce.reduce k ~axes:(Array.to_list axes) (node s what p x))
  | Scan (k, axis, x) ->
      ret (Nx.dtype x) (Lower_reduce.scan k ~axis (node s what p x))
  | Arg_reduce (k, axis, x) ->
      ret Nx_dtype.int32 (Lower_reduce.arg_reduce k ~axis (node s what p x))
  | Sort { descending; axis; x } ->
      ret (Nx.dtype x) (Lower_reduce.sort ~descending ~axis (node s what p x))
  | Argsort { descending; axis; x } ->
      ret Nx_dtype.int32
        (Lower_reduce.argsort ~descending ~axis (node s what p x))
  | Pad _ | Cat _ | Gather _ | Scatter _ | Update _ | Unfold _ | Fold _ ->
      refuse ()
  | Matmul _ | Cholesky _ | Qr _ | Lu _ | Svd _ | Solve_triangular _ ->
      refuse ()
  | Fft _ | Rfft _ | Irfft _ | Eig _ | Eigh _ -> refuse ()
  | Contiguous x -> ret (Nx.dtype x) (Ops.contiguous (node s what p x))
  | Move (x, m) ->
      let q = home x in
      let u =
        match Repr.v x with
        | Repr.Traced _ -> uop x
        | Repr.Host _ | Repr.Placed _ -> capture s what q x
      in
      ret (Nx.dtype x) (move u m)
  | Place (q, x) -> (
      match Repr.v x with
      | Repr.Traced _ -> ret (Nx.dtype x) (place s what (Nx.placement x) q x)
      | Repr.Host _ | Repr.Placed _ -> ret (Nx.dtype x) (capture s what q x))
  | Read x -> (
      match Repr.v x with
      | Repr.Traced _ ->
          jit_error
            "a compiled function read the value of a traced tensor (item, \
             to_host, or a branch on its elements): a compiled program cannot \
             depend on the values it computes"
      | Repr.Host _ | Repr.Placed _ -> Nx.Op.eval o)
