(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk
module View = Nx_array.View
module Repr = Nx.Repr
module Placement = Nx.Placement

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
  | Int4 -> `Int (Bigint.of_int v)
  | UInt4 -> `Int (Bigint.of_int v)
  | Int8 -> `Int (Bigint.of_int v)
  | UInt8 -> `Int (Bigint.of_int v)
  | Int16 -> `Int (Bigint.of_int v)
  | UInt16 -> `Int (Bigint.of_int v)
  | Int32 -> `Int (Bigint.of_int32 v)
  | UInt32 -> `Int (Bigint.of_int32_unsigned v)
  | Int64 -> `Int (Bigint.of_int64 v)
  | UInt64 -> `Int (Bigint.of_int64_unsigned v)
  | Bool -> `Bool v
  | Complex64 | Complex128 -> invalid_arg "a complex constant"

(* Views over storage

   A view reaches the elements of its storage by strides from an offset. Over a
   flat node of those elements the same view is movements: a broadcast axis is
   an expand, a negative stride a flip, and the stepped axes, by decreasing
   stride, are rows of a reshape cut by a shrink when each stride nests in the
   one before it. Overlapping axes are windows of [pool]. *)

let ints l = List.map (fun n -> Ops.Int n) l

(* [windows u axis size] is [Ops.pool] along [axis] of [u]: the windows take
   [axis]'s place, and their elements are a new last axis. *)
let windows ?(step = 1) ?(dilation = 1) u axis size =
  let r = Ops.ndim u in
  let last = List.filter (( <> ) axis) (List.init r Fun.id) @ [ axis ] in
  let w =
    Ops.pool ~stride:[ step ] ~dilation:[ dilation ] (Ops.permute u last)
      [ size ]
  in
  Ops.permute w
    (List.init r (fun a ->
         if a = axis then r - 1 else if a < axis then a else a - 1)
    @ [ r ])

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
  let size = Ops.max_numel flat in
  let flat =
    if start + n <= size then flat
    else Ops.pad flat [ Some (Ops.Int 0, Ops.Int (start + n - size)) ]
  in
  Ops.shrink flat [ Some (Ops.Int start, Ops.Int (start + n)) ]

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

(* The stepped [axes] from the innermost out, each the windows of its stride
   over the axis before it. *)
let overlapping flat start shape strides axes =
  let extent =
    List.fold_left
      (fun e a -> e + ((shape.(a) - 1) * Int.abs strides.(a)))
      1 axes
  in
  let step u a = windows ~dilation:(Int.abs strides.(a)) u 0 shape.(a) in
  let u = List.fold_left step (window flat start extent) (List.rev axes) in
  (* The innermost axis came first, and the leading axis has one element. *)
  let k = List.length axes in
  Ops.reshape
    (Ops.permute u (0 :: List.init k (fun i -> k - i)))
    (ints (List.map (fun a -> shape.(a)) axes))

(* [strided flat v origin] is the view [v] over the flat node [flat] of its
   storage from element [origin]. *)
let strided flat v origin =
  let offset = View.offset v - origin
  and shape = View.shape v
  and strides = View.strides v in
  let r = Array.length shape in
  if View.is_c_contiguous v then
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
  buffer : Ops.t;
  buffers : Nx_device.Buffer.t list;
}

type write = { result : Ops.t; into : Ops.t; regions : Lower_index.region list }

type check = {
  first : (int64, Nx_dtype.int64_elt) Nx.t;
  shape : int array;
  msg : int array -> string;
}

type scope = {
  renderer : Nx_device.t -> Renderer.t;
  mutable dtypes : (Nx_device.t * Dtype.t list) list;
  mutable names : (string * Nx_device.t) list;
  mutable captures : capture list;
  mutable writes : write list;
  mutable checks : check list;
  mutable arguments : Ops.t list;
  stuck : unit Ops.Tbl.t;
      (* Nodes that read an argument, a write, a copy or storage that is no
         capture: they cannot follow their use. *)
  followed : (Placement.t * Ops.t option) list Ops.Tbl.t;
}

let scope ~renderer =
  {
    renderer;
    dtypes = [];
    names = [];
    captures = [];
    writes = [];
    checks = [];
    arguments = [];
    stuck = Ops.Tbl.create 64;
    followed = Ops.Tbl.create 16;
  }

let devices s = List.rev s.names
let captures s = List.rev_map (fun c -> (c.buffer, c.buffers)) s.captures
let writes s = s.writes
let checks s = List.rev s.checks

let checking s f =
  let outer = s.checks in
  s.checks <- [];
  Fun.protect
    ~finally:(fun () -> s.checks <- outer)
    (fun () ->
      let v = f () in
      (v, List.rev s.checks))

(* The name that [s]'s nodes give [d]. *)
let name s d =
  let n = Nx_device.name d in
  (match List.assoc_opt n s.names with
  | Some d' when Nx_device.equal d d' -> ()
  | Some _ ->
      invalid_arg
        (Printf.sprintf "two devices are named %s in one compiled function" n)
  | None -> s.names <- (n, d) :: s.names);
  n

(* Whether [d] computes [dt]: a dtype [d]'s renderer supports, read the first
   time [s] meets [d], or one tolk emulates where the renderer lacks it. *)
let supports s d dt =
  let dtypes =
    match List.assq_opt d s.dtypes with
    | Some l -> l
    | None ->
        let l = Decomp_dtype.computes (s.renderer d) in
        s.dtypes <- (d, l) :: s.dtypes;
        l
  in
  List.exists (Dtype.equal dt) dtypes

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
          if not (supports s d tdt) then refuse (" on " ^ Nx_device.name d))
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
  List.equal Nx_device.equal dp (Placement.devices q)
  && List.for_all
       (fun d -> Placement.window p shape d = Placement.window q shape d)
       dp

let same_devices p q =
  List.equal Nx_device.equal (Placement.devices p) (Placement.devices q)

let disk p =
  match Placement.devices p with
  | [ d ] -> Nx_device.equal d Nx_device.disk
  | _ -> false

let on_disk x = disk (Nx.placement x)

(* Where an operation reads [x]: where it lies, or the host for a value on the
   disk. *)
let home x = if on_disk x then Placement.host else Nx.placement x

(* Traced values *)

(* A value made beside one at [p] is a full copy on each of its devices. *)
let context p =
  if Placement.equal p Placement.host then Placement.host
  else Placement.replicated ~backend:(Placement.backend p) (Placement.devices p)

(* [settled s what p u] is [u], a value computed at [p]'s devices, laid out as
   nx places a result at [p]. Where they differ, a value split over the devices
   is joined whole on each, keeping each element's bits, and a value whole on
   each is split as nx splits it. *)
let settled s what p u =
  let shape = Array.of_list (Ops.max_shape u) in
  match (layout what p shape, Ops.axis u) with
  | One, _ | Copies, None -> u
  | Split a, Some b when a = b -> u
  | Copies, Some _ -> Ops.copy_to_device u (device_of s p)
  | Split a, sharding ->
      let whole =
        match sharding with
        | Some _ -> Ops.copy_to_device u (device_of s p)
        | None -> u
      in
      Ops.shard ~axis:a whole (List.map (name s) (Placement.devices p))

let traced p dt u =
  Repr.Traced.v ~context:(context p) p dt
    (Array.of_list (Ops.max_shape u))
    (Uop u)

let uop : type a b. (a, b) Nx.t -> Ops.t =
 fun x ->
  match Repr.v x with
  | Repr.Traced t -> (
      match Repr.Traced.node t with
      | Uop u -> u
      | _ -> invalid_arg "a value traced by another transformation")
  | Repr.Host _ | Repr.Placed _ ->
      invalid_arg "not a value traced by a compiled function"

(* Storage *)

(* Kernels load up to 16 bytes at a time from where a buffer starts. *)
let alignment = 16

let outside () =
  invalid_arg
    "a traced tensor has no bytes; it was used outside the trace that made it"

(* The storage of a value that is not traced, compared by identity. *)
let key : type a b. (a, b) Nx.t -> storage =
 fun x ->
  match Repr.v x with
  | Repr.Host a -> Host_buffer a.buffer
  | Repr.Placed r -> Placed (Repr.Placed.storage r)
  | Repr.Traced _ -> outside ()

(* [viewed u p shape v start] is the view [v] of a value of [shape] at [p], over
   the flat node [u] of its storage from element [start]: each device's view,
   reassembled when [p] splits the value. *)
let viewed what u p shape v start =
  let local = strided u v start in
  match layout what p shape with
  | Split axis -> Ops.unshard local [ axis ]
  | One | Copies -> local

let span tdt v =
  let lo, hi = View.extent v in
  let per = Int.max 1 (alignment / Dtype.itemsize tdt) in
  let start = lo - (lo mod per) in
  (start, hi - start)

let within tdt v =
  let start, _ = span tdt v and shape = View.shape v in
  View.create
    ~offset:(View.offset v - start)
    ~strides:
      (Array.mapi (fun d s -> if shape.(d) = 1 then 0 else s) (View.strides v))
    shape

let run tdt v b =
  let start, span = span tdt v in
  Nx_device.Buffer.view b
    ~offset:(start * Dtype.itemsize tdt)
    (Nx_device.Buffer.dtype b) span

let phase dt b start =
  if Nx_device.equal (Nx_device.Buffer.device b) Nx_device.disk then 0
  else
    let at = Nx_device.Buffer.address b in
    Nativeint.(to_int (rem (add at (of_int (start * Dtype.itemsize dt))) 16n))

(* The phase of element [start] of [bufs], the buffers of one placement: the
   same for every buffer. *)
let phase_of what tdt bufs start =
  match
    List.sort_uniq Int.compare (List.map (fun b -> phase tdt b start) bufs)
  with
  | [] -> 0
  | [ p ] -> p
  | _ ->
      jit_error
        "cannot compile %s: its buffers start at different places within 16 \
         bytes"
        what

(* The constant [c] broadcast to [shape]. *)
let broadcast c shape =
  let shape = Array.to_list shape in
  Ops.expand (Ops.reshape c (ints (List.map (fun _ -> 1) shape))) (ints shape)

let param s ~slot x =
  let what = "an argument" in
  let p = Nx.placement x in
  let tdt = check s what p (Nx.dtype x) in
  let bufs, v = Nx.shards x in
  let shape = Nx.shape x in
  let u =
    if View.numel v = 0 then
      broadcast (Ops.const ~dtype:tdt (`Int Bigint.zero)) shape
    else
      let start, span = span tdt v in
      let phase = phase_of what tdt bufs start in
      let buffer = Ops.new_buffer ~slot ~phase (device_of s p) span tdt in
      s.arguments <- buffer :: s.arguments;
      viewed what buffer p shape v start
  in
  traced p (Nx.dtype x) u

(* [laid s what storage p dt shape] is a value of [dt] and [shape] at [p] in C
   order over the node [storage d n tdt] makes of [n] elements on [d]: each
   device's window, starting on 16 bytes. *)
let laid s what storage p dt shape =
  let tdt = check s what p dt in
  (* Every device holds a window of one shape. *)
  let d = List.hd (Placement.devices p) in
  let local =
    View.create
      (Array.map (fun (lo, hi) -> hi - lo) (Placement.window p shape d))
  in
  viewed what (storage (device_of s p) (View.numel local) tdt) p shape local 0

let output s ~slot p dt shape =
  laid s "a result" (fun d n tdt -> Ops.new_buffer ~slot d n tdt) p dt shape

let parameter s ~slot p dt shape =
  traced p dt
    (laid s "a loop's value"
       (fun d n tdt -> Ops.param ~shape:[ Ops.Int n ] ~device:d slot tdt)
       p dt shape)

(* The engine's devices *)

let engine s = Tolk_engine.device (devices s)

(* Captures *)

let same_view v0 v1 =
  View.offset v0 = View.offset v1
  && View.shape v0 = View.shape v1
  && View.strides v0 = View.strides v1

(* [bind s what p x] is the node of [x], a value at [p] that is not traced. *)
let bind : type a b. scope -> string -> Placement.t -> (a, b) Nx.t -> Ops.t =
 fun s what p x ->
  let dt = Nx.dtype x and shape = Nx.shape x in
  let tdt = check s what p dt in
  let key = key x and bufs, v = Nx.shards x in
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
  if View.numel v = 0 then
    broadcast (Ops.const ~dtype:tdt (`Int Bigint.zero)) shape
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
        let start, span = span tdt v in
        let phase = phase_of what tdt bufs start in
        let buffer = Ops.new_buffer ~phase (device_of s p) span tdt in
        let node = viewed what buffer p shape v start in
        let buffers = List.map (run tdt v) bufs in
        s.captures <-
          { storage = key; view = v; at = p; node; buffer; buffers }
          :: s.captures;
        node

(* [placed s what p x] is the node of [x], a value that is not traced, at [p]:
   where it lies if that is [p], placed there first otherwise. *)
let placed s what p x =
  if (not (on_disk x)) && same_layout (Nx.placement x) p (Nx.shape x) then
    bind s what p x
  else bind s what p (Nx.place p x)

(* [capture s what p x] is the node of [x], a value that is not traced, as an
   operand of an operation at [p]: where it lies if that is on [p]'s devices,
   and otherwise a copy on each of them, as nx places a host operand. *)
let capture s what p x =
  if (not (on_disk x)) && same_devices (Nx.placement x) p then
    bind s what (Nx.placement x) x
  else placed s what (context p) x

exception Uncomputed

(* [follow s q u] is [u] computed on each device of [q] when it reads only
   captures, which are then copied there once, rather than computed where it
   lies and copied on every call. [None] when [u] reads anything else, or when a
   device of [q] does not compute one of its dtypes. *)
let follow s q u =
  let capture_of v = List.find_opt (fun c -> c.buffer == v) s.captures in
  let computes v =
    let dt = Ops.dtype v in
    (not (List.exists (Dtype.equal dt) Dtype.all))
    || List.for_all (fun d -> supports s d dt) (Placement.devices q)
  in
  let seen = Ops.Tbl.create 16 and read = ref [] in
  let rec visit v =
    if Ops.Tbl.mem s.stuck v then raise Exit;
    if not (Ops.Tbl.mem seen v) then begin
      Ops.Tbl.add seen v ();
      if not (computes v) then raise Uncomputed;
      (match Ops.op v with
      | Op.Buffer -> (
          match capture_of v with
          | Some c when List.length c.buffers = 1 -> read := c :: !read
          | Some _ | None -> stuck v)
      | Op.Param | Op.Alloc | Op.Call | Op.After | Op.Store | Op.Copy
      | Op.Mselect | Op.Mstack | Op.Unshard | Op.Allreduce ->
          stuck v
      | _ -> ());
      try List.iter visit (Ops.src v) with Exit -> stuck v
    end
  and stuck v =
    Ops.Tbl.replace s.stuck v ();
    raise Exit
  in
  match visit u with
  | exception (Exit | Uncomputed) -> None
  | () ->
      let moved c =
        let same c' =
          same_storage c'.storage c.storage
          && same_view c'.view c.view && Placement.equal c'.at q
        in
        match List.find_opt same s.captures with
        | Some c' -> (c.buffer, c'.buffer)
        | None ->
            let src = List.hd c.buffers in
            let n = Nx_device.Buffer.length src in
            let buffer =
              Ops.new_buffer (device_of s q) n (Ops.dtype c.buffer)
            in
            let buffers =
              List.map
                (fun d ->
                  let dst =
                    Nx_device.Buffer.create d (Nx_device.Buffer.dtype src) n
                  in
                  Nx_device.Buffer.copy ~src ~dst;
                  dst)
                (Placement.devices q)
            in
            let node = Ops.substitute c.node [ (c.buffer, buffer) ] in
            s.captures <- { c with at = q; node; buffer; buffers } :: s.captures;
            (c.buffer, buffer)
      in
      Some (Ops.substitute u (List.map moved !read))

(* [followed s q u] is [follow s q u], once per trace. *)
let followed s q u =
  let known = Option.value ~default:[] (Ops.Tbl.find_opt s.followed u) in
  match List.find_opt (fun (q', _) -> Placement.equal q q') known with
  | Some (_, v) -> v
  | None ->
      let v = follow s q u in
      Ops.Tbl.replace s.followed u ((q, v) :: known);
      v

(* [copied u d] is [u] copied to the device [d]: the value its views of elements
   read, copied, with the views that drop no element (reshapes, broadcasts,
   permutations, flips and pads) taken on [d], so that a copy moves the bytes
   the value reads, not those its broadcasts repeat. *)
let rec copied u d =
  match Ops.op u with
  | Op.Reshape | Op.Expand | Op.Permute | Op.Flip | Op.Pad -> (
      match Ops.src u with
      | base :: rest -> Ops.replace u ~src:(copied base d :: rest)
      | [] -> Ops.copy_to_device u d)
  | _ -> Ops.copy_to_device u d

(* [node s what p x] is the node of the operand [x] of an operation at [p]. A
   traced value on other devices is computed there when it reads only captures,
   and copied there otherwise, as nx places a host operand of an operation on a
   device. *)
let node s what p x =
  match Repr.v x with
  | Repr.Traced _ -> (
      let u = uop x in
      if same_devices (Nx.placement x) p then u
      else
        match followed s (context p) u with
        | Some u -> u
        | None -> copied u (device_of s p))
  | Repr.Host _ | Repr.Placed _ -> capture s what p x

let value s x = node s "a value" (Nx.placement x) x

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
  | Window { axis; size; step } -> windows ~step u axis size

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
    | One | Copies -> (
        match followed s q u with
        | Some u -> u
        | None -> copied u (device_of s q))

(* Operations *)

let op : type r. scope -> r Nx.Op.t -> r =
 fun s o ->
  let what = Nx.Op.name o in
  (* A movement of a value on the disk is read on the host. *)
  let p =
    match Nx.Op.placement o with p when disk p -> Placement.host | p -> p
  in
  let ret dt u =
    ignore (check s what p dt);
    traced p dt (settled s what p u)
  in
  let like x u = ret (Nx.dtype x) u in
  let write ~into regions result =
    s.writes <- { result; into; regions } :: s.writes;
    result
  in
  let n x = node s what p x in
  (* A factorization of integers raises, as nx.cpu's does. *)
  let factored x =
    let dt = Nx.dtype x in
    if not (Nx_dtype.is_float dt || Nx_dtype.is_complex dt) then
      invalid_arg (what ^ ": linalg requires a float or complex dtype");
    n x
  in
  match[@warning "@4@8"] o with
  | Unary (k, x) -> like x (Lower_arith.unary k (n x))
  | Binary (k, x, y) -> like x (Lower_arith.binary k (n x) (n y))
  | Compare (k, x, y) -> ret Nx_dtype.bool (Lower_arith.compare k (n x) (n y))
  | Where (c, x, y) -> like x (Ops.where (n c) (n x) (n y))
  | Fma (a, b, c) -> like a (Lower_arith.fma (n a) (n b) (n c))
  | Convert (Cast, dt, x) -> ret dt (Lower_arith.cast (check s what p dt) (n x))
  | Convert (Bitcast, dt, x) ->
      ret dt (Lower_arith.bitcast (check s what p dt) (n x))
  | Threefry (key, counter) ->
      let k = n key and c = n counter in
      (* A parameter is an argument of a called body: a staged loop's trip. *)
      let varies u =
        let slice = Ops.backward_slice_with_self u in
        Ops.op_in_backward_slice_with_self u [ Op.Param ]
        || List.exists (fun a -> Ops.Nodes.mem a slice) s.arguments
      in
      (* An empty draw has nothing to repeat. *)
      let empty = Ops.max_numel c = 0 in
      if not (empty || varies k || varies c) then
        jit_error
          "a random draw that does not depend on the function's arguments \
           would repeat on every call; pass the key as an argument";
      ret Nx_dtype.int32 (Lower_arith.threefry k c)
  | Reduce (k, axes, x) ->
      like x (Lower_reduce.reduce k ~axes:(Array.to_list axes) (n x))
  | Scan (k, axis, x) -> like x (Lower_reduce.scan k ~axis (n x))
  | Arg_reduce (k, axis, x) ->
      ret Nx_dtype.int64 (Lower_reduce.arg_reduce k ~axis (n x))
  | Sort { descending; axis; x } ->
      like x (Lower_reduce.sort ~descending ~axis (n x))
  | Argsort { descending; axis; x } ->
      ret Nx_dtype.int64 (Lower_reduce.argsort ~descending ~axis (n x))
  | Pad (padding, fill, x) ->
      like x (Lower_index.pad padding (const (Nx.dtype x) fill) (n x))
  | Cat (axis, xs) -> (
      match xs with
      | [] -> invalid_arg "a concatenation of no values"
      | x :: rest -> like x (Lower_index.cat axis (n x) (List.map n rest)))
  | Gather (axis, indices, x) ->
      like x (Lower_index.gather axis (n indices) (n x))
  | Scatter { mode; unique; axis; indices; updates; into = x } ->
      let indices = n indices and updates = n updates and into = n x in
      let result, regions =
        Lower_index.scatter ~mode ~unique ~axis ~indices ~updates into
      in
      like x (write ~into regions result)
  | Update (x, starts, v) ->
      let into = n x and starts = n starts and v = n v in
      like x
        (write ~into
           (Option.to_list (Lower_index.update_region into ~starts v))
           (Lower_index.update into ~starts v))
  | Unfold { kernel_size; stride; dilation; padding; x } ->
      like x (Lower_index.unfold ~kernel_size ~stride ~dilation ~padding (n x))
  | Fold { output_size; kernel_size; stride; dilation; padding; x } ->
      like x
        (Lower_index.fold ~output_size ~kernel_size ~stride ~dilation ~padding
           (n x))
  | Matmul (x, y) -> like x (Lower_linalg.matmul (n x) (n y))
  | Cholesky { upper; x } -> like x (Lower_linalg.cholesky ~upper (factored x))
  | Qr { reduced; x } ->
      let q, r = Lower_linalg.qr ~reduced (factored x) in
      (like x q, like x r)
  | Lu x ->
      let lu, pivots, perm = Lower_linalg.lu (factored x) in
      (like x lu, ret Nx_dtype.int64 pivots, ret Nx_dtype.int64 perm)
  | Svd { full_matrices; x } ->
      let u, sv, vt = Lower_linalg.svd ~full_matrices (factored x) in
      (like x u, ret Nx_dtype.float64 sv, like x vt)
  | Solve_triangular { upper; transpose; unit_diag; a; b } ->
      like b
        (Lower_linalg.solve_triangular ~upper ~transpose ~unit_diag (factored a)
           (factored b))
  | Fft _ | Rfft _ | Irfft _ | Eig _ | Eigh _ ->
      jit_error "cannot compile %s" what
  | Contiguous x -> like x (Ops.contiguous (n x))
  | Move (x, m) -> like x (move (node s what (home x) x) m)
  | Place (q, x) -> (
      match Repr.v x with
      | Repr.Traced _ -> like x (place s what (Nx.placement x) q x)
      | Repr.Host _ | Repr.Placed _ -> like x (placed s what q x))
  | Read { by; x } -> (
      match Repr.v x with
      | Repr.Traced _ ->
          jit_error
            "%s: cannot read the value of a traced tensor inside jit; return \
             it from the compiled function instead"
            by
      | Repr.Host _ | Repr.Placed _ -> Nx.Op.eval o)
  | Check { ok; msg } -> (
      match Repr.v ok with
      | Repr.Traced _ ->
          (* The first false element's index in C order, or the element count:
             the least of the indices, each where its element is false and the
             count elsewhere. *)
          let shape = Nx.shape ok in
          let count = Array.fold_left ( * ) 1 shape in
          if count > 0 then begin
            (* The index lives where a reduction of [ok] over every axis would:
               placed as nx places one, a copy on each device of a split
               value. *)
            let at =
              Nx.Op.placement
                (Reduce (Sum, Array.init (Array.length shape) Fun.id, ok))
            in
            let i = Ops.arange ~dtype:Int64 count in
            let first =
              Lower_reduce.reduce Min ~axes:[ 0 ]
                (Ops.where
                   (Ops.reshape (value s ok) [ Ops.Int count ])
                   (Ops.const_like i (`Int (Bigint.of_int count)))
                   i)
            in
            ignore (check s what at Nx_dtype.int64);
            s.checks <-
              { first = traced at Nx_dtype.int64 first; shape; msg } :: s.checks
          end
      | Repr.Host _ | Repr.Placed _ -> Nx.Op.eval o)
