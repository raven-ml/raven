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
  | Int4 -> Some Int8
  | UInt4 -> Some Uint8
  | Bit -> Some Bool
  | Complex64 | Complex128 -> None

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
  (* An integer literal is reduced modulo 16, as nx stores one. *)
  | Int4 ->
      let low = v land 15 in
      `Int (Bigint.of_int (if low < 8 then low else low - 16))
  | UInt4 -> `Int (Bigint.of_int (v land 15))
  | Int8 -> `Int (Bigint.of_int v)
  | UInt8 -> `Int (Bigint.of_int v)
  | Int16 -> `Int (Bigint.of_int v)
  | UInt16 -> `Int (Bigint.of_int v)
  | Int32 -> `Int (Bigint.of_int32 v)
  | UInt32 -> `Int (Bigint.of_int32_unsigned v)
  | Int64 -> `Int (Bigint.of_int64 v)
  | UInt64 -> `Int (Bigint.of_int64_unsigned v)
  | Bool -> `Bool v
  | Bit -> `Bool v
  | Complex64 | Complex128 -> invalid_arg "a complex constant"

(* Packed elements

   [bit], [int4] and [uint4] elements lie [8 / bits] to a byte, the first in the
   lowest bits. A graph computes each at a byte, as {!dtype} says, an [int4] or
   [uint4] holding its representative. The storage nx binds holds their bytes as
   [uint8]: a graph unpacks it where it reads it and packs a value where it
   stores one, so tolk sees no dtype narrower than a byte. *)

let bits dt = Nx_dtype.Scalar.(bitsize (of_dtype dt))
let packed dt = bits dt < 8

(* The bytes that hold [n] elements of [dt]. *)
let bytes dt n = ((n * bits dt) + 7) / 8
let ints l = List.map (fun n -> Ops.Int n) l
let uint8 u n = Ops.const_like ~dtype:Uint8 u (`Int (Bigint.of_int n))

(* [code dt u] is the bits of [u]'s elements of [dt], a dtype of at most 8 bits,
   as the low bits of [uint8]s whose other bits are 0. *)
let code : type a b. (a, b) Nx_dtype.t -> Ops.t -> Ops.t =
 fun dt u ->
  match dt with
  | Bool | Bit -> Ops.cast u Uint8
  | Int4 -> Ops.bitwise_and (Ops.bitcast u Uint8) (uint8 u 15)
  | UInt4 -> Ops.bitwise_and u (uint8 u 15)
  | Float8_e4m3 | Float8_e5m2 | Int8 | UInt8 -> Ops.bitcast u Uint8
  | Float16 | Float32 | Float64 | BFloat16 | Int16 | UInt16 | Int32 | UInt32
  | Int64 | UInt64 | Complex64 | Complex128 ->
      invalid_arg "the code of an element wider than a byte"

(* [of_code dt c] is the elements of [dt] whose bits [code] gives as [c]. *)
let of_code : type a b. (a, b) Nx_dtype.t -> Ops.t -> Ops.t =
 fun dt c ->
  match dt with
  | Bool | Bit -> Ops.ne c (uint8 c 0)
  | Int4 ->
      let int n = Ops.const_like ~dtype:Int8 c (`Int (Bigint.of_int n)) in
      Ops.sub (Ops.bitwise_xor (Ops.bitcast c Int8) (int 8)) (int 8)
  | UInt4 -> c
  | Float8_e4m3 -> Ops.bitcast c Fp8e4m3
  | Float8_e5m2 -> Ops.bitcast c Fp8e5m2
  | Int8 -> Ops.bitcast c Int8
  | UInt8 -> c
  | Float16 | Float32 | Float64 | BFloat16 | Int16 | UInt16 | Int32 | UInt32
  | Int64 | UInt64 | Complex64 | Complex128 ->
      invalid_arg "the element of a code wider than a byte"

(* [modular dt u] is [u], a result of [dt] computed at a byte, reduced modulo 16
   to its representative for [int4] and [uint4]: its code read back. *)
let modular : type a b. (a, b) Nx_dtype.t -> Ops.t -> Ops.t =
 fun dt u ->
  match dt with
  | Int4 | UInt4 -> of_code dt (code dt u)
  | Float16 | Float32 | Float64 | BFloat16 | Float8_e4m3 | Float8_e5m2 | Int8
  | UInt8 | Int16 | UInt16 | Int32 | UInt32 | Int64 | UInt64 | Bool | Bit
  | Complex64 | Complex128 ->
      u

(* The shifts [0; b; ...; (k - 1) b] along the last axis of [shape]. *)
let shifts b k shape =
  let ones = List.map (fun _ -> 1) shape in
  Ops.expand
    (Ops.reshape
       (Ops.arange ~step:b ~dtype:Uint8 (k * b))
       (ints (ones @ [ k ])))
    (ints (shape @ [ k ]))

(* [split dt c k] reads each byte of [c] as [k] elements of [dt], the first in
   the lowest bits: a last axis of [m] bytes becomes one of [m * k] elements. *)
let split dt c k =
  let b = bits dt and shape = Ops.max_shape c in
  let wide = Ops.expand (Ops.unsqueeze c (-1)) (ints (shape @ [ k ])) in
  let codes =
    Ops.bitwise_and
      (Ops.shr wide (shifts b k shape))
      (uint8 wide ((1 lsl b) - 1))
  in
  let shape =
    match List.rev shape with
    | m :: rest -> List.rev ((m * k) :: rest)
    | [] -> [ k ]
  in
  of_code dt (Ops.reshape codes (ints shape))

(* [join dt x k] is [split]'s inverse: each [k] elements of [dt] along the last
   axis of [x] are one byte, the first in the lowest bits. *)
let join dt x k =
  let b = bits dt in
  let front, m =
    match List.rev (Ops.max_shape x) with
    | n :: rest -> (List.rev rest, n / k)
    | [] -> invalid_arg "a join of a scalar"
  in
  let rows = Ops.reshape (code dt x) (ints (front @ [ m; k ])) in
  Ops.rop
    (Ops.shl rows (shifts b k (front @ [ m ])))
    Op.Add
    [ List.length front + 1 ]

(* [unpack dt u] is the elements of [dt] that the bytes [u] hold, [8 / bits] to
   each. *)
let unpack dt u = split dt u (8 / bits dt)

(* [pack dt u] is the bytes that hold [u]'s elements of [dt] in C order, the
   bits past the last element 0. *)
let pack dt u =
  let n = Ops.max_numel u and per = 8 / bits dt in
  let m = (n + per - 1) / per in
  let flat = Ops.reshape u [ Ops.Int n ] in
  let flat =
    if m * per = n then flat
    else Ops.pad flat [ Some (Ops.Int 0, Ops.Int ((m * per) - n)) ]
  in
  join dt flat per

(* Views over storage

   A view reaches the elements of its storage by strides from an offset. Over a
   flat node of those elements the same view is movements: a broadcast axis is
   an expand, a negative stride a flip, and the stepped axes, by decreasing
   stride, are rows of a reshape cut by a shrink when each stride nests in the
   one before it. Overlapping axes are windows of [pool]. *)

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
  data : Nx.packed list;
  shape : int array;
  fail : int array -> Nx.packed list -> exn;
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
  bytes : Ops.t Ops.Tbl.t;
      (* A packed value that views its storage C-contiguously from a byte, and
         those bytes. *)
  mutable live : bool;
}

type (_, _) Repr.node += Uop : scope * Ops.t -> ('a, 'b) Repr.node

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
    bytes = Ops.Tbl.create 16;
    live = true;
  }

let finish s = s.live <- false
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

(* The memories of [p]'s devices, which the compiler addresses: who computes on
   a memory eagerly takes no part in compiling for it. *)
let memories p = List.map Nx.Device.memory (Placement.devices p)

(* The name that [s]'s nodes give the memory [d]. *)
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
        (memories p);
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
  match memories p with
  | [ d ] -> Ops.Single (name s d)
  | ds -> Ops.Multi (List.map (name s) ds)

(* Whether values of [shape] at [p] and [q] lie alike in the same memories,
   whatever computes on them. *)
let same_layout p q shape =
  List.equal Nx_device.equal (memories p) (memories q)
  && List.for_all2
       (fun d e -> Placement.window p shape d = Placement.window q shape e)
       (Placement.devices p) (Placement.devices q)

let same_devices p q = List.equal Nx_device.equal (memories p) (memories q)

let disk p =
  match memories p with [ d ] -> Nx_device.equal d Nx_device.disk | _ -> false

let on_disk x = disk (Nx.placement x)

(* Where an operation reads [x]: where it lies, or the host for a value on the
   disk. *)
let home x = if on_disk x then Placement.host else Nx.placement x

(* Traced values *)

(* A value made beside one at [p] is a full copy on each of its devices. *)
let context p =
  if Placement.equal p Placement.host then Placement.host
  else Placement.replicated (Placement.devices p)

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
      Ops.shard ~axis:a whole (List.map (name s) (memories p))

let traced s p dt u =
  Repr.Traced.v ~context:(context p) p dt
    (Array.of_list (Ops.max_shape u))
    (Uop (s, u))

let traces s x =
  match Repr.v x with
  | Repr.Traced t -> (
      match Repr.Traced.node t with Uop (s', _) -> s' == s | _ -> false)
  | Repr.Host _ | Repr.Placed _ -> false

let is_traced x =
  match Repr.v x with
  | Repr.Traced t -> (
      match Repr.Traced.node t with Uop _ -> true | _ -> false)
  | Repr.Host _ | Repr.Placed _ -> false

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

let span dt v =
  let lo, hi = View.extent v in
  let per = Int.max 1 (8 * alignment / bits dt) in
  let start = lo - (lo mod per) in
  (start, hi - start)

let within dt v =
  let start, _ = span dt v and shape = View.shape v in
  View.create
    ~offset:(View.offset v - start)
    ~strides:
      (Array.mapi (fun d s -> if shape.(d) = 1 then 0 else s) (View.strides v))
    shape

let run dt v b =
  let start, span = span dt v in
  Nx_device.Buffer.view b
    ~offset:(start * bits dt / 8)
    (Nx_device.Buffer.dtype b) span

let phase dt b start =
  if Nx_device.equal (Nx_device.Buffer.device b) Nx_device.disk then 0
  else
    let at = Nx_device.Buffer.address b in
    Nativeint.(to_int (rem (add at (of_int (start * bits dt / 8))) 16n))

(* The phase of element [start] of [bufs], the buffers of one placement: the
   same for every buffer. *)
let phase_of what dt bufs start =
  match
    List.sort_uniq Int.compare (List.map (fun b -> phase dt b start) bufs)
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

(* [buffer ?slot ?phase d dt tdt n] is a buffer on [d] of [n] elements of [dt],
   which a graph computes as [tdt]: their bytes when [dt] is packed. *)
let buffer ?slot ?phase d dt tdt n =
  if packed dt then Ops.new_buffer ?slot ?phase d (bytes dt n) Uint8
  else Ops.new_buffer ?slot ?phase d n tdt

(* [elements dt b] is the flat node of the elements of [dt] that the buffer [b]
   holds, past the last one for a packed [dt] up to its last byte. *)
let elements dt b = if packed dt then unpack dt b else b

(* [storage_view s what dt b p shape v start] is [viewed] over the elements of
   [dt] that the buffer [b] holds, which [s] knows the bytes of where the view
   reads them whole: C-contiguous from a byte, on one device or a copy on
   each. *)
let storage_view s what dt b p shape v start =
  let u = viewed what (elements dt b) p shape v start in
  let first = (View.offset v - start) * bits dt in
  (match layout what p shape with
  | (One | Copies) when packed dt && View.is_c_contiguous v && first mod 8 = 0
    ->
      let n = bytes dt (View.numel v) in
      let whole = Ops.max_numel b = n && first = 0 in
      Ops.Tbl.replace s.bytes u
        (if whole then b
         else
           Ops.shrink b
             [ Some (Ops.Int (first / 8), Ops.Int ((first / 8) + n)) ])
  | One | Copies | Split _ -> ());
  u

let param s ~slot x =
  let what = "an argument" in
  let p = Nx.placement x and dt = Nx.dtype x in
  let tdt = check s what p dt in
  let bufs, v = Nx.shards x in
  let shape = Nx.shape x in
  let u =
    if View.numel v = 0 then
      broadcast (Ops.const ~dtype:tdt (`Int Bigint.zero)) shape
    else
      let start, span = span dt v in
      let phase = phase_of what dt bufs start in
      let buffer = buffer ~slot ~phase (device_of s p) dt tdt span in
      s.arguments <- buffer :: s.arguments;
      storage_view s what dt buffer p shape v start
  in
  traced s p dt u

(* The view of the window each device of [p] holds of a value of [shape] in C
   order: every device holds a window of one shape. *)
let local p shape =
  let d = List.hd (Placement.devices p) in
  View.create (Array.map (fun (lo, hi) -> hi - lo) (Placement.window p shape d))

(* [laid s what storage p dt shape] is a value of [dt] and [shape] at [p] in C
   order over the node [storage d n tdt] makes of [n] elements on [d]: each
   device's window, starting on 16 bytes. *)
let laid s what storage p dt shape =
  let tdt = check s what p dt in
  let local = local p shape in
  viewed what (storage (device_of s p) (View.numel local) tdt) p shape local 0

type target = {
  node : Ops.t;
  byte_buffer : Ops.t option; (* A packed dtype's buffer. *)
  store : Ops.t -> Ops.t;
  stored : Ops.t list -> Ops.t;
}

(* Storage of one element a byte, viewed as [node]. *)
let elementwise node =
  {
    node;
    byte_buffer = None;
    store = (fun u -> Ops.store node u);
    stored = (fun stores -> Ops.after node stores);
  }

let scratch s p dt shape =
  elementwise
    (laid s "a value" (fun d n tdt -> Ops.new_buffer d n tdt) p dt shape)

let output s ~slot p dt shape =
  let what = "a result" in
  let tdt = check s what p dt and local = local p shape in
  let b = buffer ~slot (device_of s p) dt tdt (View.numel local) in
  let read b = viewed what (elements dt b) p shape local 0 in
  let node = read b in
  if not (packed dt) then elementwise node
  else
    (* Each device packs its own window. *)
    let store u =
      match layout what p shape with
      | One | Copies -> Ops.store b (pack dt u)
      | Split axis ->
          let window =
            Ops.shard_slice u axis
              (List.hd (Ops.device_range_src (Ops.device b)))
          in
          Ops.store (Ops.unshard b [ 0 ]) (Ops.unshard (pack dt window) [ 0 ])
    in
    {
      node;
      byte_buffer = Some b;
      store;
      stored = (fun stores -> read (Ops.after b stores));
    }

let view t = t.node
let store t u = t.store u
let stored t stores = t.stored stores

let regions t (w : write) =
  let into =
    (w.into, t.node)
    ::
    (match t.byte_buffer with
    | Some b -> [ (Ops.buf_uop w.into, b) ]
    | None -> [])
  in
  List.map
    (fun (r : Lower_index.region) ->
      (Ops.substitute ~calls:Skip ~pass:Fixed_point r.dest into, r.value))
    w.regions

let parameter s ~slot p dt shape =
  traced s p dt
    (laid s "a loop's value"
       (fun d n tdt -> Ops.param ~shape:[ Ops.Int n ] ~device:d slot tdt)
       p dt shape)

(* Rows

   A loop reads row [i] of a stacked input in place. A packed input whose rows
   are whole bytes passes its bytes, and the body unpacks its row; a row that
   starts within a byte is read from the input's elements, a byte each. *)

let whole dt shape =
  packed dt && Array.fold_left ( * ) 1 shape * bits dt mod 8 = 0

let stacked s dt row u =
  if not (whole dt row) then u
  else match Ops.Tbl.find_opt s.bytes u with Some b -> b | None -> pack dt u

let row s ~slot p dt shape =
  if not (whole dt shape) then parameter s ~slot p dt shape
  else
    traced s p dt
      (laid s "a loop's value"
         (fun d n _ ->
           elements dt
             (Ops.param ~shape:[ Ops.Int (bytes dt n) ] ~device:d slot Uint8))
         p dt shape)

let argument s ~slot p dt shape =
  if Array.fold_left ( * ) 1 shape = 0 then
    let tdt = check s "an argument" p dt in
    traced s p dt (broadcast (Ops.const ~dtype:tdt (`Int Bigint.zero)) shape)
  else
    let storage d n tdt =
      let b = Ops.new_buffer ~slot d n tdt in
      s.arguments <- b :: s.arguments;
      b
    in
    traced s p dt (laid s "an argument" storage p dt shape)

(* Scopes

   A traced value belongs to the scope that traced it. A value of another scope
   is an input while that scope traces, as when a compiled call's trace encloses
   the one that meets it, and has escaped once that trace is done; a constant
   belongs to no scope. *)

(* Whether [u] reads no storage: a constant, which every scope may read. *)
let constant u =
  not
    (List.exists
       (fun v ->
         match Ops.op v with
         | Op.Buffer | Op.Param | Op.Alloc | Op.After | Op.Store | Op.Call ->
             true
         | _ -> false)
       (Ops.toposort ~calls:Enter u))

let is_constant x =
  match Repr.v x with
  | Repr.Traced t -> (
      match Repr.Traced.node t with Uop (_, u) -> constant u | _ -> false)
  | Repr.Host _ | Repr.Placed _ -> false

let describe x =
  Format.asprintf "%a%a on %a" Nx.pp_dtype (Nx.dtype x) Nx.pp_shape (Nx.shape x)
    Placement.pp (Nx.placement x)

let uop : type a b. scope -> (a, b) Nx.t -> Ops.t =
 fun s x ->
  match Repr.v x with
  | Repr.Traced t -> (
      match Repr.Traced.node t with
      | Uop (s', u) when s' == s -> u
      | Uop (s', u) ->
          if s'.live || constant u then u
          else
            invalid_arg
              "Rune.jit: a traced value escaped the function that traced it; \
               return it from that function instead"
      | _ ->
          (* An operand of the trace that no installation inside it owns is a
             value a transformation around the call traced: the function
             captured it. *)
          invalid_arg
            (Printf.sprintf
               "Rune.jit: the function reads, through its closure, a value a \
                transformation tracks (%s). Pass it as an argument."
               (describe x)))
  | Repr.Host _ | Repr.Placed _ ->
      invalid_arg "not a value traced by a compiled function"

(* The engine's devices *)

let engine s = Tolk_engine.device (devices s)

(* Captures *)

let same_view v0 v1 =
  View.offset v0 = View.offset v1
  && View.shape v0 = View.shape v1
  && View.strides v0 = View.strides v1

(* Whether every element of a value of view [v] reads one element of its
   storage, as a scalar's and a broadcast scalar's do. *)
let single v =
  View.numel v > 0
  &&
  let lo, hi = View.extent v in
  hi - lo = 1

(* [scalar tdt x] is [x], a value of one element that is not traced, as a
   constant of dtype [tdt]: its element is read now. *)
let scalar tdt x =
  let shape = Nx.shape x in
  let first = Nx.item (List.map (fun _ -> 0) (Array.to_list shape)) x in
  broadcast (Ops.const ~dtype:tdt (const (Nx.dtype x) first)) shape

(* [bind s what p x] is the node of [x], a value at [p] that is not traced. *)
let bind : type a b. scope -> string -> Placement.t -> (a, b) Nx.t -> Ops.t =
 fun s what p x ->
  let dt = Nx.dtype x and shape = Nx.shape x in
  let tdt = check s what p dt in
  let key = key x and bufs, v = Nx.shards x in
  let owned =
    match key with
    | Host_buffer b -> not (Nx_device.Buffer.is_borrowed b)
    | Placed _ -> true
  in
  if View.numel v = 0 then
    broadcast (Ops.const ~dtype:tdt (`Int Bigint.zero)) shape
  else if single v && owned then scalar tdt x
  else
    let same c =
      same_storage c.storage key && same_view c.view v && Placement.equal c.at p
    in
    match List.find_opt same s.captures with
    | Some c -> c.node
    | None ->
        let start, span = span dt v in
        let phase = phase_of what dt bufs start in
        let buffer = buffer ~phase (device_of s p) dt tdt span in
        let node = storage_view s what dt buffer p shape v start in
        let buffers = List.map (run dt v) bufs in
        s.captures <-
          { storage = key; view = v; at = p; node; buffer; buffers }
          :: s.captures;
        node

(* [placed s what p x] is the node of [x], a value that is not traced, at [p]:
   where it lies if that is [p], placed there first otherwise. A value of one
   element that lies elsewhere is a constant read now: placing it would copy it
   to [p] only to read it back, and a copy waits for the work queued on [p], so
   a trace would stall behind the kernels of the calls before it. *)
let placed s what p x =
  if on_disk x then bind s what p (Nx.place p x)
  else if same_layout (Nx.placement x) p (Nx.shape x) then bind s what p x
  else if single (snd (Nx.shards x)) then scalar (check s what p (Nx.dtype x)) x
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
    || List.for_all (fun d -> supports s d dt) (memories q)
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
            let buffer =
              Ops.new_buffer (device_of s q) (Ops.max_numel c.buffer)
                (Ops.dtype c.buffer)
            in
            let buffers =
              List.map
                (fun d ->
                  let dst =
                    Nx_device.Buffer.create d
                      (Nx_device.Buffer.dtype src)
                      (Nx_device.Buffer.length src)
                  in
                  Nx_device.Buffer.copy ~src ~dst;
                  dst)
                (memories q)
            in
            let node =
              Ops.substitute ~calls:Skip ~pass:Fixed_point c.node
                [ (c.buffer, buffer) ]
            in
            s.captures <- { c with at = q; node; buffer; buffers } :: s.captures;
            (c.buffer, buffer)
      in
      Some
        (Ops.substitute ~calls:Skip ~pass:Fixed_point u (List.map moved !read))

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
      let u = uop s x in
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
  let u = uop s x and shape = Nx.shape x in
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

(* Conversions *)

(* [cast dt tdt u] is [u] converted to [dt], computed as [tdt]. A float held at
   the ends of [int4]'s or [uint4]'s range before it narrows to a byte, as a
   cast to a wider integer holds at that integer's own, so [9.] becomes [7]. *)
let cast : type a b. (a, b) Nx_dtype.t -> Dtype.t -> Ops.t -> Ops.t =
 fun dt tdt u ->
  let held lo hi =
    if not (Dtype.is_float (Ops.dtype u)) then u
    else
      let lo = Ops.const_like u (`Float lo)
      and hi = Ops.const_like u (`Float hi) in
      Ops.where (Ops.lt u lo) lo (Ops.where (Ops.lt hi u) hi u)
  in
  match dt with
  | Int4 -> modular dt (Lower_arith.cast tdt (held (-8.) 7.))
  | UInt4 -> modular dt (Lower_arith.cast tdt (held 0. 15.))
  | Float16 | Float32 | Float64 | BFloat16 | Float8_e4m3 | Float8_e5m2 | Int8
  | UInt8 | Int16 | UInt16 | Int32 | UInt32 | Int64 | UInt64 | Bool | Bit
  | Complex64 | Complex128 ->
      Lower_arith.cast tdt u

(* [bitcast src dt tdt u] is [u], of [src], read as [dt] computed as [tdt]. A
   packed dtype's elements are bits within bytes: a wider [dt] joins the
   elements along [u]'s last axis, and a narrower one splits each into a new
   last axis, through bytes where the other is wider than one. *)
let bitcast src dt tdt u =
  let sb = bits src and tb = bits dt in
  if sb >= 8 && tb >= 8 then Lower_arith.bitcast tdt u
  else if sb = tb then of_code dt (code src u)
  else if sb < tb && tb <= 8 then
    Ops.squeeze ~axis:(-1) (of_code dt (join src u (tb / sb)))
  else if sb < tb then Lower_arith.bitcast tdt (join src u (8 / sb))
  else if sb <= 8 then split dt (Ops.unsqueeze (code src u) (-1)) (sb / tb)
  else split dt (Lower_arith.bitcast Uint8 u) (8 / tb)

(* Packed writes

   A write into a packed value that the program consumes is stored by bytes,
   each by one writer: the bytes the written elements land in, each rebuilt from
   the value's elements there and the writes landing among them. A value whose
   bytes the scope does not know is stored whole. *)

(* The C-order strides of [shape]. *)
let strides shape =
  snd
    (List.fold_left
       (fun (k, acc) n -> (k * n, k :: acc))
       (1, []) (List.rev shape))

(* [laid_as u shape full] is [u] reshaped to [shape] and expanded to [full]. *)
let laid_as u shape full = Ops.expand (Ops.reshape u (ints shape)) (ints full)
let int64_like u n = Ops.const_like ~dtype:Int64 u (`Int (Bigint.of_int n))

(* The elements of [u] in C order, [8 / bits] to a row, zeros past the last. *)
let by_byte dt u =
  let per = 8 / bits dt and n = Ops.max_numel u in
  let m = (n + per - 1) / per in
  let flat = Ops.reshape u [ Ops.Int n ] in
  let flat =
    if m * per = n then flat
    else Ops.pad flat [ Some (Ops.Int 0, Ops.Int ((m * per) - n)) ]
  in
  Ops.reshape flat (ints [ m; per ])

(* The positions in C order of the elements of the [k] bytes [b], [[k; per]]. *)
let positions dt b =
  let per = 8 / bits dt and k = Ops.max_numel b in
  let rows = laid_as b [ k; 1 ] [ k; per ] in
  Ops.add
    (Ops.mul rows (int64_like rows per))
    (laid_as (Ops.arange ~dtype:Int64 per) [ 1; per ] [ k; per ])

(* [scattered s dt ~axis ~indices ~updates into] is the region of a scatter that
   sets [updates] at [indices] along [axis] of the packed [into]: the last
   update landing in a byte stores it, as the last update at a position stores
   an element, its [8 / bits] elements those of [into] with the last update at
   each. Its [k x k] comparisons of the [k] updates are chosen where they cost
   no more than the scatter's value. *)
let scattered s dt ~axis ~indices ~updates into =
  let shape = Ops.max_shape into and ishape = Ops.max_shape indices in
  let n = Ops.max_numel into and k = Ops.max_numel indices in
  match Ops.Tbl.find_opt s.bytes into with
  | Some bytes when k > 0 && k * k <= n * List.nth ishape axis ->
      let r = List.length shape and per = 8 / bits dt in
      let m = (n + per - 1) / per in
      let flat u = Ops.reshape u [ Ops.Int k ] in
      let coord d stride =
        let c =
          if d = axis then indices
          else
            Ops.expand
              (Lower_reduce.along r d
                 (Ops.arange ~dtype:Int64 (List.nth ishape d)))
              (ints ishape)
        in
        Ops.mul c (int64_like c stride)
      in
      let pos =
        match List.mapi coord (strides shape) with
        | c :: cs -> flat (List.fold_left Ops.add c cs)
        | [] -> invalid_arg "a scatter into a scalar"
      in
      let inside =
        let i = Ops.bitcast (flat indices) Uint64 in
        Ops.lt i
          (Ops.const_like ~dtype:Uint64 i
             (`Int (Bigint.of_int (List.nth shape axis))))
      in
      let key = Ops.div ~rounding:`Trunc pos (int64_like pos per) in
      let order = Ops.arange ~dtype:Int64 k in
      (* The last update landing in each byte. *)
      let pairs = [ k; k ] in
      let column u = laid_as u [ k; 1 ] pairs
      and row u = laid_as u [ 1; k ] pairs in
      let followed =
        Ops.rop
          (Ops.bitwise_and
             (Ops.eq (column key) (row key))
             (Ops.bitwise_and (Ops.lt (column order) (row order)) (row inside)))
          Op.Max [ 1 ]
      in
      let keep = Ops.bitwise_and inside (Ops.logical_not followed) in
      let at =
        Ops.maximum
          (Ops.minimum key (int64_like key (m - 1)))
          (int64_like key 0)
      in
      let old =
        Lower_index.gather 0 (laid_as at [ k; 1 ] [ k; per ]) (by_byte dt into)
      in
      (* Each element of a stored byte against every update. *)
      let lanes = [ k; per; k ] in
      let update u = laid_as u [ 1; 1; k ] lanes in
      let hit =
        Ops.bitwise_and
          (Ops.eq (update pos) (laid_as (positions dt at) [ k; per; 1 ] lanes))
          (update inside)
      in
      let latest =
        Ops.rop
          (Ops.where hit (update order) (int64_like (update order) (-1)))
          Op.Max [ 2 ]
      in
      let u = flat (Ops.expand updates (Ops.shape indices)) in
      let set =
        Lower_reduce.of_bits (Ops.dtype u)
          (Lower_reduce.pick
             (Ops.eq (update order) (laid_as latest [ k; per; 1 ] lanes))
             (Lower_reduce.bits (update u)))
      in
      let value = Ops.where (Ops.le (int64_like latest 0) latest) set old in
      [
        {
          Lower_index.dest =
            Ops.index bytes [ Ops.valid (Lower_reduce.clamped m key) keep ];
          value = Ops.reshape (join dt value per) (ints [ k ]);
        };
      ]
  | Some _ | None -> []

(* [windowed s dt into ~starts v] is the region of an update of the packed
   [into] by [v] at the window of corner [starts]: the run of bytes from the
   window's first element through its last, of one length wherever the window
   lies, its elements [v]'s inside the window and [into]'s outside. *)
let windowed s dt into ~starts v =
  let static =
    List.for_all
      (function Ops.Int _ -> true | Ops.Sym _ -> false)
      (Ops.shape v)
  in
  match Ops.Tbl.find_opt s.bytes into with
  | Some bytes when static && Ops.max_numel v > 0 ->
      let shape = Ops.max_shape into and window = Ops.max_shape v in
      let per = 8 / bits dt and n = Ops.max_numel into in
      let m = (n + per - 1) / per and steps = strides shape in
      let span =
        List.fold_left2 (fun a k st -> a + ((k - 1) * st)) 1 window steps
      in
      let w = Int.min m (((span + per - 1) / per) + 1) in
      let start d =
        Ops.cast
          (Ops.reshape
             (Ops.shrink (Ops.contiguous starts)
                [ Some (Ops.Int d, Ops.Int (d + 1)) ])
             [])
          Int64
      in
      let first =
        List.fold_left Ops.add
          (Ops.const ~dtype:Int64 (`Int Bigint.zero))
          (List.mapi
             (fun d st -> Ops.mul (start d) (int64_like (start d) st))
             steps)
      in
      let b =
        Ops.minimum
          (Ops.div ~rounding:`Trunc first (int64_like first per))
          (int64_like first (m - w))
      in
      let run = Some (Ops.Sym b, Ops.Sym (Ops.add b (Ops.int w))) in
      let q =
        positions dt
          (Ops.add (Ops.arange ~dtype:Int64 w) (laid_as b [ 1 ] [ w ]))
      in
      let full = [ w; per ] in
      (* Each element's offset from the window's corner along each axis. *)
      let offset d st =
        let c = Ops.div ~rounding:`Trunc q (int64_like q st) in
        let c =
          if d = 0 then c else Ops.fmod c (int64_like c (List.nth shape d))
        in
        Ops.sub c (laid_as (start d) [ 1; 1 ] full)
      in
      let offsets = List.mapi offset steps in
      let inside =
        List.fold_left2
          (fun acc o k ->
            let o = Ops.bitcast o Uint64 in
            Ops.bitwise_and acc
              (Ops.lt o
                 (Ops.const_like ~dtype:Uint64 o (`Int (Bigint.of_int k)))))
          (Ops.const_like ~dtype:Bool q (`Bool true))
          offsets window
      in
      let at =
        List.fold_left2
          (fun acc o st -> Ops.add acc (Ops.mul o (int64_like o st)))
          (int64_like q 0) offsets (strides window)
      in
      let vs = Ops.reshape v [ Ops.Int (Ops.max_numel v) ] in
      let read =
        Ops.reshape
          (Lower_reduce.take vs 0 (Ops.reshape at [ Ops.Int (w * per) ]))
          (ints full)
      in
      let value =
        Ops.where inside read (Ops.shrink (by_byte dt into) [ run; None ])
      in
      [
        {
          Lower_index.dest = Ops.shrink bytes [ run ];
          value = Ops.reshape (join dt value per) (ints [ w ]);
        };
      ]
  | Some _ | None -> []

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
    traced s p dt (settled s what p u)
  in
  let like x u = ret (Nx.dtype x) u in
  (* Arithmetic on [int4] and [uint4] computes at a byte and reduces. *)
  let wrapped x u = like x (modular (Nx.dtype x) u) in
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
  | Unary (k, x) -> wrapped x (Lower_arith.unary k (n x))
  | Binary (k, x, y) -> wrapped x (Lower_arith.binary k (n x) (n y))
  | Compare (k, x, y) -> ret Nx_dtype.bool (Lower_arith.compare k (n x) (n y))
  | Where (c, x, y) -> like x (Ops.where (n c) (n x) (n y))
  | Fma (a, b, c) -> wrapped a (Lower_arith.fma (n a) (n b) (n c))
  | Convert (Cast, dt, x) -> ret dt (cast dt (check s what p dt) (n x))
  | Convert (Bitcast, dt, x) ->
      ret dt (bitcast (Nx.dtype x) dt (check s what p dt) (n x))
  | Threefry (key, counter) ->
      let k = n key and c = n counter in
      (* A parameter is an argument of a called body: a staged loop's trip. *)
      let varies u =
        let slice = Ops.backward_slice_with_self ~calls:Skip u in
        Ops.op_in_backward_slice_with_self ~calls:Skip u [ Op.Param ]
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
      wrapped x (Lower_reduce.reduce k ~axes:(Array.to_list axes) (n x))
  | Scan (k, axis, x) -> wrapped x (Lower_reduce.scan k ~axis (n x))
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
      let dt = Nx.dtype x in
      let result, regions =
        Lower_index.scatter ~mode ~unique ~axis ~indices ~updates into
      in
      let result = modular dt result in
      let regions =
        match mode with
        | _ when not (packed dt) -> regions
        | `Set -> scattered s dt ~axis ~indices ~updates into
        | `Add | `Max | `Min -> []
      in
      like x (write ~into regions result)
  | Update (x, starts, v) ->
      let into = n x and starts = n starts and v = n v in
      let result = Lower_index.update into ~starts v in
      let regions =
        if packed (Nx.dtype x) then windowed s (Nx.dtype x) into ~starts v
        else Option.to_list (Lower_index.update_region into ~starts v)
      in
      like x (write ~into regions result)
  | Unfold { kernel_size; stride; dilation; padding; x } ->
      like x (Lower_index.unfold ~kernel_size ~stride ~dilation ~padding (n x))
  | Fold { output_size; kernel_size; stride; dilation; padding; x } ->
      wrapped x
        (Lower_index.fold ~output_size ~kernel_size ~stride ~dilation ~padding
           (n x))
  | Matmul (x, y) -> wrapped x (Lower_linalg.matmul (n x) (n y))
  | Cholesky { upper; x } -> like x (Lower_linalg.cholesky ~upper (factored x))
  | Qr { reduced; x } ->
      let q, r =
        Lower_linalg.qr ~device:(device_of s p) ~reduced (factored x)
      in
      (like x q, like x r)
  | Lu x ->
      let lu, pivots, perm = Lower_linalg.lu (factored x) in
      (like x lu, ret Nx_dtype.int64 pivots, ret Nx_dtype.int64 perm)
  | Svd { full_matrices; x } ->
      let u, sv, vt =
        Lower_linalg.svd ~device:(device_of s p) ~full_matrices (factored x)
      in
      (like x u, ret Nx_dtype.float64 sv, like x vt)
  | Solve_triangular { upper; transpose; unit_diag; a; b } ->
      like b
        (Lower_linalg.solve_triangular ~upper ~transpose ~unit_diag (factored a)
           (factored b))
  | Eigh { vectors; x } ->
      let w, v =
        Lower_linalg.eigh ~device:(device_of s p) ~vectors (factored x)
      in
      (ret Nx_dtype.float64 w, Option.map (like x) v)
  | Fft _ | Rfft _ | Irfft _ | Eig _ -> jit_error "cannot compile %s" what
  | Contiguous x -> like x (Ops.contiguous (n x))
  | Move (x, m) -> like x (move (node s what (home x) x) m)
  | Place (q, x) -> (
      match Repr.v x with
      | Repr.Traced _ -> like x (place s what (Nx.placement x) q x)
      | Repr.Host _ | Repr.Placed _ -> like x (placed s what q x))
  | Group { by; x } -> (
      match Repr.v x with
      | Repr.Traced _ ->
          jit_error
            "%s: cannot group a traced tensor inside jit, as the number of \
             groups is read; group outside the compiled function instead"
            by
      | Repr.Host _ | Repr.Placed _ -> Nx.Op.eval o)
  | Read { by; x } -> (
      match Repr.v x with
      | Repr.Traced _ ->
          jit_error
            "%s: cannot read the value of a traced tensor inside jit; return \
             it from the compiled function instead"
            by
      | Repr.Host _ | Repr.Placed _ -> Nx.Op.eval o)
  | Check { ok; data; fail } ->
      let traced_leaf (Nx.P x) =
        match Repr.v x with
        | Repr.Traced _ -> true
        | Repr.Host _ | Repr.Placed _ -> false
      in
      if not (List.exists traced_leaf (Nx.P ok :: data)) then Nx.Op.eval o
      else
        (* The first false element's index in C order, or the element count: the
           least of the indices, each where its element is false and the count
           elsewhere. Each leaf of [data] is read at that index, which reads a
           zero where every element holds. *)
        let shape = Nx.shape ok in
        let count = Array.fold_left ( * ) 1 shape in
        if count > 0 then begin
          (* The index lives where a reduction of [ok] over every axis would:
             placed as nx places one, a copy on each device of a split value.
             Each leaf is read where a reduction of it would live, from a copy
             of the index there. *)
          let reduced x =
            Nx.Op.placement
              (Reduce (Sum, Array.init (Array.length shape) Fun.id, x))
          in
          let at = reduced ok in
          let flat x = Ops.reshape (value s x) [ Ops.Int count ] in
          let i = Ops.arange ~dtype:Int64 count in
          let first =
            Lower_reduce.reduce Min ~axes:[ 0 ]
              (Ops.where (flat ok)
                 (Ops.const_like i (`Int (Bigint.of_int count)))
                 i)
          in
          let read (Nx.P x) =
            let dt = Nx.dtype x and q = reduced x in
            ignore (check s what q dt);
            let first = place s what at q (traced s at Nx_dtype.int64 first) in
            let e =
              Lower_reduce.take (flat x) 0 (Ops.reshape first [ Ops.Int 1 ])
            in
            Nx.P (traced s q dt (Ops.reshape e []))
          in
          ignore (check s what at Nx_dtype.int64);
          s.checks <-
            {
              first = traced s at Nx_dtype.int64 first;
              data = List.map read data;
              shape;
              fail;
            }
            :: s.checks
        end
