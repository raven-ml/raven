(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Just-in-time compilation as an effect handler over Nx operations.

   Tracing: the handler answers every intercepted operation with a fresh
   symbolic placeholder tensor of the result's shape and dtype, and records the
   corresponding node of a Tolk tensor graph in a side table keyed by tensor
   identity. Running the function once under the handler therefore turns its
   whole computation into a single graph.

   Compiling: the graph is lowered through Tolk's pipeline (allocations,
   scheduling, kernel codegen) into a compiled linear schedule with explicit
   parameters and owned constants. Compilation happens once per key: the device,
   the arguments' skeleton (every leaf's path and every report of their walks)
   and each leaf's dtype, shape and layout. A call with a new key retraces.

   Replaying: every call binds the current input leaves to the compiled
   program's buffers and runs the schedule. On the CPU device, contiguous inputs
   and captured constants are wrapped in place — kernels read the tensors' own
   memory — and outputs are computed straight into the returned tensors'
   storage. On other devices a placed input seeds the program's buffer directly,
   a view of part of a storage is read in place, and a host input is copied on
   every call. A capture placed on the device is bound; any other capture is
   uploaded once per closure. Captures are compile-time constants: mutating one
   between calls has unspecified visibility (the CPU wrapping may observe it,
   device copies never do).

   Reading the value of a traced tensor (for example [Nx.item] on a value that
   depends on the inputs) raises [Jit_error]: a compiled trace cannot branch on
   data. Operations Tolk cannot express (FFT, complex dtypes) raise [Jit_error]
   as well. Much of the rest of the eager backend lowers at trace time instead:
   many C-kernel operations — matmul and the linear-algebra factorizations among
   them — have no single Tolk Uop, so the tracer builds them as ordinary Tolk
   compositions, which compile for every Tolk device. The factorizations unroll
   into a number of steps fixed by the input shapes alone (see
   [Tolk_frontend.Linalg]); [Nx.solve] and [Nx.inv] compile through them, a
   singular system yielding infinities where the eager kernel raises. Threefry
   (the RNG primitive) compiles, but only when its key depends on the traced
   inputs: a constant key would burn one draw into the program and silently
   replay it on every call, so it raises [Jit_error] pointing at [Nx.Rng] key
   threading. *)

open Nx_effect
module F = Tolk_frontend
module U = Tolk_uop.Uop
module TD = Tolk_uop.Dtype
module ND = Nx_dtype
module NV = Nx_core.View

exception Jit_error of string

let err fmt = Printf.ksprintf (fun s -> raise (Jit_error s)) fmt

let unsupported op =
  err
    "Rune.jit: %s is not supported inside jit; move it outside the jitted \
     function"
    op

(* Dtypes *)

let tolk_dtype dt =
  match TD.of_scalar (ND.Scalar.of_dtype dt) with
  | Some tdt -> tdt
  | None -> unsupported ("a tensor of " ^ ND.to_string dt)

let scalar_of : type a b. (a, b) ND.t -> a -> F.Tensor.scalar =
 fun dt v ->
  match dt with
  | ND.Float16 -> F.Tensor.Sfloat v
  | ND.Float32 -> F.Tensor.Sfloat v
  | ND.Float64 -> F.Tensor.Sfloat v
  | ND.BFloat16 -> F.Tensor.Sfloat v
  | ND.Float8_e4m3 -> F.Tensor.Sfloat v
  | ND.Float8_e5m2 -> F.Tensor.Sfloat v
  | ND.Int8 -> F.Tensor.Sint v
  | ND.UInt8 -> F.Tensor.Sint v
  | ND.Int16 -> F.Tensor.Sint v
  | ND.UInt16 -> F.Tensor.Sint v
  | ND.Int4 -> F.Tensor.Sint v
  | ND.UInt4 -> F.Tensor.Sint v
  | ND.Int32 -> F.Tensor.Sint (Int32.to_int v)
  | ND.UInt32 -> F.Tensor.Sint (Int32.to_int v)
  | ND.Int64 -> F.Tensor.Sint64 v
  | ND.UInt64 -> F.Tensor.Sint64 v
  | ND.Bool -> F.Tensor.Sbool v
  | ND.Complex64 -> unsupported "a complex tensor"
  | ND.Complex128 -> unsupported "a complex tensor"

(* Whether [dev]'s programs can load, store and compute [dt], natively or by
   emulation. *)
let holds dev dt =
  match TD.of_scalar (ND.Scalar.of_dtype dt) with
  | Some tdt ->
      Tolk.Decomp_dtype.is_dtype_supported (Tolk.Device.renderer dev) tdt
  | None -> false

(* Identity-keyed tables over tensors, as in [Tensor_map]. *)
module Tbl = Hashtbl.Make (struct
  type t = Obj.t

  let equal = ( == )
  let hash = Hashtbl.hash
end)

type packed = Packed : ('a, 'b) ND.t * ('a, 'b) Nx_effect.t -> packed

(* Backends. Device instances live in the shared tolk registry, one per
   canonical name, so jit and the engine's multi-device schedules resolve the
   same instance. rune installs its own opener for every backend when it is
   initialised, before any device can be opened through it: the CPU is opened
   without aligned vector types, since host tensors need not be aligned, and
   unknown or unavailable names keep rune's error text. The list is the order in
   which the default device is probed. *)

let backends = [ "METAL"; "AMD"; "NV"; "CUDA"; "CPU" ]
let () = List.iter (fun b -> Tolk.Device.register b Jit_device.create) backends

let backend name =
  match String.index_opt name ':' with
  | Some i -> String.sub name 0 i
  | None -> name

let to_program dev = Tolk.Codegen.to_program dev (Tolk.Device.renderer dev)

(* Environment knobs, read when a jit closure is created (not at module
   initialization) so tests can toggle them with [Unix.putenv]. *)

let env_int name default =
  match Sys.getenv_opt name with
  | Some s -> ( match int_of_string_opt s with Some v -> v | None -> default)
  | None -> default

let jit_debug = lazy (env_int "RUNE_JIT_DEBUG" 0)

(* Transfer accounting. Cumulative byte counters for host-to-device and
   device-to-host copies made by compiled traces; the zero-copy CPU path moves
   no bytes and counts nothing. *)

type stats = {
  bytes_to_device : int;
  bytes_from_device : int;
  resident_bytes : int;
  reused_bytes : int;
}

let bytes_to_device = ref 0
let bytes_from_device = ref 0
let resident_bytes = ref 0
let reused_bytes = ref 0

let reset_stats () =
  bytes_to_device := 0;
  bytes_from_device := 0;
  reused_bytes := 0

(* Placements

   A compiled trace runs on a list of devices of one backend, and binds each
   input and output leaf at a placement over that list ([Nx.Placement]): on one
   device, that device; on several, a full copy on each or equal slices along a
   cut axis, in the list's order. The traced function always observes global
   shapes; placement is a property of the compiled signature. *)

(* The devices other than the host, by canonical name: each has one nx device
   value, whose engine is [engine] below, and one tolk device. *)

let by_name : (string, Nx.Device.t * Tolk.Device.t) Hashtbl.t = Hashtbl.create 4

(* The name of the tolk device that compiles and runs the host's programs. *)
let host_name = "CPU"

let tolk_device_of d =
  if d == Nx.Device.host then Tolk.Device.get host_name
  else
    match Hashtbl.find_opt by_name (Nx.Device.name d) with
    | Some (d', dev) when d' == d -> dev
    | _ -> invalid_arg ("Rune: " ^ Nx.Device.name d ^ " is not a rune device")

(* Whether [p] is over the devices [ds] a program runs on: the same devices, and
   a split value's slices in the program's order. *)
let over ds p =
  let dp = Nx.Placement.devices p in
  if Nx_effect.Grid.cuts p = [] then
    List.compare_lengths dp ds = 0 && List.for_all (fun d -> List.memq d ds) dp
  else List.equal ( == ) dp ds

(* How far a call's devices are decided: not at all (the default device, which a
   placed capture replaces), as a set (by copies, whose order a split capture
   may still fix), or in order (by a split value or [?devices]). Only a split
   value's order decides which slice lands where. *)
type decided = Guessed | Unordered | Ordered

(* The devices [p] decides for a program: a split value's in its order, a copy's
   as a set, listed in the order devices were opened. *)
let decided_by p =
  let ds = Nx.Placement.devices p in
  if Nx_effect.Grid.cuts p = [] then (List.sort Nx.Device.compare ds, Unordered)
  else (ds, Ordered)

let pp_devices ppf = function
  | [ d ] -> Nx.Device.pp ppf d
  | ds ->
      Format.fprintf ppf "[%a]"
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
           Nx.Device.pp)
        ds

(* A leaf's placement in a program over [ds]: its own, or a copy on each device
   for a host value. *)
let leaf_placement : type a b.
    Nx.Device.t list -> (a, b) Nx_effect.t -> Nx.Placement.t =
 fun ds x ->
  match x with Placed r -> r.r_placement | _ -> Nx.Placement.replicated ds

(* The extents of the slice each device holds of a value of [shape] at [p]. *)
let local_shape p shape =
  Nx_effect.extents
    (Nx.Placement.window p shape (List.hd (Nx.Placement.devices p)))

(* Resident storage

   Values held on a device are placed values ([Nx_effect.Placed]) whose storage
   is a list of tolk buffers, one per device of the placement. A compiled call's
   outputs on a device that does not share host memory are placed; so are
   uploads made by [Nx.place]. The storage belongs to the value's cell: a read
   copies the view's elements out and leaves it, replay seeds a compiled input
   with the buffer itself when the placement matches, and it is released when
   the cell is unreachable (by the finaliser the engine attaches) or consumed by
   a compiled call. *)

type store = {
  s_devices : Tolk.Device.t list; (* one per shard, placement order *)
  s_nbytes : int; (* summed across shards *)
  mutable s_bufs : Tolk.Device.Buffer.t list; (* [[]] once released or lent *)
  s_nolru : bool; (* bypasses the allocator's cache: a mapped file's upload *)
}

type Nx_effect.storage += Buffers of store

let store_of (c : Nx_effect.cell) =
  match c.state with Live (Buffers s) -> Some s | _ -> None

(* The tolk device of [d] and the buffer of [s] it holds. nx places a view only
   on devices that hold its storage. *)
let buffer_on s d =
  let dev = tolk_device_of d in
  match List.find_index (( == ) dev) s.s_devices with
  | Some k -> (dev, List.nth s.s_bufs k)
  | None ->
      failwith
        (Printf.sprintf
           "Rune: a value on %s views a storage that it does not hold"
           (Nx.Device.name d))

let account s sign = resident_bytes := !resident_bytes + (sign * s.s_nbytes)

(* Finalizers only record the store; buffers are released at the next safe point
   (a read, a placement or a replay), not mid-GC inside arbitrary device
   code. *)
let pending_release : store list ref = ref []

let release_store s =
  match s.s_bufs with
  | [] -> ()
  | bufs ->
      s.s_bufs <- [];
      account s (-1);
      (* Deallocation returns each buffer to its device's LRU pool, where only
         work queued after the kernels that use it can take it. A buffer that
         bypasses the pool returns to the system, so the work that may still
         read it is awaited first. A base buffer with a still-allocated
         transient view (a kernel-argument slice not yet collected) cannot be
         deallocated; those are reclaimed by the buffer's own GC finalizer
         instead. *)
      if s.s_nolru then List.iter Tolk.Device.synchronize s.s_devices;
      List.iter
        (fun buf ->
          try Tolk.Device.Buffer.deallocate buf with Invalid_argument _ -> ())
        bufs

(* Arenas

   A compiled program's planned intermediates are slices of arena buffers (see
   [held_buffers]). They live and die inside one call, and a device runs the
   calls of all programs in queue order, so the programs a device runs share its
   arenas: a program's [k]th arena is bound, at every call, to the device's
   [k]th shared buffer, on each of its devices, which grows to the largest arena
   bound to it. The buffer a slot outgrows is retired until its views are
   released and the device has finished the work that may use it. *)

let arenas : (Tolk.Device.t * (int, Tolk.Device.Buffer.t) Hashtbl.t) list ref =
  ref []

let retired_arenas : (Tolk.Device.t * Tolk.Device.Buffer.t) list ref = ref []

let free_retired_arenas () =
  retired_arenas :=
    List.filter
      (fun (dev, buf) ->
        Tolk.Device.Buffer.allocated_views buf > 0
        || begin
          Tolk.Device.synchronize dev;
          Tolk.Device.Buffer.deallocate buf;
          false
        end)
      !retired_arenas

(* Buffer views over a range of a value's storage that a finished call or a
   dropped program bound (see [seed_of]). They go at the next safe point, before
   any storage does: a base buffer with a view still allocated cannot be
   freed. *)
let pending_views : Tolk.Device.Buffer.t list ref = ref []

let drain_releases () =
  if !retired_arenas <> [] then free_retired_arenas ();
  (match !pending_views with
  | [] -> ()
  | views ->
      pending_views := [];
      List.iter Tolk.Device.Buffer.deallocate views);
  match !pending_release with
  | [] -> ()
  | stores ->
      pending_release := [];
      List.iter release_store stores

(* A query is a safe point too: retiring the collected values first keeps
   [resident_bytes] to the storage still reachable, instead of a figure that
   depends on when the GC last ran. *)
let stats () =
  drain_releases ();
  {
    bytes_to_device = !bytes_to_device;
    bytes_from_device = !bytes_from_device;
    resident_bytes = !resident_bytes;
    reused_bytes = !reused_bytes;
  }

(* The collection budget: device allocations since the last major collection,
   eager results and uploads included. The collector does not see device memory,
   so past the budget a major collection runs and the storage of the values it
   finds unreachable is released. *)
let resident_budget () =
  env_int "RUNE_JIT_RESIDENT_BUDGET" (4 * 1024 * 1024 * 1024)

let allocated = ref 0
let majors = ref 0

let collect () =
  Gc.major ();
  drain_releases ();
  majors := (Gc.quick_stat ()).major_collections;
  allocated := 0

(* The device's [k]th shared arena, of at least [nbytes] bytes. It bypasses the
   allocator's cache: an outgrown arena returns to the system. *)
let shared_arena dev k nbytes =
  let slots =
    match List.assq_opt dev !arenas with
    | Some slots -> slots
    | None ->
        let slots = Hashtbl.create 4 in
        arenas := (dev, slots) :: !arenas;
        slots
  in
  match Hashtbl.find_opt slots k with
  | Some buf when Tolk.Device.Buffer.nbytes buf >= nbytes -> buf
  | outgrown ->
      Option.iter
        (fun buf -> retired_arenas := (dev, buf) :: !retired_arenas)
        outgrown;
      let buf =
        Tolk.Device.create_buffer ~size:nbytes ~dtype:TD.int8
          ~spec:{ Tolk.Device.Buffer_spec.default with nolru = true }
          dev
      in
      (try Tolk.Device.Buffer.ensure_allocated buf
       with _ ->
         Gc.major ();
         drain_releases ();
         Tolk.Device.Buffer.ensure_allocated buf);
      Hashtbl.replace slots k buf;
      buf

(* How a program reads an input or constant from its node: from element [skip]
   on, the value's elements in C order, or a view of [strides] over the elements
   of storage it reaches (see [view_movement]). *)
type layout = { skip : int; strides : int array option }

let dense = { skip = 0; strides = None }

(* How a value seeds a program's input or constant (see [seed_of]): by the
   buffers of its storage, one per device of the program in its order, or by the
   range of them its view reaches, or by a copy. *)
type seed =
  | Whole of { cell : Nx_effect.cell; bufs : Tolk.Device.Buffer.t list }
  | Range of {
      cell : Nx_effect.cell;
      bufs : Tolk.Device.Buffer.t list;
      lo : int;
      span : int;
      layout : layout;
    }
  | Copy

(* Trace state *)

type input = {
  i_node : U.t;
      (* a BUFFER while tracing, its explicit PARAM after compilation *)
  i_place : Nx.Placement.t;
  i_bufs : Tolk.Device.Buffer.t list; (* one per device of the program *)
  i_dtype : string;
  i_numel : int;
}

type state = {
  st_id : int; (* the trace's own id, in its traced tensors *)
  st_device : Tolk.Device.t; (* the first of [st_devices], which compiles *)
  st_devices : Nx.Device.t list; (* where the program runs, in order *)
  st_decided : decided;
      (* how far the devices are decided: a capture placed elsewhere may move
         the program to its devices, or order them (see [Runs_on]) *)
  mutable refusal : exn option;
      (* the first reason the program cannot run on its device, raised once the
         function returns: raising inside a handler would drop the function's
         own cleanups *)
  st_takes_storage : int -> bool;
      (* the input positions whose storage replay may hand an output: consumed
         leaves of a program whose outputs are not in host memory *)
  st_ctx : Nx_effect.context;
  table : F.Tensor.t Tensor_map.Tbl.t;
      (* tensors with bytes (captures, constants made while tracing) -> tolk
         tensor *)
  captures : unit Tensor_map.Tbl.t; (* closure captures lifted into the trace *)
  input_tags : (int, int) Hashtbl.t;
      (* input buffer node tag -> traversal position *)
  mutable prefills : (U.t * U.t) list;
      (* a fresh buffer an indexed write lands in, and the input buffer node
         whose value it starts from (see [write_destination]) *)
  mutable consts : (U.t * Nx.Placement.t * packed) list;
      (* reverse order, each at its placement in the program *)
  bound : F.Tensor.t Tensor_map.Tbl.t; (* resident captures bound in place *)
  mutable bound_consts : (U.t * packed * seed) list;
  scan_stacks : (U.t * int * Nx.Placement.t) list Tbl.t;
      (* staged scans: the step record's identity -> the per-leaf carry-stack
         buffer nodes the forward loop wrote, with their row strides and the
         carry's placement in the loop, for the backward loop to read. The step
         record is shared between the forward staging and the tape-recorded
         backward thunk, and is fresh per [Rune.scan] call, so it identifies the
         scan. Identity-keyed: a structural table compares the record's closure
         on a hash collision. *)
  scan_closed : Nx.packed list Tbl.t;
      (* staged scans: the step record's identity -> the external inputs the
         forward staging observed the body reading (tensors it closes over), for
         the backward loop to accumulate their cotangents. Keyed as
         [scan_stacks]. *)
  mutable scan_collectors : tensor_hook list;
      (* active observers of the tensors a scan body reads, innermost first;
         consulted by [tolk_of] while a scan body is being traced *)
  mutable scan_writes : (U.t * U.t list ref) list;
  (* staged scans being traced: each carry slot's buffer node -> the buffers
     indexed writes into it land in (see [write_destination]) *)
  mutable scan_bodies : int;
      (* the staged scan bodies being traced, forward or backward: a body
         recomputes its step in the backward loop, so remat has nothing to do
         there *)
}

(* A polymorphic observer of the tensors flowing through [tolk_of]. *)
and tensor_hook = { hook : 'a 'b. ('a, 'b) Nx_effect.t -> unit }

let shape_of x = NV.shape (Nx_effect.view x)
let numel shape = Array.fold_left ( * ) 1 shape

(* A capture that decides the program's devices where nothing else did: placed
   elsewhere than the default device, or split over the program's devices in
   another order than copies listed them. The program runs at its placement
   instead. *)
exception Runs_on of Nx.Placement.t

(* The first refusal wins, except that a capture's placement replaces what the
   undecided devices refused: the program runs there instead. *)
let refuse st e =
  match (st.refusal, e) with
  | None, _ -> st.refusal <- Some e
  | Some (Runs_on _), _ -> ()
  | Some _, Runs_on _ when st.st_decided <> Ordered -> st.refusal <- Some e
  | Some _, _ -> ()

let check_dtype : type a b. state -> (a, b) ND.t -> string -> unit =
 fun st dt what ->
  if not (holds st.st_device dt) then
    refuse st
      (Jit_error
         (Printf.sprintf "Rune.jit: %s is %s, which %s cannot hold" what
            (ND.to_string dt)
            (Tolk.Device.name st.st_device)))

(* A captured value lives on the program's devices or on the host, and was not
   consumed. *)
let check_capture : type a b. state -> (a, b) Nx_effect.t -> unit =
 fun st x ->
  match x with
  | Placed { r_cell = { state = Consumed { path }; _ }; _ } ->
      refuse st
        (Invalid_argument
           (Printf.sprintf
              "Rune.jit: a captured value was consumed at %s in a compiled \
               call's arguments; capture the value the call returned"
              path))
  | Placed { r_placement = p; _ } -> (
      if not (over st.st_devices p) then
        match st.st_decided with
        | Guessed -> refuse st (Runs_on p)
        | Unordered
          when over st.st_devices
                 (Nx.Placement.replicated (Nx.Placement.devices p)) ->
            refuse st (Runs_on p)
        | Unordered | Ordered ->
            let q = Nx.Placement.replicated st.st_devices in
            refuse st
              (Invalid_argument
                 (Format.asprintf
                    "Rune.jit: a captured value is on %a and the program runs \
                     on %a; place it there, or on the host"
                    Nx.Placement.pp p Nx.Placement.pp q)))
  | _ -> ()

(* A traced tensor's payload: the trace that made it, its node, and where it
   lives in the program. *)
type Nx_effect.node +=
  | Node of { trace : int; tensor : F.Tensor.t; place : Nx.Placement.t }

let trace_counter = ref 0

(* Where the program runs: its device, or a copy on each of its devices. *)
let here st = Nx.Placement.replicated st.st_devices

(* A fresh traced tensor of [st]'s trace standing for [tt], of [tt]'s shape, at
   [place]. *)
let traced st place dt tt =
  check_dtype st dt "a value the function computes";
  let shape = Array.of_list (F.Tensor.shape tt) in
  Nx_effect.traced st.st_ctx dt shape
    (Node { trace = st.st_id; tensor = tt; place })

let is_traced = function Nx_effect.Traced _ -> true | _ -> false

(* Wrap a graph buffer node as a tolk tensor of [shape]. Buffers are 1-D on the
   graph; the reshape restores the logical shape. *)
let buffer_tensor node shape =
  F.Movement.reshape (F.Tensor.of_uop node) (Array.to_list shape)

(* Store [tt] into the flat buffer node [dst], flattening the value first. A
   multi-d value stored as is leaves the scheduler with a flat iteration range
   over the value's reshaped view, an index form the lowering cannot handle (the
   view survives into a rank-mismatched INDEX). *)
let store_flat dst n tt =
  U.store ~dst ~value:(F.Tensor.uop (F.Movement.reshape tt [ n ])) ()

(* The devices of [st]'s program, as tolk names them. *)
let program_device st =
  match st.st_devices with
  | [ _ ] -> U.Single (Tolk.Device.name st.st_device)
  | ds -> U.Multi (List.map Nx.Device.name ds)

(* A buffer of [n] elements on every device of the program. *)
let make_node st dtolk n =
  U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:dtolk
    ~shape:(F.Tensor.shape_uop [ n ]) ~device:(program_device st) ()

(* [local], each device's slice of a value at [p], as the whole value: under
   tolk's [Unshard] when [p] cuts an axis. *)
let whole_tensor p local =
  match Nx_effect.Grid.cuts p with
  | [] -> local
  | [ (axis, _) ] ->
      F.Tensor.of_uop (U.unshard ~src:(F.Tensor.uop local) ~axes:[ axis ] ())
  | _ -> unsupported "a value cut along several axes"

(* [node], a buffer of each device's slice of a value of [shape] at [p], as the
   value. *)
let placed_tensor p node shape =
  whole_tensor p (buffer_tensor node (local_shape p shape))

(* Store [tt], a value of [shape] at [p], into [dst], a buffer of each device's
   slice: flat, or for a split value at its shape, which each device stores as
   its slice's. *)
let store_placed p dst shape tt =
  match Nx_effect.Grid.cuts p with
  | [] -> store_flat dst (numel shape) tt
  | _ ->
      U.store
        ~dst:(F.Tensor.uop (placed_tensor p dst shape))
        ~value:(F.Tensor.uop tt) ()

(* The placement of one row of a stack at [p], whose leading axis is whole, and
   of a stack of rows at [p]. *)
let row p = Nx_effect.Grid.map_axes (fun a -> a - 1) p
let stacked p = Nx_effect.Grid.map_axes (fun a -> a + 1) p

(* Where a constant of the program lives in it: a placed value on the program's
   devices keeps its placement, and any other is copied to each device. *)
let const_placement : type a b. state -> (a, b) Nx_effect.t -> Nx.Placement.t =
 fun st x ->
  match x with
  | Placed { r_placement = p; _ } when over st.st_devices p -> p
  | _ -> Nx.Placement.replicated st.st_devices

(* Where [x] lives in the program: a traced value where its operation put it, a
   constant where it is bound. *)
let placement_in : type a b. state -> (a, b) Nx_effect.t -> Nx.Placement.t =
 fun st x ->
  match x with
  | Nx_effect.Traced { t_node = Node { place; _ }; _ } -> place
  | _ -> const_placement st x

(* Bind a tensor whose bytes exist outside the traced computation (a closure
   capture, or a host constant created while tracing) as a compile-time
   constant: a buffer input aliasing the tensor's memory when the device can
   share it, uploaded once when the trace compiles otherwise, each device its
   slice. *)
let lift_const (type a b) st (x : (a, b) Nx_effect.t) : F.Tensor.t =
  let dt = Nx_effect.dtype x in
  check_dtype st dt "a constant of the function";
  let p = const_placement st x in
  let local = local_shape p (shape_of x) in
  let node = make_node st (tolk_dtype dt) (numel local) in
  st.consts <- (node, p, Packed (dt, x)) :: st.consts;
  let tt = whole_tensor p (buffer_tensor node local) in
  Tensor_map.Tbl.replace st.table (Key x) tt;
  tt

(* The range of storage elements [v] reaches, [lo] to [hi] exclusive. *)
let extent v =
  let strides = NV.strides v in
  let lo = ref (NV.offset v) and hi = ref (NV.offset v) in
  Array.iteri
    (fun d n ->
      let span = strides.(d) * (n - 1) in
      if span < 0 then lo := !lo + span else hi := !hi + span)
    (NV.shape v);
  (!lo, !hi + 1)

(* Placed views

   A value placed on a program's devices seeds the program without a copy: its
   buffers when its view covers its storage, otherwise the range of storage its
   view reaches on each device, bound as a buffer view from a 16-byte boundary
   (see [alignment]). A split value's view is each slice's, so the range is the
   same on every device. A C-order window is that range as it is; any other view
   is movement over the range, which the program applies, when its axes nest. A
   value that seeds neither way is copied. *)

(* The axes a view steps along, by decreasing stride. *)
let stepped_axes shape strides =
  List.init (Array.length shape) Fun.id
  |> List.filter (fun d -> shape.(d) > 1 && strides.(d) <> 0)
  |> List.stable_sort (fun a b ->
      Int.compare (Int.abs strides.(b)) (Int.abs strides.(a)))

(* Whether each stepped axis's stride is a multiple of the next one's and
   reaches past its extent: the view is then a cut of a C-order layout of its
   range, seen through a permutation, flips and broadcasts. Overlapping windows
   are not. *)
let nests shape strides =
  let rec go = function
    | a :: (b :: _ as rest) ->
        let sa = Int.abs strides.(a) and sb = Int.abs strides.(b) in
        sa mod sb = 0 && sa >= sb * shape.(b) && go rest
    | _ -> true
  in
  go (stepped_axes shape strides)

(* The view of [shape] and [strides] over [flat], its [span]-element range: the
   range padded to whole rows of the largest stride, reshaped into the nested
   layout, cut to the view's extents, then permuted, flipped and broadcast into
   place. *)
let view_movement flat ~span shape strides =
  let axes = stepped_axes shape strides in
  let base = Array.mapi (fun d n -> if strides.(d) = 0 then 1 else n) shape in
  let t =
    match axes with
    | [] -> F.Movement.shrink flat [ (0, 1) ]
    | a0 :: _ ->
        let s0 = Int.abs strides.(a0) in
        let rows = (span + s0 - 1) / s0 in
        let flat =
          if rows * s0 = span then flat
          else F.Movement.pad flat [ (0, (rows * s0) - span) ]
        in
        let rec inner = function
          | a :: (b :: _ as rest) ->
              (Int.abs strides.(a) / Int.abs strides.(b)) :: inner rest
          | [ a ] -> [ Int.abs strides.(a) ]
          | [] -> []
        in
        let t = F.Movement.reshape flat (rows :: inner axes) in
        let t =
          F.Movement.shrink t
            (List.map (fun a -> (0, base.(a))) axes @ [ (0, 1) ])
        in
        let t = F.Movement.reshape t (List.map (fun a -> base.(a)) axes) in
        let position a =
          let rec find i = function
            | x :: rest -> if x = a then i else find (i + 1) rest
            | [] -> assert false
          in
          find 0 axes
        in
        F.Movement.permute t (List.map position (List.sort Int.compare axes))
  in
  let t = F.Movement.reshape t (Array.to_list base) in
  let flipped =
    List.filter
      (fun d -> strides.(d) < 0 && shape.(d) > 1)
      (List.init (Array.length shape) Fun.id)
  in
  let t = if flipped = [] then t else F.Movement.flip t flipped in
  F.Movement.expand t (Array.to_list shape)

(* Kernels may load 16 bytes at a time from where a buffer starts, and CUDA and
   NV fault on a vector load from an address that is not a multiple of its width
   (Metal leaves it undefined). A range is therefore bound from the element at
   or below it whose byte offset is a multiple of 16, and the program skips the
   elements in front of it: those skipped join the cache key, so ranges whose
   offsets differ by a multiple of 16 bytes share a program. *)
let alignment = 16

(* [seed_of devs x] seeds [x] in a program over [devs]. A value whose view
   covers its storage on every device that holds it seeds whole; a view of one
   slice of a split storage, which covers that slice alone, seeds as a range. *)
let seed_of : type a b. Tolk.Device.t list -> (a, b) Nx_effect.t -> seed =
 fun devs x ->
  match x with
  | Placed r -> (
      match store_of r.r_cell with
      | Some s when s.s_bufs <> [] -> (
          let holder dev =
            Option.map (List.nth s.s_bufs)
              (List.find_index (( == ) dev) s.s_devices)
          in
          match List.map holder devs with
          | bufs when List.mem None bufs -> Copy
          | bufs ->
              let bufs = List.map Option.get bufs in
              let v = r.r_view in
              let range lo n strides =
                let item =
                  TD.itemsize (Tolk.Device.Buffer.dtype (List.hd bufs))
                in
                let per = Int.max 1 (alignment / item) in
                let skip = lo mod per in
                Range
                  {
                    cell = r.r_cell;
                    bufs;
                    lo = lo - skip;
                    span = n + skip;
                    layout = { skip; strides };
                  }
              in
              if
                Nx_effect.covers r
                && List.compare_lengths
                     (Nx.Placement.devices r.r_placement)
                     s.s_devices
                   = 0
              then Whole { cell = r.r_cell; bufs }
              else if
                NV.numel v = 0
                || not (List.for_all Tolk.Device.Buffer.supports_offset bufs)
              then Copy
              else if NV.is_c_contiguous v then
                range (NV.offset v) (NV.numel v) None
              else if nests (NV.shape v) (NV.strides v) then
                let lo, hi = extent v in
                range lo (hi - lo) (Some (NV.strides v))
              else Copy)
      | _ -> Copy)
  | _ -> Copy

(* [buf]'s elements [lo] to [lo + span], as a buffer of its own. *)
let buffer_range buf ~lo ~span =
  let dtype = Tolk.Device.Buffer.dtype buf in
  let v =
    Tolk.Device.Buffer.view buf ~size:span ~dtype
      ~offset:(lo * TD.itemsize dtype)
  in
  Tolk.Device.Buffer.ensure_allocated v;
  v

let layout_of = function
  | Range { layout; _ } -> layout
  | Whole _ | Copy -> dense

(* The elements of a node read under [layout], for a value of [shape]. *)
let layout_size layout shape =
  match layout.strides with
  | None -> layout.skip + numel shape
  | Some strides ->
      let lo, hi = extent (NV.create ~strides shape) in
      layout.skip + (hi - lo)

let layout_tensor node layout shape =
  if layout = dense then buffer_tensor node shape
  else
    let size = layout_size layout shape in
    let flat = buffer_tensor node [| size |] in
    let range =
      if layout.skip = 0 then flat
      else F.Movement.shrink flat [ (layout.skip, size) ]
    in
    match layout.strides with
    | None -> F.Movement.reshape range (Array.to_list shape)
    | Some strides ->
        view_movement range ~span:(size - layout.skip) shape strides

(* A capture placed on the program's devices keeps its storage, a split one
   included: the program reads it as the constant and no bytes move. The
   compiled record keeps the value reachable and counts the binding on its cell,
   so its buffers stay while the program lives, even once a call consumes the
   storage. *)
let bind_const (type a b) st seed (x : (a, b) Nx_effect.t) : F.Tensor.t =
  let dt = Nx_effect.dtype x in
  check_dtype st dt "a constant of the function";
  let p = const_placement st x in
  let local = local_shape p (shape_of x) in
  let layout = layout_of seed in
  let node = make_node st (tolk_dtype dt) (layout_size layout local) in
  st.bound_consts <- (node, Packed (dt, x), seed) :: st.bound_consts;
  let tt = whole_tensor p (layout_tensor node layout local) in
  Tensor_map.Tbl.replace st.bound (Key x) tt;
  tt

(* A tensor entering the trace without a table entry is a closure capture. *)
let tolk_of : type a b. state -> (a, b) Nx_effect.t -> F.Tensor.t =
 fun st x ->
  (match st.scan_collectors with
  | [] -> ()
  | fs -> List.iter (fun h -> h.hook x) fs);
  match x with
  | Nx_effect.Traced { t_node = Node { trace; tensor; _ }; _ }
    when trace = st.st_id ->
      tensor
  | Nx_effect.Traced _ ->
      err
        "Rune.jit: a tensor traced by another jit entered this trace; a value \
         computed inside a jitted function exists outside it only as an output"
  | _ -> (
      check_capture st x;
      match seed_of (List.map tolk_device_of st.st_devices) x with
      | (Whole _ | Range _) as seed -> (
          match Tensor_map.Tbl.find_opt st.bound (Key x) with
          | Some t -> t
          | None -> bind_const st seed x)
      | Copy -> (
          match Tensor_map.Tbl.find_opt st.table (Key x) with
          | Some t -> t
          | None ->
              Tensor_map.Tbl.replace st.captures (Key x) ();
              lift_const st x))

(* Composed operations Tolk has no primitive for. *)

(* Round half away from zero (C [round]); Tolk's round is half to even. *)
let round_away t =
  let open F.Elementwise in
  let half = F.Creation.const_like t (F.Tensor.Sfloat 0.5) in
  trunc (add t (copysign half t))

let atan2_graph y x =
  let open F.Elementwise in
  let zero t = F.Creation.const_like t (F.Tensor.Sfloat 0.) in
  let z = atan (div y x) in
  let pi = F.Creation.const_like z (F.Tensor.Sfloat Float.pi) in
  add z (where (lt x (zero x)) (where (ge y (zero y)) pi (neg pi)) (zero z))

(* A running maximum or minimum is NaN from the first NaN on. Tolk's scan keeps
   the larger operand by comparison, which a NaN never wins. *)
let nan_from_first ~axis t scanned =
  let seen = fst (F.Op.cummax ~axis (F.Elementwise.isnan t)) in
  F.Elementwise.where seen
    (F.Creation.const_like scanned (F.Tensor.Sfloat Float.nan))
    scanned

(* A float narrower than float32. Its products and its sums compute at float32
   and round once, as eager's do. *)
let narrow_float t =
  let d = F.Tensor.dtype t in
  TD.is_float d && TD.itemsize d < 4

(* A float as the integer of its width, and the float it was read from: an 8-bit
   float is read as its float16 widening. Where 8-bit floats are emulated,
   bitcasting one re-encodes it from a wider value, which saturates infinities.
   The widening is exact except that an emulated 8-bit float reads its
   subnormals as zero, as every compiled 8-bit float operation does. *)
let float_bits x =
  let x =
    if TD.itemsize (F.Tensor.dtype x) = 1 then F.Dtype_ops.cast x TD.float16
    else x
  in
  let int =
    match TD.itemsize (F.Tensor.dtype x) with
    | 2 -> TD.int16
    | 4 -> TD.int32
    | _ -> TD.int64
  in
  (x, F.Dtype_ops.bitcast x int)

(* Integers that order like [x], and the map from such integers back to values.
   Tolk's comparisons never let a NaN win, and its sort recovers positions by
   matching values for equality, which a NaN never satisfies. A float orders as
   its bits (see [float_bits]), with a negative value's magnitude bits flipped
   so that the larger float is the larger integer. -0 is read as +0 so that
   equal zeros tie ([`Tied]), or orders below +0 ([`Ordered]). Every NaN,
   recognised on its bits (magnitude above infinity's), takes the greatest
   integer ([`Greatest]) or the least ([`Least]); no number takes either. The -0
   test compares bits: a float comparison may flush subnormals to zero. A
   non-float [x] is its own key. *)
let order_keys ~nan ~zeros x =
  let dtype = F.Tensor.dtype x in
  if not (TD.is_float dtype) then (x, Fun.id)
  else
    let open F.Elementwise in
    let x, bits = float_bits x in
    let int = F.Tensor.dtype bits in
    let bound v = F.Tensor.of_uop (U.const v) in
    let max = bound (Tolk_uop.Const.max_value int) in
    let nan_key =
      match nan with
      | `Greatest -> max
      | `Least -> bound (Tolk_uop.Const.min_value int)
    in
    let flip bits =
      where
        (lt bits (F.Creation.const_like bits (F.Tensor.Sint 0)))
        (bitwise_xor bits max) bits
    in
    let zeros =
      match zeros with
      | `Tied ->
          where
            (eq bits (bound (Tolk_uop.Const.min_value int)))
            (F.Creation.const_like bits (F.Tensor.Sint 0))
            bits
      | `Ordered -> bits
    in
    let values keys =
      F.Dtype_ops.cast
        (where (eq keys nan_key)
           (F.Creation.const_like ~dtype:(F.Tensor.dtype x) keys
              (F.Tensor.Sfloat Float.nan))
           (F.Dtype_ops.bitcast (flip keys) (F.Tensor.dtype x)))
        dtype
    in
    let infinity =
      match F.Tensor.dtype x with
      | TD.Float16 -> 0x7c00L
      | TD.Bfloat16 -> 0x7f80L
      | TD.Float32 -> Int64.of_int32 (Int32.bits_of_float Float.infinity)
      | TD.Float64 -> Int64.bits_of_float Float.infinity
      | dt -> invalid_arg ("Rune.jit: no order keys for " ^ TD.to_string dt)
    in
    let nan_bits =
      gt (bitwise_and bits max) (bound (Tolk_uop.Const.int64 int infinity))
    in
    (where nan_bits nan_key (flip zeros), values)

(* A running maximum or minimum of [t] along [axis], as eager's: NaN from the
   first NaN on, and the first of equal values, so that -0 and +0 keep their
   order. Where int64 is native and a key of at most 32 bits leaves room for the
   positions, one int64 scan orders the key, offset to be non-negative, above
   the position, the earlier first, and the sign of a zero in the lowest bit:
   the packed integer stays non-negative, since C and Metal leave a shift of a
   negative integer undefined. The scanned high bits map back to the value and
   the lowest restores a zero's sign. A float64 key, a device without int64 or a
   longer axis scans the values and then marks NaN from its first occurrence in
   a second scan. *)
let running ~packs ~axis ~op t =
  let scan = match op with `Max -> F.Op.cummax | `Min -> F.Op.cummin in
  let dtype = F.Tensor.dtype t in
  let n = List.nth (F.Tensor.shape t) axis in
  let keys, values =
    order_keys
      ~nan:(match op with `Max -> `Greatest | `Min -> `Least)
      ~zeros:`Tied t
  in
  let key_bits = TD.bitsize (F.Tensor.dtype keys) in
  let shift = 63 - key_bits in
  if not (TD.is_float dtype) then fst (scan ~axis t)
  else if not (packs && key_bits <= 32 && 2 * n <= 1 lsl shift) then
    nan_from_first ~axis t (fst (scan ~axis t))
  else
    let open F.Elementwise in
    let int t v = F.Creation.const_like t (F.Tensor.Sint v) in
    let key_dtype = F.Tensor.dtype keys in
    let least =
      F.Tensor.of_uop (U.const (Tolk_uop.Const.min_value key_dtype))
    in
    let wide, bits = float_bits t in
    let ranks =
      F.Movement.reshape
        (F.Op.arange ~dtype:TD.int64 n)
        (List.mapi (fun i _ -> if i = axis then n else 1) (F.Tensor.shape t))
    in
    let first =
      match op with `Max -> sub (int ranks (n - 1)) ranks | `Min -> ranks
    in
    let low =
      bitwise_or
        (lshift first (int first 1))
        (F.Dtype_ops.cast (eq bits least) TD.int64)
    in
    let offset = 1 lsl (key_bits - 1) in
    let high = add (F.Dtype_ops.cast keys TD.int64) (int low offset) in
    let scanned =
      fst (scan ~axis (bitwise_or (lshift high (int high shift)) low))
    in
    let key =
      F.Dtype_ops.cast
        (sub (rshift scanned (int scanned shift)) (int scanned offset))
        key_dtype
    in
    let signed_zero =
      bitwise_and
        (eq key (int key 0))
        (eq (bitwise_and scanned (int scanned 1)) (int scanned 1))
    in
    let minus_zero =
      F.Dtype_ops.cast (F.Dtype_ops.bitcast least (F.Tensor.dtype wide)) dtype
    in
    where signed_zero minus_zero (values key)

(* The greatest or least element of [t] over [axes]: NaN when any element is
   NaN, as eager's, and of -0 and +0 the greater for a maximum and the lesser
   for a minimum, as IEEE orders them, where eager keeps the first. One integer
   reduction over the keys (see [order_keys]) gives both, and keeps the order of
   subnormals that a float comparison flushes. *)
let extreme ~op ~axes t =
  let reduce, nan =
    match op with
    | `Max -> (F.Reduce.max, `Greatest)
    | `Min -> (F.Reduce.min, `Least)
  in
  if not (TD.is_float (F.Tensor.dtype t)) then
    reduce ~axis:axes ~keepdim:false t
  else
    let keys, values = order_keys ~nan ~zeros:`Ordered t in
    values (reduce ~axis:axes ~keepdim:false keys)

(* The keys a sort orders: NaN after every number in either direction, and equal
   zeros tied, so that a stable sort keeps their order. *)
let sort_keys ~descending x =
  order_keys ~nan:(if descending then `Least else `Greatest) ~zeros:`Tied x

(* Whether [st]'s device computes int64 natively, which the packed sort
   needs. *)
let packs st =
  Tolk.Renderer.supports_dtype (Tolk.Device.renderer st.st_device) TD.int64

let bit_length n =
  let rec go n acc = if n = 0 then acc else go (n lsr 1) (acc + 1) in
  go n 0

(* The stable positions that sort [x] along [dim]. Tolk's network sorts values
   and recovers each position by an n×n match of sorted values to inputs. Where
   int64 is native ([packs]), the network sorts integers that carry the position
   instead: a key of at most 32 bits in the high bits, offset to be non-negative
   since C and Metal leave a shift of a negative integer undefined, and the
   position in the low bits, complemented for a descending sort so that equal
   keys keep index order. Packed integers are distinct, so the network alone
   gives the stable order and the positions are its low bits; an int32 position
   leaves room for 32 key bits. A 64-bit key sorts as two such passes, least
   significant half first: the second pass sorts the high halves in the first
   pass's order, and being stable it keeps that order among equal high halves.
   The packed integers get a kernel of their own: fused into the padding of an
   axis that is not a power of two, the positions' arange no longer folds to an
   index and costs n^2 work. *)
let argsort_graph ~packs ~dim ~descending x =
  let keys, _ = sort_keys ~descending x in
  let key_dtype = F.Tensor.dtype keys in
  let shape = F.Tensor.shape x in
  let dim = if dim < 0 then dim + List.length shape else dim in
  let n = List.nth shape dim in
  let open F.Elementwise in
  let int t v = F.Creation.const_like t (F.Tensor.Sint v) in
  (* The stable positions of [high], non-negative int64 below 2^32. *)
  let positions high =
    let low_bits = bit_length (n - 1) in
    let low = (1 lsl low_bits) - 1 in
    let complement r = if descending then sub (int r low) r else r in
    let ranks =
      F.Movement.reshape
        (F.Op.arange ~dtype:TD.int64 n)
        (List.mapi (fun i _ -> if i = dim then n else 1) shape)
    in
    let packed =
      bitwise_or (lshift high (int high low_bits)) (complement ranks)
    in
    let sorted = fst (F.Op.sort ~dim ~descending (contiguous packed)) in
    complement (bitwise_and sorted (int sorted low))
  in
  let signed = not (TD.is_unsigned key_dtype || TD.is_bool key_dtype) in
  if not packs then snd (F.Op.sort ~dim ~descending keys)
  else if TD.bitsize key_dtype <= 32 then
    let offset = if signed then 1 lsl (TD.bitsize key_dtype - 1) else 0 in
    let wide = F.Dtype_ops.cast keys TD.int64 in
    positions (add wide (int wide offset))
  else
    let half = 1 lsl 32 in
    let lo =
      F.Dtype_ops.cast (bitwise_and keys (int keys (half - 1))) TD.int64
    in
    (* The high half by floor division, which never shifts a negative key. *)
    let hi = F.Dtype_ops.cast (floordiv keys (int keys half)) TD.int64 in
    let hi = if signed then add hi (int hi (half / 2)) else hi in
    let along p t = F.Op.gather t ~dim (F.Dtype_ops.cast p TD.int32) in
    let first = positions lo in
    along (positions (along first hi)) first

(* [x] sorted along [dim]. A float sort returns [x]'s elements at the stable
   positions that sort it, so a -0 or a NaN keeps its bits; mapped back from the
   keys, every zero would come back as +0 and every NaN as one NaN (see
   [order_keys]). Without native int64, recovering the positions costs n^2
   operations, so the keys map back instead. An integer is its own key. *)
let sort_graph ~packs ~dim ~descending x =
  if TD.is_float (F.Tensor.dtype x) && packs then
    F.Op.gather x ~dim
      (F.Dtype_ops.cast (argsort_graph ~packs ~dim ~descending x) TD.int32)
  else
    let keys, values = sort_keys ~descending x in
    values (fst (F.Op.sort ~dim ~descending keys))

(* Whether [u]'s graph reaches an input buffer node. Constants lifted during the
   trace (captures, host arrays) are buffers too, but only input nodes are in
   [st.input_tags]; a value that never touches one is a compile-time constant of
   the trace. *)
let depends_on_input st u =
  let seen = Hashtbl.create 32 in
  let rec go u =
    let tag = U.tag u in
    if Hashtbl.mem seen tag then false
    else begin
      Hashtbl.add seen tag ();
      (match U.op u with
        | Tolk_uop.Ops.Buffer ->
            Hashtbl.mem st.input_tags tag
            || List.exists (fun (b, _) -> b == u) st.prefills
        | _ -> false)
      || Array.exists go (U.src u)
    end
  in
  go u

(* The storage an indexed write of [t] lands in: tolk's write is in place, and a
   tensor is a value, so the write never lands in [t] itself. A computed [t] is
   computed into fresh storage. A [t] that is an input is not copied by the
   program at all: the write lands in an empty buffer, recorded in
   [st.prefills], and replay gives that buffer [t]'s value before the program
   runs, by handing it the input's consumed storage or by copying. A program
   whose replays never take storage copies [t] itself: a kernel does it faster
   than replay. A [t] that is a staged loop's carry is not copied either: the
   write lands in an empty buffer the loop fills (see [stage_scan]). The storage
   is at [place], [t]'s placement: a buffer of one slice per device when it is
   split. *)
let write_destination st t ~place =
  let u = F.Tensor.uop t in
  let dtype = U.commit_dtype u in
  let empty () =
    whole_tensor place
      (F.Creation.empty ~dtype ~device:(program_device st)
         (Array.to_list (local_shape place (Array.of_list (F.Tensor.shape t)))))
  in
  let carry_writes =
    if U.has_buffer_identity u then List.assq_opt (U.buf_uop u) st.scan_writes
    else None
  in
  match carry_writes with
  | Some writes ->
      (* A staged loop's carry: the loop gives the buffer the carry's value, by
         binding it to the carry's storage or by a copy (see [stage_scan]). *)
      let out = empty () in
      writes := U.buf_uop (F.Tensor.uop out) :: !writes;
      out
  | None ->
      let takes =
        U.has_buffer_identity u
        &&
        match Hashtbl.find_opt st.input_tags (U.tag (U.buf_uop u)) with
        | Some i -> st.st_takes_storage i
        | None -> false
      in
      if takes then begin
        let out = empty () in
        st.prefills <-
          (U.buf_uop (F.Tensor.uop out), U.buf_uop u) :: st.prefills;
        out
      end
      else
        let out = F.Tensor.uop (empty ()) in
        F.Tensor.of_uop
          (U.after ~src:out
             ~deps:[ U.store ~dst:out ~value:(U.cast ~src:u ~dtype) () ])

(* Threefry lowering. The trace-level operation hashes int32 (key, counter)
   pairs laid out as consecutive elements; Tolk's primitive mixes uint64
   counters with a uint64 key. Pack each pair low word first (element 0 is the
   low half, matching the C kernel's [v[0]]), apply the primitive, and unpack
   the two result halves back into consecutive int32 lanes. Tolk's decomposition
   (decomp_op.ml) computes the same 20-round Random123 function as the eager C
   kernel, so compiled draws are bit-identical to eager ones. *)
let threefry_graph key ctr =
  let open F.Elementwise in
  let shape = F.Tensor.shape key in
  let n = List.fold_left ( * ) 1 shape / 2 in
  let col t i =
    F.Movement.reshape
      (F.Movement.shrink (F.Movement.reshape t [ n; 2 ]) [ (0, n); (i, i + 1) ])
      [ n ]
  in
  let sint t v = F.Creation.const_like t (F.Tensor.Sint v) in
  let u64 t = F.Dtype_ops.cast (F.Dtype_ops.cast t TD.uint32) TD.uint64 in
  let pack t =
    let lo = u64 (col t 0) and hi = u64 (col t 1) in
    bitwise_or (lshift hi (sint hi 32)) lo
  in
  let bits = threefry (pack ctr) (pack key) in
  (* Narrowing to uint32 truncates, so the low word needs no mask. *)
  let i32 t = F.Dtype_ops.cast (F.Dtype_ops.cast t TD.uint32) TD.int32 in
  let lo = i32 bits and hi = i32 (rshift bits (sint bits 32)) in
  let lane t = F.Movement.reshape t [ n; 1 ] in
  F.Movement.reshape (F.Op.cat ~dim:1 (lane lo) [ lane hi ]) shape

(* Sliding windows. [unfold] is pad -> pool -> permute -> reshape: pure
   movement, so the extracted patches fuse into their consumer. [fold]
   (overlapping-add) has no movement expression; it becomes a scatter-add over
   the flattened padded spatial block driven by a precomputed index constant:
   entry [kf * nwin + w] is the flat padded position written by kernel offset
   [kf] of window [w]. Kernel offsets and windows both enumerate row-major,
   matching the eager implementation; contributions that land in the padding are
   shrunk away, matching its drop-padding semantics. *)

let window_indices ~spatial_padded ~kernel_size ~stride ~dilation =
  let k = Array.length kernel_size in
  let out_sp =
    Array.init k (fun d ->
        (spatial_padded.(d) - ((dilation.(d) * (kernel_size.(d) - 1)) + 1))
        / stride.(d)
        + 1)
  in
  let kernel_prod = numel kernel_size in
  let nwin = numel out_sp in
  let sp_strides = Array.make k 1 in
  for d = k - 2 downto 0 do
    sp_strides.(d) <- sp_strides.(d + 1) * spatial_padded.(d + 1)
  done;
  let idx = Nx_buffer.create ND.int32 (kernel_prod * nwin) in
  let k_pos = Array.make k 0 in
  let w_pos = Array.make k 0 in
  let bump pos limits =
    let rec go d =
      if d >= 0 then begin
        pos.(d) <- pos.(d) + 1;
        if pos.(d) = limits.(d) then begin
          pos.(d) <- 0;
          go (d - 1)
        end
      end
    in
    go (k - 1)
  in
  let p = ref 0 in
  for _kf = 0 to kernel_prod - 1 do
    Array.fill w_pos 0 k 0;
    for _w = 0 to nwin - 1 do
      let off = ref 0 in
      for d = 0 to k - 1 do
        off :=
          !off
          + ((w_pos.(d) * stride.(d)) + (k_pos.(d) * dilation.(d)))
            * sp_strides.(d)
      done;
      Nx_buffer.unsafe_set idx !p (Int32.of_int !off);
      incr p;
      bump w_pos out_sp
    done;
    bump k_pos kernel_size
  done;
  (kernel_prod, nwin, idx)

let no_padding = Array.for_all (fun (b, a) -> b = 0 && a = 0)

(* Lift the index constant and broadcast it over the leading dimensions. *)
let window_index_tensor st idx lead n =
  let it = tolk_of st (Nx_effect.from_host st.st_ctx idx) in
  let it = F.Movement.reshape it (List.map (fun _ -> 1) lead @ [ n ]) in
  F.Movement.expand it (lead @ [ n ])

let unfold_graph st t_in ~kernel_size ~stride ~dilation ~padding =
  let shape = shape_of t_in in
  let rank = Array.length shape in
  let kd = Array.length kernel_size in
  let nlead = rank - kd in
  let lead = Array.to_list (Array.sub shape 0 nlead) in
  let t = tolk_of st t_in in
  let t =
    if no_padding padding then t
    else
      F.Op.pad t
        (List.map (fun _ -> None) lead
        @ List.init kd (fun d -> Some padding.(d)))
  in
  let pooled =
    F.Movement.pool t
      ~k:(Array.to_list kernel_size)
      ~stride:(Array.to_list stride) ~dilation:(Array.to_list dilation) ()
  in
  (* (lead.., o.., k..) -> (lead.., k.., o..) -> (lead.., prod k, windows) *)
  let perm =
    List.init nlead Fun.id
    @ List.init kd (fun d -> nlead + kd + d)
    @ List.init kd (fun d -> nlead + d)
  in
  let pooled_shape = Array.of_list (F.Tensor.shape pooled) in
  let nwin = numel (Array.sub pooled_shape nlead kd) in
  F.Movement.reshape
    (F.Movement.permute pooled perm)
    (lead @ [ numel kernel_size; nwin ])

let fold_graph st t_in ~output_size ~kernel_size ~stride ~dilation ~padding =
  let shape = shape_of t_in in
  let rank = Array.length shape in
  let kd = Array.length kernel_size in
  let lead = Array.to_list (Array.sub shape 0 (rank - 2)) in
  let spatial_padded =
    Array.init kd (fun d ->
        let before, after = padding.(d) in
        output_size.(d) + before + after)
  in
  let kernel_prod, nwin, idx =
    window_indices ~spatial_padded ~kernel_size ~stride ~dilation
  in
  let src =
    F.Movement.reshape (tolk_of st t_in) (lead @ [ kernel_prod * nwin ])
  in
  let it = window_index_tensor st idx lead (kernel_prod * nwin) in
  let template =
    F.Creation.zeros
      ~dtype:(tolk_dtype (Nx_effect.dtype t_in))
      (lead @ [ numel spatial_padded ])
  in
  let scat =
    F.Op.scatter_reduce template ~dim:(List.length lead) it src ~reduce:`Sum
      ~include_self:true ()
  in
  let r = F.Movement.reshape scat (lead @ Array.to_list spatial_padded) in
  if no_padding padding then r
  else
    F.Movement.shrink r
      (List.map (fun d -> (0, d)) lead
      @ List.init kd (fun d ->
          let before, _ = padding.(d) in
          (before, before + output_size.(d))))

(* Staged scan.

   [Rune.scan] performs [Scan.E_scan]; this tracer stages it as a loop call in
   the compiled program instead of an unrolled trace. The body is traced once
   with fresh placeholder slot tensors, scheduled as its own compiled
   sub-program, and embedded as the payload of a CALL(CUSTOM_FUNCTION "loop")
   node — see [Tolk.Realize.exec_loop] for the payload encoding and the replay
   semantics. *)

(* The scheduling pipeline allocates internal kernel buffers from a counter
   seeded at the scheduled graph's maximum slot. That counter is local to the
   schedule: a fresh buffer must never collide with a live scheduled buffer
   (buffer identity is the slot), so advance the process-wide counter past every
   non-negative buffer slot the linear mentions. *)
let reserve_slots_of linear =
  U.toposort ~enter_calls:true linear
  |> List.iter (fun n ->
      match U.as_buffer n with
      | Some { buffer = { slot; _ }; _ } when slot >= 0 ->
          U.reserve_buffer_slots (slot + 1)
      | _ -> ())

(* Schedule the traced body sink unplanned, so slot identities remain available
   for the carry analysis. Its call parameters are substituted with argument
   nodes here; [loop_call] introduces the final local parameter scope before
   compiling kernels and queues. *)
let schedule_body_linear body_sink =
  let body_sink, buffer_map = Tolk.Bufferize.run body_sink in
  let resolve_node node = U.buf_uop
      (Option.value (Hashtbl.find_opt buffer_map (U.tag node)) ~default:node) in
  let body_call = Tolk.Callify.transform_to_call body_sink in
  let captured = ref None in
  Tolk.Realize.capturing :=
    (fun l v -> captured := Some (l, v)) :: !Tolk.Realize.capturing;
  Fun.protect
    ~finally:(fun () ->
      Tolk.Realize.capturing := List.tl !Tolk.Realize.capturing)
    (fun () ->
      ignore
        (Tolk.Schedule.create_linear_with_vars
           ~get_kernel_graph:Tolk.Rangeify.get_kernel_graph body_call));
  match !captured with
  | None -> err "Rune.jit: scan body scheduling captured no computation"
  | Some (body_linear, body_vars) ->
      if body_vars <> [] then
        err
          "Rune.jit: the scan body uses symbolic variables, which staged loops \
           do not support yet";
      let body_linear =
        match U.as_call body_call with
        | Some { args; _ } ->
            let mappings =
              U.toposort ~enter_calls:true body_linear
              |> List.filter_map (fun n ->
                  match U.as_param n with
                  | Some { param = { slot; _ }; _ }
                    when slot >= 0 && slot < List.length args ->
                      Some (n, List.nth args slot)
                  | _ -> None)
            in
            if mappings = [] then body_linear
            else U.substitute ~walk:true mappings body_linear
        | None -> assert false
      in
      reserve_slots_of body_linear;
      body_linear, resolve_node

(* Schedule analyses, shared by buffer reuse at the jit boundary and inside a
   staged loop's body. *)

(* An operation that keeps every element at its index. *)
let keeps_index u =
  let op = U.op u in
  Tolk_uop.Ops.Group.is_elementwise op
  && (op <> Tolk_uop.Ops.Cast
     || Array.length (U.src u) = 0
     || TD.itemsize (U.dtype u) = TD.itemsize (U.dtype (U.src u).(0)))
  || op = Tolk_uop.Ops.Reshape || op = Tolk_uop.Ops.Stage
  || op = Tolk_uop.Ops.Contiguous_backward
  || op = Tolk_uop.Ops.Detach

(* Every path from [inode] to [u] stays at the same element index. *)
let same_index_paths ~(inode : U.t) (u : U.t) =
  let memo : (int, bool * bool) Hashtbl.t = Hashtbl.create 64 in
  (* (reaches the input, reaches it through a disallowed op) *)
  let rec go u =
    match Hashtbl.find_opt memo (U.tag u) with
    | Some r -> r
    | None ->
        let r =
          if U.tag u = U.tag inode then (true, false)
          else begin
            let reaches = ref false and bad = ref false in
            Array.iter
              (fun s ->
                let r, b = go s in
                reaches := !reaches || r;
                bad := !bad || b)
              (U.src u);
            (!reaches, !bad || (!reaches && not (keeps_index u)))
          end
        in
        Hashtbl.replace memo (U.tag u) r;
        r
  in
  let reaches, bad = go u in
  (reaches, not bad)

(* Whether a kernel call stores more than one buffer. The scheduler gives each
   store its own kernel ([split_store]), so a kernel reads an input for the one
   buffer it writes; one that stores several may read an input at other indices
   for another, and reuse treats it so. *)
let stores_several call =
  match U.as_call call with
  | Some { body; _ } -> (
      match U.as_program_info body with
      | Some info -> List.length info.outs > 1
      | None -> false)
  | None -> false

(* The schedule's calls in execution order, expanding queue access metadata: a
   kernel's buffer arguments, and whether it stores more than one buffer. A
   queue submission's [k]th access list and [k]th fallback call are the same
   kernel's. [Opaque] marks a call whose inner order is unknown (a staged loop):
   it may read and write its arguments in any order. *)
type scheduled = Kernel of { args : U.t list; several : bool } | Opaque of U.t

let schedule_calls linear =
  List.concat_map
    (fun call ->
      let call = U.without_after call in
      match U.as_call call with
      | Some { body; args } -> (
          match U.arg call with
          | U.Arg.Call_info { aux = Some info; _ } ->
              List.map2
                (fun slots original ->
                  Kernel
                    {
                      args = List.map (List.nth args) slots;
                      several = stores_several original;
                    })
                info.accesses info.fallback
          | _ when U.op body = Tolk_uop.Ops.Custom_function -> [ Opaque call ]
          | _ -> [ Kernel { args; several = stores_several call } ])
      | None -> [])
    (U.children linear)

(* The buffers among a call's arguments, by tag. *)
let buffer_tags args =
  List.filter_map
    (fun a ->
      if U.is_bound_var a || U.is_variable a then None
      else Some (U.tag (U.buf_uop a)))
    args

(* No kernel reads the buffer [itag] after the first kernel that writes [otag],
   and neither buffer is touched by an opaque call. That first kernel may read
   [itag] itself when it writes each element where it read it and stores nothing
   else. An indexed write does not, so under [indexed] it must not read [itag]
   either. *)
let schedule_allows ?(indexed = false) ~linear ~itag ~otag () =
  let mentions args tag = List.mem tag (buffer_tags args) in
  let calls = schedule_calls linear in
  let opaque_touch =
    List.exists
      (function
        | Opaque c -> (
            match U.as_call c with
            | Some { args; _ } -> mentions args itag || mentions args otag
            | None -> false)
        | Kernel _ -> false)
      calls
  in
  if opaque_touch then false
  else begin
    let first_o = ref None and last_i = ref None and several = ref false in
    List.iteri
      (fun k -> function
        | Opaque _ -> ()
        | Kernel { args; several = s } ->
            if !first_o = None && mentions args otag then begin
              first_o := Some k;
              several := s
            end;
            if mentions args itag then last_i := Some k)
      calls;
    match (!first_o, !last_i) with
    | Some o, Some i -> if indexed || !several then i < o else i <= o
    | Some _, None -> true
    | None, _ -> false
  end

(* Loop calls.

   A staged loop replays its compiled body once per iteration and rebinds the
   body's slot nodes between iterations; see [Tolk.Realize.exec_loop]. A [loop]
   collects the call's argument buffers and slots of two kinds.

   A row slot binds a body node to row [i] of a stacked argument, [i] the
   iteration's data index. Rows are [stride] elements apart, [stride] padded to
   a whole number of 16 bytes, so every row starts where a vectorized access
   may.

   A carry binds the body nodes that read it and the nodes that write it to a
   buffer pair that alternates by iteration: iteration [j] reads buffer [j mod
   2] and writes the other, so after [n] iterations the value is in buffer [n
   mod 2]. A carry the body may update in place binds every node to one buffer.
   The buffer the loop starts from is written with the initial value before the
   loop.

   The loop launches the body and nothing else: the schedule writes every buffer
   it starts from, and every result is a buffer the body wrote. *)

type loop_slot = {
  node : U.t;
  pos0 : int;
  pos1 : int;
  size : int;
  stride : int;
}

type loop = {
  mutable args : U.t list; (* in position order, reversed *)
  mutable n_args : int;
  mutable ins : loop_slot list; (* reversed *)
  mutable outs : loop_slot list; (* reversed *)
}

let loop () = { args = []; n_args = 0; ins = []; outs = [] }

let add_arg l u =
  let pos = l.n_args in
  l.args <- u :: l.args;
  l.n_args <- pos + 1;
  pos

let row_stride dt numel =
  let unit = Int.max 1 (16 / TD.itemsize dt) in
  (numel + unit - 1) / unit * unit

(* A row slot [slot] of [numel] elements over the stacked buffer [node], whose
   rows are [stride] elements apart. *)
let add_rows_in l ~slot ~numel ~stride node =
  let pos = add_arg l node in
  l.ins <- { node = slot; pos0 = pos; pos1 = -1; size = numel; stride } :: l.ins

(* A row output: the body writes [slot], [numel] elements of [dt], to row [i] of
   a fresh stacked buffer. Returns the buffer and its row stride. *)
let add_rows_out st l ~slot ~dt ~numel ~n =
  let stride = row_stride dt numel in
  let buf = make_node st dt (n * stride) in
  let pos = add_arg l buf in
  l.outs <-
    { node = slot; pos0 = pos; pos1 = -1; size = numel; stride } :: l.outs;
  (buf, stride)

type carry = Pair of U.t * U.t | In_place of U.t

(* A carry of [shape] and [dt] at [place] starting from [init], read through the
   body nodes [reads] and written through [writes]: in one buffer under
   [in_place], in a pair otherwise. *)
let add_carry st l ?(in_place = false) ~reads ~writes ~dt ~place ~shape init =
  let numel = numel (local_shape place shape) in
  let start = make_node st dt numel in
  let pos0 =
    add_arg l (U.after ~src:start ~deps:[ store_placed place start shape init ])
  in
  let slot pos1 node = { node; pos0; pos1; size = numel; stride = 0 } in
  if in_place then begin
    l.ins <- List.rev_append (List.map (slot (-1)) reads) l.ins;
    l.outs <- List.rev_append (List.map (slot (-1)) writes) l.outs;
    In_place start
  end
  else begin
    let other = make_node st dt numel in
    let pos1 = add_arg l other in
    l.ins <- List.rev_append (List.map (slot pos1) reads) l.ins;
    l.outs <- List.rev_append (List.map (slot pos1) writes) l.outs;
    Pair (start, other)
  end

(* The buffer holding a carry's value after [n] iterations. *)
let final_carry ~n = function
  | In_place b -> b
  | Pair (b0, b1) -> if n mod 2 = 0 then b0 else b1

let loop_call st l ~body_linear ~resolve_node ~reversed ~n =
  let cint v = U.const (Tolk_uop.Const.int Tolk_uop.Dtype.weakint v) in
  let ins = List.rev l.ins and outs = List.rev l.outs in
  let targets = List.map (fun s -> resolve_node s.node) (ins @ outs)
    |> List.sort_uniq U.compare in
  let args = ref (List.rev l.args) in
  (* Captures cross the same CALL argument boundary as row and carry storage.
     Nested bodies retain their own PARAM namespace. *)
  let captures =
    U.toposort ~enter_calls:false body_linear
    |> List.filter (fun node ->
        U.op node = Tolk_uop.Ops.Buffer
        && U.addrspace node = Some TD.Global
        && not (List.exists (U.equal node) targets))
    |> List.map (fun node ->
        let slot = match List.find_index (U.equal node) !args with
          | Some slot -> slot
          | None ->
              let slot = List.length !args in
              args := !args @ [node];
              slot in
        node, U.param_like node ~slot) in
  let bindings = List.mapi (fun i node ->
      node, U.param_like node ~slot:(List.length !args + i)) targets in
  let body_linear = U.substitute ~walk:true (captures @ bindings) body_linear in
  let body_linear = Tolk.Realize.compile_linear ~device:st.st_device
      ~to_program body_linear in
  let slots ss =
    cint (List.length ss)
    :: List.concat_map
         (fun s ->
           [ List.assq (resolve_node s.node) bindings;
             cint s.pos0; cint s.pos1; cint s.size; cint s.stride ])
         ss
  in
  let payload =
    U.custom_function ~name:"loop"
      ~srcs:
        ([ body_linear; cint n; cint (if reversed then 1 else 0) ]
        @ slots ins @ slots outs)
  in
  let info =
    {
      U.grad_fxn = None;
      name = None;
      precompile = false;
      precompile_backward = false;
      dtype = TD.void;
      aux = None;
    }
  in
  (* Assembled with [replace], like the graph batcher's calls: the compiled body
     carries its launch ranges, which [U.call]'s range check rejects. *)
  match !args with
  | [] -> assert false
  | hd :: _ as args ->
      U.replace
        (U.call
           ~body:(U.custom_function ~name:"loop" ~srcs:[])
           ~args:[ hd ] ~info)
        ~src:(Array.of_list (payload :: args))
        ()

(* The buffer [b] once the loop [call] has written it. *)
let written_by call b = U.after ~src:b ~deps:[ call ]

(* The stacked rows of [shape] at [place] in [buf], each device's slices
   [stride] elements apart, as an [n :: shape] tensor. *)
let rows_tensor buf ~n ~stride ~place shape =
  let local = local_shape place shape in
  let numel = numel local in
  let t = buffer_tensor buf [| n; stride |] in
  let t =
    if stride = numel then t else F.Movement.shrink t [ (0, n); (0, numel) ]
  in
  whole_tensor (stacked place) (F.Movement.reshape t (n :: Array.to_list local))

(* The buffer [u] is, when it is a whole buffer after the effects that wrote it,
   under any reshape, or each device's slice of one when [u] is split. *)
let rec written_buffer u =
  match U.op u with
  | Tolk_uop.Ops.Reshape | Tolk_uop.Ops.Unshard -> written_buffer (U.src u).(0)
  | Tolk_uop.Ops.After when U.has_buffer_identity ~after_ok:true u -> Some u
  | _ -> None

(* Whether [tt] is whole storage under any reshape, which a kernel argument
   reads in place. *)
let is_storage tt =
  let rec go u =
    match U.op u with
    | Tolk_uop.Ops.Reshape -> go (U.src u).(0)
    | _ -> U.has_buffer_identity ~after_ok:true u
  in
  go (F.Tensor.uop tt)

(* A loop-call argument must resolve to a buffer. Buffer-identity nodes pass
   through; a computed value is realized; a device-less constant (e.g. a scalar
   carry init) is stored into a fresh buffer once, before the loop. *)
let realize_arg st (tt : F.Tensor.t) : U.t =
  let u = F.Tensor.uop tt in
  if U.has_buffer_identity u then u
  else if Option.is_none (U.device_of u) then
    let dt = F.Tensor.val_dtype tt in
    let n = numel (Array.of_list (F.Tensor.shape tt)) in
    let buf = make_node st dt n in
    U.after ~src:buf ~deps:[ U.store ~dst:buf ~value:u () ]
  else U.contiguous ~force:true ~src:u ()

let in_scan_body st f =
  st.scan_bodies <- st.scan_bodies + 1;
  Fun.protect ~finally:(fun () -> st.scan_bodies <- st.scan_bodies - 1) f

(* Gradient checkpointing (see [Remat]) *)

(* The storage [u] reads through views, each device's slice of it when [u] is
   split, when it reads one. *)
let viewed_storage u =
  let base = U.base u in
  let base =
    if U.op base = Tolk_uop.Ops.Unshard then U.base (U.src base).(0) else base
  in
  if U.has_buffer_identity ~after_ok:true base then Some base else None

(* The storage behind [tt]'s views, a value at [p]. A value that reads none is
   stored into a buffer of each device's slice first and [tt] repointed at it:
   the tensors that read [tt] later read the buffer, and the nodes built before
   keep the computation they read. *)
let storage st p tt =
  match viewed_storage (F.Tensor.uop tt) with
  | Some s -> s
  | None ->
      let shape = Array.of_list (F.Tensor.shape tt) in
      let buf =
        make_node st (F.Tensor.val_dtype tt) (numel (local_shape p shape))
      in
      let s = U.after ~src:buf ~deps:[ store_placed p buf shape tt ] in
      F.Tensor.set_uop tt (F.Tensor.uop (placed_tensor p s shape));
      s

(* [u] with [base], the node its views read, replaced by [f base]. *)
let rec reroot base f u =
  if u == base then f u
  else
    U.replace u
      ~src:
        (Array.mapi (fun i s -> if i = 0 then reroot base f s else s) (U.src u))
      ()

(* A row slot [slot] of [numel] elements, each device's, over the rows of the
   [n; ...] value [tt], padded to the loop's row stride when a row falls short
   of it (a whole row: see [stage_scan]). *)
let add_rows_in_value st l ~slot ~numel ~n tt =
  let stride = row_stride (F.Tensor.val_dtype tt) numel in
  let node =
    if stride = numel then realize_arg st tt
    else
      realize_arg st
        (F.Movement.pad
           (F.Movement.reshape tt [ n; numel ])
           [ (0, 0); (0, stride - numel) ])
  in
  add_rows_in l ~slot ~numel ~stride node

(* A staged body's slot: the placeholder the body receives for one leaf, of
   [s_shape] at [s_place], bound to a buffer node of each device's slice that
   the loop rebinds per iteration. *)
type body_slot = {
  s_ph : Nx.packed;
  s_node : U.t;
  s_dt : TD.t;
  s_shape : int array;
  s_place : Nx.Placement.t;
}

let slot_numel s = numel (local_shape s.s_place s.s_shape)

(* One slot per tensor of [leaves], a row of it (its leading axis dropped) under
   [rows], at [places] or where it lives. *)
let body_slots st ?(rows = false) ?places leaves =
  let places =
    match places with
    | Some places -> places
    | None -> List.map (fun (Nx.P leaf) -> placement_in st leaf) leaves
  in
  List.map2
    (fun (Nx.P leaf) place ->
      let shape = shape_of leaf in
      let shape, place =
        if rows then (Array.sub shape 1 (Array.length shape - 1), row place)
        else (shape, place)
      in
      let dt = tolk_dtype (Nx_effect.dtype leaf) in
      let node = make_node st dt (numel (local_shape place shape)) in
      let ph =
        traced st place (Nx_effect.dtype leaf) (placed_tensor place node shape)
      in
      {
        s_ph = Nx.P ph;
        s_node = node;
        s_dt = dt;
        s_shape = shape;
        s_place = place;
      })
    leaves places

let slot_values slots = List.map (fun s -> s.s_ph) slots

(* Fresh placeholders standing for [values], one [(shape, place, value)] per
   tensor of [leaves], of that tensor's dtype. *)
let placeholders st leaves values =
  List.map2
    (fun (Nx.P leaf) (shape, place, value) ->
      Nx.P
        (Nx_effect.traced st.st_ctx (Nx_effect.dtype leaf) shape
           (Node { trace = st.st_id; tensor = value; place })))
    leaves values

(* Whether [leaves] are what [slots] stand for: of their shapes, at their
   placements. *)
let same_slots st leaves slots =
  List.for_all2
    (fun (Nx.P l) s ->
      shape_of l = s.s_shape && Nx.Placement.equal (placement_in st l) s.s_place)
    leaves slots

(* Placements in a program over several devices

   Every value of such a program lives where nx's rules put it
   ([Nx_effect.routing], [Nx_effect.result]), decided as the function traces: an
   operation over operands split differently raises as it does eagerly, instead
   of the compiler resharding them, and so does a movement that would move
   elements between devices. A cut of the split axis within one slice is where
   compiled and eager code differ: tolk copies a whole slice to every device,
   and a cut strictly inside one would place it on one device, which a program
   over several devices cannot hold. *)

(* Where a value at [p] of [shape] lives once moved by [m]: where eager movement
   puts it ([Nx_effect.moved_placement]), unless that keeps fewer devices. A cut
   of one whole slice of a split axis is then copied to every device, as tolk
   lowers it; one strictly inside a slice raises. *)
let moved p shape m =
  let q = Nx_effect.moved_placement p shape m in
  if List.compare_lengths (Nx.Placement.devices q) (Nx.Placement.devices p) = 0
  then q
  else
    let slice =
      Nx.Placement.window p shape (List.hd (Nx.Placement.devices q))
    in
    match m with
    | Nx_effect.Shrink limits ->
        List.fold_left
          (fun q' (axis, _) ->
            if List.mem_assoc axis (Nx_effect.Grid.cuts q) then q'
            else if limits.(axis) = slice.(axis) then
              Nx_effect.Grid.uncut q' ~axis
            else
              invalid_arg
                (Printf.sprintf
                   "Nx: a cut inside one slice of the split axis %d of shape \
                    %s would place it on one device, which a compiled program \
                    over several devices cannot; place the value on one device \
                    first"
                   axis
                   (Nx_core.Shape.to_string shape)))
          p (Nx_effect.Grid.cuts p)
    | _ -> q

(* [tt], a value at [p] in [st]'s program over several devices, at [q]: a split
   value is gathered by a copy to every device, and a copy is split by each
   device keeping its slice (tolk's copy and shard nodes). *)
let reshard st p q tt =
  let names = List.map Nx.Device.name st.st_devices in
  let whole =
    if Nx_effect.Grid.cuts p = [] then tt
    else
      F.Tensor.of_uop (U.copy ~src:(F.Tensor.uop tt) ~device:(U.Multi names) ())
  in
  match Nx_effect.Grid.cuts q with
  | [] -> whole
  | [ (axis, _) ] -> F.Creation.shard ~axis ~devices:names whole
  | _ -> unsupported "a value cut along several axes"

(* [x] at [p] in [st]'s program, resharded there from where it lives. *)
let resharded st p x =
  let q = placement_in st x and tt = tolk_of st x in
  if Nx.Placement.equal p q then tt else reshard st q p tt

(* [x] as a kernel over values at [q] reads it: at [q], or as it is when it is a
   copy on each device with one element along every axis [q] cuts. *)
let aligned st q x =
  if
    Nx_effect.Grid.cuts (placement_in st x) = []
    && List.for_all (fun (a, _) -> (shape_of x).(a) = 1) (Nx_effect.Grid.cuts q)
  then tolk_of st x
  else resharded st q x

(* The placement over [ds] of [u] as tolk lays it out: split along the axis its
   [Unshard] cuts by the device range, or a copy on each device. A value after
   the effects that wrote it is laid out as its storage. *)
let laid_out ds u =
  let u = if U.op u = Tolk_uop.Ops.After then (U.src u).(0) else u in
  match U.sharding u with
  | [] -> Nx.Placement.replicated ds
  | [ (axis, _) ] -> Nx.Placement.sharded ~axis ds
  | _ -> unsupported "a value cut along several axes"

(* A kernel's result [tt] in [st]'s program, where tolk's kernel builder put
   it. *)
let kernel_result st dt tt =
  traced st (laid_out st.st_devices (F.Tensor.uop tt)) dt tt

(* Where the result of the operation performing [eff] lives in [st]'s program:
   raises [Invalid_argument] as nx does when its operands cannot meet. *)
let result_placement : type c. state -> c Effect.t -> Nx.Placement.t =
 fun st eff ->
  match Nx_effect.routing eff with
  | Some (op, rule, xs) ->
      Nx_effect.result op rule
        (List.map
           (fun (Nx.P x) ->
             (Some (placement_in st x), Array.length (shape_of x)))
           xs)
  | None -> (
      match Nx_effect.movement_of eff with
      | Some (Nx.P x, m) -> moved (placement_in st x) (shape_of x) m
      | None -> here st)

(* Whether a scan over the stacks [xs] cannot be staged in [st]'s program: a row
   of a stack split along its leading axis lies on one device, and a split row
   is read in place only as a whole number of 16-byte units (see
   [add_rows_in_value]). Such a scan unrolls, as one whose carry changes
   does. *)
let unstageable st xs =
  List.exists
    (fun (Nx.P x) ->
      let p = placement_in st x and shape = shape_of x in
      let cuts = Nx_effect.Grid.cuts p in
      List.mem_assoc 0 cuts
      || cuts <> []
         &&
         let local =
           numel
             (local_shape (row p) (Array.sub shape 1 (Array.length shape - 1)))
         in
         row_stride (tolk_dtype (Nx_effect.dtype x)) local <> local)
    xs

(* Handler *)

let rec handler : type r. state -> (r, r) Effect.Deep.handler =
 fun st ->
  let open Effect.Deep in
  let dt x = Nx_effect.dtype x in
  let go x = tolk_of st x in
  let refuse k op =
    discontinue k
      (Jit_error
         (Printf.sprintf
            "Rune.jit: %s is not supported inside jit; move it outside the \
             jitted function"
            op))
  in
  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
   fun eff ->
    (* Answer an intercepted operation: record the graph node and continue with
       a fresh placeholder carrying the result's shape, dtype and placement.
       Operands that cannot meet raise into the function. *)
    let ret : type a b r.
        ((a, b) Nx_effect.t, r) continuation -> (a, b) ND.t -> F.Tensor.t -> r =
     fun k dt tt ->
      match result_placement st eff with
      | p -> continue k (traced st p dt tt)
      | exception (Invalid_argument _ as e) -> discontinue k e
    in
    (* Like [ret] for a two-result operation: one placeholder per result, both
       carrying the operation's shared dtype (qr's factors, e.g.). *)
    let ret2 : type a b r.
        ((a, b) Nx_effect.t * (a, b) Nx_effect.t, r) continuation ->
        (a, b) ND.t ->
        F.Tensor.t ->
        F.Tensor.t ->
        r =
     fun k dt tq tr ->
      match result_placement st eff with
      | p -> continue k (traced st p dt tq, traced st p dt tr)
      | exception (Invalid_argument _ as e) -> discontinue k e
    in
    match eff with
    (* Metadata reads fall back to the placeholder, whose view is the
       result's. *)
    | E_view _ -> None
    (* Reading data is allowed only for tensors whose bytes are real (constants
       created during the trace); reading a traced value would burn data into
       the compiled program. *)
    | E_to_host x ->
        if is_traced x then
          Some
            (fun k ->
              discontinue k
                (Jit_error
                   "Rune.jit: the value of a traced tensor was read during jit \
                    tracing (item, to_host, or a data-dependent branch); \
                    jitted code cannot branch on tensor values"))
        else None
    (* Creation *)
    | E_buffer { dtype; size_in_elements; _ } ->
        Some
          (fun k ->
            let ph = Nx_effect.buffer st.st_ctx dtype [| size_in_elements |] in
            ignore (lift_const st ph);
            continue k ph)
    | E_const_scalar { value; dtype; _ } ->
        Some
          (fun k ->
            check_dtype st dtype "a constant of the function";
            (* [buffer:false] keeps the scalar an immediate constant: it folds
               into consuming kernels instead of being stored into a one-element
               buffer by a kernel of its own. *)
            let tt =
              F.Creation.full ~buffer:false ~dtype:(tolk_dtype dtype) []
                (scalar_of dtype value)
            in
            let ph = Nx_effect.const_scalar st.st_ctx value dtype in
            Tensor_map.Tbl.replace st.table (Key ph) tt;
            continue k ph)
    | E_from_host { array; _ } ->
        Some
          (fun k ->
            let ph = Nx_effect.from_host st.st_ctx array in
            ignore (lift_const st ph);
            continue k ph)
    (* Binary arithmetic *)
    | E_add { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.add (go a) (go b)))
    | E_sub { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.sub (go a) (go b)))
    | E_mul { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.mul (go a) (go b)))
    | E_idiv { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.cdiv (go a) (go b)))
    | E_fdiv { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.div (go a) (go b)))
    | E_max { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.maximum (go a) (go b)))
    | E_min { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.minimum (go a) (go b)))
    | E_mod { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.fmod (go a) (go b)))
    | E_pow { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.pow (go a) (go b)))
    | E_xor { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.bitwise_xor (go a) (go b)))
    | E_or { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.bitwise_or (go a) (go b)))
    | E_and { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.bitwise_and (go a) (go b)))
    | E_atan2 { a; b } ->
        Some (fun k -> ret k (dt a) (atan2_graph (go a) (go b)))
    (* Comparisons *)
    | E_cmpeq { a; b } ->
        Some (fun k -> ret k ND.bool (F.Elementwise.eq (go a) (go b)))
    | E_cmpne { a; b } ->
        Some (fun k -> ret k ND.bool (F.Elementwise.ne (go a) (go b)))
    | E_cmplt { a; b } ->
        Some (fun k -> ret k ND.bool (F.Elementwise.lt (go a) (go b)))
    | E_cmple { a; b } ->
        Some (fun k -> ret k ND.bool (F.Elementwise.le (go a) (go b)))
    (* Unary arithmetic *)
    | E_neg { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.neg (go t_in)))
    | E_sin { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.sin (go t_in)))
    | E_sqrt { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.sqrt (go t_in)))
    | E_recip { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.reciprocal (go t_in)))
    | E_log { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.log (go t_in)))
    | E_exp { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.exp (go t_in)))
    | E_cos { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.cos (go t_in)))
    | E_abs { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.abs (go t_in)))
    | E_sign { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.sign (go t_in)))
    | E_tan { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.tan (go t_in)))
    | E_asin { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.asin (go t_in)))
    | E_acos { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.acos (go t_in)))
    | E_atan { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.atan (go t_in)))
    | E_sinh { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.sinh (go t_in)))
    | E_cosh { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.cosh (go t_in)))
    | E_tanh { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.tanh (go t_in)))
    | E_erf { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.erf (go t_in)))
    (* Rounding: identity on integers, as eagerly. *)
    | E_trunc { t_in } ->
        Some
          (fun k ->
            let t = go t_in in
            ret k (dt t_in)
              (if ND.is_float (dt t_in) then F.Elementwise.trunc t else t))
    | E_ceil { t_in } ->
        Some
          (fun k ->
            let t = go t_in in
            ret k (dt t_in)
              (if ND.is_float (dt t_in) then F.Elementwise.ceil t else t))
    | E_floor { t_in } ->
        Some
          (fun k ->
            let t = go t_in in
            ret k (dt t_in)
              (if ND.is_float (dt t_in) then F.Elementwise.floor t else t))
    | E_round { t_in } ->
        Some
          (fun k ->
            let t = go t_in in
            ret k (dt t_in) (if ND.is_float (dt t_in) then round_away t else t))
    (* Ternary *)
    | E_where { condition; if_true; if_false } ->
        Some
          (fun k ->
            ret k (dt if_true)
              (F.Elementwise.where (go condition) (go if_true) (go if_false)))
    (* Reductions. The accumulator dtype is pinned to the input's so results
       match eager execution. *)
    | E_reduce_sum { t_in; axes } ->
        Some
          (fun k ->
            let t = go t_in in
            let axis = Array.to_list axes in
            (* A float sum accumulates at float32 or wider and rounds once, as
               the eager one does; an integer sum wraps at its own width. *)
            ret k (dt t_in)
              (if ND.is_float (dt t_in) then F.Reduce.sum ~axis ~keepdim:false t
               else
                 F.Reduce.sum ~axis ~keepdim:false ~dtype:(F.Tensor.val_dtype t)
                   t))
    | E_reduce_prod { t_in; axes } ->
        Some
          (fun k ->
            let t = go t_in in
            let axis = Array.to_list axes in
            ret k (dt t_in)
              (if narrow_float t then
                 F.Dtype_ops.cast
                   (F.Reduce.prod ~axis ~keepdim:false ~dtype:TD.float32 t)
                   (F.Tensor.dtype t)
               else
                 F.Reduce.prod ~axis ~keepdim:false
                   ~dtype:(F.Tensor.val_dtype t) t))
    | E_reduce_max { t_in; axes } ->
        Some
          (fun k ->
            ret k (dt t_in)
              (extreme ~op:`Max ~axes:(Array.to_list axes) (go t_in)))
    | E_reduce_min { t_in; axes } ->
        Some
          (fun k ->
            ret k (dt t_in)
              (extreme ~op:`Min ~axes:(Array.to_list axes) (go t_in)))
    (* Over integer keys the first NaN is the extreme and zeros tie, so the
       first of equal extremes is eager's position. *)
    | E_argmax { t_in; axis; keepdims } ->
        Some
          (fun k ->
            let keys, _ = order_keys ~nan:`Greatest ~zeros:`Tied (go t_in) in
            ret k ND.int32
              (F.Dtype_ops.cast
                 (F.Op.argmax ~axis ~keepdim:keepdims keys)
                 TD.int32))
    | E_argmin { t_in; axis; keepdims } ->
        Some
          (fun k ->
            let keys, _ = order_keys ~nan:`Least ~zeros:`Tied (go t_in) in
            ret k ND.int32
              (F.Dtype_ops.cast
                 (F.Op.argmin ~axis ~keepdim:keepdims keys)
                 TD.int32))
    | E_sort { t_in; axis; descending } ->
        Some
          (fun k ->
            ret k (dt t_in)
              (sort_graph ~packs:(packs st) ~dim:axis ~descending (go t_in)))
    | E_argsort { t_in; axis; descending } ->
        Some
          (fun k ->
            ret k ND.int32
              (F.Dtype_ops.cast
                 (argsort_graph ~packs:(packs st) ~dim:axis ~descending
                    (go t_in))
                 TD.int32))
    | E_associative_scan { t_in; axis; op } ->
        Some
          (fun k ->
            let t = go t_in in
            let r =
              match op with
              | `Sum -> F.Op.cumsum ~axis t
              | `Prod -> F.Op.cumprod ~axis t
              | (`Max | `Min) as op -> running ~packs:(packs st) ~axis ~op t
            in
            (* A sum over small integers accumulates wider; the scan keeps its
               input's dtype. *)
            ret k (dt t_in) (F.Dtype_ops.cast r (tolk_dtype (dt t_in))))
    (* Movement *)
    | E_permute { t_in; axes } ->
        Some
          (fun k ->
            ret k (dt t_in) (F.Movement.permute (go t_in) (Array.to_list axes)))
    | E_reshape { t_in; new_shape } ->
        Some
          (fun k ->
            ret k (dt t_in)
              (F.Movement.reshape (go t_in) (Array.to_list new_shape)))
    | E_expand { t_in; new_target_shape } ->
        Some
          (fun k ->
            ret k (dt t_in)
              (F.Movement.expand (go t_in) (Array.to_list new_target_shape)))
    | E_pad { t_in; padding_config; fill_value } ->
        Some
          (fun k ->
            let pads =
              Array.to_list (Array.map (fun p -> Some p) padding_config)
            in
            ret k (dt t_in)
              (F.Op.pad ~value:(scalar_of (dt t_in) fill_value) (go t_in) pads))
    | E_shrink { t_in; limits } ->
        Some
          (fun k ->
            ret k (dt t_in) (F.Movement.shrink (go t_in) (Array.to_list limits)))
    | E_flip { t_in; dims_to_flip } ->
        Some
          (fun k ->
            let axes = ref [] in
            Array.iteri (fun i f -> if f then axes := i :: !axes) dims_to_flip;
            ret k (dt t_in) (F.Movement.flip (go t_in) (List.rev !axes)))
    (* Unlike the eager movement, which is view metadata, the lowering pools
       through pad/reshape/shrink and materializes: a step narrower than the
       window writes each overlapped element once per window that reads it. *)
    | E_sliding_window { t_in; axis; window; step } ->
        Some
          (fun k ->
            ret k (dt t_in)
              (F.Movement.unfold (go t_in) axis ~size:window ~step))
    | E_cat { t_list; axis } ->
        Some
          (fun k ->
            match t_list with
            | [] -> err "Rune.jit: cat of an empty list"
            | hd :: tl ->
                ret k (dt hd)
                  (F.Op.cat ~dim:axis (go hd) (List.map (fun t -> go t) tl)))
    (* Cast and copies *)
    | E_cast { t_in; target_dtype } ->
        Some
          (fun k ->
            ret k target_dtype
              (F.Dtype_ops.cast (go t_in) (tolk_dtype target_dtype)))
    (* tolk emulates float8 on every device it runs, decoding each element to a
       wider float and back, which flushes subnormals and clamps infinities: a
       float8 bitcast would change bits. *)
    | E_bitcast { t_in; target_dtype } ->
        let float8 (type c d) (d : (c, d) ND.t) =
          match d with ND.Float8_e4m3 | ND.Float8_e5m2 -> true | _ -> false
        in
        if float8 (dt t_in) || float8 target_dtype then
          unsupported "a bitcast to or from float8"
        else
          Some
            (fun k ->
              ret k target_dtype
                (F.Dtype_ops.bitcast (go t_in) (tolk_dtype target_dtype)))
    (* A written buffer is contiguous storage already, and a [contiguous] over
       it would copy it out. *)
    | E_contiguous { t_in } ->
        Some
          (fun k ->
            let tt = go t_in in
            ret k (dt t_in)
              (match written_buffer (F.Tensor.uop tt) with
              | Some _ -> tt
              | None -> F.Elementwise.contiguous tt))
    | E_copy { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.contiguous (go t_in)))
    (* Staged scans *)
    | Scan.E_scan_probe -> Some (fun k -> continue k true)
    | Scan.E_scan req ->
        Some
          (fun k ->
            if unstageable st req.req_xs then discontinue k Scan.Not_staged
            else stage_scan st req k)
    | Scan.E_scan_bwd bwd -> Some (fun k -> stage_scan_bwd st bwd k)
    (* Gradient checkpointing. A differentiated remat's arguments are the
       residuals of its backward pass, so they are materialised. The backward
       pass reads them through an AFTER on their storage whose dependencies are
       the cotangents of the result, materialised too: that node is one the
       forward pass never built, so the recomputation shares nothing with it,
       and the kernels reading it wait for the cotangents, so the recomputation
       runs in the backward pass. Storage the program does not write (an input
       of the compiled function, a constant) takes no AFTER: a kernel reading it
       in two states would be a read/write cycle to the scheduler, and no
       intermediate of the forward pass depends on it alone. An argument backed
       by such storage is read as it is, so a remat whose arguments are all
       inputs or constants shares the forward pass's nodes, as does one whose
       cotangents are storage from the start: the AFTER has nothing to wait for.
       A staged scan body, whose backward loop recomputes each step already,
       keeps the plain function. *)
    | Remat.E_remat (Remat.Call { params_s; params; f; residuals; _ }) ->
        Some
          (fun k ->
            if residuals && st.scan_bodies = 0 then
              Nx.Ptree.fold params_s
                (fun _ leaf () ->
                  ignore (storage st (placement_in st leaf) (go leaf) : U.t))
                params ();
            continue k (Effect.Deep.match_with f params (handler st)))
    | Remat.E_barrier { values; after } ->
        Some
          (fun k ->
            if st.scan_bodies > 0 then continue k values
            else
              let deps =
                List.filter_map
                  (fun (Nx.P a) ->
                    let s = storage st (placement_in st a) (go a) in
                    if U.op s = Tolk_uop.Ops.After then Some s else None)
                  after
              in
              continue k
                (List.map
                   (fun (Nx.P v) ->
                     let tt = go v in
                     let s = storage st (placement_in st v) tt in
                     if U.op s <> Tolk_uop.Ops.After then Nx.P v
                     else
                       Nx.P
                         (traced st (placement_in st v) (dt v)
                            (F.Tensor.of_uop
                               (reroot s
                                  (fun s -> U.after ~src:s ~deps)
                                  (F.Tensor.uop tt)))))
                   values))
    (* Indexed access *)
    | E_gather { data; indices; axis } ->
        Some
          (fun k ->
            ret k (dt data) (F.Op.gather (go data) ~dim:axis (go indices)))
    | E_scatter { data_template; indices; updates; axis; mode; unique_indices }
      ->
        Some
          (fun k ->
            (* Over a split destination each device writes its slice: the
               updates and their positions are split alike off the write axis
               and whole along it. *)
            let place = placement_in st data_template in
            let along = Nx_effect.Grid.uncut place ~axis in
            ret k (dt data_template)
              (F.Op.scatter_indexed
                 (write_destination st (go data_template) ~place)
                 ~dim:axis (aligned st along indices) (aligned st along updates)
                 ~mode ~unique:unique_indices))
    (* The window write. A constant corner is a padded [v] selected over [t] in
       one pass. A traced corner is a scatter of [v]'s elements at their flat
       positions in [t], which are distinct and, the corner being clamped by the
       frontend, inside [t]: its cost is [v]. Over a [t] split along its first
       axis, the flat positions keep that split, and each device writes the
       positions in its slice. Flat positions are int32, and a [t] split along a
       later axis has none across its devices, so beyond either the window is
       read through a clamped gather per axis and masked. *)
    | E_update { t_in; starts; v } ->
        Some
          (fun k ->
            let tshape = shape_of t_in and vshape = shape_of v in
            let rank = Array.length tshape in
            let tt = go t_in and tv = go v in
            let zero = scalar_of (dt v) (ND.zero (dt v)) in
            if rank = 0 then ret k (dt t_in) tv
            else if not (is_traced starts) then begin
              let host = Nx_effect.to_host starts in
              let sv = Nx_effect.view starts in
              let s k =
                Int32.to_int
                  (Nx_buffer.get host
                     (NV.offset sv + (k * (NV.strides sv).(0))))
              in
              let pads =
                List.init rank (fun k ->
                    Some (s k, tshape.(k) - s k - vshape.(k)))
              in
              let ones =
                F.Creation.full ~buffer:false ~dtype:TD.bool
                  (Array.to_list vshape) (F.Tensor.Sbool true)
              in
              let mask = F.Op.pad ~value:(F.Tensor.Sbool false) ones pads in
              ret k (dt t_in)
                (F.Elementwise.where mask (F.Op.pad ~value:zero tv pads) tt)
            end
            else if
              Array.fold_left ( * ) 1 tshape <= Int32.to_int Int32.max_int
              && List.for_all
                   (fun (a, _) -> a = 0)
                   (Nx_effect.Grid.cuts (placement_in st t_in))
            then begin
              let whole = Nx_effect.Grid.uncut (placement_in st t_in) ~axis:0 in
              let st_t = aligned st whole starts and tv = aligned st whole v in
              let i32 n =
                F.Creation.full ~buffer:false ~dtype:TD.int32 []
                  (F.Tensor.Sint n)
              in
              let flat = ref (i32 0) and stride = ref 1 in
              for ax = rank - 1 downto 0 do
                let start =
                  F.Movement.reshape
                    (F.Movement.shrink st_t [ (ax, ax + 1) ])
                    []
                in
                let along =
                  List.init rank (fun d -> if d = ax then -1 else 1)
                in
                let coord =
                  F.Elementwise.add start
                    (F.Movement.reshape
                       (F.Op.arange ~dtype:TD.int32 vshape.(ax))
                       along)
                in
                flat :=
                  F.Elementwise.add !flat
                    (F.Elementwise.mul coord (i32 !stride));
                stride := !stride * tshape.(ax)
              done;
              let count = Array.fold_left ( * ) 1 vshape in
              let index =
                F.Movement.reshape
                  (F.Movement.expand !flat (Array.to_list vshape))
                  [ count ]
              in
              let src = F.Movement.reshape tv [ count ] in
              let written =
                F.Op.scatter_indexed
                  (F.Movement.reshape
                     (write_destination st tt ~place:(placement_in st t_in))
                     [ Array.fold_left ( * ) 1 tshape ])
                  ~dim:0 index src ~mode:`Set ~unique:true
              in
              ret k (dt t_in)
                (F.Movement.reshape written (Array.to_list tshape))
            end
            else begin
              let st_t = go starts in
              let win = ref tv in
              let mask = ref None in
              for ax = 0 to rank - 1 do
                let n = tshape.(ax) and len = vshape.(ax) in
                let start =
                  F.Movement.reshape
                    (F.Movement.shrink st_t [ (ax, ax + 1) ])
                    []
                in
                let ar = F.Op.arange ~dtype:TD.int32 n in
                let rel = F.Elementwise.sub ar start in
                let lo =
                  F.Creation.full ~buffer:false ~dtype:TD.int32 []
                    (F.Tensor.Sint 0)
                in
                let hi =
                  F.Creation.full ~buffer:false ~dtype:TD.int32 []
                    (F.Tensor.Sint (len - 1))
                in
                let idx = F.Elementwise.clamp ~min:lo ~max:hi rel in
                let inside =
                  F.Elementwise.bitwise_and (F.Elementwise.ge rel lo)
                    (F.Elementwise.lt rel
                       (F.Creation.full ~buffer:false ~dtype:TD.int32 []
                          (F.Tensor.Sint len)))
                in
                let along =
                  List.init rank (fun d -> if d = ax then -1 else 1)
                in
                let shp =
                  List.mapi
                    (fun d s -> if d = ax then n else s)
                    (F.Tensor.shape !win)
                in
                let idx =
                  F.Movement.expand (F.Movement.reshape idx along) shp
                in
                win := F.Op.gather !win ~dim:ax idx;
                let inside = F.Movement.reshape inside along in
                mask :=
                  Some
                    (match !mask with
                    | None -> inside
                    | Some m -> F.Elementwise.bitwise_and m inside)
              done;
              let mask = Option.get !mask in
              ret k (dt t_in) (F.Elementwise.where mask !win tt)
            end)
    (* Matrix multiplication *)
    | E_matmul { a; b } ->
        Some
          (fun k ->
            let ta = go a and tb = go b in
            ret k (dt a)
              (if narrow_float ta then
                 F.Dtype_ops.cast
                   (F.Op.matmul
                      (F.Dtype_ops.cast ta TD.float32)
                      (F.Dtype_ops.cast tb TD.float32))
                   (F.Tensor.dtype ta)
               else F.Op.matmul ta tb))
    (* Quantised products lower to Nx compositions and tolk's kernels over the
       traced values, traced under this handler like the function's own
       operations. A block or instance names its matrix by id, which over
       matrices split along their first axis may be another device's: its rows
       and ids are brought whole, and each device multiplies the ones whose
       matrix it holds (tolk's partial product). Otherwise rows are split with
       their ids. *)
    | Nx_quant.Effect.E_quant { w; op } ->
        let operands ids x matrices =
          if
            Option.is_some ids
            && List.mem_assoc 0 (Nx_effect.Grid.cuts (placement_in st matrices))
          then
            (resharded st (here st) x, Option.map (resharded st (here st)) ids)
          else
            match Option.map (placement_in st) ids with
            | Some p when Nx_effect.Grid.cuts p <> [] ->
                (aligned st p x, Option.map go ids)
            | _ -> (go x, Option.map go ids)
        in
        let quant_matmul ?ids x ~codes ~scales =
          let part name t =
            let tt = go t in
            if Lazy.force jit_debug >= 1 && not (is_storage tt) then
              Printf.eprintf
                "rune.jit: quantised product: %s is a view, copied on every call\n\
                 %!"
                name;
            tt
          in
          let rows, ids = operands ids x codes in
          let codes = part "codes" codes and scales = part "scales" scales in
          kernel_result st (dt x) (F.Op.quant_matmul ?ids rows ~codes ~scales)
        in
        let block_matmul ~transpose x w ~ids =
          let rows, ids = operands (Some ids) x w in
          kernel_result st (dt x)
            (F.Op.block_matmul ~transpose rows (go w) ~ids:(Option.get ids))
        in
        let kernels =
          { Quant.device = st.st_device; quant_matmul; block_matmul }
        in
        Some
          (fun k ->
            continue k
              (Effect.Deep.match_with
                 (fun () -> Quant.lower kernels w op)
                 () (handler st)))
    (* A placement inside a program: the identity at the value's own placement,
       a [reshard] to another placement over the program's devices; any other
       target raises. *)
    | E_place { placement = q; t_in } ->
        Some
          (fun k ->
            let p = placement_in st t_in in
            match
              Nx_effect.Placement.check_shape "Nx.place" q (shape_of t_in)
            with
            | exception (Invalid_argument _ as e) -> discontinue k e
            | () ->
                if Nx.Placement.equal p q then continue k t_in
                else if not (over st.st_devices q) then
                  discontinue k
                    (Jit_error
                       (Format.asprintf
                          "Rune.jit: a program on %a cannot place a value on \
                           %a; place it outside the compiled function"
                          pp_devices st.st_devices Nx.Placement.pp q))
                else
                  continue k (traced st q (dt t_in) (reshard st p q (go t_in))))
    (* A traced value lives where its operation put it, which is what the
       gradient of a placement asks of its primal. *)
    | E_placement x when is_traced x ->
        Some (fun k -> continue k (placement_in st x))
    | E_placement _ -> None
    (* Random bits compile only from a key that depends on the traced inputs. A
       constant key (implicit RNG such as [Nx.rand], or a captured key) would
       freeze one draw into the compiled program and silently replay it on every
       call — the worst failure mode, wrong without erring. *)
    | E_threefry { key; ctr } ->
        Some
          (fun k ->
            let kt = go key in
            if not (depends_on_input st (F.Tensor.uop kt)) then
              discontinue k
                (Jit_error
                   "Rune.jit: random number generation from a constant key \
                    inside jit: the key does not depend on the jitted \
                    function's inputs, so every call would replay the same \
                    values. Use Nx.Rng and pass the key as an input of the \
                    jitted function (derive per-call keys with Nx.Rng.split or \
                    Nx.Rng.fold_in); implicit RNG (Nx.rand and friends) is not \
                    supported inside jit")
            else ret k ND.int32 (threefry_graph kt (go ctr)))
    | E_unfold { t_in; kernel_size; stride; dilation; padding } ->
        Some
          (fun k ->
            ret k (dt t_in)
              (unfold_graph st t_in ~kernel_size ~stride ~dilation ~padding))
    | E_fold { t_in; output_size; kernel_size; stride; dilation; padding } ->
        Some
          (fun k ->
            ret k (dt t_in)
              (fold_graph st t_in ~output_size ~kernel_size ~stride ~dilation
                 ~padding))
    | E_fft _ -> Some (fun k -> refuse k "fft")
    | E_ifft _ -> Some (fun k -> refuse k "ifft")
    | E_rfft _ -> Some (fun k -> refuse k "rfft")
    | E_irfft _ -> Some (fun k -> refuse k "irfft")
    (* Linear algebra without a single Tolk Uop lowers at trace time into
       ordinary Tolk compositions, like matmul and the other lowered C-kernel
       ops; the factorizations unroll a number of steps fixed by the input
       shapes (see [Tolk_frontend.Linalg]). Complex inputs cannot be traced at
       all; non-float dtypes are refused here. A non-positive-definite Cholesky
       input, which the eager kernel reports as [Linalg_error], yields nans in
       the compiled program. *)
    | E_cholesky { t_in; upper } ->
        Some
          (fun k ->
            if ND.is_float (dt t_in) then
              ret k (dt t_in) (F.Linalg.cholesky ~upper (go t_in))
            else refuse k "cholesky")
    | E_qr { t_in; reduced } ->
        Some
          (fun k ->
            if ND.is_float (dt t_in) then
              let q, r = F.Linalg.qr ~reduced (go t_in) in
              ret2 k (dt t_in) q r
            else refuse k "qr")
    | E_svd _ -> Some (fun k -> refuse k "svd")
    | E_eigvals _ -> Some (fun k -> refuse k "eigvals")
    | E_eig _ -> Some (fun k -> refuse k "eig")
    | E_eigvalsh _ -> Some (fun k -> refuse k "eigvalsh")
    | E_eigh _ -> Some (fun k -> refuse k "eigh")
    | E_solve_triangular { a; b; upper; transpose; unit_diag } ->
        Some
          (fun k ->
            if ND.is_float (dt a) then
              ret k (dt a)
                (F.Linalg.solve_triangular ~upper ~transpose ~unit_diag (go a)
                   (go b))
            else refuse k "solve_triangular")
    | E_psum _ ->
        Some
          (fun k ->
            discontinue k
              (Jit_error "Rune.jit: psum is only meaningful under vmap"))
    | _ -> None
  in
  { retc = Fun.id; exnc = raise; effc }

(* The forward scan: trace the body once, compile it as a sub-program, and emit
   the loop call. The carry, the rows and the outputs arrive as their tensors in
   walk order: every tensor has its own slot, and slots pair with their buffers
   by position. A carry leaf is a buffer pair, a row leaf is read at row [i] of
   its stacked input, and an output leaf is written at row [i] of its stack.
   When a staged transpose will read them ([req_record]), the body also writes
   the carry it receives to a carry stack. *)
and stage_scan : type r.
    ?places:Nx.Placement.t list ->
    ?seen:Nx.Placement.t list list ->
    state ->
    Scan.scan_req ->
    (Scan.scan_res, r) Effect.Deep.continuation ->
    r =
 fun ?places ?(seen = []) st req k ->
  let Scan.{ req_carry; req_xs; req_step = step; req_record } = req in
  let n = Scan.length req_xs in
  (* Discover the body's external inputs — the differentiable tensors it closes
     over: everything the body runs through [tolk_of] that predates its trace is
     a free input of the loop. The backward loop accumulates their cotangents,
     so record them against the scan's identity. The snapshot is taken before
     the slot placeholders exist, and tensors first touched by the body itself
     (constants it captures) are absent from it — they can never be tracked by
     an enclosing grad, whose tape only sees traced tensors. The float filter
     stands for differentiability: complex would qualify too, but [tolk_dtype]
     refuses complex tensors long before one could reach this hook — widen the
     filter if tolk ever takes them. *)
  let before = Tensor_map.Tbl.create 16 in
  Tensor_map.Tbl.iter (fun k _ -> Tensor_map.Tbl.replace before k ()) st.table;
  let horizon = Nx_effect.next_traced_id () in
  let predates : type a b. (a, b) Nx_effect.t -> bool = function
    | Nx_effect.Traced { t_id; _ } -> t_id < horizon
    | t -> Tensor_map.Tbl.mem before (Key t)
  in
  let closed = ref [] in
  let collect =
    {
      hook =
        (fun (type a b) (t : (a, b) Nx_effect.t) ->
          let k = Obj.repr t in
          if
            ND.is_float (Nx_effect.dtype t)
            && predates t
            && not (List.exists (fun (Nx.P g) -> Obj.repr g == k) !closed)
          then closed := Nx.P t :: !closed);
    }
  in
  let c_slots = body_slots st ?places req_carry in
  let x_slots = body_slots st ~rows:true req_xs in
  (* Trace the body once under a nested copy of this tracer, collecting its
     external inputs and the buffers its indexed writes into the carry land
     in. *)
  let writes = List.map (fun s -> (s.s_node, ref [])) c_slots in
  let outer_writes = st.scan_writes in
  st.scan_collectors <- collect :: st.scan_collectors;
  st.scan_writes <- writes @ outer_writes;
  let c_next, y =
    in_scan_body st (fun () ->
        Fun.protect
          ~finally:(fun () ->
            st.scan_collectors <- List.tl st.scan_collectors;
            st.scan_writes <- outer_writes)
          (fun () ->
            Effect.Deep.match_with
              (fun () -> step.run (slot_values c_slots) (slot_values x_slots))
              () (handler st)))
  in
  Tbl.replace st.scan_closed (Obj.repr step) !closed;
  (* A loop can only be compiled from a stable carry (the single prototype trace
     stands for every step). A body that changes a carry's shape or placement
     declines staging — the scan folds eagerly and unrolls into this trace, as
     every jit did before staging existed. The traced body's nodes are
     unreachable from any output and never get scheduled. *)
  if not (same_slots st c_next c_slots) then
    (* A body that places the carry elsewhere is staged again with the carry
       where it puts it, its initial value resharded there, until the placements
       are a fixed point of the body, the placement the unrolled fold reaches.
       Placements already seen are a cycle, which unrolls. *)
    let current = List.map (fun s -> s.s_place) c_slots
    and next = List.map (fun (Nx.P c) -> placement_in st c) c_next in
    let seen = current :: seen in
    if
      List.for_all2 (fun (Nx.P c) s -> shape_of c = s.s_shape) c_next c_slots
      && not (List.exists (List.equal Nx.Placement.equal next) seen)
    then stage_scan ~places:next ~seen st req k
    else Effect.Deep.discontinue k Scan.Not_staged
  else
    let y_outs =
      List.map
        (fun (Nx.P y) ->
          let shape = shape_of y and place = placement_in st y in
          let dt = tolk_dtype (Nx_effect.dtype y) in
          ( dt,
            shape,
            place,
            make_node st dt (numel (local_shape place shape)),
            tolk_of st y ))
        y
    in
    let stack_outs =
      if req_record then
        List.map (fun s -> make_node st s.s_dt (slot_numel s)) c_slots
      else []
    in
    (* In-place carries, by RFC 0001's reuse rule applied to the body. An
       indexed write into a carry lands in its own buffer (see
       [write_destination]), which the loop binds to the carry's storage: the
       body then writes only the elements it updates. A carry whose next value
       is that write needs no other buffer, nor does one whose next value reads
       it only at the index it writes. Both hold only when the body's schedule
       reads the carry no later than the write, which is known once the body is
       scheduled: a write that fails it is filled with the carry by a kernel
       instead, the other carries stay pairs, and the body is scheduled again
       until every remaining candidate holds. *)
    let next_values =
      List.map (fun (Nx.P c) -> F.Tensor.uop (tolk_of st c)) c_next
    in
    let slot_writes = List.map (fun (_, w) -> !w) writes in
    let c_outs =
      List.map (fun s -> make_node st s.s_dt (slot_numel s)) c_slots
    in
    let body ~copied ~same_index =
      let aliased es = List.filter (fun e -> not (List.memq e copied)) es in
      let written es value =
        match (aliased es, written_buffer value) with
        | [ e ], Some a when U.buf_uop a == e -> Some a
        | _ -> None
      in
      let modes =
        List.map2
          (fun (es, c_out) value ->
            match written es value with
            | Some a -> `Written (a, List.hd (aliased es))
            | None ->
                if es = [] && List.memq c_out same_index then `Same_index c_out
                else `Pair (aliased es, c_out))
          (List.combine slot_writes c_outs)
          next_values
      in
      let fill e =
        let s, _ =
          List.find
            (fun (_, es) -> List.memq e es)
            (List.combine c_slots slot_writes)
        in
        ( e,
          U.after ~src:e
            ~deps:
              [
                store_placed s.s_place e s.s_shape
                  (placed_tensor s.s_place s.s_node s.s_shape);
              ] )
      in
      let sink =
        U.sink
          (List.map2
             (fun (s, mode) value ->
               match mode with
               | `Written (a, _) -> a
               | `Same_index c_out | `Pair (_, c_out) ->
                   U.after ~src:c_out
                     ~deps:
                       [
                         store_placed s.s_place c_out s.s_shape
                           (F.Tensor.of_uop value);
                       ])
             (List.combine c_slots modes)
             next_values
          @ List.map
              (fun (_, shape, place, y_out, value) ->
                U.after ~src:y_out
                  ~deps:[ store_placed place y_out shape value ])
              y_outs
          @ List.map2
              (fun s stack_out ->
                U.after ~src:stack_out
                  ~deps:
                    [
                      store_placed s.s_place stack_out s.s_shape
                        (placed_tensor s.s_place s.s_node s.s_shape);
                    ])
              (if req_record then c_slots else [])
              stack_outs)
      in
      let sink =
        if copied = [] then sink
        else U.substitute ~walk:true (List.map fill copied) sink
      in
      (modes, schedule_body_linear sink)
    in
    let rec settle ~copied ~same_index =
      let modes, (linear, resolve_node) = body ~copied ~same_index in
      let allows ?indexed (s : body_slot) o =
        schedule_allows ?indexed ~linear ~itag:(U.tag s.s_node)
          ~otag:(U.tag (resolve_node o))
          ()
      in
      let failed_writes =
        List.concat_map
          (fun (s, mode) ->
            match mode with
            | `Written (_, e) -> if allows ~indexed:true s e then [] else [ e ]
            | `Pair (es, _) ->
                List.filter (fun e -> not (allows ~indexed:true s e)) es
            | `Same_index _ -> [])
          (List.combine c_slots modes)
      in
      let failed_updates =
        List.concat_map
          (fun (s, mode) ->
            match mode with
            | `Same_index c_out -> if allows s c_out then [] else [ c_out ]
            | `Written _ | `Pair _ -> [])
          (List.combine c_slots modes)
      in
      if failed_writes = [] && failed_updates = [] then
        (modes, linear, resolve_node)
      else
        settle ~copied:(failed_writes @ copied)
          ~same_index:
            (List.filter (fun c -> not (List.memq c failed_updates)) same_index)
    in
    let same_index =
      List.filter_map
        (fun ((s, c_out), value) ->
          if snd (same_index_paths ~inode:s.s_node value) then Some c_out
          else None)
        (List.combine (List.combine c_slots c_outs) next_values)
    in
    let copied =
      List.concat_map
        (fun es -> if List.length es > 1 then es else [])
        slot_writes
    in
    let modes, body_linear, resolve_node = settle ~copied ~same_index in
    let l = loop () in
    List.iter2
      (fun s (Nx.P x) ->
        add_rows_in_value st l ~slot:s.s_node ~numel:(slot_numel s) ~n
          (tolk_of st x))
      x_slots req_xs;
    let pairs =
      List.map2
        (fun (s, mode) (Nx.P c) ->
          let add =
            add_carry st l ~dt:s.s_dt ~place:s.s_place ~shape:s.s_shape
          in
          let init = resharded st s.s_place c in
          match mode with
          | `Written (_, e) ->
              add ~in_place:true ~reads:[ s.s_node; e ] ~writes:[] init
          | `Same_index c_out ->
              add ~in_place:true ~reads:[ s.s_node ] ~writes:[ c_out ] init
          | `Pair (es, c_out) ->
              add ~reads:(s.s_node :: es) ~writes:[ c_out ] init)
        (List.combine c_slots modes)
        req_carry
    in
    let ys_rows =
      List.map
        (fun (dt, shape, place, y_out, _) ->
          ( shape,
            place,
            add_rows_out st l ~slot:y_out ~dt
              ~numel:(numel (local_shape place shape))
              ~n ))
        y_outs
    in
    let stacks =
      List.map2
        (fun s stack_out ->
          add_rows_out st l ~slot:stack_out ~dt:s.s_dt ~numel:(slot_numel s) ~n)
        (if req_record then c_slots else [])
        stack_outs
    in
    let call = loop_call st l ~body_linear ~resolve_node ~reversed:false ~n in
    (* Register the carry stacks as outputs of the forward loop: the backward
       loop reads them, and only a graph-visible dependency keeps the forward
       loop reachable (and so scheduled) when the scan's declared outputs are
       dead — e.g. under [grad], which discards the loss value. *)
    if req_record then
      Tbl.replace st.scan_stacks (Obj.repr step)
        (List.map2
           (fun (buf, stride) s -> (written_by call buf, stride, s.s_place))
           stacks c_slots);
    let r_carry =
      placeholders st req_carry
        (List.map2
           (fun s pair ->
             ( s.s_shape,
               s.s_place,
               placed_tensor s.s_place
                 (written_by call (final_carry ~n pair))
                 s.s_shape ))
           c_slots pairs)
    in
    let r_ys =
      placeholders st y
        (List.map
           (fun (shape, place, (buf, stride)) ->
             ( Array.append [| n |] shape,
               stacked place,
               rows_tensor (written_by call buf) ~n ~stride ~place shape ))
           ys_rows)
    in
    Effect.Deep.continue k { Scan.r_carry; r_ys }

(* The backward scan (the transpose): capture the body's pullback against
   placeholder slot tensors — recomputing the forward step inside the body to
   recover its residuals — and emit a reversed loop carrying the carry's
   cotangent. Reads the carry of step i from the forward loop's stack, and
   writes the cotangent of row i of every floating-point row leaf.

   The body's external inputs (the differentiable tensors it closes over,
   observed by the forward staging) are tracked on the private tape, so the
   captured pullback also emits their per-step cotangent contributions; the loop
   totals each in a carry that starts at zero — unlike the carry, an external
   input's cotangent is a sum over the steps, not a thread through them. *)
and stage_scan_bwd : type r.
    state ->
    Scan.scan_bwd ->
    (Scan.scan_bwd_res, r) Effect.Deep.continuation ->
    r =
 fun st bwd k ->
  let Scan.{ bwd_step = step; bwd_carry; bwd_xs; bwd_dc; bwd_dys } = bwd in
  let n = Scan.length bwd_xs in
  (* The body's external inputs and carry stacks, recorded by the forward
     staging of this scan. *)
  let not_staged () =
    (* Reachable only if a handler claimed [E_scan] without answering
       [E_scan_probe]: reverse then recorded a transpose for a scan this trace
       never staged with its carries recorded. *)
    err
      "Rune.jit: backward scan for a scan this trace did not stage with its \
       carries recorded (a handler claimed E_scan without answering \
       E_scan_probe)"
  in
  let closed =
    match Tbl.find_opt st.scan_closed (Obj.repr step) with
    | Some closed -> closed
    | None -> not_staged ()
  in
  let stacks =
    match Tbl.find_opt st.scan_stacks (Obj.repr step) with
    | Some stacks -> stacks
    | None -> not_staged ()
  in
  (* The carry and its cotangent live where the forward loop kept the carry. *)
  let places = List.map (fun (_, _, p) -> p) stacks in
  let c_slots = body_slots st ~places bwd_carry in
  let x_slots = body_slots st ~rows:true bwd_xs in
  let dc_slots = body_slots st ~places bwd_dc in
  let dy_slots = body_slots st ~rows:true bwd_dys in
  let differentiable s =
    let (Nx.P ph) = s.s_ph in
    ND.is_float (Nx_effect.dtype ph)
  in
  (* Capture the pullback: run the body once under a private reverse tape, then
     replay the tape against the placeholder cotangents — every op lands in the
     trace, forming the backward body (the forward step's ops are recomputed
     inside it to recover residuals). The external inputs are tracked too, so
     the pullback emits their per-step contributions. *)
  let tape = Tape.create () in
  let track (Nx.P t) = Tape.track tape t in
  List.iter (fun s -> track s.s_ph) c_slots;
  List.iter (fun s -> if differentiable s then track s.s_ph) x_slots;
  List.iter track closed;
  let c_next, y =
    in_scan_body st (fun () ->
        Effect.Deep.match_with
          (fun () ->
            Effect.Deep.match_with
              (fun () -> step.run (slot_values c_slots) (slot_values x_slots))
              () (Reverse.handler tape))
          () (handler st))
  in
  if not (same_slots st c_next c_slots) then
    err
      "Rune.jit: the scan body must return a carry of the same shapes and \
       placements it receives (stable carry)";
  let cotangent (Nx.P t) = Nx.P (Tape.cotangent tape t) in
  let dc_i, dx_i, dgs =
    in_scan_body st (fun () ->
        Effect.Deep.match_with
          (fun () ->
            let seed (Nx.P v) s =
              Tape.accumulate tape v (Nx.unpack (Nx_effect.dtype v) s.s_ph)
            in
            List.iter2 seed c_next dc_slots;
            List.iter2 seed y dy_slots;
            Tape.backward tape;
            ( List.map (fun s -> cotangent s.s_ph) c_slots,
              List.map
                (fun s ->
                  if differentiable s then Some (cotangent s.s_ph) else None)
                x_slots,
              List.map
                (fun (Nx.P g) -> Scan.Closed_ctan (g, Tape.cotangent tape g))
                closed ))
          () (handler st))
  in
  (* The backward body: per-leaf carry cotangents, row cotangents, and the
     external inputs' accumulators. The accumulation is elementwise, so it runs
     on the flat buffers directly. The tensor itself stays packed — unpacked,
     its type would escape its scope in the tuple. *)
  let dc_outs =
    List.map (fun s -> make_node st s.s_dt (slot_numel s)) c_slots
  in
  let dx_outs =
    List.map
      (fun s ->
        if differentiable s then Some (make_node st s.s_dt (slot_numel s))
        else None)
      x_slots
  in
  let g_outs =
    List.map
      (fun (Scan.Closed_ctan (g, dg)) ->
        let g_shape = shape_of g and g_place = placement_in st g in
        let gdt = tolk_dtype (Nx_effect.dtype g) in
        let gn = numel (local_shape g_place g_shape) in
        ( Nx.P g,
          g_shape,
          g_place,
          gdt,
          make_node st gdt gn,
          make_node st gdt gn,
          resharded st g_place dg ))
      dgs
  in
  (* Cotangents are stored where their primals live; stacks are read where they
     live. *)
  let value p (Nx.P t) = resharded st p t in
  let stack (Nx.P t) = tolk_of st t in
  let body_sink =
    U.sink
      (List.map2
         (fun (s, dc_out) dc ->
           U.after ~src:dc_out
             ~deps:
               [ store_placed s.s_place dc_out s.s_shape (value s.s_place dc) ])
         (List.combine c_slots dc_outs)
         dc_i
      @ List.concat_map
          (fun ((s, dx_out), dx) ->
            match (dx_out, dx) with
            | Some dx_out, Some dx ->
                [
                  U.after ~src:dx_out
                    ~deps:
                      [
                        store_placed s.s_place dx_out s.s_shape
                          (value s.s_place dx);
                      ];
                ]
            | _ -> [])
          (List.combine (List.combine x_slots dx_outs) dx_i)
      @ List.map
          (fun (_, g_shape, g_place, _, g_in, g_out, dg_tt) ->
            U.after ~src:g_out
              ~deps:
                [
                  store_placed g_place g_out g_shape
                    (F.Elementwise.add
                       (placed_tensor g_place g_in g_shape)
                       dg_tt);
                ])
          g_outs)
  in
  let body_linear, resolve_node = schedule_body_linear body_sink in
  let l = loop () in
  List.iter2
    (fun s (stack, stride, _) ->
      add_rows_in l ~slot:s.s_node ~numel:(slot_numel s) ~stride stack)
    c_slots stacks;
  List.iter2
    (fun s x ->
      add_rows_in_value st l ~slot:s.s_node ~numel:(slot_numel s) ~n (stack x))
    x_slots bwd_xs;
  List.iter2
    (fun s dy ->
      add_rows_in_value st l ~slot:s.s_node ~numel:(slot_numel s) ~n (stack dy))
    dy_slots bwd_dys;
  let dc_pairs =
    List.map2
      (fun ((s, dc_out), d) dc ->
        add_carry st l ~reads:[ d.s_node ] ~writes:[ dc_out ] ~dt:s.s_dt
          ~place:s.s_place ~shape:s.s_shape (value s.s_place dc))
      (List.combine (List.combine c_slots dc_outs) dc_slots)
      bwd_dc
  in
  let dx_rows =
    List.map2
      (fun s dx_out ->
        Option.map
          (fun dx_out ->
            add_rows_out st l ~slot:dx_out ~dt:s.s_dt ~numel:(slot_numel s) ~n)
          dx_out)
      x_slots dx_outs
  in
  let g_pairs =
    List.map
      (fun (_, g_shape, g_place, gdt, g_in, g_out, _) ->
        add_carry st l ~reads:[ g_in ] ~writes:[ g_out ] ~dt:gdt ~place:g_place
          ~shape:g_shape
          (F.Creation.zeros ~buffer:false ~dtype:gdt (Array.to_list g_shape)))
      g_outs
  in
  let call = loop_call st l ~body_linear ~resolve_node ~reversed:true ~n in
  let br_carry =
    placeholders st bwd_carry
      (List.map2
         (fun s pair ->
           ( s.s_shape,
             s.s_place,
             placed_tensor s.s_place
               (written_by call (final_carry ~n pair))
               s.s_shape ))
         c_slots dc_pairs)
  in
  (* A row leaf that is not floating point gets a zero cotangent. *)
  let br_xs =
    placeholders st bwd_xs
      (List.map2
         (fun s rows ->
           let shape = Array.append [| n |] s.s_shape in
           match rows with
           | Some (buf, stride) ->
               ( shape,
                 stacked s.s_place,
                 rows_tensor (written_by call buf) ~n ~stride ~place:s.s_place
                   s.s_shape )
           | None ->
               ( shape,
                 here st,
                 F.Creation.zeros ~buffer:false ~dtype:s.s_dt
                   (Array.to_list shape) ))
         x_slots dx_rows)
  in
  (* Each external input's total cotangent, as outputs of the loop. *)
  let br_closed =
    List.map2
      (fun (Nx.P g, g_shape, g_place, _, _, _, _) pair ->
        let after = written_by call (final_carry ~n pair) in
        let ph =
          traced st g_place (Nx_effect.dtype g)
            (placed_tensor g_place after g_shape)
        in
        Scan.Closed_ctan (g, ph))
      g_outs g_pairs
  in
  Effect.Deep.continue k { Scan.br_carry; br_xs; br_closed }

(* Host transfers *)

(* A buffer with no bytes is never allocated: a device has no storage of size
   zero, and a program never reads a value without elements. *)
let ensure_storage buf =
  if Tolk.Device.Buffer.nbytes buf > 0 then
    Tolk.Device.Buffer.ensure_allocated buf

(* Wrap host memory as a device buffer without copying. The caller must keep the
   memory's owner reachable while the buffer can still be read or written. *)
let wrap_ptr dev dtolk n ptr =
  let buf =
    Tolk.Device.create_buffer ~size:n ~dtype:dtolk
      ~spec:{ Tolk.Device.Buffer_spec.default with external_ptr = Some ptr }
      dev
  in
  ensure_storage buf;
  buf

(* Wrap a tensor's memory, or [None] when its elements are not a contiguous
   span. Also returns the value that keeps the memory reachable. *)
let wrap_tensor : type a b.
    Tolk.Device.t -> (a, b) Nx_effect.t -> (Tolk.Device.Buffer.t * Obj.t) option
    =
 fun dev x ->
  let x =
    match x with
    | Nx_effect.Placed _ -> Nx_effect.Host (Nx_effect.host_of x)
    | x -> x
  in
  let v = Nx_effect.view x in
  if not (NV.is_c_contiguous v) then None
  else
    let dt = Nx_effect.dtype x in
    let host = Nx_effect.to_host x in
    let ptr =
      Nativeint.add
        (Nx_buffer.unsafe_data_ptr host)
        (Nativeint.of_int (NV.offset v * ND.itemsize dt))
    in
    Some (wrap_ptr dev (tolk_dtype dt) (numel (NV.shape v)) ptr, Obj.repr host)

(* A host buffer wired as a kernel output: the computed tensor is built on it
   directly. *)
type host_out = Host : ('a, 'b) ND.t * ('a, 'b) Nx_buffer.t -> host_out

(* Chunked transfers

   Every copy between host and device moves at most [chunk_bytes] at a time,
   through a window of the device buffer, so no transfer stages a whole leaf on
   the host. *)

let chunk_bytes = 64 * 1024 * 1024

(* A device may park host staging per pending copy until it next synchronizes
   (CUDA pins one buffer per copy), so a long run of copies synchronizes every
   [sync_bytes]. *)
let sync_bytes = 256 * 1024 * 1024
let unsynced_bytes = ref 0

let note_copied dev n =
  unsynced_bytes := !unsynced_bytes + n;
  if !unsynced_bytes >= sync_bytes then begin
    Tolk.Device.synchronize dev;
    unsynced_bytes := 0
  end

(* Byte staging, keyed by size and reused across calls, so that a compiled
   program does not repopulate the page tables with a fresh [Bytes] per leaf on
   every replay. A transfer primitive wants bytes of exactly its length; every
   full chunk shares one [Bytes], and the table holds the shorter lengths.
   Compiled functions are not thread-safe, and each use completes before the
   next lookup. *)
type scratch = (int, Bytes.t) Hashtbl.t

let full_chunk = lazy (Bytes.create chunk_bytes)

let scratch_bytes tbl size =
  if size = chunk_bytes then Lazy.force full_chunk
  else
    match Hashtbl.find_opt tbl size with
    | Some b -> b
    | None ->
        let b = Bytes.create size in
        Hashtbl.add tbl size b;
        b

(* The window of [buf] that starts at byte [off] and spans [len] bytes, as a
   buffer of bytes, so that two windows of the same length copy into each other.
   A window is released before its base can be. *)
let with_window buf ~off ~len f =
  let w =
    Tolk.Device.Buffer.view buf ~size:len ~dtype:Tolk_uop.Dtype.uint8
      ~offset:off
  in
  Tolk.Device.Buffer.ensure_allocated w;
  Fun.protect
    ~finally:(fun () -> Tolk.Device.Buffer.deallocate w)
    (fun () -> f w)

(* File-backed sources

   A host buffer over a mapped file is copied fastest by reading the file: a
   read is bound by the disk, a walk of the mapping by the page-fault path,
   several times slower once the file no longer fits in the cache beside the
   device buffers it is copied into. The path may name another file by now, so
   the file opened must be the one that was mapped. *)

let open_file (file : Nx_buffer.file) =
  match Unix.openfile file.path [ Unix.O_RDONLY; Unix.O_CLOEXEC ] 0 with
  | exception Unix.Unix_error _ -> None
  | fd -> (
      match Unix.LargeFile.fstat fd with
      | st
        when st.st_size = Int64.of_int file.size
             && st.st_mtime = file.mtime && st.st_ino = file.inode ->
          Some fd
      | _ | (exception Unix.Unix_error _) ->
          Unix.close fd;
          None)

(* [read_at fd ~pos bytes len] fills [bytes] with the [len] bytes of [fd] at
   [pos], or is [false] if the file is shorter or cannot be read. *)
let read_at fd ~pos bytes len =
  match Unix.LargeFile.lseek fd (Int64.of_int pos) Unix.SEEK_SET with
  | exception Unix.Unix_error _ -> false
  | _ ->
      let rec fill off =
        off = len
        ||
        match Unix.read fd bytes off (len - off) with
        | 0 -> false
        | n -> fill (off + n)
        | exception Unix.Unix_error _ -> false
      in
      fill 0

(* [with_file_source host f] runs [f] with a reader of [host]'s bytes from its
   file, [None] when [host] is not a mapped file that can still be read. *)
let with_file_source host f =
  match Nx_buffer.file_range host with
  | None -> f None
  | Some (file, base) -> (
      match open_file file with
      | None -> f None
      | Some fd ->
          Fun.protect
            ~finally:(fun () -> Unix.close fd)
            (fun () ->
              f
                (Some
                   (fun ~pos bytes len ->
                     read_at fd ~pos:(base + pos) bytes len))))

(* [base_layout v] is [Some (shape, axes)] when the elements of the view [v] are
   exactly a contiguous run of its buffer seen through a permutation of axes:
   [shape] is the run's own shape and permuting it by [axes] gives [v]. *)
let base_layout v =
  let shape = NV.shape v and strides = NV.strides v in
  let rank = Array.length shape in
  let order = Array.init rank Fun.id in
  Array.stable_sort
    (fun a b ->
      match (shape.(a) = 1, shape.(b) = 1) with
      | true, true -> 0
      | true, false -> -1
      | false, true -> 1
      | false, false -> compare strides.(b) strides.(a))
    order;
  let expected = ref 1 and ok = ref true in
  for i = rank - 1 downto 0 do
    let a = order.(i) in
    if shape.(a) <> 1 && strides.(a) <> !expected then ok := false;
    expected := !expected * shape.(a)
  done;
  if not !ok then None
  else begin
    let axes = Array.make rank 0 in
    Array.iteri (fun i a -> axes.(a) <- i) order;
    Some (Array.map (fun a -> shape.(a)) order, axes)
  end

(* A strided view of a mapped file whose elements are a contiguous run of the
   file seen through a permutation of axes (a transposed weight): the run, read
   from the file into host memory, under the same permutation. It costs a host
   copy of the run for the length of the upload; walking the mapping in the
   view's order instead faults its pages in at a fraction of the disk's
   speed. *)
let read_base : type a b.
    scratch -> (a, b) Nx_effect.t -> (a, b) Nx_effect.t option =
 fun sc x ->
  let v = Nx_effect.view x in
  let host = Nx_effect.to_host x in
  match (Nx_buffer.file_range host, base_layout v) with
  | None, _ | _, None -> None
  | Some _, Some (shape, axes) ->
      with_file_source host @@ fun read ->
      Option.bind read @@ fun read ->
      let dt = Nx_effect.dtype x in
      let item = ND.itemsize dt in
      let n = numel shape in
      let run = Nx_buffer.create dt n in
      let chunk = chunk_bytes / item in
      let pos = ref 0 and ok = ref true in
      while !ok && !pos < n do
        let len = Int.min chunk (n - !pos) in
        let bytes = scratch_bytes sc (len * item) in
        ok := read ~pos:((NV.offset v + !pos) * item) bytes (len * item);
        if !ok then Nx_buffer.blit_from_bytes ~dst_off:!pos ~len bytes run;
        pos := !pos + len
      done;
      if not !ok then None
      else
        Some
          (Nx_effect.permute
             (Nx_effect.reshape
                (Nx_effect.from_host (Nx_effect.context x) run)
                shape)
             axes)

(* Copy a tensor's logical contents into [buf] from byte [off] on. A contiguous
   source, offset or not, is read in place chunk by chunk, from its file when it
   is a mapped one. A strided one is cut along its leading axis into pieces of
   at most a chunk, each made contiguous on its own. *)
let rec copyin_at : type a b.
    scratch ->
    Tolk.Device.t ->
    Tolk.Device.Buffer.t ->
    off:int ->
    (a, b) Nx_effect.t ->
    unit =
 fun sc dev buf ~off x ->
  let v = Nx_effect.view x in
  let shape = NV.shape v in
  let item = ND.itemsize (Nx_effect.dtype x) in
  let nbytes = numel shape * item in
  if nbytes = 0 then ()
  else if NV.is_c_contiguous v then begin
    let host = Nx_effect.to_host x in
    with_file_source host @@ fun read ->
    let read = ref read in
    let chunk = chunk_bytes / item in
    let n = numel shape in
    let pos = ref 0 in
    while !pos < n do
      let len = Int.min chunk (n - !pos) in
      let bytes = scratch_bytes sc (len * item) in
      let src_off = NV.offset v + !pos in
      let from_file =
        match !read with
        | Some read -> read ~pos:(src_off * item) bytes (len * item)
        | None -> false
      in
      (* A file that can no longer be read is left for the mapping. *)
      if not from_file then begin
        read := None;
        Nx_buffer.blit_to_bytes ~src_off ~len host bytes
      end;
      with_window buf
        ~off:(off + (!pos * item))
        ~len:(len * item)
        (fun w -> Tolk.Device.Buffer.copyin w bytes);
      bytes_to_device := !bytes_to_device + (len * item);
      note_copied dev (len * item);
      pos := !pos + len
    done
  end
  else
    match read_base sc x with
    | Some base -> copyin_at sc dev buf ~off base
    | None when nbytes <= chunk_bytes ->
        copyin_at sc dev buf ~off (Nx_effect.contiguous x)
    | None ->
        (* Axes of size one before [axis] do not change the row-major order. *)
        let axis = ref 0 in
        while shape.(!axis) = 1 do
          incr axis
        done;
        let axis = !axis in
        let row = nbytes / shape.(axis) in
        let rows = Int.max 1 (chunk_bytes / row) in
        let r = ref 0 in
        while !r < shape.(axis) do
          let stop = Int.min shape.(axis) (!r + rows) in
          let ranges =
            Array.mapi
              (fun d n -> if d = axis then (!r, stop) else (0, n))
              shape
          in
          copyin_at sc dev buf
            ~off:(off + (!r * row))
            (Nx_effect.shrink x ranges);
          r := stop
        done

(* Copy a tensor's logical contents into a device buffer. *)
let copyin_tensor sc dev buf x =
  ensure_storage buf;
  copyin_at sc dev buf ~off:0
    (match x with
    | Nx_effect.Placed _ -> Nx_effect.Host (Nx_effect.host_of x)
    | x -> x)

(* Copy a device buffer's contents into [host] from element [dst_off] on. *)
let copyout_into : type a b.
    scratch -> Tolk.Device.Buffer.t -> dst_off:int -> (a, b) Nx_buffer.t -> unit
    =
 fun sc buf ~dst_off host ->
  let item = ND.itemsize (Nx_buffer.dtype host) in
  let n = Tolk.Device.Buffer.nbytes buf / item in
  let chunk = chunk_bytes / item in
  let pos = ref 0 in
  while !pos < n do
    let len = Int.min chunk (n - !pos) in
    let bytes = scratch_bytes sc (len * item) in
    with_window buf ~off:(!pos * item) ~len:(len * item) (fun w ->
        Tolk.Device.Buffer.copyout w bytes);
    Nx_buffer.blit_from_bytes ~dst_off:(dst_off + !pos) ~len bytes host;
    bytes_from_device := !bytes_from_device + (len * item);
    pos := !pos + len
  done

(* [x] with a host value in place of a placed one: its view's elements. *)
let on_host : type a b. (a, b) Nx_effect.t -> (a, b) Nx_effect.t = function
  | Placed _ as x -> Nx_effect.Host (Nx_effect.host_of x)
  | x -> x

(* Copy [x] into [bufs], one per device of [on], each its window of [x] at [p]:
   the whole value for a copy, its slice for a split. *)
let upload_windows : type a b.
    scratch ->
    Nx.Placement.t ->
    (a, b) Nx_effect.t ->
    (Nx.Device.t * Tolk.Device.t) list ->
    Tolk.Device.Buffer.t list ->
    unit =
 fun sc p x on bufs ->
  let x = on_host x in
  let shape = shape_of x in
  List.iter2
    (fun (d, dev) buf ->
      copyin_tensor sc dev buf
        (Nx_effect.shrink x (Nx.Placement.window p shape d)))
    on bufs

(* Build a fresh tensor of [dt]/[shape] from a device buffer's contents. *)
let read_out : type a b.
    scratch ->
    Nx_effect.context ->
    (a, b) ND.t ->
    int array ->
    Tolk.Device.Buffer.t ->
    (a, b) Nx_effect.t =
 fun sc ctx dtv shape buf ->
  let host = Nx_buffer.create dtv (numel shape) in
  copyout_into sc buf ~dst_off:0 host;
  Nx_effect.reshape (Nx_effect.from_host ctx host) shape

(* The device engine

   Reading copies a placed value's view's elements to the host and leaves its
   storage. On devices whose memory the host addresses (Metal, the CPU device),
   the elements are copied straight out of the buffer's own memory; elsewhere
   the range the view reaches is copied out first. *)

(* Copy the elements [v] reaches into [dst] in C order, from [src], which holds
   the storage's elements from [base] on. *)
let gather_elements : type a b.
    (a, b) Nx_buffer.t -> base:int -> NV.t -> (a, b) Nx_buffer.t -> unit =
 fun src ~base v dst ->
  let shape = NV.shape v and strides = NV.strides v in
  let rank = Array.length shape in
  let idx = Array.make rank 0 and off = ref (NV.offset v - base) in
  for i = 0 to Nx_buffer.length dst - 1 do
    Nx_buffer.unsafe_set dst i (Nx_buffer.unsafe_get src !off);
    let d = ref (rank - 1) in
    while !d >= 0 do
      idx.(!d) <- idx.(!d) + 1;
      off := !off + strides.(!d);
      if idx.(!d) < shape.(!d) then d := -1
      else begin
        off := !off - (strides.(!d) * shape.(!d));
        idx.(!d) <- 0;
        decr d
      end
    done
  done

(* [gather_elements] for words of a bigarray kind: when the view's last axis is
   contiguous, each run along it is copied whole. *)
let gather_words : type a b.
    (a, b) Nx_buffer.t -> base:int -> NV.t -> (a, b) Nx_buffer.t -> unit =
 fun src ~base v dst ->
  let shape = NV.shape v and strides = NV.strides v in
  let rank = Array.length shape in
  if rank = 0 || strides.(rank - 1) <> 1 || Nx_buffer.length dst = 0 then
    gather_elements src ~base v dst
  else begin
    let s = Nx_buffer.to_bigarray1 src and d = Nx_buffer.to_bigarray1 dst in
    let run = shape.(rank - 1) in
    let idx = Array.make rank 0 and off = ref (NV.offset v - base) in
    for row = 0 to (Nx_buffer.length dst / run) - 1 do
      Bigarray.Array1.blit
        (Bigarray.Array1.sub s !off run)
        (Bigarray.Array1.sub d (row * run) run);
      let a = ref (rank - 2) in
      while !a >= 0 do
        idx.(!a) <- idx.(!a) + 1;
        off := !off + strides.(!a);
        if idx.(!a) < shape.(!a) then a := -1
        else begin
          off := !off - (strides.(!a) * shape.(!a));
          idx.(!a) <- 0;
          decr a
        end
      done
    done
  end

(* [gather_elements] over the elements' bits, read as integers of their width: a
   float read into an OCaml float would quiet a signalling NaN. An element of 16
   bytes is two 8-byte words; 4-bit elements are copied as values. *)
let gather_view : type a b.
    (a, b) Nx_buffer.t -> base:int -> NV.t -> (a, b) Nx_buffer.t -> unit =
 fun src ~base v dst ->
  let as_words (type c d) (word : (c, d) ND.t) w =
    let shape = NV.shape v and strides = NV.strides v in
    let words =
      if w = 1 then v
      else
        NV.create
          ~offset:(NV.offset v * w)
          ~strides:(Array.append (Array.map (fun s -> s * w) strides) [| 1 |])
          (Array.append shape [| w |])
    in
    gather_words
      (Nx_buffer.reinterpret word src)
      ~base:(base * w) words
      (Nx_buffer.reinterpret word dst)
  in
  match Nx_buffer.dtype src with
  | ND.Int4 | ND.UInt4 -> gather_elements src ~base v dst
  | kind -> (
      match ND.itemsize kind with
      | 1 -> as_words ND.Int8 1
      | 2 -> as_words ND.Int16 1
      | 4 -> as_words ND.Int32 1
      | 8 -> as_words ND.Int64 1
      | n -> as_words ND.Int64 (n / 8))

(* [with_storage_range dt buf ~lo ~hi f] is [f src how], [src] the elements of
   [buf]'s storage from [lo] to [hi] on the host: [`Borrowed] from the buffer's
   memory when the host addresses it, [`Copied] otherwise. A buffer's finaliser
   frees its memory, so [buf] stays reachable until [f] returns, and a borrowed
   [src] must not outlive [f]. *)
let with_storage_range : type a b c.
    (a, b) ND.t ->
    Tolk.Device.Buffer.t ->
    lo:int ->
    hi:int ->
    ((a, b) Nx_buffer.t -> [ `Borrowed | `Copied ] -> c) ->
    c =
 fun dt buf ~lo ~hi f ->
  let item = ND.itemsize dt in
  match Tolk.Device.Buffer.as_buffer buf with
  | Some mem ->
      let bytes = Bigarray.Array1.sub mem (lo * item) ((hi - lo) * item) in
      let r =
        f (Nx_buffer.reinterpret dt (Nx_buffer.of_bigarray1 bytes)) `Borrowed
      in
      ignore (Sys.opaque_identity buf);
      r
  | None ->
      let host = Nx_buffer.create dt (hi - lo) in
      with_window buf ~off:(lo * item)
        ~len:((hi - lo) * item)
        (fun w -> copyout_into (Hashtbl.create 1) w ~dst_off:0 host);
      f host `Copied

(* The elements of view [v] of one buffer's storage. *)
let read_window : type a b.
    (a, b) ND.t -> Tolk.Device.Buffer.t -> NV.t -> (a, b) Nx_buffer.t =
 fun dt buf v ->
  let n = NV.numel v in
  if n = 0 then Nx_buffer.create dt 0
  else
    let lo, hi = extent v in
    with_storage_range dt buf ~lo ~hi @@ fun src how ->
    if how = `Borrowed then
      bytes_from_device := !bytes_from_device + (n * ND.itemsize dt);
    if NV.is_c_contiguous v && how = `Copied then src
    else begin
      let dst = Nx_buffer.create dt n in
      if NV.is_c_contiguous v then Nx_buffer.blit ~src ~dst
      else gather_view src ~base:lo v dst;
      dst
    end

let read : type a b. (a, b) Nx_effect.resident -> (a, b) Nx_buffer.t =
 fun r ->
  drain_releases ();
  match store_of r.r_cell with
  | None -> assert false (* nx reads held and consumed values itself *)
  | Some { s_bufs = []; _ } ->
      Nx_buffer.create r.r_dtype 0 (* an empty value has no buffer *)
  | Some s ->
      let shape = Nx_effect.global r.r_placement (NV.shape r.r_view) in
      Nx_effect.assemble r
        (Array.map (fun n -> (0, n)) shape)
        (fun d v ->
          let dev, buf = buffer_on s d in
          Tolk.Device.synchronize dev;
          read_window r.r_dtype buf v)

(* Allocate [buf] on [dev], whose device value is [d], collecting first past the
   budget. An allocation that fails collects and retries once (the LRU allocator
   has flushed its own cache by then); a second failure is the device's
   [Out_of_memory]. *)
let allocate d buf =
  drain_releases ();
  let n = Tolk.Device.Buffer.nbytes buf in
  let m = (Gc.quick_stat ()).major_collections in
  if m <> !majors then begin
    majors := m;
    allocated := 0
  end;
  if !allocated + n > resident_budget () then collect ();
  allocated := !allocated + n;
  (* Allocators report an exhausted device with [Failure]. *)
  try Tolk.Device.Buffer.ensure_allocated buf
  with Failure _ -> (
    collect ();
    try Tolk.Device.Buffer.ensure_allocated buf
    with Failure _ -> raise (Nx.Device.Out_of_memory (d, n)))

(* Raise unless [dev] can hold [dt]. *)
let check_holds (type a b) d dev (dt : (a, b) ND.t) =
  if not (holds dev dt) then
    invalid_arg
      (Printf.sprintf "Nx.place: %s cannot hold %s" (Nx.Device.name d)
         (ND.to_string dt))

(* Release buffers that no cell owns yet. *)
let release_unowned bufs = List.iter Tolk.Device.Buffer.deallocate bufs

(* [allocate_all ds devs ~size dt ~nolru] is a buffer of [size] elements of [dt]
   on each device, allocated; if one fails, those allocated before it are
   released before the failure is raised. *)
let allocate_all ds devs ~size dt ~nolru =
  let made = ref [] in
  try
    List.map2
      (fun d dev ->
        let buf =
          Tolk.Device.create_buffer ~size ~dtype:(tolk_dtype dt)
            ~spec:{ Tolk.Device.Buffer_spec.default with nolru }
            dev
        in
        allocate d buf;
        made := buf :: !made;
        buf)
      ds devs
  with e ->
    release_unowned !made;
    raise e

(* Move the placed value [r], whose storage is [s], into [bufs], one per device
   of [devs] holding the window of [windows]. Each window takes, from every tile
   of the source that it meets, the elements they share, read from a device
   holding that tile: its own device when it holds it. When every such piece is
   a contiguous run of both storages, tolk copies it between the buffers, device
   to device where the backend can and through the host in chunks otherwise.

   A window with a piece that is not contiguous is gathered on the host a block
   of rows at a time (at least one row, else at most a chunk) through the
   engine's own read, and uploaded. This is the one transfer rune makes itself,
   an exception to moves belonging to tolk: tolk copies contiguous buffers only,
   and a strided piece needs a copy compiled on its source device first, which
   stage 3 brings. *)
let transfer : type a b.
    scratch ->
    (a, b) Nx_effect.resident ->
    store ->
    Tolk.Device.t list ->
    (int * int) array list ->
    Tolk.Device.Buffer.t list ->
    unit =
 fun sc r s devs windows bufs ->
  let q = r.r_placement and dt = r.r_dtype in
  let shape = Nx_effect.global q (NV.shape r.r_view) in
  let item = ND.itemsize dt in
  (* The source's distinct tiles, each with the buffers holding it. *)
  let tiles =
    List.fold_left
      (fun tiles d ->
        let holder = buffer_on s d in
        let w = Nx.Placement.window q shape d in
        match List.assoc_opt w tiles with
        | Some holders -> (w, holders @ [ holder ]) :: List.remove_assoc w tiles
        | None -> tiles @ [ (w, [ holder ]) ])
      [] (Nx.Placement.devices q)
  in
  List.iter
    (fun (_, holders) ->
      List.iter (fun (dev, _) -> Tolk.Device.synchronize dev) holders)
    tiles;
  let offset v = NV.offset v * item in
  List.iter2
    (fun (dev, buf) w ->
      let local = NV.create (Nx_effect.extents w) in
      let pieces =
        List.filter_map
          (fun (t, holders) ->
            Option.map
              (fun i ->
                let src =
                  match List.assq_opt dev holders with
                  | Some b -> b
                  | None -> snd (List.hd holders)
                in
                ( src,
                  NV.shrink r.r_view (Nx_effect.within t i),
                  NV.shrink local (Nx_effect.within w i) ))
              (Nx_effect.intersect w t))
          tiles
      in
      if
        List.for_all
          (fun (_, sv, dv) -> NV.is_c_contiguous sv && NV.is_c_contiguous dv)
          pieces
      then
        List.iter
          (fun (src, sv, dv) ->
            let len = NV.numel sv * item in
            with_window src ~off:(offset sv) ~len (fun src ->
                with_window buf ~off:(offset dv) ~len (fun dst ->
                    Tolk.Device.Buffer.copy_from ~dst ~src)))
          pieces
      else begin
        let e = Nx_effect.extents w in
        let row = numel e / e.(0) in
        let rows = Int.max 1 (chunk_bytes / (row * item)) in
        let r0 = ref 0 in
        while !r0 < e.(0) do
          let r1 = Int.min e.(0) (!r0 + rows) in
          let block = Array.copy w in
          block.(0) <- (fst w.(0) + !r0, fst w.(0) + r1);
          let host =
            Nx_effect.assemble r block (fun d v ->
                read_window dt (snd (buffer_on s d)) v)
          in
          copyin_at sc dev buf
            ~off:(!r0 * row * item)
            (Nx_effect.from_host Nx_effect.host_tensor_context host);
          r0 := r1
        done
      end)
    (List.combine devs bufs) windows;
  List.iter Tolk.Device.synchronize devs

(* The engine of rune's devices, one value per tolk backend, so that nx refuses
   a placement over two backends. [make_placed] wraps buffers already on the
   devices as a placed value whose cell releases them when it is unreachable;
   [place_on] puts each device's window of a value on it: from the host, an
   upload of the window alone, read from its file when it is a mapped one; from
   rune's devices, a [transfer]. An upload from a mapped file bypasses the
   allocator's cache, so a dropped model returns to the system rather than
   staying parked in it. *)
let rec make_placed : type a b.
    Nx_effect.placement ->
    Tolk.Device.t list ->
    nolru:bool ->
    (a, b) ND.t ->
    NV.t ->
    Tolk.Device.Buffer.t list ->
    (a, b) Nx_effect.t =
 fun placement devices ~nolru dt view bufs ->
  let s =
    {
      s_devices = devices;
      s_nbytes =
        List.fold_left (fun a b -> a + Tolk.Device.Buffer.nbytes b) 0 bufs;
      s_bufs = bufs;
      s_nolru = nolru;
    }
  in
  account s 1;
  let cell = Nx_effect.cell ~placement ~length:(NV.numel view) (Buffers s) in
  if bufs <> [] then
    Gc.finalise
      (fun (c : Nx_effect.cell) ->
        match c.state with
        | Live (Buffers s) when s.s_bufs <> [] ->
            pending_release := s :: !pending_release
        | _ -> ())
      cell;
  Nx_effect.placed placement dt view cell

and place_on : type a b.
    Nx_effect.placement -> (a, b) Nx_effect.t -> (a, b) Nx_effect.t =
 fun p x ->
  let ds = Nx.Placement.devices p in
  let devs = List.map tolk_device_of ds in
  let dt = Nx_effect.dtype x and shape = shape_of x in
  List.iter2 (fun d dev -> check_holds d dev dt) ds devs;
  let windows = List.map (fun d -> Nx.Placement.window p shape d) ds in
  let local = Nx_effect.extents (List.hd windows) in
  let source =
    match x with
    | Placed ({ r_cell; _ } as r) -> (
        match store_of r_cell with
        | Some s when s.s_bufs <> [] -> `Stored (r, s)
        | _ -> `Host (on_host x))
    | _ -> `Host x
  in
  let nolru =
    match source with
    | `Host h -> Nx_buffer.file_range (Nx_effect.to_host h) <> None
    | `Stored _ -> false
  in
  let bufs =
    if numel local = 0 then []
    else allocate_all ds devs ~size:(numel local) dt ~nolru
  in
  if bufs <> [] then (
    let sc = Hashtbl.create 1 in
    try
      match source with
      | `Host h ->
          List.iter2
            (fun (dev, buf) w ->
              copyin_tensor sc dev buf (Nx_effect.shrink h w))
            (List.combine devs bufs) windows
      | `Stored (r, s) -> transfer sc r s devs windows bufs
    with e ->
      release_unowned bufs;
      raise e);
  make_placed p devs ~nolru dt (NV.create local) bufs

(* One engine per tolk backend. *)
let engines : (string, Nx_effect.engine) Hashtbl.t = Hashtbl.create 4

let engine_for backend =
  match Hashtbl.find_opt engines backend with
  | Some e -> e
  | None ->
      let e = { Nx_effect.read; place = place_on } in
      Hashtbl.add engines backend e;
      e

(* Devices, opened by name *)

(* A device's canonical name: its backend in capitals, and its index unless it
   is 0. Metal has one device, whatever the index its runtime is opened with. *)
let canonical name =
  let b = String.uppercase_ascii (backend name) in
  match String.index_opt name ':' with
  | None -> b
  | Some i -> (
      match
        int_of_string_opt (String.sub name (i + 1) (String.length name - i - 1))
      with
      | Some 0 -> b
      | Some _ when String.equal b "METAL" ->
          invalid_arg "Rune.device: Metal has one device, METAL"
      | Some k when k > 0 -> b ^ ":" ^ string_of_int k
      | _ ->
          invalid_arg
            (Printf.sprintf "Rune.device: %s: the index is not a natural number"
               name))

let device name =
  let name = canonical name in
  if String.equal name host_name then Nx.Device.host
  else
    match Hashtbl.find_opt by_name name with
    | Some (d, _) -> d
    | None ->
        if not (List.mem (backend name) backends) then
          invalid_arg (Printf.sprintf "Rune.device: unknown device %s" name);
        let dev =
          try Tolk.Device.get name
          with Failure msg ->
            invalid_arg
              (Printf.sprintf "Rune.device: device %s unavailable: %s" name msg)
        in
        let d = Nx_effect.Device.make name (engine_for (backend name)) in
        Hashtbl.add by_name name (d, dev);
        d

(* Metal exposes one device, whatever the index; the other backends number
   theirs from 0, and the first index that does not open ends the list. *)
let devices name =
  let name = String.uppercase_ascii name in
  if String.contains name ':' then
    invalid_arg
      (Printf.sprintf
         "Rune.devices: %s names one device; pass its backend, such as %s" name
         (backend name));
  if String.equal name host_name then [ Nx.Device.host ]
  else if String.equal name "METAL" then [ device name ]
  else
    let rec from i =
      match device (Printf.sprintf "%s:%d" name i) with
      | d -> d :: from (i + 1)
      | exception Invalid_argument _ -> []
    in
    device name :: from 1

(* The [DEV] environment variable names the default backend; without it, the
   first backend that opens, in the order of [backends], the host last. *)
let default_device =
  let chosen =
    lazy
      (match Tolk.Helpers.Context_var.get Tolk.Helpers.dev with
      | target :: _ when target.Tolk_uop.Target.device <> "" ->
          device target.device
      | _ ->
          Tolk.Helpers.select_first_inited ~message:"Rune: no usable device"
            (List.map (fun b () -> device b) backends))
  in
  fun () -> Lazy.force chosen

let create_fresh_buffer d dev dtolk n =
  let buf = Tolk.Device.create_buffer ~size:n ~dtype:dtolk dev in
  allocate d buf;
  buf

(* Compiled traces *)

(* A value the traced function returns: its placeholder (dtype and shape), the
   buffer node realization assigned it ([None] when it has no elements: every
   call returns a fresh empty tensor), and its placement. *)
type output = {
  o_value : packed;
  o_node : U.t option;
  o_place : Nx.Placement.t;
}

(* An output that may take an input's storage: its buffer node's tag, the
   input's position, and the result's path as [RUNE_JIT_DEBUG] names it. *)
type lend = { l_otag : int; l_input : int; l_result : string }

type 'q compiled = {
  cp_device : Tolk.Device.t; (* the first of [cp_devices], which compiles *)
  cp_devices : (Nx.Device.t * Tolk.Device.t) list; (* where it runs *)
  cp_zero_copy : bool;
  cp_ctx : Nx_effect.context;
  cp_linear : U.t;
  cp_vars : (string * int) list;
  cp_input_uops : U.t array;
      (* default storage for the schedule's PARAM slots; replay copies this
         array and supplies its input, output and shared-arena owners *)
  cp_inputs : input array; (* one per leaf visit, in traversal order *)
  cp_consumed : bool array;
      (* per input position, whether a call consumes it: its storage is marked
         consumed before the first kernel, and released or handed to an output
         once the call has run *)
  cp_consumptions : Nx_effect.consumption array;
      (* per input position, its argument and path, which a consumed storage
         records *)
  cp_names : string array;
      (* per input position, its argument and path as errors name them *)
  cp_wrapped : (packed * Obj.t) array;
      (* captures bound by aliasing host memory: kernels read that memory on
         every call, so it must stay reachable while the trace can run *)
  cp_captures : Nx_effect.cell array;
      (* the cells of the placed values the program captures, bound or copied: a
         consumed leaf may reach none of them (rule 4) *)
  cp_bound : (Nx_effect.cell * packed) array;
      (* resident captures whose device buffers are this program's constants:
         the values stay reachable while the trace can run, and their cells
         count this binding until the record is collected *)
  cp_outputs : output array;
      (* the distinct values the traced function returns, in the order its
         result first walks them *)
  cp_results : int array;
      (* per result leaf in walk order, the output it is (an index into
         [cp_outputs]) *)
  cp_first : bool array;
      (* per result leaf, whether it is the first in walk order whose output
         resolves to its buffer node: it takes the node's storage, and each
         other one is a copy *)
  cp_lends : lend list;
      (* the consumed inputs whose storage an output may take, chosen once when
         the program compiles (see Lending below) *)
  cp_prefills : (U.t * int) list;
      (* output PARAM -> traversal position of the input leaf whose value
         the output starts from: an indexed write lands in it, and the program
         never copies the input into it *)
  cp_reserved : (int, unit) Hashtbl.t;
      (* tags of input arguments and owned constants: pass-through outputs
         copy their value or take a consumed input's storage *)
  cp_arenas : U.t list;
      (* the memory planner's arena PARAMs, supplied at every call with their
         devices' shared arenas (see [shared_arena]) *)
  cp_skeleton : 'q; (* the traced result, the template results are rebuilt in *)
  cp_scratch : scratch; (* staging bytes reused across replays *)
}

module Ops = Tolk_uop.Ops

let owned_uop node bufs =
  match U.device_of node, bufs with
  | Some (U.Multi _), _ -> U.mstack (List.map U.from_buffer bufs)
  | _, [buf] -> U.from_buffer buf
  | _ -> invalid_arg "Rune.jit: buffer placement disagrees with its graph node"

let argument_slot node =
  match U.as_param node with
  | Some {param; _} -> param.slot
  | None -> invalid_arg "Rune.jit: replay storage must be a PARAM"

(* Planned arenas are byte buffers reached through views, excluding storage
   held by the caller. Their owners can grow between calls to different JITs. *)
let arena_nodes bound linear =
    let seen = U.Tbl.create 4 in
    List.iter (fun node -> U.Tbl.replace seen node ()) bound;
    U.toposort ~enter_calls:true linear
    |> List.filter_map (fun u ->
        match U.contiguous_view u with
        | Some (src, _)
          when U.op src = Ops.Buffer && TD.equal (U.dtype src) TD.int8
               && not (U.Tbl.mem seen src) ->
            U.Tbl.add seen src ();
            Some src
        | _ -> None)

let parameterize nodes =
  let seen = U.Tbl.create (List.length nodes) in
  List.filter_map (fun node ->
      if U.Tbl.mem seen node then None
      else begin
        let param = U.param_like node ~slot:(U.Tbl.length seen) in
        U.Tbl.add seen node ();
        Some (node, param)
      end) nodes

(* Lending: writing an output over a consumed input's storage.

   With tensors as values a compiled carry (parameters, optimizer state, a KV
   cache) returns fresh outputs every call, and a call releases the inputs it
   consumes afterwards, so it holds two generations of state on the device. When
   an output may safely take a consumed input's buffer, it is bound to that
   buffer instead of a fresh one and the input's storage moves to the output
   value: one generation, no copy, and results identical.

   An output may take a consumed input's storage when their dtypes, byte sizes
   and placements are equal, no kernel reads the input after the first kernel
   that writes the output, and that kernel reads the input only when the output
   derives from it at its own index: every path from the input's buffer node to
   the output's node passes only through elementwise operations, casts of equal
   width, reshapes and contiguous markers, so the kernel storing the output
   reads the input only at the index it writes, whatever the scheduler fuses
   into it (RFC 0001's conditions). An indexed write into an input is a buffer
   the program writes at loaded indices and never fills: replay gives it the
   input's value (see [write_destination]), and taking that input's storage is
   how it gets it for free; the kernel that writes it reads at indices of its
   own, so it must not read the input at all.

   Partners are chosen once per program, in three passes over the outputs in
   walk order: the outputs of an indexed write, each of which may take only the
   input it starts from; then the outputs that derive from a consumed input at
   their own index, a consumed input returned unchanged included, each taking
   the first such input in walk order; then the rest, in increasing order of
   their first write, each taking the free input read last longest ago, which
   lends as much storage as any pairing can, since an input free for one output
   is free for every later-written one. An input some other output returns
   unchanged lends only to it: that output copies it after the kernels. Each
   storage lends at most once. A kernel stores one buffer ([split_store]); one
   that stores several is treated as reading its inputs at other indices, so it
   lends only what it does not read. The analyses are one pass over the schedule
   and one over the outputs' graph, with the consumed inputs as bitsets. Replay
   adds what only it knows: the input must have seeded from storage no program
   binds. Over several devices a storage is one buffer per device, all of which
   an output takes, so the partners' placements must be equal too. *)

(* The buffers the memory planner must leave alone: those under the nodes replay
   binds (inputs, constants, outputs), and every buffer a staged loop mentions,
   since its compiled body addresses them by node. *)
let held_buffers bound linear =
  let buffers n =
    List.filter (fun n -> U.op n = Ops.Buffer) (U.toposort ~enter_calls:true n)
  in
  let opaque =
    List.concat_map
      (function Kernel _ -> [] | Opaque c -> buffers c)
      (schedule_calls linear)
  in
  List.concat_map buffers bound @ opaque

(* The kernels that first and last mention each buffer, by tag, the buffers an
   opaque call mentions, and the kernels that store more than one buffer. *)
type mentions = {
  first : (int, int) Hashtbl.t;
  last : (int, int) Hashtbl.t;
  opaque : (int, unit) Hashtbl.t;
  several : (int, unit) Hashtbl.t;
}

let mentions_of linear =
  let m =
    {
      first = Hashtbl.create 64;
      last = Hashtbl.create 64;
      opaque = Hashtbl.create 8;
      several = Hashtbl.create 4;
    }
  in
  let tags call =
    match U.as_call call with
    | Some { args; _ } -> buffer_tags args
    | None -> []
  in
  List.iteri
    (fun k -> function
      | Opaque c -> List.iter (fun t -> Hashtbl.replace m.opaque t ()) (tags c)
      | Kernel { args; several } ->
          if several then Hashtbl.replace m.several k ();
          List.iter
            (fun t ->
              if not (Hashtbl.mem m.first t) then Hashtbl.add m.first t k;
              Hashtbl.replace m.last t k)
            (buffer_tags args))
    (schedule_calls linear);
  m

(* No kernel reads the buffer [itag] after the first kernel that writes [otag],
   and neither buffer is touched by an opaque call. Unless [strict], that first
   kernel may read [itag] itself, when it stores [otag] alone. *)
let allows m ~strict ~itag ~otag =
  (not (Hashtbl.mem m.opaque itag))
  && (not (Hashtbl.mem m.opaque otag))
  &&
  match (Hashtbl.find_opt m.first otag, Hashtbl.find_opt m.last itag) with
  | Some o, Some i ->
      if strict || Hashtbl.mem m.several o then i < o else i <= o
  | Some _, None -> true
  | None, _ -> false

(* Bitsets over the consumed inputs. *)
let bits_empty = [||]

let bits_union a b =
  if a == b || b == bits_empty then a
  else if a == bits_empty then b
  else Array.map2 ( lor ) a b

let bits_mem a i = a != bits_empty && a.(i / 62) land (1 lsl (i mod 62)) <> 0

let bits_single words i =
  let a = Array.make words 0 in
  a.(i / 62) <- 1 lsl (i mod 62);
  a

(* For every node of [roots]' graphs, by tag, the inputs of [inodes] it reaches
   and those it reaches through an operation that moves elements: [roots]
   derives from input [b] at its own index exactly when it reaches [b] and not
   through such an operation. One pass, children first. *)
let derivations roots inodes =
  let words = (Array.length inodes + 61) / 62 in
  let index = Hashtbl.create 16 in
  Array.iteri (fun b n -> Hashtbl.replace index (U.tag n) b) inodes;
  let table = Hashtbl.create 256 in
  List.iter
    (fun u ->
      let tag = U.tag u in
      let r =
        match Hashtbl.find_opt index tag with
        | Some b -> (bits_single words b, bits_empty)
        | None ->
            let reach = ref bits_empty and moved = ref bits_empty in
            Array.iter
              (fun s ->
                match Hashtbl.find_opt table (U.tag s) with
                | Some (r, m) ->
                    reach := bits_union !reach r;
                    moved := bits_union !moved m
                | None -> ())
              (U.src u);
            if !reach != bits_empty && not (keeps_index u) then
              moved := bits_union !moved !reach;
            (!reach, !moved)
      in
      Hashtbl.replace table tag r)
    (U.toposort ~enter_calls:false (U.sink roots));
  table

(* Program keys

   Two calls share a program exactly when they run on the same devices in the
   same order, their arguments visit the same leaves at the same paths and make
   the same reports ([Nx.Ptree.Skeleton]), and their leaves have equal dtypes,
   shapes, placements and layouts, a host leaf counting as a copy on each
   device. The hash only finds candidates. *)

type leaf_sig = {
  l_dtype : string;
  l_shape : int array;
  l_place : Nx.Placement.t;
  l_layout : layout;
}

type key = {
  k_devices : Nx.Device.t list;
  k_skeleton : Nx.Ptree.Skeleton.t;
  k_leaves : leaf_sig array;
  k_hash : int;
}

let key_of ds ~hash skeleton leaves shapes seeds =
  {
    k_devices = ds;
    k_skeleton = skeleton;
    k_leaves =
      Array.mapi
        (fun i (Nx.P x) ->
          {
            l_dtype = ND.to_string (Nx_effect.dtype x);
            l_shape = shapes.(i);
            l_place = leaf_placement ds x;
            l_layout = layout_of seeds.(i);
          })
        leaves;
    k_hash = hash;
  }

(* The hash of a call's key, computed without building the key. *)
let hash_call skeleton leaves shapes =
  let h = ref (Nx.Ptree.Skeleton.hash skeleton) in
  for i = 0 to Array.length leaves - 1 do
    let (Nx.P x) = leaves.(i) in
    h := (31 * !h) + Hashtbl.hash (Nx_effect.dtype x);
    let shape = shapes.(i) in
    for d = 0 to Array.length shape - 1 do
      h := (31 * !h) + shape.(d)
    done
  done;
  !h land max_int

(* Whether [key] is the key of a call on [ds] whose argument flattened to
   [skeleton] and [leaves], of [shapes], seeded as [seeds]. *)
let matches key ds ~hash skeleton leaves shapes seeds =
  let n = Array.length leaves in
  let rec same_leaves i =
    i = n
    ||
    let l = key.k_leaves.(i) in
    let (Nx.P x) = leaves.(i) in
    String.equal l.l_dtype (ND.to_string (Nx_effect.dtype x))
    && l.l_shape = shapes.(i)
    && Nx.Placement.equal l.l_place (leaf_placement ds x)
    && l.l_layout = layout_of seeds.(i)
    && same_leaves (i + 1)
  in
  key.k_hash = hash
  && List.equal ( == ) key.k_devices ds
  && Array.length key.k_leaves = n
  && same_leaves 0
  && Nx.Ptree.Skeleton.equal key.k_skeleton skeleton

let same_sig a b =
  String.equal a.l_dtype b.l_dtype
  && a.l_shape = b.l_shape
  && Nx.Placement.equal a.l_place b.l_place
  && a.l_layout = b.l_layout

let describe_leaf l =
  Format.asprintf "%s [%s]%s%s" l.l_dtype
    (String.concat "; " (Array.to_list (Array.map string_of_int l.l_shape)))
    (match Nx.Placement.devices l.l_place with
    | [ _ ] -> ""
    | _ -> Format.asprintf " %a" Nx.Placement.pp l.l_place)
    (if l.l_layout = dense then ""
     else
       Printf.sprintf " viewed from element %d%s" l.l_layout.skip
         (match l.l_layout.strides with
         | None -> ""
         | Some s ->
             Printf.sprintf " with strides [%s]"
               (String.concat "; " (Array.to_list (Array.map string_of_int s)))))

(* The first difference between [key], a call's, and [previous], the closure's
   previous call's: the device, the first differing visit, or the first leaf
   whose signature differs, named by [names]. *)
let key_difference ~names key previous =
  if not (List.equal ( == ) key.k_devices previous.k_devices) then
    Format.asprintf "the devices: %a here, %a in the previous key" pp_devices
      key.k_devices pp_devices previous.k_devices
  else
    match
      Nx.Ptree.Skeleton.diff ~this:"here" key.k_skeleton
        ~that:"in the previous key" previous.k_skeleton
    with
    | Some m -> m
    | None -> (
        let n = Array.length key.k_leaves in
        let rec first i =
          if i >= n then None
          else if not (same_sig key.k_leaves.(i) previous.k_leaves.(i)) then
            Some i
          else first (i + 1)
        in
        match first 0 with
        | Some i ->
            Printf.sprintf "%s: %s here, %s in the previous key" names.(i)
              (describe_leaf key.k_leaves.(i))
              (describe_leaf previous.k_leaves.(i))
        | None -> "no difference")

(* The paths of [x]'s leaves under [s], in walk order, as printed. *)
let leaf_paths s x =
  Array.of_list
    (List.filter_map
       (function
         | Nx.Ptree.Leaf p -> (
             match Nx.Ptree.Path.segments p with
             | [] -> Some "the root"
             | _ -> Some (Nx.Ptree.Path.to_string p))
         | Report _ -> None)
       (Nx.Ptree.visits s x))

(* What a program knows of each leaf of its arguments, in walk order: its
   argument, whether the call consumes it, where, and its name in errors, its
   path in the arguments. *)
type leaf_info = {
  arguments : int array;
  consumed : bool array;
  consumptions : Nx_effect.consumption array;
  names : string array;
}

(* [trace_compile ~devices f params leaves] traces and compiles [f] for a call
   over [devices], each leaf seeding the program at its placement in
   [placements] and read under its layout in [layouts]. *)
let trace_compile (type p q) ~devices:(ds, devs) ~zero_copy ~info ~const_cache
    ~decided ~placements ~layouts ?beam ?beam_parallel (p : p Nx.Ptree.t)
    (q : q Nx.Ptree.t) (f : p -> q) (params : p) (leaves : Nx.packed array) :
    q compiled =
  let consumed i = info.consumed.(i) in
  let dev = List.hd devs in
  let multi = List.compare_length_with ds 1 > 0 in
  incr trace_counter;
  let st =
    {
      st_id = !trace_counter;
      st_device = dev;
      st_devices = ds;
      st_decided = decided;
      refusal = None;
      st_takes_storage = (fun i -> consumed i && not zero_copy);
      st_ctx = Nx_effect.create_context ();
      table = Tensor_map.Tbl.create 64;
      captures = Tensor_map.Tbl.create 16;
      input_tags = Hashtbl.create 16;
      prefills = [];
      consts = [];
      bound = Tensor_map.Tbl.create 16;
      bound_consts = [];
      scan_stacks = Tbl.create 4;
      scan_closed = Tbl.create 4;
      scan_collectors = [];
      scan_writes = [];
      scan_bodies = 0;
    }
  in
  (* One placeholder and one input record per leaf visit, in traversal order, so
     replay pairs current leaves positionally: a tensor behind two leaves is two
     inputs, equal on this call and free to differ on the next. *)
  let inputs = ref [] and placeholders = ref [] and tensors = ref [] in
  let pos = ref 0 in
  Array.iter
    (fun (Nx.P leaf) ->
      let dtolk = tolk_dtype (Nx_effect.dtype leaf) in
      (* On a guessed device the refusal waits for the trace, whose captures may
         move the program where the dtype is held. *)
      if not (holds dev (Nx_effect.dtype leaf)) then begin
        let e =
          Invalid_argument
            (Printf.sprintf
               "Rune.jit: the argument at %s is %s, which %s cannot hold"
               info.names.(!pos)
               (ND.to_string (Nx_effect.dtype leaf))
               (Tolk.Device.name dev))
        in
        if decided = Guessed then refuse st e else raise e
      end;
      (* The trace-level tensor carries the global shape: a split leaf is a
         buffer of one slice on each device under tolk's [Unshard], which
         multiplies the cut axis back up, and a copy is the whole value on each
         device. A view of part of a storage binds the range of storage it
         reaches. *)
      let place = placements.(!pos) and layout = layouts.(!pos) in
      let local = local_shape place (shape_of leaf) in
      let size = layout_size layout local in
      let node = make_node st dtolk size in
      let tt = whole_tensor place (layout_tensor node layout local) in
      let bufs =
        List.map (fun d -> Tolk.Device.create_buffer ~size ~dtype:dtolk d) devs
      in
      let ph =
        Nx_effect.traced st.st_ctx (Nx_effect.dtype leaf) (shape_of leaf)
          (Node { trace = st.st_id; tensor = tt; place })
      in
      placeholders := Nx.P ph :: !placeholders;
      tensors := tt :: !tensors;
      Hashtbl.replace st.input_tags (U.tag node) !pos;
      inputs :=
        {
          i_node = node;
          i_place = place;
          i_bufs = bufs;
          i_dtype = ND.to_string (Nx_effect.dtype leaf);
          i_numel = size;
        }
        :: !inputs;
      incr pos)
    leaves;
  let ph_params = Nx.Ptree.rebuild p ~like:params (List.rev !placeholders) in
  let y =
    Gate.with_transform (fun () ->
        Effect.Deep.match_with f ph_params (handler st))
  in
  (* Collect the values the result returns, each once, and which of them each
     result leaf is; a value the trace never saw is a constant passing through
     unchanged. *)
  let index = Tensor_map.Tbl.create 16
  and distinct = ref []
  and count = ref 0 in
  let results =
    Nx.Ptree.fold q
      (fun (type a b) _ (leaf : (a, b) Nx_effect.t) acc ->
        let k =
          match Tensor_map.Tbl.find_opt index (Key leaf) with
          | Some k -> k
          | None ->
              let k = !count in
              incr count;
              Tensor_map.Tbl.replace index (Key leaf) k;
              distinct :=
                (k, Packed (Nx_effect.dtype leaf, leaf), tolk_of st leaf)
                :: !distinct;
              k
        in
        k :: acc)
      y []
    |> List.rev |> Array.of_list
  in
  Option.iter raise st.refusal;
  let empty, outs =
    List.partition
      (fun (_, Packed (_, ph), _) -> numel (shape_of ph) = 0)
      (List.rev !distinct)
  in
  (* A result computed purely from trace-time constants (say, the zero gradient
     of an unused parameter) has no device anywhere in its graph, so the
     scheduler would materialize nothing for it. Anchor such results with a
     bitwise-identity multiply by a device-resident scalar one. *)
  let anchor : type a b. (a, b) ND.t -> F.Tensor.t -> F.Tensor.t =
   fun dt tt ->
    match U.device_of (F.Tensor.uop tt) with
    | Some _ -> tt
    | None ->
        let one = Nx_effect.const_scalar st.st_ctx (ND.one dt) dt in
        F.Elementwise.mul tt (tolk_of st one)
  in
  let out_anch =
    List.map
      (fun (key, (Packed (dt, ph) as pk), tt) ->
        (key, pk, anchor dt tt, placement_in st ph))
      outs
  in
  (* RFC 0006's rule 5 pairing, once per program: the consumed leaf each result
     continues, each leaf paired with at most one result, of its dtype and byte
     size. First each result an indexed write lands in takes the leaf it starts
     from; then, in walk order, each result that derives from a consumed leaf at
     its own index takes the first such leaf still free. Lending writes a paired
     result over its leaf's storage where the schedule allows (see Lending), and
     over several devices a paired result keeps its leaf's placement. *)
  let pairing =
    let pairing = Hashtbl.create 8 in
    let carried =
      Array.of_list
        (List.filter consumed (List.init (Array.length leaves) Fun.id))
    in
    if carried <> [||] then begin
      let taken = Array.make (Array.length leaves) false in
      let fits i (Packed (dt, ph)) =
        let (Nx.P leaf) = leaves.(i) in
        (not taken.(i))
        && String.equal (ND.to_string (Nx_effect.dtype leaf)) (ND.to_string dt)
        && numel (shape_of leaf) = numel (shape_of ph)
      in
      let pair key i =
        taken.(i) <- true;
        Hashtbl.replace pairing key i
      in
      List.iter
        (fun (key, pk, tt, _) ->
          match written_buffer (F.Tensor.uop tt) with
          | Some v -> (
              match
                List.find_opt (fun (b, _) -> U.buf_uop v == b) st.prefills
              with
              | Some (_, input) ->
                  let i = Hashtbl.find st.input_tags (U.tag input) in
                  if consumed i && fits i pk then pair key i
              | None -> ())
          | None -> ())
        out_anch;
      let tensors = Array.of_list (List.rev !tensors) in
      let table =
        derivations
          (List.map (fun (_, _, tt, _) -> F.Tensor.uop tt) out_anch)
          (Array.map (fun i -> F.Tensor.uop tensors.(i)) carried)
      in
      List.iter
        (fun (key, pk, tt, _) ->
          if not (Hashtbl.mem pairing key) then
            let reach, moved =
              Option.value ~default:(bits_empty, bits_empty)
                (Hashtbl.find_opt table (U.tag (F.Tensor.uop tt)))
            in
            match
              List.find_opt
                (fun b ->
                  bits_mem reach b
                  && (not (bits_mem moved b))
                  && fits carried.(b) pk)
                (List.init (Array.length carried) Fun.id)
            with
            | Some b -> pair key carried.(b)
            | None -> ())
        out_anch
    end;
    pairing
  in
  (* A paired result of its leaf's shape keeps the leaf's placement: where nx's
     rules put it elsewhere, the program reshards it at its end, so a consumed
     carry keeps its placement from call to call, and one that starts on the
     host stays a copy on each device. *)
  let out_anch =
    List.map
      (fun ((key, (Packed (_, ph) as pk), tt, place) as out) ->
        match Hashtbl.find_opt pairing key with
        | Some i ->
            let (Nx.P leaf) = leaves.(i) in
            let q = placements.(i) in
            if shape_of leaf <> shape_of ph || Nx.Placement.equal place q then
              out
            else begin
              if Lazy.force jit_debug >= 1 then
                Printf.eprintf "rune.jit: %s\n%!"
                  (Format.asprintf
                     "the result paired with the argument at %s is resharded \
                      from %a to %a"
                     info.names.(i) Nx.Placement.pp place Nx.Placement.pp q);
              (key, pk, reshard st place q tt, q)
            end
        | None -> out)
      out_anch
  in
  let out_uops =
    let outs_u = List.map (fun (_, _, tt, _) -> F.Tensor.uop tt) out_anch in
    (* Only an output can be given its starting value by replay. An indexed
       write into any other buffer starts from its input by a copy in the
       program, as a computed destination does. *)
    let is_output b =
      List.exists
        (fun u ->
          match written_buffer u with
          | Some v -> U.buf_uop v == b
          | None -> false)
        outs_u
    in
    let kept, copied = List.partition (fun (b, _) -> is_output b) st.prefills in
    st.prefills <- kept;
    let filled (b, input) =
      (b, U.after ~src:b ~deps:[ U.store ~dst:b ~value:input () ])
    in
    let outs_u =
      if copied = [] then outs_u
      else
        U.children
          (U.substitute ~walk:true (List.map filled copied) (U.sink outs_u))
    in
    (* Canonicalize sharding before allocation: rewrite the multi-device rules
       over the whole output graph now, so every split value reaching a sink is
       a syntactic [Unshard] and buffer allocation sizes its output per slice
       (copies allocate full-size on every device). Scheduling reapplies the
       same rules; the rewrite is idempotent. *)
    U.children (U.graph_rewrite Tolk.Multi.multi_pm (U.sink outs_u))
  in
  (* Resolve each output to the buffer node realization assigned it. An output
     whose node is a graph buffer under identity wrappers (an input or constant
     returned unchanged: [U.contiguous] elides itself on buffer-identity
     sources, so such outputs are never scheduled) reads that buffer directly.
     Over several devices the mapped node may wrap the buffer in [Unshard];
     follow it down. *)
  let rec strip_identity u =
    match U.op u with
    | Tolk_uop.Ops.Buffer -> Some u
    | Tolk_uop.Ops.Reshape | Tolk_uop.Ops.Unshard ->
        if Array.length (U.src u) > 0 then strip_identity (U.src u).(0)
        else None
    | _ -> None
  in
  let rec moves u =
    match U.op u with
    | Tolk_uop.Ops.Buffer -> true
    | op when Tolk_uop.Ops.Group.is_movement op ->
        Array.length (U.src u) > 0 && moves (U.src u).(0)
    | _ -> false
  in
  (* A buffer after the effects that wrote it is already the result: it is sunk
     as it stands, under no reshape, since a [contiguous] over it would copy it
     out. Any other movement of a buffer (a shrink of an input, say) is copied
     into an output of its own: the schedule would leave a [contiguous] of it a
     view of that buffer, which no output node stands for. *)
  let out_conts =
    List.map2
      (fun (key, pk, _, tracked) u ->
        let c =
          match written_buffer u with
          | Some v -> v
          | None when strip_identity u = None && moves u ->
              Option.get
                (written_buffer
                   (F.Tensor.uop (F.Creation.clone (F.Tensor.of_uop u))))
          | None -> U.contiguous ~src:u ()
        in
        let place = laid_out ds u in
        if not (Nx.Placement.equal tracked place) then
          err "Rune.jit: a result lands at %s where nx's rules put it at %s"
            (Format.asprintf "%a" Nx.Placement.pp place)
            (Format.asprintf "%a" Nx.Placement.pp tracked);
        (key, pk, u, place, c))
      out_anch out_uops
  in
  let sink = U.sink (List.map (fun (_, _, _, _, c) -> c) out_conts) in
  let sink, buffer_map = Tolk.Bufferize.run sink in
  st.prefills <-
    List.map
      (fun (node, input) ->
        let node =
          Option.value (Hashtbl.find_opt buffer_map (U.tag node)) ~default:node
        in
        (U.buf_uop node, input))
      st.prefills;
  let call = Tolk.Callify.transform_to_call sink in
  let resolve what u c =
    let unwrap node = U.buf_uop node in
    match Hashtbl.find_opt buffer_map (U.tag c) with
    | Some node -> unwrap node
    | None -> (
        match strip_identity u with
        | Some b -> b
        | None -> err "Rune.jit: %s was not scheduled to a buffer" what)
  in
  let cp_outputs =
    let a = Array.make !count None in
    List.iter
      (fun (k, pk, u, place, c) ->
        a.(k) <-
          Some
            {
              o_value = pk;
              o_node = Some (resolve "an output of the traced function" u c);
              o_place = place;
            })
      out_conts;
    List.iter
      (fun (k, pk, _) ->
        a.(k) <-
          Some
            {
              o_value = pk;
              o_node = None;
              o_place = Nx.Placement.replicated ds;
            })
      empty;
    Array.map Option.get a
  in
  let output_nodes =
    Array.to_list cp_outputs |> List.filter_map (fun o -> o.o_node)
  in
  let input_nodes = List.map (fun inp -> inp.i_node) (List.rev !inputs) in
  let constant_nodes = List.map (fun (node, _, _) -> node) st.consts
      @ List.map (fun (node, _, _) -> node) st.bound_consts in
  let held = input_nodes @ constant_nodes @ output_nodes in
  let replay_nodes arenas =
    input_nodes
    @ List.filter (fun node -> not (List.exists (U.equal node) constant_nodes))
        output_nodes
    @ arenas in
  (* Persistent compile cache: a hit replaces scheduling and kernel compilation
     with an import of the stored compiled linear, rebound to this trace's fresh
     buffer nodes. Programs over several devices are not cached. *)
  (* The effective beam width: the per-call override when it enables search,
     otherwise the BEAM environment variable. Part of the persistent cache key
     because it changes the compiled kernels. *)
  let effective_beam =
    match beam with
    | Some b when b >= 1 -> b
    | Some _ | None -> env_int "BEAM" 0
  in
  let cache_key =
    if multi then None else Jit_cache.key ~device:dev ~beam:effective_beam call
  in
  let cached = Option.bind cache_key (fun key -> Jit_cache.load ~key call) in
  let linear, var_vals =
    match cached with
    | Some ((linear, _) as hit) ->
        reserve_slots_of linear;
        hit
    | None ->
        (* Schedule under a capture hook, which hands the linear over
           unplanned. *)
        let linear, var_vals =
          let captured = ref None in
          Tolk.Realize.capturing :=
            [ (fun linear var_vals -> captured := Some (linear, var_vals)) ];
          Fun.protect
            ~finally:(fun () -> Tolk.Realize.capturing := [])
            (fun () ->
              ignore
                (Tolk.Schedule.create_linear_with_vars
                   ~get_kernel_graph:Tolk.Rangeify.get_kernel_graph call));
          match !captured with
          | Some lv -> lv
          | None -> err "Rune.jit: scheduling captured no computation"
        in
        let linear =
          Tolk.Schedule.memory_plan_rewrite linear (held_buffers held linear)
        in
        let linear =
          let parameters = parameterize
              (replay_nodes (arena_nodes held linear)) in
          let linear = U.substitute ~walk:true parameters linear in
          let compile () =
            Tolk.Realize.compile_linear ~device:dev ?beam ~to_program linear
          in
          let compiled = match beam_parallel with
            | None -> compile ()
            | Some n ->
                Tolk.Helpers.Context_var.(
                  with_context [ B (Tolk.Search.beam_parallel, n) ] compile) in
          (* Persistent serialization normalizes this trace's external names;
             queue compilation already saw their PARAM semantics. *)
          U.substitute ~walk:true
            (List.map (fun (node, param) -> param, node) parameters) compiled
        in
        (* The scheduler's internal buffer slots come from a counter local to
           this schedule; reserve them globally so later traces (and scan
           bodies) never hand out a colliding buffer slot. *)
        reserve_slots_of linear;
        Option.iter
          (fun key -> Jit_cache.store ~key call linear var_vals)
          cache_key;
        (linear, var_vals)
  in
  let cp_arenas = arena_nodes held linear in
  let parameters = parameterize (replay_nodes cp_arenas) in
  let cp_input_uops = Array.of_list (List.map fst parameters) in
  let constants = ref [] in
  let own node bufs = constants := (node, owned_uop node bufs) :: !constants in
  let reserved = Hashtbl.create 16 in
  List.iter
    (fun inp ->
      Hashtbl.replace reserved (U.tag inp.i_node) ();
      let param = List.assq inp.i_node parameters in
      cp_input_uops.(argument_slot param) <- owned_uop inp.i_node inp.i_bufs)
    !inputs;
  (* Give each constant an owner at compile time: alias its memory when the device
     shares host memory and the tensor is contiguous, copy each device its slice
     otherwise. The staging of these one-time uploads is dropped after
     compilation. *)
  let scratch = Hashtbl.create 8 in
  let wrapped = ref [] in
  List.iter
    (fun (node, cp, (Packed (cdt, src) as pk)) ->
      Hashtbl.replace reserved (U.tag node) ();
      match if zero_copy then wrap_tensor dev src else None with
      | Some (buf, keep) ->
          own node [buf];
          wrapped := (pk, keep) :: !wrapped
      | None ->
          (* One device copy of a capture serves every signature of the closure:
             the bytes are uploaded when the capture is first compiled and later
             compilations reuse the buffers. *)
          let bufs =
            match Tensor_map.Tbl.find_opt const_cache (Key src) with
            | Some bufs -> bufs
            | None ->
                let n = numel (local_shape cp (shape_of src)) in
                let bufs =
                  List.map
                    (fun d ->
                      Tolk.Device.create_buffer ~size:n ~dtype:(tolk_dtype cdt)
                        d)
                    devs
                in
                upload_windows scratch cp src (List.combine ds devs) bufs;
                Tensor_map.Tbl.replace const_cache (Key src) bufs;
                bufs
          in
          own node bufs)
    st.consts;
  (* A bound capture owns the resident buffers through its constant:
     reads leave storage in place, and the value is reachable from the trace. *)
  let views = ref [] in
  let bound =
    List.map
      (fun (node, pk, seed_) ->
        Hashtbl.replace reserved (U.tag node) ();
        match seed_ with
        | Range { cell; bufs; lo; span; _ } ->
            let bufs = List.map (fun buf -> buffer_range buf ~lo ~span) bufs in
            views := bufs @ !views;
            own node bufs;
            (cell, pk)
        | Whole { cell; bufs } ->
            own node bufs;
            (cell, pk)
        | Copy -> assert false (* only a seeded capture is bound *))
      st.bound_consts
  in
  let cp_inputs = Array.of_list (List.rev !inputs) in
  let cp_lends =
    let consumed_inputs =
      List.filter consumed (List.init (Array.length cp_inputs) Fun.id)
      |> Array.of_list
    in
    (* On the host, outputs are host buffers the kernels write in place. *)
    if zero_copy || consumed_inputs = [||] then []
    else begin
      let m = mentions_of linear in
      let position = Hashtbl.create 16 in
      Array.iteri
        (fun i inp -> Hashtbl.replace position (U.tag inp.i_node) i)
        cp_inputs;
      let result_names = leaf_paths q y in
      let first_result = Array.make (Array.length cp_outputs) "" in
      for j = Array.length results - 1 downto 0 do
        first_result.(results.(j)) <- result_names.(j)
      done;
      (* The outputs, each node once, in walk order. *)
      let candidates =
        let seen = Hashtbl.create 8 in
        List.filter_map
          (fun (k, _, _, _, _) ->
            match cp_outputs.(k) with
            | { o_node = Some node; _ } ->
                let otag = U.tag node in
                if Hashtbl.mem seen otag then None
                else begin
                  Hashtbl.replace seen otag ();
                  Some (k, otag)
                end
            | _ -> None)
          out_conts
      in
      (* An input some output returns unchanged is read by that output's copy
         after the kernels, so only that output may take it. *)
      let returned = Hashtbl.create 8 in
      List.iter
        (fun (_, otag) ->
          if Hashtbl.mem position otag then Hashtbl.replace returned otag ())
        candidates;
      let starts_from = Hashtbl.create 4 in
      List.iter
        (fun (b, input) -> Hashtbl.replace starts_from (U.tag b) (U.tag input))
        st.prefills;
      let taken = Array.make (Array.length cp_inputs) false in
      let lendable i k =
        let { o_value = Packed (odt, ph); o_place; _ } = cp_outputs.(k) in
        consumed i
        && (not taken.(i))
        && cp_inputs.(i).i_dtype = ND.to_string odt
        && cp_inputs.(i).i_numel = numel (local_shape o_place (shape_of ph))
        && Nx.Placement.equal cp_inputs.(i).i_place o_place
      in
      let lends = ref [] and paired = Hashtbl.create 8 in
      let pair k otag i =
        taken.(i) <- true;
        Hashtbl.replace paired otag ();
        lends :=
          { l_otag = otag; l_input = i; l_result = first_result.(k) } :: !lends
      in
      (* The pairing's outputs take their partner where the schedule allows: an
         output of an indexed write only when the kernel writing it reads the
         partner at no other index, and an output that returns its partner
         unchanged always. *)
      List.iter
        (fun (k, otag) ->
          match Hashtbl.find_opt pairing k with
          | Some i ->
              let itag = U.tag cp_inputs.(i).i_node in
              let returns_it = Hashtbl.find_opt position otag = Some i in
              if
                lendable i k
                && (returns_it
                   || (not (Hashtbl.mem returned itag))
                      && (not (Hashtbl.mem reserved otag))
                      && allows m
                           ~strict:(Hashtbl.mem starts_from otag)
                           ~itag ~otag)
              then pair k otag i
          | None -> ())
        candidates;
      (* The rest, in increasing order of their first write, take the free input
         read last longest ago. *)
      let written otag =
        Option.value ~default:max_int (Hashtbl.find_opt m.first otag)
      in
      let read_last i =
        Option.value ~default:(-1)
          (Hashtbl.find_opt m.last (U.tag cp_inputs.(i).i_node))
      in
      let rest =
        List.filter
          (fun (_, otag) ->
            not
              (Hashtbl.mem paired otag
              || Hashtbl.mem starts_from otag
              || Hashtbl.mem reserved otag))
          candidates
        |> List.stable_sort (fun (_, a) (_, b) ->
            Int.compare (written a) (written b))
      in
      let free =
        List.stable_sort
          (fun a b -> Int.compare (read_last a) (read_last b))
          (Array.to_list consumed_inputs)
      in
      List.iter
        (fun (k, otag) ->
          match
            List.find_opt
              (fun i ->
                let itag = U.tag cp_inputs.(i).i_node in
                lendable i k
                && (not (Hashtbl.mem returned itag))
                && allows m ~strict:true ~itag ~otag)
              free
          with
          | Some i -> pair k otag i
          | None -> ())
        rest;
      List.rev !lends
    end
  in
  let cp_prefills =
    List.map
      (fun (b, input) ->
        if not (List.exists (fun n -> U.tag n = U.tag b) output_nodes) then
          err "Rune.jit: an indexed write's buffer is not an output";
        let position = ref None in
        Array.iteri
          (fun i inp ->
            if !position = None && inp.i_node == input then position := Some i)
          cp_inputs;
        (b, Option.get !position))
      st.prefills
  in
  (* A result leaf whose output resolves to a node an earlier leaf's did is a
     copy. *)
  let cp_first =
    let seen = Hashtbl.create 8 in
    Array.map
      (fun k ->
        match cp_outputs.(k).o_node with
        | None -> true
        | Some node ->
            let tag = U.tag node in
            (not (Hashtbl.mem seen tag))
            &&
            (Hashtbl.replace seen tag ();
             true))
      results
  in
  let mappings = parameters @ !constants in
  let mapped node = Option.value (List.assq_opt node mappings) ~default:node in
  let linear = U.substitute ~walk:true mappings linear in
  let linear = Tolk.Realize.link_linear
      ~ctx:(Tolk.Realize.exec_context ~input_uops:cp_input_uops ()) linear in
  let cp_inputs = Array.map
      (fun inp -> {inp with i_node = mapped inp.i_node}) cp_inputs in
  let cp_outputs = Array.map
      (fun output -> {output with o_node = Option.map mapped output.o_node})
      cp_outputs in
  let cp_lends = List.map (fun lend ->
      let node = List.find (fun node -> U.tag node = lend.l_otag) output_nodes in
      {lend with l_otag = U.tag (mapped node)}) cp_lends in
  let cp_prefills = List.map (fun (node, input) -> mapped node, input) cp_prefills in
  let cp_arenas = List.map mapped cp_arenas in
  let reserved = Hashtbl.create (List.length input_nodes + List.length constant_nodes) in
  List.iter (fun node -> Hashtbl.replace reserved (U.tag (mapped node)) ())
    (input_nodes @ constant_nodes);
  List.iter (fun ((c : Nx_effect.cell), _) -> c.bound <- c.bound + 1) bound;
  let cp_captures =
    Array.of_list
      (List.map fst bound
      @ List.filter_map
          (fun (_, _, Packed (_, x)) ->
            match x with Nx_effect.Placed r -> Some r.r_cell | _ -> None)
          st.consts)
  in
  let compiled =
    {
      cp_device = dev;
      cp_devices = List.combine ds devs;
      cp_zero_copy = zero_copy;
      cp_ctx = st.st_ctx;
      cp_linear = linear;
      cp_vars = var_vals;
      cp_input_uops;
      cp_inputs;
      cp_consumed = info.consumed;
      cp_consumptions = info.consumptions;
      cp_names = info.names;
      cp_wrapped = Array.of_list !wrapped;
      cp_captures;
      cp_bound = Array.of_list bound;
      cp_outputs;
      cp_results = results;
      cp_first;
      cp_lends;
      cp_prefills;
      cp_reserved = reserved;
      cp_arenas;
      cp_skeleton = y;
      cp_scratch = Hashtbl.create 8;
    }
  in
  (* A bound storage that a call consumed stays for the programs that bind it;
     the last of them to go releases it. *)
  let cells = List.map (fun (cell, _) -> (cell, store_of cell)) bound in
  let views = !views in
  Gc.finalise_last
    (fun () ->
      pending_views := views @ !pending_views;
      List.iter
        (fun ((cell : Nx_effect.cell), store) ->
          cell.bound <- cell.bound - 1;
          match (cell.state, store) with
          | Consumed _, Some s when cell.bound = 0 ->
              pending_release := s :: !pending_release
          | _ -> ())
        cells)
    compiled;
  compiled

let replay (type q) (q : q Nx.Ptree.t) (c : q compiled)
    (leaves : Nx.packed array) (seeds : seed array) : q =
  drain_releases ();
  let input_uops = Array.copy c.cp_input_uops in
  let context = Tolk.Realize.exec_context ~input_uops () in
  let supply node bufs =
    input_uops.(argument_slot node) <- owned_uop node bufs
  in
  let in0 = !bytes_to_device and out0 = !bytes_from_device in
  (* Seed the inputs. A leaf placed on this device seeds its input node with
     its buffer directly — no transfer, and the value stays resident (inputs
     are read-only). Otherwise wrap the current leaf's memory when the device
     shares host memory and the leaf is contiguous, and copy its bytes if not.
     Seeded leaves and wrapped hosts are kept reachable until the run
     completes, so no finalizer can release a buffer the kernels still read. *)
  (* Consumption, checked before anything moves (RFC 0006, rule 4). A consumed
     leaf's view must cover its storage on every device that holds it, and no
     other leaf of the call, nor a capture of the program, may reach that
     storage. A host leaf has no storage to consume: it is uploaded and stays
     usable. *)
  let consumed = ref [] in
  Array.iteri
    (fun i (Nx.P leaf) ->
      if c.cp_consumed.(i) then
        match leaf with
        | Placed r ->
            (match store_of r.r_cell with
            | Some s
              when List.compare_lengths
                     (Nx_effect.Placement.devices r.r_placement)
                     s.s_devices
                   < 0 ->
                invalid_arg
                  (Printf.sprintf
                     "Rune.jit: the argument at %s is a view of one shard of a \
                      split storage, so it cannot be consumed; pass Nx.copy of \
                      it"
                     c.cp_names.(i))
            | _ -> ());
            if not (Nx_effect.covers r) then
              invalid_arg
                (Printf.sprintf
                   "Rune.jit: the argument at %s views part of its storage (a \
                    slice, a transpose or a broadcast), so it cannot be \
                    consumed; pass Nx.copy of it"
                   c.cp_names.(i));
            consumed := (i, r.r_cell) :: !consumed
        | Host _ | Traced _ -> ())
    leaves;
  let consumed = List.rev !consumed in
  List.iter
    (fun (i, (cell : Nx_effect.cell)) ->
      Array.iteri
        (fun j (Nx.P leaf) ->
          match leaf with
          | Placed r when j <> i && r.r_cell == cell ->
              invalid_arg
                (Printf.sprintf
                   "Rune.jit: the arguments at %s and %s reach one storage, \
                    which a consumed argument must hold alone; pass Nx.copy of \
                    one of them"
                   c.cp_names.(Int.min i j)
                   c.cp_names.(Int.max i j))
          | _ -> ())
        leaves;
      if Array.exists (fun cell' -> cell' == cell) c.cp_captures then
        invalid_arg
          (Printf.sprintf
             "Rune.jit: the argument at %s and a capture of the function reach \
              one storage, which a consumed argument must hold alone; pass \
              Nx.copy of it"
             c.cp_names.(i)))
    consumed;
  let seed_entry = Array.make (Array.length c.cp_inputs) None in
  (* The buffer views of this call's ranges are released once it has run, or has
     raised: not at a safe point inside it, where the launch would allocate them
     again. *)
  let ranges = ref [] in
  Fun.protect ~finally:(fun () -> pending_views := !ranges @ !pending_views)
  @@ fun () ->
  (* Rebind arenas before queue replay patches their addresses: another compiled
     function may have grown the shared storage since the last call. An arena
     over the program's devices is a view of each device's shared buffer of
     exactly its size, the shards of one buffer being equal, while each device's
     buffer grows with its own programs only. An arena on one device of several,
     which a collective's copies use, is that device's, found by the name the
     program gave it. *)
  let devs = List.map snd c.cp_devices in
  let names = List.map Tolk.Device.name devs in
  List.iteri
    (fun k node ->
      let nbytes = U.max_numel node in
      let seed dev =
        supply node [shared_arena dev k nbytes]
      in
      match (U.device_of node, devs) with
      | Some (U.Single _), [ dev ] -> seed dev
      | Some (U.Single name), _ -> (
          match List.find_index (String.equal name) names with
          | Some i -> seed (List.nth devs i)
          | None -> invalid_arg "Rune.jit: an arena off the program's devices")
      | Some (U.Multi ns), _ when List.equal String.equal ns names ->
          let exactly dev =
            let buf = shared_arena dev k nbytes in
            if Tolk.Device.Buffer.nbytes buf = nbytes then buf
            else begin
              let view = buffer_range buf ~lo:0 ~span:nbytes in
              ranges := view :: !ranges;
              view
            end
          in
          supply node (List.map exactly devs)
      | _ -> invalid_arg "Rune.jit: an arena off the program's devices")
    c.cp_arenas;
  let keep = ref [] in
  Array.iteri
    (fun i (Nx.P leaf) ->
      let inp = c.cp_inputs.(i) in
      match seeds.(i) with
      | Whole { cell; bufs } ->
          keep := Obj.repr leaf :: !keep;
          seed_entry.(i) <- Some (cell, bufs);
          supply inp.i_node bufs
      | Range { lo; span; bufs; _ } ->
          keep := Obj.repr leaf :: !keep;
          let range = List.map (fun buf -> buffer_range buf ~lo ~span) bufs in
          ranges := range @ !ranges;
          supply inp.i_node range
      | Copy -> (
          match
            if c.cp_zero_copy then wrap_tensor c.cp_device leaf else None
          with
          | Some (buf, ka) ->
              keep := ka :: !keep;
              supply inp.i_node [buf]
          | None ->
              supply inp.i_node inp.i_bufs;
              upload_windows c.cp_scratch inp.i_place leaf c.cp_devices
                inp.i_bufs))
    leaves;
  (* Lending claims: an output takes the storage of its partner, its buffer on
     each device in the program's order, when the partner seeded from storage
     that no program binds. *)
  let claims :
      (int, Nx_effect.cell * store * Tolk.Device.Buffer.t list) Hashtbl.t =
    Hashtbl.create 4
  in
  List.iter
    (fun { l_otag; l_input; _ } ->
      match seed_entry.(l_input) with
      | Some ((e : Nx_effect.cell), bufs) when e.bound = 0 -> (
          match store_of e with
          | Some s ->
              reused_bytes := !reused_bytes + s.s_nbytes;
              Hashtbl.replace claims l_otag (e, s, bufs)
          | None -> ())
      | _ -> ())
    c.cp_lends;
  (* Wire the outputs' storage, all of it before the first kernel. On the
     zero-copy device, fresh host buffers become the kernels' output storage, so
     results are written straight into the tensors returned to the caller; nodes
     backed by an input or constant buffer keep their binding and are read back
     through a copy instead. On other devices, every distinct output node is
     bound to fresh storage for this call, or to the storage it claimed, so
     values from earlier calls keep their own storage and never alias a later
     call's outputs. A node backed by an input or constant keeps its binding,
     and its value is copied after the kernels into storage allocated here. *)
  let out_hosts : (int, host_out) Hashtbl.t = Hashtbl.create 8 in
  let out_bufs : (int, Tolk.Device.Buffer.t list) Hashtbl.t =
    Hashtbl.create 8
  in
  let copies = ref [] in
  (* Buffers for an output of [ph] at [place]: each device's slice of a split
     one, the whole value on each device for a copy. *)
  let fresh odt ph place =
    let n = numel (local_shape place (shape_of ph)) in
    List.map
      (fun (d, dev) -> create_fresh_buffer d dev (tolk_dtype odt) n)
      c.cp_devices
  in
  Array.iter
    (fun { o_value = Packed (odt, ph); o_node; o_place = place } ->
      match o_node with
      | None -> ()
      | Some node -> (
          let tag = U.tag node in
          let reserved = Hashtbl.mem c.cp_reserved tag in
          if c.cp_zero_copy then
            begin if (not reserved) && not (Hashtbl.mem out_hosts tag) then begin
              let n = numel (shape_of ph) in
              let host = Nx_buffer.create odt n in
              let buf =
                wrap_ptr c.cp_device (tolk_dtype odt) n
                  (Nx_buffer.unsafe_data_ptr host)
              in
              supply node [buf];
              Hashtbl.add out_hosts tag (Host (odt, host))
            end
            end
          else if not (Hashtbl.mem out_bufs tag) then
            match Hashtbl.find_opt claims tag with
            | Some (_, _, bufs) ->
                if not reserved then supply node bufs;
                Hashtbl.add out_bufs tag bufs
            | None ->
                let bufs = fresh odt ph place in
                if reserved then copies := (node, bufs) :: !copies
                else supply node bufs;
                Hashtbl.add out_bufs tag bufs))
    c.cp_outputs;
  (* Storage of its own for each result leaf that repeats an earlier leaf's
     node, filled from that node's after the kernels. *)
  let repeats =
    Array.mapi
      (fun j k ->
        let { o_value = Packed (odt, ph); o_node; o_place = place } =
          c.cp_outputs.(k)
        in
        match o_node with
        | Some _ when not c.cp_first.(j) ->
            if c.cp_zero_copy then None else Some (fresh odt ph place)
        | _ -> None)
      c.cp_results
  in
  (* An output an indexed write lands in starts from its input. One that claimed
     that input's storage already holds the value; any other is given it by a
     copy. *)
  let node_bufs node =
    match Tolk.Realize.resolve_buffer context node with
    | Tolk.Realize.Single b -> [ b ]
    | Tolk.Realize.Multi m -> Tolk.Device.Multi_buffer.bufs m
  in
  let copy_into dsts srcs =
    List.iter2 (fun dst src -> Tolk.Device.Buffer.copy_from ~dst ~src) dsts srcs
  in
  List.iter
    (fun (node, i) ->
      let claimed =
        match (Hashtbl.find_opt claims (U.tag node), seed_entry.(i)) with
        | Some (e, _, _), Some (e', _) -> e == e'
        | _ -> false
      in
      if not claimed then
        copy_into (node_bufs node) (node_bufs c.cp_inputs.(i).i_node))
    c.cp_prefills;
  (* Before the first kernel, every storage a consumed leaf reaches is marked
     consumed; nothing unmarks it. Its buffers stay until the kernels have read
     them. A call that fails from here on has consumed its arguments and returns
     nothing. *)
  let marked =
    List.map
      (fun (i, (cell : Nx_effect.cell)) ->
        let s = store_of cell in
        cell.state <- Consumed c.cp_consumptions.(i);
        (i, cell, s))
      consumed
  in
  let release () =
    List.iter
      (fun (_, (cell : Nx_effect.cell), s) ->
        match s with Some s when cell.bound = 0 -> release_store s | _ -> ())
      marked
  in
  (match
     Tolk.Realize.run_linear ~device:c.cp_device ~to_program ~input_uops
       ~var_vals:c.cp_vars ~jit:true c.cp_linear
   with
  | () -> ()
  | exception e ->
      release ();
      raise e);
  (* An output that is an input or a capture returned unchanged keeps its
     reserved binding: its value is copied into the storage allocated for it, so
     it never aliases an input and survives later calls. A consumed input
     returned unchanged instead hands its storage over: no copy. *)
  List.iter (fun (node, dsts) -> copy_into dsts (node_bufs node)) !copies;
  Array.iteri
    (fun j -> function
      | Some dsts ->
          let node = Option.get c.cp_outputs.(c.cp_results.(j)).o_node in
          copy_into dsts (Hashtbl.find out_bufs (U.tag node))
      | None -> ())
    repeats;
  (* The call returns while its kernels may still run, on every device. An
     output is a placed value whose reads wait for its devices, and a buffer
     released below goes back to its device's pool, where only work queued after
     these kernels can take it. The host's programs still wait here: their
     outputs are host tensors the kernels write in place, and wrapped buffers
     alias caller memory (seeded inputs and wrapped captures) that the caller
     may touch as soon as the call returns. *)
  if c.cp_zero_copy then Tolk.Device.synchronize c.cp_device;
  ignore (Sys.opaque_identity !keep);
  ignore (Sys.opaque_identity c.cp_wrapped);
  ignore (Sys.opaque_identity c.cp_bound);
  let placed_on place dt shape ~nolru bufs =
    make_placed place
      (List.map snd c.cp_devices)
      ~nolru dt
      (NV.create (local_shape place shape))
      bufs
  in
  (* The result's leaves, in walk order. *)
  let values =
    Array.to_list
      (Array.mapi
         (fun j k ->
           let { o_value = Packed (dt, ph); o_node; o_place = place } =
             c.cp_outputs.(k)
           in
           let shape = shape_of ph in
           match o_node with
           | None ->
               Nx.P
                 (Nx_effect.reshape
                    (Nx_effect.from_host c.cp_ctx (Nx_buffer.create dt 0))
                    shape)
           | Some node -> (
               let tag = U.tag node in
               match Hashtbl.find_opt out_hosts tag with
               | Some (Host (hdt, host)) -> (
                   match ND.equal_witness hdt dt with
                   | Some Type.Equal ->
                       let host =
                         if c.cp_first.(j) then host
                         else begin
                           let copy =
                             Nx_buffer.create hdt (Nx_buffer.length host)
                           in
                           Nx_buffer.blit ~src:host ~dst:copy;
                           copy
                         end
                       in
                       Nx.P
                         (Nx_effect.reshape
                            (Nx_effect.from_host c.cp_ctx host)
                            shape)
                   | None -> assert false)
               | None -> (
                   if c.cp_zero_copy then
                     let buf =
                       Tolk.Realize.resolve context node
                     in
                     Nx.P (read_out c.cp_scratch c.cp_ctx dt shape buf)
                   else
                     match repeats.(j) with
                     | Some bufs ->
                         Nx.P (placed_on place dt shape ~nolru:false bufs)
                     | None ->
                         (* Storage lent by a consumed input keeps its way back:
                            past the allocator's cache for an upload from a
                            mapped file. *)
                         let nolru =
                           match Hashtbl.find_opt claims tag with
                           | Some (_, s, _) -> s.s_nolru
                           | None -> false
                         in
                         Nx.P
                           (placed_on place dt shape ~nolru
                              (Hashtbl.find out_bufs tag)))))
         c.cp_results)
  in
  let y = Nx.Ptree.rebuild q ~like:c.cp_skeleton values in
  (* The storage of each consumed leaf is now owned by the output that claimed
     it, or returned to the allocator, where the next call's fresh outputs reuse
     it in queue order, after the kernels of this call. A storage a program
     binds stays for it (see [trace_compile]). *)
  let claimed cell =
    Hashtbl.fold (fun _ (e, _, _) a -> a || e == cell) claims false
  in
  List.iter
    (fun (_, cell, s) ->
      match s with
      | Some s when claimed cell ->
          s.s_bufs <- [];
          account s (-1)
      | Some s when cell.Nx_effect.bound = 0 -> release_store s
      | _ -> ())
    marked;
  if Lazy.force jit_debug >= 1 then begin
    Printf.eprintf
      "rune.jit: replay on %s: %d bytes to device, %d bytes from device, %d \
       bytes resident\n\
       %!"
      (String.concat ", "
         (List.map (fun (d, _) -> Nx.Device.name d) c.cp_devices))
      (!bytes_to_device - in0)
      (!bytes_from_device - out0)
      !resident_bytes;
    List.iter
      (fun (i, cell, s) ->
        match
          List.find_opt
            (fun l -> l.l_input = i && Hashtbl.mem claims l.l_otag)
            c.cp_lends
        with
        | Some l ->
            Printf.eprintf "rune.jit: %s -> result %s reused\n%!" c.cp_names.(i)
              (if l.l_result = "the root" then "(the root)" else l.l_result)
        | None ->
            Printf.eprintf "rune.jit: %s consumed, %s\n%!" c.cp_names.(i)
              (match s with
              | None -> "held by nx"
              | Some _ when cell.Nx_effect.bound > 0 ->
                  "kept for the programs that bind it"
              | Some _ -> "storage released"))
      marked
  end;
  y

(* Public entry points *)

(* A call's placed leaves that are not on its devices: the leaves, by position,
   and the message given their names. *)
exception Misplaced of int list * (string list -> string)

(* The devices a call runs on, as its placed leaves decide them, and how far:
   [requested] when given, else those of its first split leaf, whose order
   decides which slice lands where, else those of its first placed leaf as a set
   ([decided_by]). Every placed leaf must be over them ([over]); otherwise
   raises [Misplaced]. *)
let leaves_devices ~requested leaves =
  let placed =
    List.concat
      (List.mapi
         (fun i (Nx.P leaf) ->
           match leaf with
           | Nx_effect.Placed r -> [ (i, r.r_placement) ]
           | _ -> [])
         (Array.to_list leaves))
  in
  let decided =
    match requested with
    | Some ds -> Some (None, (ds, Ordered))
    | None -> (
        let split =
          List.filter (fun (_, p) -> Nx_effect.Grid.cuts p <> []) placed
        in
        match split @ placed with
        | (i, p) :: _ -> Some (Some (i, p), decided_by p)
        | [] -> None)
  in
  Option.map
    (fun (by, ((ds, _) as d)) ->
      List.iter
        (fun (i, p) ->
          if not (over ds p) then
            raise
              (match by with
              | None ->
                  Misplaced
                    ( [ i ],
                      fun names ->
                        Format.asprintf
                          "Rune.jit: the argument at %s is on %a and ~devices \
                           names %a; place it there, or on the host"
                          (List.hd names) Nx.Placement.pp p pp_devices ds )
              | Some (j, q) ->
                  let (a, pa), (b, pb) =
                    if j < i then ((j, q), (i, p)) else ((i, p), (j, q))
                  in
                  Misplaced
                    ( [ a; b ],
                      fun names ->
                        Format.asprintf
                          "Rune.jit: the arguments at %s and %s are on %a and \
                           %a; place them on the same devices, in one order"
                          (List.nth names 0) (List.nth names 1) Nx.Placement.pp
                          pa Nx.Placement.pp pb )))
        placed;
      d)
    decided

(* What a program knows of the leaves of [args], a value of a signature's
   arguments with [roles]: each leaf's path is its name, and its first segment
   the leaf's argument. *)
let leaf_info ~roles args v =
  let paths =
    List.filter_map
      (function Nx.Ptree.Leaf p -> Some p | Report _ -> None)
      (Nx.Ptree.visits args v)
    |> Array.of_list
  in
  let names = Array.map Nx.Ptree.Path.to_string paths in
  {
    arguments = Array.map Structure.argument_of paths;
    consumed =
      Array.map
        (fun p -> roles.(Structure.argument_of p) = Nx.Ptree.Consumed)
        paths;
    consumptions = Array.map (fun path -> { Nx_effect.path }) names;
    names;
  }

(* A call's leaves must be live: a consumed one raises before anything traces or
   moves. *)
let check_live leaves =
  Array.iter
    (fun (Nx.P x) ->
      match x with
      | Nx_effect.Placed { r_cell = { state = Consumed k; _ }; _ } ->
          Nx_effect.consumed k
      | _ -> ())
    leaves

(* The compiled function over [args], a signature's arguments with [roles].

   A call runs on the requested devices, else where its placed leaves live, else
   where a capture lives, else on the default device. Captures are found by
   tracing: a trace on the default device that meets a capture placed elsewhere
   runs again on the capture's devices, and one over devices that copies listed
   as a set, which meets a capture split over them in another order, runs again
   in that order; every later call then takes them.

   A call flattens its arguments once: the leaves seed the program, and the
   skeleton and the leaves' signatures are its key. [RUNE_JIT_DEBUG=1] reports
   each retrace with the first difference from the previous call's key. *)
let compile_fn (type a r) ?requested ?beam ?beam_parallel ~roles
    (args : a Nx.Ptree.t) (result : r Nx.Ptree.t) (f : a -> r) : a -> r =
  let captured_on = ref None in
  let programs : (key * r compiled) list ref = ref [] in
  let previous = ref None in
  (* Device copies of captured tensors, per device list, shared by every
     signature of this closure and keyed by capture identity. *)
  let const_caches = ref [] in
  let const_cache ds =
    match
      List.find_opt (fun (ds', _) -> List.equal ( == ) ds ds') !const_caches
    with
    | Some (_, t) -> t
    | None ->
        let t = Tensor_map.Tbl.create 4 in
        const_caches := (ds, t) :: !const_caches;
        t
  in
  fun v ->
    if Gate.transforming () then f v
    else
      let leaves, skeleton = Nx.Ptree.flatten args v in
      let leaves = Array.of_list leaves in
      check_live leaves;
      let ds, decided =
        match
          ( (try leaves_devices ~requested leaves
             with Misplaced (is, message) ->
               let { names; _ } = leaf_info ~roles args v in
               invalid_arg (message (List.map (fun i -> names.(i)) is))),
            !captured_on )
        with
        | Some (ds, Unordered), Some (c, Ordered)
          when over ds (Nx.Placement.replicated c) ->
            (c, Ordered)
        | Some d, _ | None, Some d -> d
        | None, None -> ([ default_device () ], Guessed)
      in
      let shapes = Array.map (fun (Nx.P x) -> shape_of x) leaves in
      let hash = hash_call skeleton leaves shapes in
      let program ds ~decided =
        let devs = List.map tolk_device_of ds in
        let seeds = Array.map (fun (Nx.P x) -> seed_of devs x) leaves in
        match
          List.find_opt
            (fun (k, _) -> matches k ds ~hash skeleton leaves shapes seeds)
            !programs
        with
        | Some (key, c) ->
            previous := Some key;
            replay result c leaves seeds
        | None ->
            let key = key_of ds ~hash skeleton leaves shapes seeds in
            let info = leaf_info ~roles args v in
            (if Lazy.force jit_debug >= 1 then
               match !previous with
               | Some prev ->
                   Printf.eprintf "rune.jit: retrace: %s\n%!"
                     (key_difference ~names:info.names key prev)
               | None -> ());
            (* The host's programs run over host memory. *)
            let c =
              trace_compile ~devices:(ds, devs)
                ~zero_copy:(List.equal ( == ) ds [ Nx.Device.host ])
                ~info ~const_cache:(const_cache ds) ~decided
                ~placements:
                  (Array.map (fun (Nx.P x) -> leaf_placement ds x) leaves)
                ~layouts:(Array.map layout_of seeds)
                ?beam ?beam_parallel args result f v leaves
            in
            (* Captures it binds make the devices the closure's. *)
            if c.cp_bound <> [||] && !captured_on = None then
              captured_on :=
                Some (ds, if decided = Guessed then Unordered else decided);
            programs := (key, c) :: !programs;
            previous := Some key;
            replay result c leaves seeds
      in
      (* Each retry decides more (guessed, then a set, then an order), so it
         ends. *)
      let rec run ds decided =
        match program ds ~decided with
        | y -> y
        | exception Runs_on p ->
            let ds, decided = decided_by p in
            captured_on := Some (ds, decided);
            run ds decided
      in
      run ds decided

(* [devices], checked as a placement's devices are: at least one, distinct, of
   one backend. *)
let checked_devices what devices =
  match Nx.Placement.replicated devices with
  | _ -> devices
  | exception Invalid_argument m ->
      invalid_arg (Printf.sprintf "%s: ~devices: %s" what m)

let jit ?devices ?beam ?beam_parallel sg f =
  let (Structure.Uncurried u) = Structure.signature "Rune.jit" sg in
  let requested = Option.map (checked_devices "Rune.jit") devices in
  u.curry
    (compile_fn ?requested ?beam ?beam_parallel ~roles:u.roles u.args u.result
       (u.apply f))

let jit' ?devices ?beam ?beam_parallel f =
  jit ?devices ?beam ?beam_parallel Nx.Ptree.(tensor @-> returns tensor) f
