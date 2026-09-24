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
   scheduling, kernel codegen) into a compiled linear schedule with a persistent
   buffer binding. Compilation happens once per input signature (the leaf dtypes
   and shapes in traversal order); a call with a new signature retraces.

   Replaying: every call binds the current input leaves to the compiled
   program's buffers and runs the schedule. On the CPU device, contiguous inputs
   and captured constants are wrapped in place — kernels read the tensors' own
   memory — and outputs are computed straight into the returned tensors'
   storage. Other devices and non-contiguous tensors go through byte copies:
   inputs are re-copied on every call, captured tensors are uploaded once when
   the trace compiles and stay resident on the device. Captures are compile-time
   constants: mutating one between calls has unspecified visibility (the CPU
   wrapping may observe it, device copies never do).

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
module ND = Nx_core.Dtype
module NV = Nx_core.View

exception Jit_error of string

let err fmt = Printf.ksprintf (fun s -> raise (Jit_error s)) fmt

let unsupported op =
  err
    "Rune.jit: %s is not supported inside jit; move it outside the jitted \
     function"
    op

(* Dtype bridges *)

let tolk_dtype : type a b. (a, b) ND.t -> TD.t = function
  | ND.Float16 -> TD.float16
  | ND.Float32 -> TD.float32
  | ND.Float64 -> TD.float64
  | ND.BFloat16 -> TD.bfloat16
  | ND.Float8_e4m3 -> TD.fp8e4m3
  | ND.Float8_e5m2 -> TD.fp8e5m2
  | ND.Int8 -> TD.int8
  | ND.UInt8 -> TD.uint8
  | ND.Int16 -> TD.int16
  | ND.UInt16 -> TD.uint16
  | ND.Int32 -> TD.int32
  | ND.UInt32 -> TD.uint32
  | ND.Int64 -> TD.int64
  | ND.UInt64 -> TD.uint64
  | ND.Bool -> TD.bool
  | ND.Int4 -> unsupported "an int4 tensor"
  | ND.UInt4 -> unsupported "a uint4 tensor"
  | ND.Complex64 -> unsupported "a complex tensor"
  | ND.Complex128 -> unsupported "a complex tensor"

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
let holds : type a b. Tolk.Device.t -> (a, b) ND.t -> bool =
 fun dev dt ->
  match tolk_dtype dt with
  | tdt -> Tolk.Decomp_dtype.is_dtype_supported (Tolk.Device.renderer dev) tdt
  | exception Jit_error _ -> false

(* Identity-keyed tables over tensors, as in [Tensor_map]. *)
module Tbl = Hashtbl.Make (struct
  type t = Obj.t

  let equal = ( == )
  let hash = Hashtbl.hash
end)

type packed = Packed : ('a, 'b) ND.t * ('a, 'b) Nx_effect.t -> packed

(* Backends. Device instances live in the shared tolk registry, one per
   canonical name, so jit, pmap and the engine's multi-device schedules resolve
   the same instance. rune installs its own opener for every backend when it is
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

   A compiled trace binds each input and output leaf to a placement: the trace's
   single device, or — under [pmap] — every device of a tuple, either replicated
   (a full copy per device) or sharded (split along one axis into equal
   per-device slices). The traced function always observes global shapes;
   placement is a property of the compiled signature. *)

type leaf_place = P_single | P_replicated | P_sharded of int

let place_axis = function
  | P_sharded a -> Some a
  | P_replicated | P_single -> None

(* A pmap device tuple: canonical names and their registry instances. *)
type multi_spec = { md_names : string list; md_devs : Tolk.Device.t list }

(* The devices other than the host, by canonical name: each has one nx device
   value, whose engine is [engine] below, and one tolk device. *)

let by_name : (string, Nx.Device.t * Tolk.Device.t) Hashtbl.t = Hashtbl.create 4

(* Resident storage

   Values held on a device are placed values ([Nx_effect.Placed]) whose storage
   is a list of tolk buffers, one per device of the placement. A compiled call's
   outputs on a device that does not share host memory are placed; so are
   uploads made by [Nx.place]. The storage belongs to the value's cell: a read
   copies the view's elements out and leaves it, replay seeds a compiled input
   with the buffer itself when the placement matches, and it is released when
   the cell is unreachable (by the finaliser the engine attaches) or donated to
   a [jit_step] call. *)

type store = {
  s_devices : Tolk.Device.t list; (* one per shard, placement order *)
  s_nbytes : int; (* summed across shards *)
  mutable s_bufs : Tolk.Device.Buffer.t list; (* [[]] once released or lent *)
  s_nolru : bool; (* bypasses the allocator's cache: a mapped file's upload *)
}

type Nx_effect.storage += Buffers of store

let store_of (c : Nx_effect.cell) =
  match c.state with Live (Buffers s) -> Some s | _ -> None

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
   calls of all programs in queue order, so the single-device programs a device
   runs share its arenas: a program's [k]th arena is bound, at every call, to
   the device's [k]th shared buffer, which grows to the largest arena bound to
   it. The buffer a slot outgrows is retired: device graphs recorded over it
   keep views of it for as long as their program lives, and it is freed once no
   view remains, after the device has finished the work that may use it. *)

let arenas : (string * int, Tolk.Device.Buffer.t) Hashtbl.t = Hashtbl.create 4
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
  let key = (Tolk.Device.name dev, k) in
  match Hashtbl.find_opt arenas key with
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
      Hashtbl.replace arenas key buf;
      buf

(* How a program reads an input or constant from its node: from element [skip]
   on, the value's elements in C order, or a view of [strides] over the elements
   of storage it reaches (see [view_movement]). *)
type layout = { skip : int; strides : int array option }

let dense = { skip = 0; strides = None }

(* How a value seeds a program's input or constant: see [seed_of]. *)
type seed =
  | Whole of Nx_effect.cell
  | Range of { cell : Nx_effect.cell; lo : int; span : int; layout : layout }
  | Copy

(* Trace state *)

type input = {
  i_node : U.t;
  i_place : leaf_place;
  i_bufs : Tolk.Device.Buffer.t list; (* one per device of the placement *)
  i_dtype : string;
  i_numel : int;
}

type state = {
  st_id : int; (* the trace's own id, in its traced tensors *)
  st_device : Tolk.Device.t;
  st_multi : string list option; (* pmap device tuple, [None] = single *)
  st_placement : Nx.Placement.t option; (* a single device's placement *)
  st_may_move : bool;
      (* nothing decided the device: a capture placed elsewhere moves the
         program to it (see [Runs_on]) *)
  mutable refusal : exn option;
      (* the first reason the program cannot run on its device, raised once the
         function returns: raising inside a handler would drop the function's
         own cleanups *)
  st_takes_storage : int -> bool;
      (* the input positions whose storage replay may hand an output: consumed
         leaves of a single-device program whose outputs are not in host
         memory *)
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
  mutable consts : (U.t * packed) list; (* reverse order *)
  bound : F.Tensor.t Tensor_map.Tbl.t; (* resident captures bound in place *)
  mutable bound_consts : (U.t * Nx_effect.cell * packed * seed) list;
  mutable axis_index : U.t option; (* pmap: per-device index buffer, once *)
  scan_stacks : (U.t * int) list Tbl.t;
      (* staged scans: the step record's identity -> the per-leaf carry-stack
         buffer nodes the forward loop wrote, for the backward loop to read. The
         step record is shared between the forward staging and the tape-recorded
         backward thunk, and is fresh per [Rune.scan] call, so it identifies the
         scan. Identity-keyed: a structural table compares the record's closure
         on a hash collision. *)
  scan_closed : Scan.packed_t list Tbl.t;
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
}

(* A polymorphic observer of the tensors flowing through [tolk_of]. *)
and tensor_hook = { hook : 'a 'b. ('a, 'b) Nx_effect.t -> unit }

let shape_of x = NV.shape (Nx_effect.view x)
let numel shape = Array.fold_left ( * ) 1 shape

(* A capture placed on another device than the program's, when nothing else
   decided where the program runs: the program runs there instead. *)
exception Runs_on of Nx.Device.t

(* The first refusal wins, except that while nothing decided the device, a
   capture's device replaces what the guessed device refused: the program runs
   there instead. *)
let refuse st e =
  match (st.refusal, e) with
  | None, _ -> st.refusal <- Some e
  | Some (Runs_on _), _ -> ()
  | Some _, Runs_on _ when st.st_may_move -> st.refusal <- Some e
  | Some _, _ -> ()

let check_dtype : type a b. state -> (a, b) ND.t -> string -> unit =
 fun st dt what ->
  if not (holds st.st_device dt) then
    refuse st
      (Jit_error
         (Printf.sprintf "Rune.jit: %s is %s, which %s cannot hold" what
            (ND.to_string dt)
            (Tolk.Device.name st.st_device)))

(* A captured value lives on the program's device or on the host. A pmap reads
   any other through the host. *)
let check_capture : type a b. state -> (a, b) Nx_effect.t -> unit =
 fun st x ->
  match (st.st_placement, x) with
  | Some (Device d), Placed { r_placement = Device d'; _ } when d' == d -> ()
  | Some _, Placed { r_placement = Device d'; _ } when st.st_may_move ->
      refuse st (Runs_on d')
  | Some p, Placed { r_placement = p'; _ } ->
      refuse st
        (Invalid_argument
           (Format.asprintf
              "Rune.jit: a captured value is on %a and the program runs on %a; \
               place it on %a, or on the host"
              Nx.Placement.pp p' Nx.Placement.pp p Nx.Placement.pp p))
  | _ -> ()

(* A traced tensor's payload: the trace that made it and its node. *)
type Nx_effect.node += Node of { trace : int; tensor : F.Tensor.t }

let trace_counter = ref 0

(* A fresh traced tensor of [st]'s trace standing for [tt], of [tt]'s shape. *)
let traced st dt tt =
  check_dtype st dt "a value the function computes";
  let shape = Array.of_list (F.Tensor.shape tt) in
  Nx_effect.traced st.st_ctx dt shape (Node { trace = st.st_id; tensor = tt })

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
  U.store ~dst
    ~value:(U.contiguous ~src:(F.Tensor.uop (F.Movement.reshape tt [ n ])) ())
    ()

let make_node st dtolk n =
  let device =
    match st.st_multi with
    | Some names -> U.Multi names (* pmap: constants replicate on the tuple *)
    | None -> U.Single (Tolk.Device.name st.st_device)
  in
  U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:dtolk
    ~shape:(F.Tensor.shape_uop [ n ]) ~device ()

(* Bind a tensor whose bytes exist outside the traced computation (a closure
   capture, or a host constant created while tracing) as a compile-time
   constant: a buffer input aliasing the tensor's memory when the device can
   share it, uploaded once when the trace compiles otherwise. *)
let lift_const (type a b) st (x : (a, b) Nx_effect.t) : F.Tensor.t =
  let dt = Nx_effect.dtype x in
  check_dtype st dt "a constant of the function";
  let shape = shape_of x in
  let node = make_node st (tolk_dtype dt) (numel shape) in
  st.consts <- (node, Packed (dt, x)) :: st.consts;
  let tt = buffer_tensor node shape in
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

   A value placed on a program's device seeds the program without a copy: its
   buffer when its view covers its storage, otherwise the range of storage its
   view reaches, bound as a buffer view from a 16-byte boundary (see
   [alignment]). A C-order window is that range as it is; any other view is
   movement over the range, which the program applies, when its axes nest. A
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

let seed_of : type a b. Tolk.Device.t -> (a, b) Nx_effect.t -> seed =
 fun dev x ->
  match x with
  | Placed r -> (
      match store_of r.r_cell with
      | Some { s_devices = [ d ]; s_bufs = [ buf ]; _ } when d == dev ->
          let v = r.r_view in
          let range lo n strides =
            let per =
              Int.max 1 (alignment / TD.itemsize (Tolk.Device.Buffer.dtype buf))
            in
            let skip = lo mod per in
            Range
              {
                cell = r.r_cell;
                lo = lo - skip;
                span = n + skip;
                layout = { skip; strides };
              }
          in
          if Nx_effect.covers r then Whole r.r_cell
          else if NV.numel v = 0 || not (Tolk.Device.Buffer.supports_offset buf)
          then Copy
          else if NV.is_c_contiguous v then
            range (NV.offset v) (NV.numel v) None
          else if nests (NV.shape v) (NV.strides v) then
            let lo, hi = extent v in
            range lo (hi - lo) (Some (NV.strides v))
          else Copy
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

(* A capture placed on a single-device program's device keeps its storage: the
   program reads it as the constant and no bytes move. The compiled record keeps
   the value reachable and counts the binding on its cell, so the storage is
   never donated while the program lives. *)
let bind_const (type a b) st seed cell (x : (a, b) Nx_effect.t) : F.Tensor.t =
  let dt = Nx_effect.dtype x in
  check_dtype st dt "a constant of the function";
  let shape = shape_of x in
  let layout = layout_of seed in
  let node = make_node st (tolk_dtype dt) (layout_size layout shape) in
  st.bound_consts <- (node, cell, Packed (dt, x), seed) :: st.bound_consts;
  let tt = layout_tensor node layout shape in
  Tensor_map.Tbl.replace st.bound (Key x) tt;
  tt

(* A tensor entering the trace without a table entry is a closure capture. *)
let tolk_of : type a b. state -> (a, b) Nx_effect.t -> F.Tensor.t =
 fun st x ->
  (match st.scan_collectors with
  | [] -> ()
  | fs -> List.iter (fun h -> h.hook x) fs);
  match x with
  | Nx_effect.Traced { t_node = Node { trace; tensor }; _ }
    when trace = st.st_id ->
      tensor
  | Nx_effect.Traced _ ->
      err
        "Rune.jit: a tensor traced by another jit entered this trace; a value \
         computed inside a jitted function exists outside it only as an output"
  | _ -> (
      check_capture st x;
      match if st.st_multi = None then seed_of st.st_device x else Copy with
      | (Whole cell | Range { cell; _ }) as seed -> (
          match Tensor_map.Tbl.find_opt st.bound (Key x) with
          | Some t -> t
          | None -> bind_const st seed cell x)
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
   so that the larger float is the larger integer, and -0 read as +0 so that
   equal zeros tie. Every NaN takes the greatest integer ([`Greatest]) or the
   least ([`Least]); no number takes either. The -0 test compares bits: a float
   comparison may flush subnormals to zero. A non-float [x] is its own key. *)
let order_keys ~nan x =
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
    let merged =
      where
        (eq bits (bound (Tolk_uop.Const.min_value int)))
        (F.Creation.const_like bits (F.Tensor.Sint 0))
        bits
    in
    let values keys =
      F.Dtype_ops.cast
        (where (eq keys nan_key)
           (F.Creation.const_like x (F.Tensor.Sfloat Float.nan))
           (F.Dtype_ops.bitcast (flip keys) (F.Tensor.dtype x)))
        dtype
    in
    (where (isnan x) nan_key (flip merged), values)

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
    order_keys ~nan:(match op with `Max -> `Greatest | `Min -> `Least) t
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

(* The keys a sort orders: NaN after every number in either direction, and equal
   zeros tied, so that a stable sort keeps their order. *)
let sort_keys ~descending x =
  order_keys ~nan:(if descending then `Least else `Greatest) x

(* Whether [st]'s device computes int64 natively, which the packed sort
   needs. *)
let packs st =
  Tolk.Renderer.supports_dtype (Tolk.Device.renderer st.st_device) TD.int64

let bit_length n =
  let rec go n acc = if n = 0 then acc else go (n lsr 1) (acc + 1) in
  go n 0

(* [x] sorted along [dim]. Only the values are demanded, so Tolk's recovery of
   positions is never computed. *)
let sort_graph ~dim ~descending x =
  let keys, values = sort_keys ~descending x in
  values (fst (F.Op.sort ~dim ~descending keys))

(* The stable positions that sort [x] along [dim]. Tolk's network sorts values
   and recovers each position by an n×n match of sorted values to inputs. When a
   key and its position fit together in a non-negative int64, the network sorts
   that integer instead: the key in the high bits, offset to be non-negative
   since C and Metal leave a shift of a negative integer undefined, and the
   position in the low bits, complemented for a descending sort so that equal
   keys keep index order. Packed integers are distinct, so the network alone
   gives the stable order and the positions are its low bits. 32-bit keys fit at
   every length an int32 position reaches; 64-bit keys never fit and, like a
   device without native int64 ([packs] is false), keep the match. The packed
   integers get a kernel of their own: fused into the padding of an axis that is
   not a power of two, the positions' arange no longer folds to an index and
   costs n^2 work. *)
let argsort_graph ~packs ~dim ~descending x =
  let keys, _ = sort_keys ~descending x in
  let key_dtype = F.Tensor.dtype keys in
  let shape = F.Tensor.shape x in
  let dim = if dim < 0 then dim + List.length shape else dim in
  let n = List.nth shape dim in
  let low_bits = bit_length (n - 1) in
  if not (packs && TD.bitsize key_dtype + low_bits <= 63) then
    snd (F.Op.sort ~dim ~descending keys)
  else
    let open F.Elementwise in
    let int t v = F.Creation.const_like t (F.Tensor.Sint v) in
    let offset =
      if TD.is_unsigned key_dtype || TD.is_bool key_dtype then 0
      else 1 lsl (TD.bitsize key_dtype - 1)
    in
    let low = (1 lsl low_bits) - 1 in
    let complement r = if descending then sub (int r low) r else r in
    let ranks =
      F.Movement.reshape
        (F.Op.arange ~dtype:TD.int64 n)
        (List.mapi (fun i _ -> if i = dim then n else 1) shape)
    in
    let wide = F.Dtype_ops.cast keys TD.int64 in
    let high = add wide (int wide offset) in
    let packed =
      bitwise_or (lshift high (int high low_bits)) (complement ranks)
    in
    let sorted = fst (F.Op.sort ~dim ~descending (contiguous packed)) in
    complement (bitwise_and sorted (int sorted low))

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
   runs, by handing it the input's donated storage or by copying. A program
   whose replays never take storage copies [t] itself: a kernel does it faster
   than replay. A [t] that is a staged loop's carry is not copied either: the
   write lands in an empty buffer the loop fills (see [stage_scan]). [operands]
   place the storage when [t] is a constant. *)
let write_destination st t ~operands =
  let device = List.find_map F.Tensor.device (t :: operands) in
  let u = F.Tensor.uop t in
  let empty () =
    F.Creation.empty ~dtype:(F.Tensor.dtype t) ?device (F.Tensor.shape t)
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
      else F.Creation.clone ?device t

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
  let idx = Nx_buffer.create Nx_buffer.int32 (kernel_prod * nwin) in
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

(* Schedule the traced body sink as its own compiled linear. Captured unplanned,
   so the loop's slot buffers stay intact (the memory planner would rewrite them
   into arenas); body PARAMs are substituted with the body call's argument
   nodes, which the loop executor rebinds per iteration. *)
let schedule_body_linear st body_sink =
  let body_sink, _ = Tolk.Bufferize.run body_sink in
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
      (* Batched like a compiled call's linear: each iteration replays the
         body's graphs with the rebound slot buffers patched in. *)
      Tolk.Jit.batch_graphs ~device:st.st_device
        (Tolk.Realize.pm_compile ~device:st.st_device
           ~to_program:(to_program st.st_device) body_linear)

(* Schedule analyses, shared by buffer reuse at the jit boundary and inside a
   staged loop's body. *)

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
            let op = U.op u in
            let allowed =
              Tolk_uop.Ops.Group.is_elementwise op
              && (op <> Tolk_uop.Ops.Cast
                 || Array.length (U.src u) = 0
                 || TD.itemsize (U.dtype u)
                    = TD.itemsize (U.dtype (U.src u).(0)))
              || op = Tolk_uop.Ops.Reshape
              || op = Tolk_uop.Ops.Stage
              || op = Tolk_uop.Ops.Contiguous_backward
              || op = Tolk_uop.Ops.Detach
            in
            (!reaches, !bad || (!reaches && not allowed))
          end
        in
        Hashtbl.replace memo (U.tag u) r;
        r
  in
  let reaches, bad = go u in
  (reaches, not bad)

(* The schedule's calls in execution order, descending into batched graph calls.
   [Opaque] marks a call whose inner order is unknown (a staged loop): it may
   read and write its arguments in any order. *)
type scheduled = Kernel of U.t | Opaque of U.t

let rec schedule_calls linear =
  List.concat_map
    (fun call ->
      match U.as_call call with
      | Some { body; _ } when U.op body = Tolk_uop.Ops.Custom_function -> (
          match (U.Arg.as_string (U.arg body), U.src body) with
          | Some "graph", [| inner |] -> schedule_calls inner
          | _ -> [ Opaque call ])
      | _ -> [ Kernel call ])
    (U.children linear)

(* No kernel reads the buffer [itag] after the first kernel that writes [otag],
   and neither buffer is touched by an opaque call. That first kernel may read
   [itag] itself when it writes each element where it read it. An indexed write
   does not, so under [indexed] it must not read [itag] either. *)
let schedule_allows ?(indexed = false) ~linear ~itag ~otag () =
  let mentions call tag =
    match U.as_call call with
    | Some { args; _ } ->
        List.exists
          (fun a ->
            match U.op a with
            | _ when U.is_bound_var a || U.is_variable a -> false
            | _ -> U.tag (U.buf_uop a) = tag)
          args
    | None -> false
  in
  let calls = schedule_calls linear in
  let opaque_touch =
    List.exists
      (function
        | Opaque c -> mentions c itag || mentions c otag | Kernel _ -> false)
      calls
  in
  if opaque_touch then false
  else begin
    let first_o = ref None and last_i = ref None in
    List.iteri
      (fun k -> function
        | Opaque _ -> ()
        | Kernel c ->
            if !first_o = None && mentions c otag then first_o := Some k;
            if mentions c itag then last_i := Some k)
      calls;
    match (!first_o, !last_i) with
    | Some o, Some i -> if indexed then i < o else i <= o
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

(* A carry of [numel] elements of [dt] starting from [init], read through the
   body nodes [reads] and written through [writes]: in one buffer under
   [in_place], in a pair otherwise. *)
let add_carry st l ?(in_place = false) ~reads ~writes ~dt ~numel init =
  let start = make_node st dt numel in
  let pos0 =
    add_arg l (U.after ~src:start ~deps:[ store_flat start numel init ])
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

let loop_call l ~body_linear ~reversed ~n =
  let cint v = U.const (Tolk_uop.Const.int Tolk_uop.Dtype.weakint v) in
  let slots ss =
    cint (List.length ss)
    :: List.concat_map
         (fun s ->
           [ s.node; cint s.pos0; cint s.pos1; cint s.size; cint s.stride ])
         (List.rev ss)
  in
  let payload =
    U.custom_function ~name:"loop"
      ~srcs:
        ([ body_linear; cint n; cint (if reversed then 1 else 0) ]
        @ slots l.ins @ slots l.outs)
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
  match List.rev l.args with
  | [] -> assert false
  | hd :: _ as args ->
      U.replace
        (U.call
           ~body:(U.custom_function ~name:"loop" ~srcs:[])
           ~args:[ hd ] ~info)
        ~src:(Array.of_list (payload :: args))
        ()

(* The buffer [b] once the loop [call] has written it. *)
let written_by call b = U.after ~src:b ~deps:[ U.store ~dst:b ~value:call () ]

(* The stacked rows of [shape] in [buf], [stride] elements apart, as an [n ::
   shape] tensor. *)
let rows_tensor buf ~n ~stride shape =
  let numel = numel shape in
  let t = buffer_tensor buf [| n; stride |] in
  let t =
    if stride = numel then t else F.Movement.shrink t [ (0, n); (0, numel) ]
  in
  F.Movement.reshape t (n :: Array.to_list shape)

(* The buffer [u] is, when it is a whole buffer after the effects that wrote it,
   under any reshape. *)
let rec written_buffer u =
  match U.op u with
  | Tolk_uop.Ops.Reshape -> written_buffer (U.src u).(0)
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

(* A row slot [slot] of [numel] elements over the rows of the [n; ...] value
   [tt], padded to the loop's row stride when a row falls short of it. *)
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

(* A staged body's slot: the placeholder the body receives for one leaf, bound
   to a buffer node the loop rebinds per iteration. *)
type body_slot = {
  s_ph : Scan.packed_t;
  s_node : U.t;
  s_dt : TD.t;
  s_shape : int array;
}

(* One slot per leaf of [tree], a row of it (its leading axis dropped) under
   [row], and the structure over the slots' placeholders. *)
let body_slots st ?(row = false) (Scan.Tree (m, t) as tree) =
  let slots =
    List.map
      (fun (Scan.Packed_t leaf) ->
        let shape = shape_of leaf in
        let shape =
          if row then Array.sub shape 1 (Array.length shape - 1) else shape
        in
        let dt = tolk_dtype (Nx_effect.dtype leaf) in
        let node = make_node st dt (numel shape) in
        let ph = traced st (Nx_effect.dtype leaf) (buffer_tensor node shape) in
        { s_ph = Scan.Packed_t ph; s_node = node; s_dt = dt; s_shape = shape })
      (Scan.leaves tree)
  in
  (Scan.Tree (m, Scan.unflatten m t (List.map (fun s -> s.s_ph) slots)), slots)

(* [tree]'s structure over fresh placeholders standing for [values], one
   [(shape, value)] per leaf, of the leaf's dtype. *)
let placeholders st (Scan.Tree (m, t) as tree) values =
  let phs =
    List.map2
      (fun (Scan.Packed_t leaf) (shape, value) ->
        let ph =
          Nx_effect.traced st.st_ctx (Nx_effect.dtype leaf) shape
            (Node { trace = st.st_id; tensor = value })
        in
        Scan.Packed_t ph)
      (Scan.leaves tree) values
  in
  Scan.Tree (m, Scan.unflatten m t phs)

let same_shapes leaves slots =
  List.for_all2 (fun (Scan.Packed_t l) s -> shape_of l = s.s_shape) leaves slots

(* Handler *)

let rec handler : type r. state -> (r, r) Effect.Deep.handler =
 fun st ->
  let open Effect.Deep in
  (* Answer an intercepted operation: record the graph node and continue with a
     fresh placeholder carrying the result's shape and dtype. *)
  let ret : type a b r.
      ((a, b) Nx_effect.t, r) continuation -> (a, b) ND.t -> F.Tensor.t -> r =
   fun k dt tt -> continue k (traced st dt tt)
  in
  let dt x = Nx_effect.dtype x in
  let go x = tolk_of st x in
  (* Like [ret] for a two-result operation: one placeholder per result, both
     carrying the operation's shared dtype (qr's factors, e.g.). *)
  let ret2 : type a b r.
      ((a, b) Nx_effect.t * (a, b) Nx_effect.t, r) continuation ->
      (a, b) ND.t ->
      F.Tensor.t ->
      F.Tensor.t ->
      r =
   fun k dt tq tr -> continue k (traced st dt tq, traced st dt tr)
  in
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
            (* A half-precision product multiplies at float32 and rounds once,
               as the eager one does. *)
            let narrow =
              ND.is_float (dt t_in) && TD.itemsize (F.Tensor.dtype t) < 4
            in
            ret k (dt t_in)
              (if narrow then
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
              (F.Reduce.max ~axis:(Array.to_list axes) ~keepdim:false (go t_in)))
    | E_reduce_min { t_in; axes } ->
        Some
          (fun k ->
            ret k (dt t_in)
              (F.Reduce.min ~axis:(Array.to_list axes) ~keepdim:false (go t_in)))
    | E_argmax { t_in; axis; keepdims } ->
        Some
          (fun k ->
            ret k ND.int32
              (F.Dtype_ops.cast
                 (F.Op.argmax ~axis ~keepdim:keepdims (go t_in))
                 TD.int32))
    | E_argmin { t_in; axis; keepdims } ->
        Some
          (fun k ->
            ret k ND.int32
              (F.Dtype_ops.cast
                 (F.Op.argmin ~axis ~keepdim:keepdims (go t_in))
                 TD.int32))
    | E_sort { t_in; axis; descending } ->
        Some
          (fun k ->
            ret k (dt t_in) (sort_graph ~dim:axis ~descending (go t_in)))
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
    (* Staged scans. A multi-device (pmap) trace cannot stage a loop yet: it
       answers the probe with [false] — so reverse-mode below tapes the eager
       fold per step and never records an [E_scan_bwd] — and unrolls a directly
       performed scan into the trace, as every jit did before staging. *)
    | Scan.E_scan_probe ->
        Some (fun k -> continue k (Option.is_none st.st_multi))
    | Scan.E_scan req ->
        Some
          (fun k ->
            if st.st_multi <> None then
              let res : Scan.scan_res =
                Effect.Deep.match_with
                  (fun () -> Scan.eager req)
                  () (handler st)
              in
              continue k res
            else stage_scan st req k)
    | Scan.E_scan_bwd bwd -> Some (fun k -> stage_scan_bwd st bwd k)
    (* Indexed access *)
    | E_gather { data; indices; axis } ->
        Some
          (fun k ->
            ret k (dt data) (F.Op.gather (go data) ~dim:axis (go indices)))
    | E_scatter { data_template; indices; updates; axis; mode; unique_indices }
      ->
        Some
          (fun k ->
            let t = go data_template and index = go indices in
            let src = go updates in
            let r =
              match (st.st_multi, mode) with
              | None, _ ->
                  F.Op.scatter_indexed
                    (write_destination st t ~operands:[ index; src ])
                    ~dim:axis index src ~mode ~unique:unique_indices
              | Some _, `Set -> F.Op.scatter t ~dim:axis index src
              | Some _, `Add ->
                  F.Op.scatter_reduce t ~dim:axis index src ~reduce:`Sum
                    ~include_self:true ()
            in
            ret k (dt data_template) r)
    (* The window write. A constant corner is a padded [v] selected over [t] in
       one pass. A traced corner is a scatter of [v]'s elements at their flat
       positions in [t], which are distinct and, the corner being clamped by the
       frontend, inside [t]: its cost is [v]. Flat positions are int32, and a
       sharded trace keeps the one-hot scatter, so beyond either the window is
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
              st.st_multi = None
              && Array.fold_left ( * ) 1 tshape <= Int32.to_int Int32.max_int
            then begin
              let st_t = go starts in
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
                     (write_destination st tt ~operands:[ index; src ])
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
        Some (fun k -> ret k (dt a) (F.Op.matmul (go a) (go b)))
    (* Quantised products lower to Nx compositions, traced under this handler
       like the function's own operations, and on a single device to tolk's
       kernels over the traced values. *)
    | Nx_quant.Effect.E_quant { w; op } ->
        let kernels =
          match st.st_multi with
          | Some _ -> None
          | None ->
              let quant_matmul ?ids x ~codes ~scales =
                let part name t =
                  let tt = go t in
                  if Lazy.force jit_debug >= 1 && not (is_storage tt) then
                    Printf.eprintf
                      "rune.jit: quantised product: %s is a view, copied on \
                       every call\n\
                       %!"
                      name;
                  tt
                in
                let codes = part "codes" codes
                and scales = part "scales" scales in
                traced st (dt x)
                  (F.Op.quant_matmul ?ids:(Option.map go ids) (go x) ~codes
                     ~scales)
              in
              let block_matmul ~transpose x w ~ids =
                traced st (dt x)
                  (F.Op.block_matmul ~transpose (go x) (go w) ~ids:(go ids))
              in
              Some { Quant.device = st.st_device; quant_matmul; block_matmul }
        in
        Some
          (fun k ->
            continue k
              (Effect.Deep.match_with
                 (fun () -> Quant.lower kernels w op)
                 () (handler st)))
    (* Device movement is the identity on the single jit device. *)
    (* A placement inside a program is the compiler's: placing a value where
       the program runs is the identity, and a program moves nothing between
       devices yet. *)
    | E_place { placement; t_in } -> (
        match st.st_placement with
        | Some p when Nx.Placement.equal p placement ->
            Some (fun k -> continue k t_in)
        | _ ->
            Some
              (fun k ->
                discontinue k
                  (Jit_error
                     (Format.asprintf
                        "Rune.jit: a compiled program cannot place a value on \
                         %a"
                        Nx.Placement.pp placement))))
    (* A traced value lives where the program runs, which is what the gradient
       of a placement asks of its primal. Several devices are the compiler's to
       split. *)
    | E_placement x when is_traced x -> (
        match st.st_placement with
        | Some p -> Some (fun k -> continue k p)
        | None ->
            Some
              (fun k ->
                discontinue k
                  (Jit_error
                     "Rune.jit: placement inside a program over several \
                      devices is the compiler's; query it outside the jitted \
                      function")))
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
    (* The mapped-axis index. Under pmap it is the device's own index, bound as
       a per-device scalar input buffer (each device's buffer holds its index)
       exactly as sharded input slices are bound. A key folded with it
       ([Nx.Rng.fold_in_axis]) therefore decorrelates the devices while keeping
       the key's global (scalar-broadcast) shape, so downstream samplers are
       unchanged. A buffer value (not a symbolic offset) survives the sharding,
       allreduce, and grad rewrites intact. Single device jit has one lane: fall
       through to the eager index 0. *)
    | E_axis_index -> (
        match st.st_multi with
        | None -> None
        | Some _ ->
            Some
              (fun k ->
                let node =
                  match st.axis_index with
                  | Some node -> node
                  | None ->
                      let node = make_node st TD.int32 1 in
                      st.axis_index <- Some node;
                      node
                in
                ret k ND.int32 (buffer_tensor node [||])))
    | _ -> None
  in
  { retc = Fun.id; exnc = raise; effc }

(* The forward scan: trace the body once, compile it as a sub-program, and emit
   the loop call. The carry, the rows and the outputs are structures: every leaf
   has its own slot, and slots pair with their buffers by traversal position
   ([Scan.leaves], [Scan.unflatten]). A carry leaf is a buffer pair, a row leaf
   is read at row [i] of its stacked input, and an output leaf is written at row
   [i] of its stack. When a staged transpose will read them ([req_record]), the
   body also writes the carry it receives to a carry stack. *)
and stage_scan : type r.
    state -> Scan.scan_req -> (Scan.scan_res, r) Effect.Deep.continuation -> r =
 fun st req k ->
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
            && not
                 (List.exists
                    (fun (Scan.Packed_t g) -> Obj.repr g == k)
                    !closed)
          then closed := Scan.Packed_t t :: !closed);
    }
  in
  let slot_c, c_slots = body_slots st req_carry in
  let slot_x, x_slots = body_slots st ~row:true req_xs in
  (* Trace the body once under a nested copy of this tracer, collecting its
     external inputs and the buffers its indexed writes into the carry land
     in. *)
  let writes = List.map (fun s -> (s.s_node, ref [])) c_slots in
  let outer_writes = st.scan_writes in
  st.scan_collectors <- collect :: st.scan_collectors;
  st.scan_writes <- writes @ outer_writes;
  let c_next, y =
    Fun.protect
      ~finally:(fun () ->
        st.scan_collectors <- List.tl st.scan_collectors;
        st.scan_writes <- outer_writes)
      (fun () ->
        Effect.Deep.match_with
          (fun () -> step.run slot_c slot_x)
          () (handler st))
  in
  Tbl.replace st.scan_closed (Obj.repr step) !closed;
  (* A loop can only be compiled from a shape-stable carry (the single prototype
     trace stands for every step). A body that changes a carry shape declines
     staging — the scan folds eagerly and unrolls into this trace, as every jit
     did before staging existed. The traced body's nodes are unreachable from
     any output and never get scheduled. *)
  let c_next = Scan.leaves c_next in
  if not (same_shapes c_next c_slots) then
    Effect.Deep.discontinue k Scan.Not_staged
  else
    let ys = Scan.leaves y in
    let y_outs =
      List.map
        (fun (Scan.Packed_t y) ->
          let shape = shape_of y in
          ( tolk_dtype (Nx_effect.dtype y),
            shape,
            make_node st (tolk_dtype (Nx_effect.dtype y)) (numel shape),
            tolk_of st y ))
        ys
    in
    let stack_outs =
      if req_record then
        List.map (fun s -> make_node st s.s_dt (numel s.s_shape)) c_slots
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
      List.map (fun (Scan.Packed_t c) -> F.Tensor.uop (tolk_of st c)) c_next
    in
    let slot_writes = List.map (fun (_, w) -> !w) writes in
    let c_outs =
      List.map (fun s -> make_node st s.s_dt (numel s.s_shape)) c_slots
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
                store_flat e (numel s.s_shape)
                  (buffer_tensor s.s_node s.s_shape);
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
                         store_flat c_out (numel s.s_shape)
                           (F.Tensor.of_uop value);
                       ])
             (List.combine c_slots modes)
             next_values
          @ List.map
              (fun (_, shape, y_out, value) ->
                U.after ~src:y_out
                  ~deps:[ store_flat y_out (numel shape) value ])
              y_outs
          @ List.map2
              (fun s stack_out ->
                U.after ~src:stack_out
                  ~deps:
                    [
                      store_flat stack_out (numel s.s_shape)
                        (buffer_tensor s.s_node s.s_shape);
                    ])
              (if req_record then c_slots else [])
              stack_outs)
      in
      let sink =
        if copied = [] then sink
        else U.substitute ~walk:true (List.map fill copied) sink
      in
      (modes, schedule_body_linear st sink)
    in
    let rec settle ~copied ~same_index =
      let modes, linear = body ~copied ~same_index in
      let allows ?indexed (s : body_slot) o =
        schedule_allows ?indexed ~linear ~itag:(U.tag s.s_node) ~otag:(U.tag o)
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
      if failed_writes = [] && failed_updates = [] then (modes, linear)
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
    let modes, body_linear = settle ~copied ~same_index in
    let l = loop () in
    List.iter2
      (fun s (Scan.Packed_t x) ->
        add_rows_in_value st l ~slot:s.s_node ~numel:(numel s.s_shape) ~n
          (tolk_of st x))
      x_slots (Scan.leaves req_xs);
    let pairs =
      List.map2
        (fun (s, mode) (Scan.Packed_t c) ->
          let add = add_carry st l ~dt:s.s_dt ~numel:(numel s.s_shape) in
          let init = tolk_of st c in
          match mode with
          | `Written (_, e) ->
              add ~in_place:true ~reads:[ s.s_node; e ] ~writes:[] init
          | `Same_index c_out ->
              add ~in_place:true ~reads:[ s.s_node ] ~writes:[ c_out ] init
          | `Pair (es, c_out) ->
              add ~reads:(s.s_node :: es) ~writes:[ c_out ] init)
        (List.combine c_slots modes)
        (Scan.leaves req_carry)
    in
    let ys_rows =
      List.map
        (fun (dt, shape, y_out, _) ->
          (shape, add_rows_out st l ~slot:y_out ~dt ~numel:(numel shape) ~n))
        y_outs
    in
    let stacks =
      List.map2
        (fun s stack_out ->
          add_rows_out st l ~slot:stack_out ~dt:s.s_dt ~numel:(numel s.s_shape)
            ~n)
        (if req_record then c_slots else [])
        stack_outs
    in
    let call = loop_call l ~body_linear ~reversed:false ~n in
    (* Register the carry stacks as outputs of the forward loop: the backward
       loop reads them, and only a graph-visible dependency keeps the forward
       loop reachable (and so scheduled) when the scan's declared outputs are
       dead — e.g. under [grad], which discards the loss value. *)
    if req_record then
      Tbl.replace st.scan_stacks (Obj.repr step)
        (List.map (fun (buf, stride) -> (written_by call buf, stride)) stacks);
    let r_carry =
      placeholders st req_carry
        (List.map2
           (fun s pair ->
             ( s.s_shape,
               buffer_tensor (written_by call (final_carry ~n pair)) s.s_shape
             ))
           c_slots pairs)
    in
    let r_ys =
      placeholders st y
        (List.map
           (fun (shape, (buf, stride)) ->
             ( Array.append [| n |] shape,
               rows_tensor (written_by call buf) ~n ~stride shape ))
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
  let slot_c, c_slots = body_slots st bwd_carry in
  let slot_x, x_slots = body_slots st ~row:true bwd_xs in
  let _, dc_slots = body_slots st bwd_dc in
  let _, dy_slots = body_slots st ~row:true bwd_dys in
  let differentiable s =
    let (Scan.Packed_t ph) = s.s_ph in
    ND.is_float (Nx_effect.dtype ph)
  in
  (* The body's external inputs, discovered by the forward staging of this
     scan. *)
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
  (* Capture the pullback: run the body once under a private reverse tape, then
     replay the tape against the placeholder cotangents — every op lands in the
     trace, forming the backward body (the forward step's ops are recomputed
     inside it to recover residuals). The external inputs are tracked too, so
     the pullback emits their per-step contributions. *)
  let tape = Tape.create () in
  let track (Scan.Packed_t t) = Tape.track tape t in
  List.iter (fun s -> track s.s_ph) c_slots;
  List.iter (fun s -> if differentiable s then track s.s_ph) x_slots;
  List.iter track closed;
  let c_next, y =
    Effect.Deep.match_with
      (fun () ->
        Effect.Deep.match_with
          (fun () -> step.run slot_c slot_x)
          () (Reverse.handler tape))
      () (handler st)
  in
  let c_next = Scan.leaves c_next in
  if not (same_shapes c_next c_slots) then
    err
      "Rune.jit: the scan body must return a carry of the same shapes it \
       receives (shape-stable carry)";
  let cotangent (Scan.Packed_t t) = Scan.Packed_t (Tape.cotangent tape t) in
  let dc_i, dx_i, dgs =
    Effect.Deep.match_with
      (fun () ->
        let seed (Scan.Packed_t v) s =
          let (Scan.Packed_t d) = s.s_ph in
          Tape.accumulate tape v (Obj.magic d)
        in
        List.iter2 seed c_next dc_slots;
        List.iter2 seed (Scan.leaves y) dy_slots;
        Tape.backward tape;
        ( List.map (fun s -> cotangent s.s_ph) c_slots,
          List.map
            (fun s ->
              if differentiable s then Some (cotangent s.s_ph) else None)
            x_slots,
          List.map
            (fun (Scan.Packed_t g) ->
              Scan.Closed_ctan (g, Tape.cotangent tape g))
            closed ))
      () (handler st)
  in
  (* The backward body: per-leaf carry cotangents, row cotangents, and the
     external inputs' accumulators. The accumulation is elementwise, so it runs
     on the flat buffers directly. The tensor itself stays packed — unpacked,
     its type would escape its scope in the tuple. *)
  let dc_outs =
    List.map (fun s -> make_node st s.s_dt (numel s.s_shape)) c_slots
  in
  let dx_outs =
    List.map
      (fun s ->
        if differentiable s then Some (make_node st s.s_dt (numel s.s_shape))
        else None)
      x_slots
  in
  let g_outs =
    List.map
      (fun (Scan.Closed_ctan (g, dg)) ->
        let g_shape = shape_of g in
        let gdt = tolk_dtype (Nx_effect.dtype g) in
        let gn = numel g_shape in
        ( Scan.Packed_t g,
          g_shape,
          gdt,
          gn,
          make_node st gdt gn,
          make_node st gdt gn,
          tolk_of st dg ))
      dgs
  in
  let value (Scan.Packed_t t) = tolk_of st t in
  let body_sink =
    U.sink
      (List.map2
         (fun (s, dc_out) dc ->
           U.after ~src:dc_out
             ~deps:[ store_flat dc_out (numel s.s_shape) (value dc) ])
         (List.combine c_slots dc_outs)
         dc_i
      @ List.concat_map
          (fun ((s, dx_out), dx) ->
            match (dx_out, dx) with
            | Some dx_out, Some dx ->
                [
                  U.after ~src:dx_out
                    ~deps:[ store_flat dx_out (numel s.s_shape) (value dx) ];
                ]
            | _ -> [])
          (List.combine (List.combine x_slots dx_outs) dx_i)
      @ List.map
          (fun (_, _, _, gn, g_in, g_out, dg_tt) ->
            U.after ~src:g_out
              ~deps:
                [
                  store_flat g_out gn
                    (F.Elementwise.add (F.Tensor.of_uop g_in)
                       (F.Movement.reshape dg_tt [ gn ]));
                ])
          g_outs)
  in
  let body_linear = schedule_body_linear st body_sink in
  let l = loop () in
  List.iter2
    (fun s (stack, stride) ->
      add_rows_in l ~slot:s.s_node ~numel:(numel s.s_shape) ~stride stack)
    c_slots stacks;
  List.iter2
    (fun s x ->
      add_rows_in_value st l ~slot:s.s_node ~numel:(numel s.s_shape) ~n
        (value x))
    x_slots (Scan.leaves bwd_xs);
  List.iter2
    (fun s dy ->
      add_rows_in_value st l ~slot:s.s_node ~numel:(numel s.s_shape) ~n
        (value dy))
    dy_slots (Scan.leaves bwd_dys);
  let dc_pairs =
    List.map2
      (fun ((s, dc_out), d) dc ->
        add_carry st l ~reads:[ d.s_node ] ~writes:[ dc_out ] ~dt:s.s_dt
          ~numel:(numel s.s_shape) (value dc))
      (List.combine (List.combine c_slots dc_outs) dc_slots)
      (Scan.leaves bwd_dc)
  in
  let dx_rows =
    List.map2
      (fun s dx_out ->
        Option.map
          (fun dx_out ->
            add_rows_out st l ~slot:dx_out ~dt:s.s_dt ~numel:(numel s.s_shape)
              ~n)
          dx_out)
      x_slots dx_outs
  in
  let g_pairs =
    List.map
      (fun (_, _, gdt, gn, g_in, g_out, _) ->
        add_carry st l ~reads:[ g_in ] ~writes:[ g_out ] ~dt:gdt ~numel:gn
          (F.Creation.zeros ~dtype:gdt [ gn ]))
      g_outs
  in
  let call = loop_call l ~body_linear ~reversed:true ~n in
  let br_carry =
    placeholders st bwd_carry
      (List.map2
         (fun s pair ->
           ( s.s_shape,
             buffer_tensor (written_by call (final_carry ~n pair)) s.s_shape ))
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
               (shape, rows_tensor (written_by call buf) ~n ~stride s.s_shape)
           | None ->
               (shape, F.Creation.zeros ~dtype:s.s_dt (Array.to_list shape)))
         x_slots dx_rows)
  in
  (* Each external input's total cotangent, as outputs of the loop. *)
  let br_closed =
    List.map2
      (fun (Scan.Packed_t g, g_shape, _, _, _, _, _) pair ->
        let after = written_by call (final_carry ~n pair) in
        let ph = traced st (Nx_effect.dtype g) (buffer_tensor after g_shape) in
        Scan.Closed_ctan (g, ph))
      g_outs g_pairs
  in
  Effect.Deep.continue k { Scan.br_carry; br_xs; br_closed }

(* Host transfers *)

let itemsize dt = Nx_buffer.kind_size_in_bytes dt

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
        (Nativeint.of_int (NV.offset v * itemsize dt))
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

(* The window of [buf] that starts at byte [off] and spans [len] bytes; [buf]
   itself when the window covers it. A window is released before its base can
   be. *)
let with_window buf ~off ~len f =
  if off = 0 && len = Tolk.Device.Buffer.nbytes buf then f buf
  else begin
    let w =
      Tolk.Device.Buffer.view buf ~size:len ~dtype:Tolk_uop.Dtype.uint8
        ~offset:off
    in
    Tolk.Device.Buffer.ensure_allocated w;
    Fun.protect
      ~finally:(fun () -> Tolk.Device.Buffer.deallocate w)
      (fun () -> f w)
  end

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
      let item = itemsize dt in
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
  let item = itemsize (Nx_effect.dtype x) in
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
  let item = itemsize (Nx_buffer.kind host) in
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

(* Upload a host tensor to a device tuple: the whole value to every device
   (replication), or to each device its slice of the shard axis. *)
let upload_multi : type a b.
    scratch ->
    leaf_place ->
    Tolk.Device.t list ->
    Tolk.Device.Buffer.t list ->
    (a, b) Nx_effect.t ->
    unit =
 fun sc place devs bufs x ->
  match place with
  | P_replicated | P_single ->
      List.iter2 (fun dev buf -> copyin_tensor sc dev buf x) devs bufs
  | P_sharded axis ->
      let shape = shape_of x in
      let part = shape.(axis) / List.length bufs in
      List.iteri
        (fun k (dev, buf) ->
          let ranges =
            Array.mapi
              (fun d n ->
                if d = axis then (k * part, (k + 1) * part) else (0, n))
              shape
          in
          copyin_tensor sc dev buf (Nx_effect.shrink x ranges))
        (List.combine devs bufs)

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

(* [gather_elements] over the elements' bits, read as integers of their width: a
   float read into an OCaml float would quiet a signalling NaN. An element of 16
   bytes is two 8-byte words; 4-bit elements are copied as values. *)
let gather_view : type a b.
    (a, b) Nx_buffer.t -> base:int -> NV.t -> (a, b) Nx_buffer.t -> unit =
 fun src ~base v dst ->
  let as_words (type c d) (word : (c, d) Nx_buffer.kind) w =
    let shape = NV.shape v and strides = NV.strides v in
    let words =
      NV.create
        ~offset:(NV.offset v * w)
        ~strides:(Array.append (Array.map (fun s -> s * w) strides) [| 1 |])
        (Array.append shape [| w |])
    in
    gather_elements
      (Nx_buffer.reinterpret word src)
      ~base:(base * w) words
      (Nx_buffer.reinterpret word dst)
  in
  match Nx_buffer.kind src with
  | Nx_buffer.Int4 | Nx_buffer.UInt4 -> gather_elements src ~base v dst
  | kind -> (
      match Nx_buffer.kind_size_in_bytes kind with
      | 1 -> as_words Nx_buffer.Int8 1
      | 2 -> as_words Nx_buffer.Int16 1
      | 4 -> as_words Nx_buffer.Int32 1
      | 8 -> as_words Nx_buffer.Int64 1
      | n -> as_words Nx_buffer.Int64 (n / 8))

(* The elements of [buf]'s storage from [lo] to [hi] on the host: borrowed from
   the buffer's memory when the host addresses it, copied otherwise. *)
let storage_range : type a b.
    (a, b) ND.t ->
    Tolk.Device.Buffer.t ->
    lo:int ->
    hi:int ->
    (a, b) Nx_buffer.t * [ `Borrowed | `Copied ] =
 fun dt buf ~lo ~hi ->
  let item = itemsize dt in
  match Tolk.Device.Buffer.as_buffer buf with
  | Some mem ->
      let bytes = Bigarray.Array1.sub mem (lo * item) ((hi - lo) * item) in
      (Nx_buffer.reinterpret dt (Nx_buffer.of_bigarray1 bytes), `Borrowed)
  | None ->
      let host = Nx_buffer.create dt (hi - lo) in
      with_window buf ~off:(lo * item)
        ~len:((hi - lo) * item)
        (fun w -> copyout_into (Hashtbl.create 1) w ~dst_off:0 host);
      (host, `Copied)

(* The elements of view [v] of one buffer's storage. *)
let read_window : type a b.
    (a, b) ND.t -> Tolk.Device.Buffer.t -> NV.t -> (a, b) Nx_buffer.t =
 fun dt buf v ->
  let n = NV.numel v in
  if n = 0 then Nx_buffer.create dt 0
  else
    let lo, hi = extent v in
    let src, how = storage_range dt buf ~lo ~hi in
    if how = `Borrowed then
      bytes_from_device := !bytes_from_device + (n * itemsize dt);
    if NV.is_c_contiguous v && how = `Copied then src
    else begin
      let dst = Nx_buffer.create dt n in
      if NV.is_c_contiguous v then Nx_buffer.blit ~src ~dst
      else gather_view src ~base:lo v dst;
      dst
    end

(* A split value's shards, each a C-contiguous [shard] of the whole [shape],
   gathered in global order. *)
let read_shards : type a b.
    (a, b) ND.t ->
    int array ->
    axis:int ->
    NV.t ->
    Tolk.Device.Buffer.t list ->
    (a, b) Nx_buffer.t =
 fun dt shape ~axis shard bufs ->
  let n = numel shape and shard_n = NV.numel shard in
  let host = Nx_buffer.create dt n in
  let outer = ref 1 in
  for d = 0 to axis - 1 do
    outer := !outer * shape.(d)
  done;
  let shard_row = shard_n / !outer in
  let full_row = List.length bufs * shard_row in
  List.iteri
    (fun k buf ->
      let part = read_window dt buf shard in
      for o = 0 to !outer - 1 do
        for i = 0 to shard_row - 1 do
          Nx_buffer.unsafe_set host
            ((o * full_row) + (k * shard_row) + i)
            (Nx_buffer.unsafe_get part ((o * shard_row) + i))
        done
      done)
    bufs;
  host

let read : type a b. (a, b) Nx_effect.resident -> (a, b) Nx_buffer.t =
 fun r ->
  drain_releases ();
  match (store_of r.r_cell, r.r_placement) with
  | None, _ -> assert false (* nx reads held and donated values itself *)
  | Some s, Sharded { axis; devices } ->
      List.iter Tolk.Device.synchronize s.s_devices;
      let shape = Array.copy (NV.shape r.r_view) in
      shape.(axis) <- shape.(axis) * List.length devices;
      read_shards r.r_dtype shape ~axis r.r_view s.s_bufs
  | Some s, (Device _ | Replicated _) -> (
      match (s.s_devices, s.s_bufs) with
      | dev :: _, buf :: _ ->
          Tolk.Device.synchronize dev;
          read_window r.r_dtype buf r.r_view
      | _ -> Nx_buffer.create r.r_dtype 0 (* an empty value has no buffer *))

(* [x] with a host value in place of a placed one: its view's elements. *)
let on_host : type a b. (a, b) Nx_effect.t -> (a, b) Nx_effect.t = function
  | Placed _ as x -> Nx_effect.Host (Nx_effect.host_of x)
  | x -> x

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

(* The name of the tolk device that compiles and runs the host's programs. *)
let host_name = "CPU"

let tolk_device_of d =
  if d == Nx.Device.host then Tolk.Device.get host_name
  else
    match Hashtbl.find_opt by_name (Nx.Device.name d) with
    | Some (d', dev) when d' == d -> dev
    | _ -> invalid_arg ("Rune: " ^ Nx.Device.name d ^ " is not a rune device")

(* Raise unless [dev] can hold [dt]. *)
let check_holds (type a b) d dev (dt : (a, b) ND.t) =
  if not (holds dev dt) then
    invalid_arg
      (Printf.sprintf "Nx.place: %s cannot hold %s" (Nx.Device.name d)
         (ND.to_string dt))

(* The engine of rune's devices. [make_placed] wraps buffers already on the
   devices as a placed value whose cell releases them when it is unreachable;
   [place_on] uploads a value, reading it first if it is placed elsewhere. An
   upload from a mapped file bypasses the allocator's cache, so a dropped model
   returns to the system rather than staying parked in it. *)
let rec engine = { Nx_effect.read; place = (fun p x -> place_on p x) }

and make_placed : type a b.
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
  let cell = Nx_effect.cell engine ~length:(NV.numel view) (Buffers s) in
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
  match p with
  | Replicated _ | Sharded _ ->
      invalid_arg
        (Format.asprintf
           "Nx.place: placing a value on several devices (%a) is not supported \
            yet"
           Nx.Placement.pp p)
  | Device d ->
      let dev = tolk_device_of d in
      check_holds d dev (Nx_effect.dtype x);
      let x = on_host x in
      let dt = Nx_effect.dtype x and shape = shape_of x in
      let n = numel shape in
      let nolru = Nx_buffer.file_range (Nx_effect.to_host x) <> None in
      let bufs =
        if n = 0 then []
        else begin
          let buf =
            Tolk.Device.create_buffer ~size:n ~dtype:(tolk_dtype dt)
              ~spec:{ Tolk.Device.Buffer_spec.default with nolru }
              dev
          in
          allocate d buf;
          copyin_tensor (Hashtbl.create 1) dev buf x;
          [ buf ]
        end
      in
      make_placed p [ dev ] ~nolru dt (NV.create shape) bufs

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
        let d = Nx_effect.Device.make name engine in
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

(* The device value of a tolk device. *)
let nx_device dev = device (Tolk.Device.name dev)

let create_fresh_buffer dev dtolk n =
  let buf = Tolk.Device.create_buffer ~size:n ~dtype:dtolk dev in
  allocate (nx_device dev) buf;
  buf

(* Compiled traces *)

type 'q compiled = {
  cp_device : Tolk.Device.t;
  cp_multi : multi_spec option; (* pmap device tuple, [None] = single *)
  cp_zero_copy : bool;
  cp_ctx : Nx_effect.context;
  cp_linear : U.t;
  cp_vars : (string * int) list;
  cp_binding : Tolk.Realize.Buffers.t;
  cp_inputs : input array; (* one per leaf visit, in traversal order *)
  cp_consumed : int -> bool;
      (* the input positions a call consumes: their resident entries are
         released or handed to an output once the call has run *)
  cp_wrapped : (packed * Obj.t) array;
      (* captures bound by aliasing host memory: kernels read that memory on
         every call, so it must stay reachable while the trace can run *)
  cp_bound : (Nx_effect.cell * packed) array;
      (* resident captures whose device buffers are this program's constants:
         the values stay reachable while the trace can run, and their cells
         count this binding until the record is collected *)
  cp_outputs : (Obj.t * packed * U.t * leaf_place) list;
      (* output leaf -> its placeholder (dtype and shape), buffer node, and
         placement *)
  cp_empty : Obj.t list;
      (* output leaves with no elements: no buffer holds them, and every call
         returns a fresh empty tensor of the leaf's dtype and shape *)
  cp_aliases : (int * int) list;
      (* output buffer node tag -> traversal position of the consumed input leaf
         whose storage the output may take, the one at the output's own position
         in the state (see [elision] below); an output tag equal to the input's
         own node tag is an input returned unchanged, whose storage moves to the
         output *)
  cp_prefills : (U.t * int) list;
      (* output buffer node -> traversal position of the input leaf whose value
         the output starts from: an indexed write lands in it, and the program
         never copies the input into it *)
  cp_reserved : (int, unit) Hashtbl.t;
      (* tags of input and constant buffer nodes: outputs must not reseed
         them *)
  cp_arenas : U.t list;
      (* the memory planner's arena buffer nodes, bound at every call to the
         device's shared arenas (see [shared_arena]); none under a pmap *)
  cp_skeleton : 'q; (* trace-time output structure *)
  cp_scratch : scratch; (* staging bytes reused across replays *)
}

module Ops = Tolk_uop.Ops

(* Elision: writing an output over a donated input's storage.

   With tensors as values a jitted carry (parameters, optimizer state, a KV
   cache) returns fresh outputs every call, and a step releases the inputs it
   consumes afterwards, so it holds two generations of state on the device. When
   an output may safely take a donated input's buffer, it is bound to that
   buffer instead of a fresh one and the input's storage moves to the output
   value: one generation, no copy, and results identical.

   Safe means two things, both decided once at compile time. First, in the
   traced graph every path from the input's buffer node to the output's node
   passes only through elementwise operations, casts of equal width, reshapes,
   and contiguous markers, so the kernel storing the output reads the input only
   at the index it writes, whatever the scheduler fuses into it. Second, in the
   linear schedule no kernel reads the input after the one that first writes the
   output, so the old value is never read through the new one. An output that
   never reads an input meets the first condition vacuously and may take that
   input's buffer under the second: a bf16 copy of f32 master weights takes the
   previous copy's storage this way. The one candidate is the consumed leaf at
   the output's own position in the state: a step returns its state in the order
   it took it. An indexed write into an input is the third case. Its output is a
   buffer the program writes at loaded indices and never fills: replay gives it
   the input's value (see [write_destination]). Taking the input's storage is
   how it gets that value for free, so it is taken only when that input is the
   output's candidate, and the kernel that writes it reads at indices of its
   own, so under the second condition it must not read the input either. Replay
   adds what only it knows: the input must have seeded from a donated cell that
   seeds no other leaf of the call, and nothing else may claim the same buffer.
   Single-device programs only; a pmap carry keeps two generations. *)

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

(* Donation. A consumed cell is donated: every view of it raises from now on.
   Its storage passes to the output at its position ([lend]) or goes back to the
   allocator, where only work queued after this call can take it. *)
let bufs_of c = match store_of c with Some s -> s.s_bufs | None -> []

let donate (c : Nx_effect.cell) ~lend =
  (match store_of c with
  | Some s when lend ->
      s.s_bufs <- [];
      account s (-1)
  | Some s -> release_store s
  | None -> ());
  c.state <- Donated

let signature_of (type p) (module P : Nx.Ptree.S with type t = p) (params : P.t)
    =
  let acc = ref [] in
  P.iter
    (fun leaf ->
      acc := (ND.to_string (Nx_effect.dtype leaf), shape_of leaf) :: !acc)
    params;
  List.rev !acc

let trace_compile (type p q) ~device:dev ~zero_copy ~consumed_from ~const_cache
    ?(may_move = false) ?layouts ?multi ?beam ?beam_parallel
    (module P : Nx.Ptree.S with type t = p)
    (module Q : Nx.Ptree.S with type t = q) (f : P.t -> Q.t) (params : P.t) :
    Q.t compiled =
  (* Input leaves from position [n] on are consumed, and output leaf [k] is the
     state's leaf at input position [n + k]. *)
  let consumed i =
    match consumed_from with Some n -> i >= n | None -> false
  in
  incr trace_counter;
  let st =
    {
      st_id = !trace_counter;
      st_device = dev;
      st_multi = Option.map (fun (spec, _) -> spec.md_names) multi;
      st_placement =
        (match multi with
        | None -> Some (Nx.Placement.device (nx_device dev))
        | Some _ -> None);
      st_may_move = may_move;
      refusal = None;
      st_takes_storage =
        (fun i -> consumed i && (not zero_copy) && multi = None);
      st_ctx = Nx_effect.create_context ();
      table = Tensor_map.Tbl.create 64;
      captures = Tensor_map.Tbl.create 16;
      input_tags = Hashtbl.create 16;
      prefills = [];
      consts = [];
      bound = Tensor_map.Tbl.create 16;
      bound_consts = [];
      axis_index = None;
      scan_stacks = Tbl.create 4;
      scan_closed = Tbl.create 4;
      scan_collectors = [];
      scan_writes = [];
    }
  in
  (* One placeholder and one input record per leaf visit, in traversal order, so
     replay pairs current leaves positionally: a tensor behind two leaves is two
     inputs, equal on this call and free to differ on the next. *)
  let inputs = ref [] and placeholders = ref [] in
  let pos = ref 0 in
  P.iter
    (fun leaf ->
      let dtolk = tolk_dtype (Nx_effect.dtype leaf) in
      (* On a guessed device the refusal waits for the trace, whose captures may
         move the program where the dtype is held. *)
      if not (holds dev (Nx_effect.dtype leaf)) then begin
        let e =
          Invalid_argument
            (Printf.sprintf
               "Rune.jit: input leaf %d is %s, which %s cannot hold" !pos
               (ND.to_string (Nx_effect.dtype leaf))
               (Tolk.Device.name dev))
        in
        if may_move then refuse st e else raise e
      end;
      let shape = shape_of leaf in
      let n = numel shape in
      let place =
        match multi with None -> P_single | Some (_, places) -> places.(!pos)
      in
      (* The trace-level tensor carries the global shape. A sharded leaf becomes
         a per-shard buffer on the device tuple wrapped in MULTI (whose shape
         multiplies the axis back up); a replicated leaf is a full-size buffer
         on the tuple with no wrapper. *)
      (* A view of part of a storage binds the range of storage it reaches. *)
      let layout = match layouts with Some a -> a.(!pos) | None -> dense in
      let size = layout_size layout shape in
      let node, tt, bufs =
        match (place, multi) with
        | P_single, _ ->
            let node = make_node st dtolk size in
            ( node,
              layout_tensor node layout shape,
              [ Tolk.Device.create_buffer ~size ~dtype:dtolk dev ] )
        | P_replicated, Some (spec, _) ->
            let node = make_node st dtolk n in
            ( node,
              buffer_tensor node shape,
              List.map
                (fun d -> Tolk.Device.create_buffer ~size:n ~dtype:dtolk d)
                spec.md_devs )
        | P_sharded a, Some (spec, _) ->
            let ndev = List.length spec.md_names in
            let per = n / ndev in
            let node = make_node st dtolk per in
            let shard_shape =
              Array.to_list shape
              |> List.mapi (fun i d -> if i = a then d / ndev else d)
            in
            let inner =
              U.reshape ~src:node ~shape:(F.Tensor.shape_uop shard_shape)
            in
            ( node,
              F.Tensor.of_uop (U.multi ~src:inner ~axis:a),
              List.map
                (fun d -> Tolk.Device.create_buffer ~size:per ~dtype:dtolk d)
                spec.md_devs )
        | (P_replicated | P_sharded _), None -> assert false
      in
      let ph =
        Nx_effect.traced st.st_ctx (Nx_effect.dtype leaf) shape
          (Node { trace = st.st_id; tensor = tt })
      in
      placeholders := Scan.Packed_t ph :: !placeholders;
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
    params;
  let ph_params = Scan.unflatten (module P) params (List.rev !placeholders) in
  let y =
    Gate.with_transform (fun () ->
        Effect.Deep.match_with f ph_params (handler st))
  in
  (* Collect the output leaves; a leaf the trace never saw is a constant passing
     through unchanged. *)
  let out_assoc = ref [] in
  Q.iter
    (fun (type a b) (leaf : (a, b) Nx_effect.t) ->
      let key = Obj.repr leaf in
      if not (List.exists (fun (k, _, _) -> k == key) !out_assoc) then
        out_assoc :=
          (key, Packed (Nx_effect.dtype leaf, leaf), tolk_of st leaf)
          :: !out_assoc)
    y;
  Option.iter raise st.refusal;
  let empty, outs =
    List.partition
      (fun (_, Packed (_, ph), _) -> numel (shape_of ph) = 0)
      (List.rev !out_assoc)
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
      (fun (key, (Packed (dt, _) as pk), tt) -> (key, pk, anchor dt tt))
      outs
  in
  (* Canonicalize sharding before allocation: rewrite the multi-device rules
     over the whole output graph now, so every sharded value reaching a sink is
     a syntactic MULTI and buffer allocation sizes its output per shard
     (replicated values allocate full-size on every device). Scheduling
     reapplies the same rules; the rewrite is idempotent. *)
  let out_uops =
    let outs_u = List.map (fun (_, _, tt) -> F.Tensor.uop tt) out_anch in
    match multi with
    | None ->
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
        let kept, copied =
          List.partition (fun (b, _) -> is_output b) st.prefills
        in
        st.prefills <- kept;
        if copied = [] then outs_u
        else
          let filled (b, input) =
            (b, U.after ~src:b ~deps:[ U.store ~dst:b ~value:input () ])
          in
          U.children
            (U.substitute ~walk:true (List.map filled copied) (U.sink outs_u))
    | Some _ ->
        let shapes n =
          match U.max_shape n with
          | s -> Some s
          | exception Invalid_argument _ -> None
        in
        let pre = U.sink outs_u in
        let pre =
          U.graph_rewrite (Tolk.Multi.multi_pm ~shapes ~devices:U.device_of) pre
        in
        U.children pre
  in
  let place_of u =
    match U.device_of u with
    | Some (U.Multi _) -> (
        match (U.op u, U.axis u) with
        | Tolk_uop.Ops.Unshard, Some a -> P_sharded a
        | _ -> P_replicated)
    | Some (U.Single _) | Some (U.Index _) | None -> P_single
  in
  (* Resolve each output to the buffer node realization assigned it. An output
     whose node is a graph buffer under identity wrappers (an input or constant
     returned unchanged: [U.contiguous] elides itself on buffer-identity
     sources, so such outputs are never scheduled) reads that buffer directly.
     In multi mode the mapped node may wrap the buffer in MULTI; follow it
     down. *)
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
      (fun (key, pk, _) u ->
        let c =
          match written_buffer u with
          | Some v -> v
          | None when multi = None && strip_identity u = None && moves u ->
              Option.get
                (written_buffer
                   (F.Tensor.uop (F.Creation.clone (F.Tensor.of_uop u))))
          | None -> U.contiguous ~src:u ()
        in
        (key, pk, u, place_of u, c))
      out_anch out_uops
  in
  let sink = U.sink (List.map (fun (_, _, _, _, c) -> c) out_conts) in
  let sink, buffer_map = Tolk.Bufferize.run sink in
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
    List.map
      (fun (key, pk, u, place, c) ->
        (key, pk, resolve "an output of the traced function" u c, place))
      out_conts
  in
  (* Persistent compile cache: a hit replaces scheduling and kernel compilation
     with an import of the stored compiled linear, rebound to this trace's fresh
     buffer nodes. Multi-device placements are not cached. *)
  (* The effective beam width: the per-call override when it enables search,
     otherwise the BEAM environment variable. Part of the persistent cache key
     because it changes the compiled kernels. *)
  let effective_beam =
    match beam with
    | Some b when b >= 1 -> b
    | Some _ | None -> env_int "BEAM" 0
  in
  let cache_key =
    match multi with
    | Some _ -> None
    | None -> Jit_cache.key ~device:dev ~beam:effective_beam call
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
          let bound =
            List.map (fun inp -> inp.i_node) !inputs
            @ List.map fst st.consts
            @ List.map (fun (node, _, _, _) -> node) st.bound_consts
            @ Option.to_list st.axis_index
            @ List.map (fun (_, _, node, _) -> node) cp_outputs
          in
          Tolk.Schedule.memory_plan_rewrite linear (held_buffers bound linear)
        in
        let linear =
          let compile () =
            Tolk.Realize.pm_compile ~device:dev ?beam
              ~to_program:(to_program dev) linear
          in
          match beam_parallel with
          | None -> compile ()
          | Some n ->
              Tolk.Helpers.Context_var.(
                with_context [ B (Tolk.Search.beam_parallel, n) ] compile)
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
  (* Batch consecutive graph-compatible kernels into device execution graphs
     (CUDA graphs, Metal indirect command buffers), so replay dispatches each
     batch as one launch instead of one launch per kernel. Buffers rebound
     between replays (inputs, fresh per-call outputs) are diff-patched into the
     recorded graph by [Realize.run_linear]'s graph runner. Honors JIT (>= 2
     disables) and JIT_BATCH_SIZE. *)
  let linear = Tolk.Jit.batch_graphs ~device:dev linear in
  (* The planner's arenas are the int8 buffers its slices view; every other
     buffer a slice views is an input, a constant or an output. *)
  let cp_arenas =
    if multi <> None then []
    else begin
      let bound = Hashtbl.create 16 in
      List.iter
        (fun n -> Hashtbl.replace bound (U.tag n) ())
        (List.map (fun inp -> inp.i_node) !inputs
        @ List.map fst st.consts
        @ List.map (fun (node, _, _, _) -> node) st.bound_consts
        @ List.map (fun (_, _, node, _) -> node) cp_outputs);
      let seen = Hashtbl.create 4 and acc = ref [] in
      List.iter
        (fun u ->
          match U.contiguous_view u with
          | Some (src, _)
            when U.op src = Ops.Buffer
                 && TD.equal (U.dtype src) TD.int8
                 && (not (Hashtbl.mem bound (U.tag src)))
                 && not (Hashtbl.mem seen (U.tag src)) ->
              Hashtbl.replace seen (U.tag src) ();
              acc := src :: !acc
          | _ -> ())
        (U.toposort ~enter_calls:true linear);
      List.rev !acc
    end
  in
  let binding = Tolk.Realize.Buffers.create () in
  let reserved = Hashtbl.create 16 in
  List.iter
    (fun inp ->
      Hashtbl.replace reserved (U.tag inp.i_node) ();
      match inp.i_bufs with
      | [ buf ] when inp.i_place = P_single ->
          Tolk.Realize.Buffers.seed binding inp.i_node buf
      | bufs ->
          Tolk.Realize.Buffers.seed_multi binding inp.i_node
            (Tolk.Device.Multi_buffer.of_bufs bufs))
    !inputs;
  (* Bind each constant once, at compile time: alias its memory when the device
     shares host memory and the tensor is contiguous, copy its bytes to the
     device otherwise (to every device of a pmap tuple). The staging of these
     one-time uploads is dropped with this table. *)
  let scratch = Hashtbl.create 8 in
  let wrapped = ref [] in
  List.iter
    (fun (node, (Packed (cdt, src) as pk)) ->
      Hashtbl.replace reserved (U.tag node) ();
      match if zero_copy then wrap_tensor dev src else None with
      | Some (buf, keep) ->
          Tolk.Realize.Buffers.seed binding node buf;
          wrapped := (pk, keep) :: !wrapped
      | None -> (
          (* One device copy of a capture serves every signature of the closure:
             the bytes are uploaded when the capture is first compiled and later
             compilations reuse the buffer. *)
          let buf =
            match Tensor_map.Tbl.find_opt const_cache (Key src) with
            | Some buf -> buf
            | None ->
                let n = numel (shape_of src) in
                let dtolk = tolk_dtype cdt in
                let buf =
                  match multi with
                  | None ->
                      let buf =
                        Tolk.Device.create_buffer ~size:n ~dtype:dtolk dev
                      in
                      copyin_tensor scratch dev buf src;
                      Tolk.Realize.Single buf
                  | Some (spec, _) ->
                      let bufs =
                        List.map
                          (fun d ->
                            let buf =
                              Tolk.Device.create_buffer ~size:n ~dtype:dtolk d
                            in
                            copyin_tensor scratch d buf src;
                            buf)
                          spec.md_devs
                      in
                      Tolk.Realize.Multi (Tolk.Device.Multi_buffer.of_bufs bufs)
                in
                Tensor_map.Tbl.replace const_cache (Key src) buf;
                buf
          in
          match buf with
          | Tolk.Realize.Single buf ->
              Tolk.Realize.Buffers.seed binding node buf
          | Tolk.Realize.Multi mbuf ->
              Tolk.Realize.Buffers.seed_multi binding node mbuf))
    st.consts;
  (* A bound capture seeds its constant with the resident buffer itself: reads
     leave storage in place, and the value is reachable from the trace. *)
  let views = ref [] in
  let bound =
    List.map
      (fun (node, cell, pk, seed) ->
        Hashtbl.replace reserved (U.tag node) ();
        (match (store_of cell, seed) with
        | Some { s_bufs = [ buf ]; _ }, Range { lo; span; _ } ->
            let view = buffer_range buf ~lo ~span in
            views := view :: !views;
            Tolk.Realize.Buffers.seed binding node view
        | Some { s_bufs = [ buf ]; _ }, (Whole _ | Copy) ->
            Tolk.Realize.Buffers.seed binding node buf
        | _ -> err "Rune.jit: a bound capture was donated while tracing");
        (cell, pk))
      st.bound_consts
  in
  (* The per-device axis index ([Nx.Rng.fold_in_axis] under pmap): one scalar
     buffer per device holding that device's own index. *)
  (match (st.axis_index, multi) with
  | Some node, Some (spec, _) ->
      Hashtbl.replace reserved (U.tag node) ();
      let bufs =
        List.mapi
          (fun i d ->
            let buf = Tolk.Device.create_buffer ~size:1 ~dtype:TD.int32 d in
            let idx = Nx_buffer.create Nx_buffer.int32 1 in
            Nx_buffer.unsafe_set idx 0 (Int32.of_int i);
            copyin_tensor scratch d buf (Nx_effect.from_host st.st_ctx idx);
            buf)
          spec.md_devs
      in
      Tolk.Realize.Buffers.seed_multi binding node
        (Tolk.Device.Multi_buffer.of_bufs bufs)
  | _ -> ());
  let cp_inputs = Array.of_list (List.rev !inputs) in
  let cp_aliases =
    if multi <> None then []
    else begin
      let seen = Hashtbl.create 8 in
      (* An input some output returns unchanged is read by that output's copy
         after the kernels ran, so only the pass-through itself may take it. *)
      let returned_unchanged itag =
        List.exists (fun (_, _, n, _) -> U.tag n = itag) cp_outputs
      in
      (* The first traversal position of an output leaf. *)
      let output_position key =
        let k = ref 0 and at = ref None in
        Q.iter
          (fun leaf ->
            if !at = None && Obj.repr leaf == key then at := Some !k;
            incr k)
          y;
        !at
      in
      List.concat_map
        (fun ((key, Packed (odt, ph), node, place), (_, _, u, _, _)) ->
          let otag = U.tag node in
          if place <> P_single || Hashtbl.mem seen otag then []
          else begin
            Hashtbl.replace seen otag ();
            let odtype = ND.to_string odt and onumel = numel (shape_of ph) in
            (* The input an indexed write into this output starts from: replay
               gives the output that input's value, so it takes that input's
               storage or none. *)
            let starts_from =
              List.find_map
                (fun (b, input) ->
                  if U.tag b = otag then Some (U.tag input) else None)
                st.prefills
            in
            (* [Some true] when the output derives from the input, [Some false]
               when it never reads it; either may take its storage. *)
            let fits inp =
              let itag = U.tag inp.i_node in
              if
                inp.i_place <> P_single || inp.i_dtype <> odtype
                || inp.i_numel <> onumel
              then None
              else if itag = otag then Some true
              else if Hashtbl.mem reserved otag || returned_unchanged itag then
                None
              else
                match starts_from with
                | Some from ->
                    if
                      from = itag
                      && schedule_allows ~indexed:true ~linear ~itag ~otag ()
                    then Some true
                    else None
                | None ->
                    let reaches, ok = same_index_paths ~inode:inp.i_node u in
                    if ok && schedule_allows ~linear ~itag ~otag () then
                      Some reaches
                    else None
            in
            let partner =
              match (consumed_from, output_position key) with
              | Some n, Some k when n + k < Array.length cp_inputs ->
                  Some (n + k)
              | _ -> None
            in
            match partner with
            | Some i when fits cp_inputs.(i) <> None -> [ (otag, i) ]
            | _ -> []
          end)
        (List.combine cp_outputs out_conts)
    end
  in
  let cp_prefills =
    List.map
      (fun (b, input) ->
        if not (List.exists (fun (_, _, n, _) -> U.tag n = U.tag b) cp_outputs)
        then err "Rune.jit: an indexed write's buffer is not an output";
        let position = ref None in
        Array.iteri
          (fun i inp ->
            if !position = None && inp.i_node == input then position := Some i)
          cp_inputs;
        (b, Option.get !position))
      st.prefills
  in
  List.iter (fun ((c : Nx_effect.cell), _) -> c.bound <- c.bound + 1) bound;
  let compiled =
    {
      cp_device = dev;
      cp_multi = Option.map fst multi;
      cp_zero_copy = zero_copy;
      cp_ctx = st.st_ctx;
      cp_linear = linear;
      cp_vars = var_vals;
      cp_binding = binding;
      cp_inputs;
      cp_consumed = consumed;
      cp_wrapped = Array.of_list !wrapped;
      cp_bound = Array.of_list bound;
      cp_outputs;
      cp_empty = List.map (fun (key, _, _) -> key) empty;
      cp_aliases;
      cp_prefills;
      cp_reserved = reserved;
      cp_arenas;
      cp_skeleton = y;
      cp_scratch = Hashtbl.create 8;
    }
  in
  let cells = List.map fst bound and views = !views in
  Gc.finalise_last
    (fun () ->
      pending_views := views @ !pending_views;
      List.iter
        (fun (cell : Nx_effect.cell) -> cell.bound <- cell.bound - 1)
        cells)
    compiled;
  compiled

let replay (type p q) (module P : Nx.Ptree.S with type t = p)
    (module Q : Nx.Ptree.S with type t = q) (c : Q.t compiled) (params : P.t) :
    Q.t =
  drain_releases ();
  (* Bind the arenas before the first run records a device graph over them, so
     the graph re-patches their addresses when a shared arena grows. *)
  List.iteri
    (fun k node ->
      let nbytes = List.fold_left ( * ) 1 (U.max_shape node) in
      Tolk.Realize.Buffers.seed c.cp_binding node
        (shared_arena c.cp_device k nbytes))
    c.cp_arenas;
  let in0 = !bytes_to_device and out0 = !bytes_from_device in
  (* Seed the inputs. A leaf placed on this device seeds its input node with
     its buffer directly — no transfer, and the value stays resident (inputs
     are read-only). Otherwise
     wrap the current leaf's memory when the device shares host memory and the
     leaf is contiguous, and copy its bytes if not. Seeded leaves and wrapped
     hosts are kept reachable until the run completes, so no finalizer can
     release a buffer the kernels still read. *)
  (* A placed value on a single device seeds the compiled input as [seed_of]
     says. On a device tuple, it seeds only when its view covers its storage and
     its placement matches the input's: the same tuple with the same shard axis.
     Any other value is read by the copy path, which leaves it where it is, and
     re-split. *)
  let resident_multi : type a b.
      multi_spec -> leaf_place -> (a, b) Nx_effect.t -> Nx_effect.cell option =
   fun spec place -> function
     | Placed r when Nx_effect.covers r -> (
         let axis =
           match r.r_placement with
           | Sharded { axis; _ } -> Some axis
           | _ -> None
         in
         match store_of r.r_cell with
         | Some s
           when s.s_bufs <> []
                && List.equal ( == ) s.s_devices spec.md_devs
                && axis = place_axis place ->
             Some r.r_cell
         | _ -> None)
     | _ -> None
  in
  (* Cells that seeded a consumed leaf of this call. Their buffers are released
     back to the allocator once the call completes — never during it: the
     schedule has no aliasing knowledge, so a consumed buffer must stay intact
     until every kernel has read it. A value whose placement mismatched is read
     by the copy path instead and is never consumed. A bound cell is some
     program's constant, and a cell that also seeded a read leaf is read:
     neither is consumed. *)
  let seeded = ref [] and read = ref [] in
  let seed_entry = Array.make (Array.length c.cp_inputs) None in
  let note i (e : Nx_effect.cell) =
    if not (c.cp_consumed i) then read := e :: !read
    else if e.bound = 0 && not (List.memq e !seeded) then seeded := e :: !seeded
  in
  (* The buffer views of this call's ranges are released once it has run, or has
     raised: not at a safe point inside it, where the launch would allocate them
     again. *)
  let ranges = ref [] in
  Fun.protect ~finally:(fun () -> pending_views := !ranges @ !pending_views)
  @@ fun () ->
  let keep = ref [] in
  let i = ref 0 in
  P.iter
    (fun leaf ->
      let inp = c.cp_inputs.(!i) in
      (* A storage a read leaf reaches, through any view, is read. Read leaves
         come first in traversal. *)
      (match leaf with
      | Placed r when not (c.cp_consumed !i) -> read := r.r_cell :: !read
      | _ -> ());
      (match c.cp_multi with
      | Some spec -> (
          match resident_multi spec inp.i_place leaf with
          | Some e ->
              keep := Obj.repr leaf :: !keep;
              note !i e;
              Tolk.Realize.Buffers.seed_multi c.cp_binding inp.i_node
                (Tolk.Device.Multi_buffer.of_bufs (bufs_of e))
          | None ->
              Tolk.Realize.Buffers.seed_multi c.cp_binding inp.i_node
                (Tolk.Device.Multi_buffer.of_bufs inp.i_bufs);
              upload_multi c.cp_scratch inp.i_place spec.md_devs inp.i_bufs
                (on_host leaf))
      | None -> (
          match seed_of c.cp_device leaf with
          | Whole e ->
              keep := Obj.repr leaf :: !keep;
              note !i e;
              seed_entry.(!i) <- Some e;
              Tolk.Realize.Buffers.seed c.cp_binding inp.i_node
                (List.hd (bufs_of e))
          (* A view of part of a storage is read, never consumed. *)
          | Range { cell = e; lo; span; _ }
            when (not (c.cp_consumed !i)) || List.memq e !read ->
              keep := Obj.repr leaf :: !keep;
              note !i e;
              seed_entry.(!i) <- Some e;
              let range = buffer_range (List.hd (bufs_of e)) ~lo ~span in
              ranges := range :: !ranges;
              Tolk.Realize.Buffers.seed c.cp_binding inp.i_node range
          | Range _ | Copy -> (
              (* A state leaf is donated by cell: a held value is consumed, and
                 a view of part of its storage cannot be. *)
              (match leaf with
              | Placed r when c.cp_consumed !i && not (List.memq r.r_cell !read)
                -> (
                  match r.r_cell.state with
                  | Live (Nx_effect.Held _) -> note !i r.r_cell
                  | Live _ when not (Nx_effect.covers r) ->
                      invalid_arg
                        (Printf.sprintf
                           "Rune.jit_step: state leaf %d is a view of part of \
                            its storage, which cannot be donated; pass \
                            [Nx.copy] of it"
                           !i)
                  | Live _ | Donated -> ())
              | _ -> ());
              match
                if c.cp_zero_copy then wrap_tensor c.cp_device leaf else None
              with
              | Some (buf, ka) ->
                  keep := ka :: !keep;
                  Tolk.Realize.Buffers.seed c.cp_binding inp.i_node buf
              | None -> (
                  match inp.i_bufs with
                  | [ buf ] ->
                      Tolk.Realize.Buffers.seed c.cp_binding inp.i_node buf;
                      copyin_tensor c.cp_scratch c.cp_device buf leaf
                  | _ -> assert false))));
      incr i)
    params;
  let read = !read in
  let seeded = List.filter (fun e -> not (List.memq e read)) !seeded in
  (* Wire the outputs' storage. On the zero-copy device, fresh host buffers
     become the kernels' output storage, so results are written straight into
     the tensors returned to the caller; nodes backed by an input or constant
     buffer keep their binding and are read back through a copy instead. On
     other devices, every distinct non-reserved output node is bound to a fresh
     device buffer for this call, so values from earlier calls keep their own
     storage and never alias a later call's outputs. *)
  let out_hosts : (int, host_out) Hashtbl.t = Hashtbl.create 8 in
  let out_bufs : (int, Tolk.Device.Buffer.t list) Hashtbl.t =
    Hashtbl.create 8
  in
  (* Elision claims: an output takes the storage of the consumed input at its
     position when that input seeded from a cell seeding no other leaf of this
     call, and no earlier output claimed the cell. *)
  let claims : (int, Nx_effect.cell) Hashtbl.t = Hashtbl.create 4 in
  if not c.cp_zero_copy then begin
    let uses e =
      Array.fold_left
        (fun n -> function Some e' when e' == e -> n + 1 | _ -> n)
        0 seed_entry
    in
    let claimed = ref [] in
    List.iter
      (fun (otag, i) ->
        match seed_entry.(i) with
        | Some e
          when (not (Hashtbl.mem claims otag))
               && e.bound = 0
               && (not (List.memq e read))
               && uses e = 1
               && not (List.memq e !claimed) ->
            claimed := e :: !claimed;
            reused_bytes :=
              !reused_bytes
              + List.fold_left
                  (fun a b -> a + Tolk.Device.Buffer.nbytes b)
                  0 (bufs_of e);
            Hashtbl.replace claims otag e
        | _ -> ())
      c.cp_aliases
  end;
  if c.cp_zero_copy then
    List.iter
      (fun (_, Packed (odt, ph), node, _) ->
        let tag = U.tag node in
        if
          (not (Hashtbl.mem c.cp_reserved tag))
          && not (Hashtbl.mem out_hosts tag)
        then begin
          let shape = shape_of ph in
          let n = numel shape in
          let host = Nx_buffer.create odt n in
          let buf =
            wrap_ptr c.cp_device (tolk_dtype odt) n
              (Nx_buffer.unsafe_data_ptr host)
          in
          Tolk.Realize.Buffers.seed c.cp_binding node buf;
          Hashtbl.add out_hosts tag (Host (odt, host))
        end)
      c.cp_outputs
  else
    List.iter
      (fun (_, Packed (odt, ph), node, place) ->
        let tag = U.tag node in
        if
          (not (Hashtbl.mem c.cp_reserved tag))
          && not (Hashtbl.mem out_bufs tag)
        then
          begin match (place, c.cp_multi) with
          | P_single, _ ->
              let buf =
                match Hashtbl.find_opt claims tag with
                | Some e -> List.hd (bufs_of e)
                | None ->
                    create_fresh_buffer c.cp_device (tolk_dtype odt)
                      (numel (shape_of ph))
              in
              Tolk.Realize.Buffers.seed c.cp_binding node buf;
              Hashtbl.add out_bufs tag [ buf ]
          | (P_replicated | P_sharded _), Some spec ->
              (* The node's own shape is the per-device size: the shard for a
                 sharded output, the full value for a replicated one. *)
              let n = List.fold_left ( * ) 1 (U.max_shape node) in
              let bufs =
                List.map
                  (fun d -> create_fresh_buffer d (tolk_dtype odt) n)
                  spec.md_devs
              in
              Tolk.Realize.Buffers.seed_multi c.cp_binding node
                (Tolk.Device.Multi_buffer.of_bufs bufs);
              Hashtbl.add out_bufs tag bufs
          | (P_replicated | P_sharded _), None -> assert false
          end)
      c.cp_outputs;
  (* An output an indexed write lands in starts from its input. One that claimed
     that input's storage already holds the value; any other is given it by a
     copy. *)
  List.iter
    (fun (node, i) ->
      let claimed =
        match (Hashtbl.find_opt claims (U.tag node), seed_entry.(i)) with
        | Some e, Some e' -> e == e'
        | _ -> false
      in
      if not claimed then begin
        let dst = Tolk.Realize.Buffers.of_buffer_node c.cp_binding node in
        let src =
          Tolk.Realize.Buffers.of_buffer_node c.cp_binding
            c.cp_inputs.(i).i_node
        in
        if not (Tolk.Device.Buffer.transfer ~dst ~src) then
          Tolk.Device.Buffer.copy_between ~dst ~src
      end)
    c.cp_prefills;
  Tolk.Realize.run_linear ~device:c.cp_device
    ~to_program:(to_program c.cp_device) c.cp_binding ~var_vals:c.cp_vars
    ~jit:true c.cp_linear;
  (* An output that is an input or a capture returned unchanged keeps its
     reserved binding; copy it into a fresh buffer on the device, so its value
     never aliases an input and survives later calls. A donated input returned
     unchanged instead hands its storage over: no copy. *)
  if not c.cp_zero_copy then
    List.iter
      (fun (_, Packed (odt, ph), node, _) ->
        let tag = U.tag node in
        if Hashtbl.mem c.cp_reserved tag && not (Hashtbl.mem out_bufs tag) then
          match Hashtbl.find_opt claims tag with
          | Some e -> Hashtbl.add out_bufs tag (bufs_of e)
          | None -> begin
              let copy_to dev n src =
                let dst = create_fresh_buffer dev (tolk_dtype odt) n in
                if not (Tolk.Device.Buffer.transfer ~dst ~src) then
                  Tolk.Device.Buffer.copy_between ~dst ~src;
                dst
              in
              match Tolk.Realize.Buffers.buffer_of_node c.cp_binding node with
              | Tolk.Realize.Single src ->
                  Hashtbl.add out_bufs tag
                    [ copy_to c.cp_device (numel (shape_of ph)) src ]
              | Tolk.Realize.Multi m ->
                  let devs =
                    match c.cp_multi with
                    | Some spec -> spec.md_devs
                    | None -> assert false
                  in
                  Hashtbl.add out_bufs tag
                    (List.map2
                       (fun dev src ->
                         copy_to dev (Tolk.Device.Buffer.size src) src)
                       devs
                       (Tolk.Device.Multi_buffer.bufs m))
            end)
      c.cp_outputs;
  (* The call returns while its kernels may still run. An output is a placed
     value whose reads wait for the device, and a buffer released below goes
     back to the device's pool, where only work queued after these kernels can
     take it. Two programs still wait here. On the zero-copy device the outputs
     are host tensors the kernels write in place, and wrapped buffers alias
     caller memory (seeded inputs and wrapped captures) that the caller may
     touch as soon as the call returns. A pmap waits too. *)
  (match c.cp_multi with
  | Some spec -> List.iter Tolk.Device.synchronize spec.md_devs
  | None -> if c.cp_zero_copy then Tolk.Device.synchronize c.cp_device);
  ignore (Sys.opaque_identity !keep);
  ignore (Sys.opaque_identity c.cp_wrapped);
  ignore (Sys.opaque_identity c.cp_bound);
  (* Output leaves resolving to the same buffer node share one value, so each
     device buffer has a single owner. *)
  let handles : (int, packed) Hashtbl.t = Hashtbl.create 8 in
  let y =
    Q.map
      (fun (type a b) (leaf : (a, b) Nx_effect.t) : (a, b) Nx_effect.t ->
        match
          List.find_opt (fun (k, _, _, _) -> k == Obj.repr leaf) c.cp_outputs
        with
        | Some (_, _, node, place) -> (
            let tag = U.tag node in
            match Hashtbl.find_opt out_hosts tag with
            | Some (Host (hdt, host)) -> (
                match ND.equal_witness hdt (Nx_effect.dtype leaf) with
                | Some Type.Equal ->
                    Nx_effect.reshape
                      (Nx_effect.from_host c.cp_ctx host)
                      (shape_of leaf)
                | None -> assert false)
            | None -> (
                if c.cp_zero_copy then
                  let buf =
                    Tolk.Realize.Buffers.of_buffer_node c.cp_binding node
                  in
                  read_out c.cp_scratch c.cp_ctx (Nx_effect.dtype leaf)
                    (shape_of leaf) buf
                else
                  match Hashtbl.find_opt handles tag with
                  | Some (Packed (hdt, h)) -> (
                      match ND.equal_witness hdt (Nx_effect.dtype leaf) with
                      | Some Type.Equal -> h
                      | None -> assert false)
                  | None ->
                      let bufs =
                        match Hashtbl.find_opt out_bufs tag with
                        | Some bufs -> bufs
                        | None -> assert false (* every node was bound above *)
                      in
                      let shape = shape_of leaf in
                      let devices, placement, view =
                        match (place, c.cp_multi) with
                        | P_single, _ ->
                            ( [ c.cp_device ],
                              Nx.Placement.device (nx_device c.cp_device),
                              NV.create shape )
                        | P_replicated, Some spec ->
                            ( spec.md_devs,
                              Nx.Placement.replicated
                                (List.map nx_device spec.md_devs),
                              NV.create shape )
                        | P_sharded axis, Some spec ->
                            let shard = Array.copy shape in
                            shard.(axis) <-
                              shard.(axis) / List.length spec.md_devs;
                            ( spec.md_devs,
                              Nx.Placement.sharded ~axis
                                (List.map nx_device spec.md_devs),
                              NV.create shard )
                        | (P_replicated | P_sharded _), None -> assert false
                      in
                      let dt = Nx_effect.dtype leaf in
                      (* Storage lent by a donated input keeps its way back:
                         past the allocator's cache for an upload from a mapped
                         file. *)
                      let nolru =
                        match Hashtbl.find_opt claims tag with
                        | Some e -> (
                            match store_of e with
                            | Some s -> s.s_nolru
                            | None -> false)
                        | None -> false
                      in
                      let h =
                        make_placed placement devices ~nolru dt view bufs
                      in
                      Hashtbl.add handles tag (Packed (dt, h));
                      h))
        | None ->
            assert (List.memq (Obj.repr leaf) c.cp_empty);
            Nx_effect.reshape
              (Nx_effect.from_host c.cp_ctx
                 (Nx_buffer.create (Nx_effect.dtype leaf) 0))
              (shape_of leaf))
      c.cp_skeleton
  in
  (* Consumption. The storage of each consumed input that seeded from a cell is
     either owned by an output now (a claimed cell: its storage moved) or
     returned to the allocator, where the next call's fresh outputs reuse it in
     queue order, after the kernels of this call. The cell becomes [Donated]:
     every view of it raises from now on. *)
  Hashtbl.iter (fun _ e -> donate e ~lend:true) claims;
  List.iter
    (fun e ->
      if not (Hashtbl.fold (fun _ e' a -> a || e' == e) claims false) then
        donate e ~lend:false)
    seeded;
  if Lazy.force jit_debug >= 1 then begin
    Printf.eprintf
      "rune.jit: replay on %s: %d bytes to device, %d bytes from device, %d \
       bytes resident\n\
       %!"
      (Tolk.Device.name c.cp_device)
      (!bytes_to_device - in0)
      (!bytes_from_device - out0)
      !resident_bytes;
    let n = Array.length seed_entry in
    if List.exists c.cp_consumed (List.init n Fun.id) then
      Array.iteri
        (fun i -> function
          | None -> Printf.eprintf "rune.jit: input leaf %d: not resident\n%!" i
          | Some e ->
              let reused =
                Hashtbl.fold (fun _ e' acc -> acc || e' == e) claims false
              in
              Printf.eprintf "rune.jit: input leaf %d: %s\n%!" i
                (if e.bound > 0 then "bound"
                 else if (not (c.cp_consumed i)) || List.memq e read then "read"
                 else if reused then "storage reused"
                 else "storage copied"))
        seed_entry
  end;
  y

(* Public entry points *)

(* The device a call's placed leaves share. Each must be on [requested] when a
   device is requested, on the same device as the others, and on one device. *)
let leaves_device (type p) (module P : Nx.Ptree.S with type t = p) ~requested
    (params : P.t) =
  let name = Nx.Device.name in
  let found = ref None and i = ref 0 in
  P.iter
    (fun leaf ->
      (match leaf with
      | Nx_effect.Placed { r_placement = Device d; _ } -> (
          (match requested with
          | Some r when r != d ->
              invalid_arg
                (Printf.sprintf
                   "Rune.jit: input leaf %d is on %s and ~device names %s; \
                    place it on %s, or on the host"
                   !i (name d) (name r) (name r))
          | _ -> ());
          match !found with
          | Some (d0, i0) when d0 != d ->
              invalid_arg
                (Printf.sprintf
                   "Rune.jit: input leaves %d and %d are on %s and %s; place \
                    them on one device"
                   i0 !i (name d0) (name d))
          | Some _ -> ()
          | None -> found := Some (d, !i))
      | Nx_effect.Placed { r_placement = p; _ } ->
          invalid_arg
            (Format.asprintf
               "Rune.jit: input leaf %d is on %a; a compiled function runs on \
                one device, so place it on one device, or on the host"
               !i Nx.Placement.pp p)
      | _ -> ());
      incr i)
    params;
  Option.map fst !found

(* The compiled function over [P]. With [consumed_from p = Some n], the leaves
   from position [n] on are the state, which the call consumes and [Q] returns
   in the same order. Two splits of one leaf sequence compile apart: which
   inputs an indexed write lands in and which outputs may take storage differ.

   A call runs on the requested device, else where its placed leaves live, else
   where a capture lives, else on the default device. Captures are found by
   tracing: a trace on the default device that meets a capture placed elsewhere
   runs again on the capture's device, which every later call then takes. *)
let compile_fn (type p q) ?device:name ?beam ?beam_parallel ~consumed_from
    (module P : Nx.Ptree.S with type t = p)
    (module Q : Nx.Ptree.S with type t = q) (f : P.t -> Q.t) : P.t -> Q.t =
  let requested = Option.map device name in
  let captured_on = ref None in
  let cache : (_, Q.t compiled) Hashtbl.t = Hashtbl.create 4 in
  (* Device copies of captured tensors, per device, shared by every signature of
     this closure and keyed by capture identity. *)
  let const_caches : (string, Tolk.Realize.buffer Tensor_map.Tbl.t) Hashtbl.t =
    Hashtbl.create 1
  in
  let compile d ~may_move ~layouts consumed_from params =
    let const_cache =
      match Hashtbl.find_opt const_caches (Nx.Device.name d) with
      | Some t -> t
      | None ->
          let t = Tensor_map.Tbl.create 4 in
          Hashtbl.add const_caches (Nx.Device.name d) t;
          t
    in
    (* The host's programs run over host memory. *)
    trace_compile ~device:(tolk_device_of d) ~zero_copy:(d == Nx.Device.host)
      ~consumed_from ~const_cache ~may_move ~layouts ?beam ?beam_parallel
      (module P)
      (module Q)
      f params
  in
  fun params ->
    if Gate.transforming () then f params
    else
      let d, may_move =
        match
          (requested, leaves_device (module P) ~requested params, !captured_on)
        with
        | Some d, _, _ | None, Some d, _ | None, None, Some d -> (d, false)
        | None, None, None -> (default_device (), true)
      in
      let consumed_from = consumed_from params in
      let signature = signature_of (module P) params in
      (* A leaf bound as a view reads its range in the program, so how it does
         joins the key; any other leaf is its elements in C order. *)
      let layouts =
        let dev = tolk_device_of d and acc = ref [] in
        P.iter (fun leaf -> acc := layout_of (seed_of dev leaf) :: !acc) params;
        Array.of_list (List.rev !acc)
      in
      let key d = (Nx.Device.name d, consumed_from, signature, layouts) in
      let compiled d ~may_move =
        match Hashtbl.find_opt cache (key d) with
        | Some c -> c
        | None ->
            let c = compile d ~may_move ~layouts consumed_from params in
            (* Captures it binds make the device the closure's. *)
            if c.cp_bound <> [||] && !captured_on = None then
              captured_on := Some d;
            Hashtbl.add cache (key d) c;
            c
      in
      let c =
        match compiled d ~may_move with
        | c -> c
        | exception Runs_on d' ->
            captured_on := Some d';
            compiled d' ~may_move:false
      in
      replay (module P) (module Q) c params

let jit2 ?device ?beam ?beam_parallel p q f =
  compile_fn ?device ?beam ?beam_parallel ~consumed_from:(fun _ -> None) p q f

let jit_step (type r s) ?device ?beam ?beam_parallel
    (module R : Nx.Ptree.S with type t = r)
    (module S : Nx.Ptree.S with type t = s) (f : R.t -> S.t -> S.t) :
    R.t -> S.t -> S.t =
  let module P = struct
    type t = R.t * S.t

    let map (f : 'a 'b. ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t) (r, s) =
      let r = R.map f r in
      (r, S.map f s)

    let map2
        (f :
          'a 'b.
          ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t)
        (r, s) (r', s') =
      let r = R.map2 f r r' in
      (r, S.map2 f s s')

    let iter (f : 'a 'b. ('a, 'b) Nx_effect.t -> unit) (r, s) =
      R.iter f r;
      S.iter f s
  end in
  let consumed_from (r, _) =
    let n = ref 0 in
    R.iter (fun _ -> incr n) r;
    Some !n
  in
  let g =
    compile_fn ?device ?beam ?beam_parallel ~consumed_from
      (module P)
      (module S)
      (fun (r, s) -> f r s)
  in
  fun r s -> g (r, s)

let jit (type p c d) ?device ?beam ?beam_parallel
    (module P : Nx.Ptree.S with type t = p) (f : P.t -> (c, d) Nx_effect.t) :
    P.t -> (c, d) Nx_effect.t =
  let module Q = struct
    type t = (c, d) Nx_effect.t

    let map (f : 'a 'b. ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t) t = f t

    let map2
        (f :
          'a 'b.
          ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t)
        a b =
      f a b

    let iter (f : 'a 'b. ('a, 'b) Nx_effect.t -> unit) t = f t
  end in
  jit2 ?device ?beam ?beam_parallel (module P) (module Q) f

let jit' (type a b c d) ?device ?beam ?beam_parallel
    (f : (a, b) Nx_effect.t -> (c, d) Nx_effect.t) :
    (a, b) Nx_effect.t -> (c, d) Nx_effect.t =
  let module L = struct
    type t = (a, b) Nx_effect.t

    let map (f : 'a 'b. ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t) t = f t

    let map2
        (f :
          'a 'b.
          ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t)
        a b =
      f a b

    let iter (f : 'a 'b. ('a, 'b) Nx_effect.t -> unit) t = f t
  end in
  jit ?device ?beam ?beam_parallel (module L) f

(* pmap: multi-device parallel jit. The compiled core is [trace_compile] /
   [replay] with a multi-device placement; pmap only derives the placement from
   [devices] and [in_axes] and validates it against the leaves. *)

let pmap_names devices =
  if devices = [] then
    invalid_arg "Rune.pmap: devices must name at least one device";
  let names = List.map canonical devices in
  let p0 = backend (List.hd names) in
  List.iter
    (fun n ->
      if String.equal n host_name then
        invalid_arg
          "Rune.pmap: CPU is the host, which holds no placed value; use CPU:1, \
           CPU:2, ...";
      if not (String.equal (backend n) p0) then
        invalid_arg
          (Printf.sprintf
             "Rune.pmap: devices must share one backend, got %s and %s"
             (List.hd names) n))
    names;
  names

let pmap2 (type p q) ~devices ?in_axes ?(donate = false) ?beam ?beam_parallel
    (module P : Nx.Ptree.S with type t = p)
    (module Q : Nx.Ptree.S with type t = q) (f : P.t -> Q.t) : P.t -> Q.t =
  let names = pmap_names devices in
  let devs = List.map (fun n -> tolk_device_of (device n)) names in
  let spec = { md_names = names; md_devs = devs } in
  let dev = List.hd devs in
  let ndev = List.length names in
  let cache : (_, Q.t compiled) Hashtbl.t = Hashtbl.create 4 in
  let const_cache : Tolk.Realize.buffer Tensor_map.Tbl.t =
    Tensor_map.Tbl.create 4
  in
  fun params ->
    if Gate.transforming () then f params
    else begin
      let shapes = ref [] in
      P.iter (fun leaf -> shapes := shape_of leaf :: !shapes) params;
      let shapes = Array.of_list (List.rev !shapes) in
      let nleaves = Array.length shapes in
      let axes =
        match in_axes with
        | None -> Array.make nleaves (Some 0)
        | Some l ->
            if List.length l <> nleaves then
              invalid_arg
                (Printf.sprintf
                   "Rune.pmap: in_axes has %d entries but the input has %d \
                    leaves"
                   (List.length l) nleaves);
            Array.of_list l
      in
      let places =
        Array.mapi
          (fun i ax ->
            match ax with
            | None -> P_replicated
            | Some a ->
                let shape = shapes.(i) in
                if a < 0 || a >= Array.length shape then
                  invalid_arg
                    (Printf.sprintf
                       "Rune.pmap: in_axes maps leaf %d to axis %d, but the \
                        leaf has rank %d"
                       i a (Array.length shape));
                if shape.(a) mod ndev <> 0 then
                  invalid_arg
                    (Printf.sprintf
                       "Rune.pmap: leaf %d has dimension %d along axis %d, \
                        which does not divide into %d equal shards"
                       i shape.(a) a ndev);
                P_sharded a)
          axes
      in
      let sg = signature_of (module P) params in
      let c =
        match Hashtbl.find_opt cache sg with
        | Some c -> c
        | None ->
            let c =
              trace_compile ~device:dev ~zero_copy:false
                ~consumed_from:(if donate then Some 0 else None)
                ~const_cache ?beam ?beam_parallel ~multi:(spec, places)
                (module P)
                (module Q)
                f params
            in
            Hashtbl.add cache sg c;
            c
      in
      replay (module P) (module Q) c params
    end

let pmap (type p c d) ~devices ?in_axes ?donate ?beam ?beam_parallel
    (module P : Nx.Ptree.S with type t = p) (f : P.t -> (c, d) Nx_effect.t) :
    P.t -> (c, d) Nx_effect.t =
  let module Q = struct
    type t = (c, d) Nx_effect.t

    let map (f : 'a 'b. ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t) t = f t

    let map2
        (f :
          'a 'b.
          ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t)
        a b =
      f a b

    let iter (f : 'a 'b. ('a, 'b) Nx_effect.t -> unit) t = f t
  end in
  pmap2 ~devices ?in_axes ?donate ?beam ?beam_parallel (module P) (module Q) f
