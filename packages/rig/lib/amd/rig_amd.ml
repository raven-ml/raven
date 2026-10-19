(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind

(* Sources: the Linux kernel's amdgpu driver for the PM4 and SDMA packet
   headers (nvd.h, soc15d.h, sdma_v*_0_pkt_open.h), its fences (gfx_v12_0.c,
   sdma_v7_0.c) and its flush of the host data path
   (amdgpu_hdp_generic_flush); ROCm's runtime for the AQL queue's descriptor
   (amd_queue.h) and its creation (amd_aql_queue.cpp). *)

module Abi = Rig_amd_abi
module Packet = Abi.Packet
module Aql = Abi.Aql
module Pm4 = Abi.Pm4
module Sdma = Abi.Sdma

exception Fault of string

external reraise : exn -> 'a = "%reraise"

(* Locks

   Each lock counts among the locks its thread holds (rig_hold in rig_edge.h),
   from before its take to after its give: a call of rig's or of this driver's
   from a signal handler, finaliser or Gc.Memprof callback inside it is refused
   as it starts, and never waits for the lock its own thread holds. Locks nest
   in this order: an image's, which [entry] holds while it grows the scratch; a
   device's growth lock, which [grow] holds; a device's guard; on the PCI path,
   rig.pci's driver lock and a hold's ([Rig_pci.Driver]), which an open, a
   reset and a stop hold; a path's; then rig.pci's memory locks. A device's
   HDP lock holds no other.

   An image's lock is held while the scratch grows, a device's guard while
   its scratch and traces change, and the PCI path's lock across register
   sequences that wait for the GPU to answer: each must see its state whole,
   and a sequence left half done leaves the GPU in an unknown state. A call
   of rig's from a callback during one raises. *)

module Lock = struct
  external holds : unit -> (int[@untagged])
    = "caml_rig_edge_holds_byte" "caml_rig_edge_holds"
  [@@noalloc]

  external hold : unit -> unit = "caml_rig_edge_hold" [@@noalloc]
  external unhold : unit -> unit = "caml_rig_edge_unhold" [@@noalloc]

  type t = Mutex.t

  let create = Mutex.create

  (* [take m] takes [m], counted from before the take, so that a signal handler
     that runs inside a contended take sees it held. An exception that ends the
     take gives the count back first, having taken nothing: one a handler
     raised, or the error-checking [Mutex]'s [Sys_error] for a take by the
     thread that holds [m], which only a call that skipped its check makes and
     which raises [Invalid_argument]. *)
  let take m =
    hold ();
    match Mutex.lock m with
    | () -> ()
    | exception e -> (
        unhold ();
        match e with
        | Sys_error why when String.starts_with ~prefix:"Mutex.lock:" why ->
            invalid_arg
              "Rig_amd: a lock taken again by the thread that holds it, from a \
               signal handler, finaliser or Gc.Memprof callback"
        | e -> reraise e)

  (* Gives [m] back, then its count, with nothing between that allocates or
     polls. *)
  let give m =
    Mutex.unlock m;
    unhold ()

  (* [protect m f] is [f ()] holding [m], taken as the last step before the
     handler and given back as the first step of both exits, where nothing
     allocates or polls, so an exception anywhere in [f], an asynchronous one
     too, leaves [m] free and the count right. *)
  let protect m f =
    take m;
    match f () with
    | x ->
        give m;
        x
    | exception e ->
        give m;
        reraise e

  (* Raises [Invalid_argument] naming [fn] while the thread holds a lock: the
     call comes from a signal handler, finaliser or Gc.Memprof callback inside
     that lock, and would wait for it or see its section half done. *)
  let check fn =
    if holds () <> 0 then
      invalid_arg
        (fn
       ^ ": called from a signal handler, finaliser or Gc.Memprof callback \
          inside a lock of the calling thread's")
end

type 'm memory = { address : int; host : int option; data : 'm }
type clock = Source of int | Round_trip

type refusal = Lacking of string | Full of string | Refused of string

type 'm path = {
  key : 'm Type.Id.t;
  index : int;
  gpu : Abi.Gpu.t;
  waves : int;
  lds : int;
  clock_hz : int;
  clock : clock;
  mec : int;
  wgps : int array array;
  budget : int;
  alloc : Rig_edge.memory -> int -> 'm memory option;
  map_host : (int -> int -> 'm memory option) option;
  reaches : int -> bool;
  map_peer : 'm memory -> 'm memory option;
  free : 'm memory -> unit;
  queue :
    [ `Pm4 | `Aql | `Sdma ] ->
    ring:int ->
    bytes:int ->
    read:int ->
    write:int ->
    (int, refusal) result;
  retire : int -> unit;
  most_ring : [ `Pm4 | `Aql | `Sdma ] -> int;
  hdp : int option;
  interrupt : int;
  hang_ms : int option;
  sleep : ms:int -> unit;
  stable_power : unit -> (unit, string) result;
  stop : fault:string option -> [ `Stopped | `Unknown ];
}

(* Whose a path's memory is, by the path's key, and the path's [free]. Memory
   holds its owner, so that a free after stop needs nothing of the device. *)
type 'm owner = { key : 'm Type.Id.t; free : 'm memory -> unit }

(* A path's memory, with its owner. *)
type mem = Mem : 'm owner * 'm memory -> mem

let mem_address (Mem (_, m)) = m.address
let mem_host (Mem (_, m)) = m.host
let free_mem (Mem (o, m)) = o.free m

(* The path's functions over memory whose type the owner hides. *)
type ops = {
  alloc : Rig_edge.memory -> int -> mem option;
  map_host : (int -> int -> mem option) option;
  map_peer : mem -> mem option;
  queue :
    [ `Pm4 | `Aql | `Sdma ] ->
    ring:int ->
    bytes:int ->
    read:int ->
    write:int ->
    (int, refusal) result;
  retire : int -> unit;
  most_ring : [ `Pm4 | `Aql | `Sdma ] -> int;
  sleep : ms:int -> unit;
  stable_power : unit -> (unit, string) result;
  stop : fault:string option -> [ `Stopped | `Unknown ];
}

let ops (type m) (p : m path) =
  let owner = { key = p.key; free = p.free } in
  let pack m = Mem (owner, m) in
  let own (Mem (o, m)) : m memory option =
    match Type.Id.provably_equal o.key p.key with
    | Some Type.Equal -> Some m
    | None -> None
  in
  {
    alloc = (fun k n -> Option.map pack (p.alloc k n));
    map_host = Option.map (fun f a n -> Option.map pack (f a n)) p.map_host;
    map_peer =
      (fun m -> Option.bind (own m) (fun m -> Option.map pack (p.map_peer m)));
    queue = p.queue;
    retire = p.retire;
    most_ring = p.most_ring;
    sleep = p.sleep;
    stable_power = p.stable_power;
    stop = p.stop;
  }

(* Memory *)

type region = {
  bytes : int;
  mem : mem;
  flush : int option;
      (* the device whose HDP register this region keeps flushed, by its
         [self] *)
}

let region ?flush n m = { bytes = n; mem = m; flush }

(* C state *)

external create : unit -> int = "caml_rig_amd_create"
external destroy : int -> unit = "caml_rig_amd_destroy" [@@noalloc]
external end_state : int -> unit = "caml_rig_amd_end" [@@noalloc]

external set_memory : int -> int -> int -> unit = "caml_rig_amd_memory"

external set_segment : int -> int -> int -> int -> unit = "caml_rig_amd_segment"

external set_ring :
  int -> int -> int -> int -> int -> int -> int -> int -> int -> unit
  = "caml_rig_amd_ring_byte" "caml_rig_amd_ring"

external set_descriptor : int -> int -> int -> unit = "caml_rig_amd_descriptor"

(* The code of a ring's kind in rig_amd_stubs.h. *)
let ring_kind = function `Pm4 -> 0 | `Aql -> 1 | `Sdma -> 2

external set_template : int -> int -> string -> string -> unit
  = "caml_rig_amd_template"

external set_copies : int -> int -> int -> unit = "caml_rig_amd_copies"

external set_stamps : int -> int -> int -> int -> int -> unit
  = "caml_rig_amd_stamps"

external set_clock : int -> int -> bool = "caml_rig_amd_clock"
external hdp_count : int -> int -> int -> bool = "caml_rig_amd_hdp"
[@@noalloc]

external set_hdp : int -> int -> unit = "caml_rig_amd_set_hdp" [@@noalloc]
external drop_hdp : int -> unit = "caml_rig_amd_drop" [@@noalloc]
external zero : int -> int -> unit = "caml_rig_amd_zero"
external poke32 : int -> int -> int -> unit = "caml_rig_amd_poke32"

external publish_scratch : int -> int -> int array -> int array -> int
  = "caml_rig_amd_scratch"

external scratch_taken : int -> int = "caml_rig_amd_scratch_taken" [@@noalloc]
external signaled_word : int -> int = "caml_rig_amd_signaled" [@@noalloc]
external renumber_device : int -> int -> unit = "caml_rig_amd_renumber"
external copy_word : int -> int -> unit = "caml_rig_amd_copy_word" [@@noalloc]

external c_launch_pm4 :
  string -> string -> int array -> int array -> int array -> int
  = "caml_rig_amd_launch_pm4"

external c_launch_aql :
  string -> string -> int array -> int array -> int array -> int
  = "caml_rig_amd_launch_aql"

external free_launch : int -> unit = "caml_rig_amd_launch_free" [@@noalloc]

external free_launches : int list -> unit = "caml_rig_amd_launches_free"
[@@noalloc]

external set_copy_launch : int -> int -> unit = "caml_rig_amd_copy_launch"
[@@noalloc]

external write_code : int -> string -> unit = "caml_rig_amd_write_code"
[@@noalloc]

(* Device state *)

(* What growing the scratch answered: [Short bytes] if the GPU has no memory for
   the [bytes] of scratch it needs. *)
type grown = Served | Stopped | Short of int

(* A replaced scratch buffer, freed once the word reaches [until], the value
   that installed its successor. *)
type retired = { buffer : mem; mutable until : int }

(* The device's scratch, which every kernel the compute queue runs takes, one at
   a time: installed, published for the next submission to take, and replaced,
   each freed once the word reaches the value that installed its successor. An
   AQL queue's descriptor names it; a launch's dispatch does on a PM4 queue.
   Each change stores values made first, with nothing between its stores that
   allocates or polls, so an exception leaves no buffer both installed and
   retired, or held by neither. *)
type scratch = {
  mutable installed : (mem * int) option;
      (* the buffer and its bytes per lane *)
  mutable pending : (mem * int) option;
  mutable retired : retired list;
}

(* What a growth made, or the device's first of its kind: a segment, or a
   queue, by its doorbell, with its ring and positions. It is named by the
   GPU address of the segment or the ring, as the C state's growth names it
   (rig_amd_stubs.h). *)
type part = { id : int; mems : mem list; mutable doorbell : int option }

(* What only a running device uses, under the device's guard but for
   [parts], under its growth lock. The handle outlives the device in its
   properties, so stop drops it. *)
type live = {
  ops : ops;
  own : mem list; (* the first queues' positions, private code *)
  compute : [ `Pm4 | `Aql ]; (* the kind of the compute queue *)
  mutable parts : part list;
      (* the segments and queues, until neither installed nor handed
         (rig_amd_ring.c's Growth) *)
  copy_launch : int; (* private launch until confirmed stop; 0 when absent *)
  scratch : scratch;
  mutable traces : (Abi.Capability.trace * mem list) option;
      (* the trace buffers, made at the first trace: the capability's record and
         the memory under it *)
}

(* A device: its facts, what [peer] uses for the life of the process, and its
   live state. *)
type t = {
  self : int;
  path : int; (* the path's key, as an integer *)
  index : int; (* the GPU's number in bus order *)
  gpu : Abi.Gpu.t;
  aql : bool; (* the compute queue reads AQL packets *)
  lds : int;
  waves : int;
  hdp : int option;
  budget : int;
  hang_ms : int option;
  waits64 : bool; (* its queues run 64-bit waits *)
  maps_host : bool;
  word : region; (* rig frees a word of the device's own after stop *)
  reaches : int -> bool; (* the path's *)
  capability : Abi.Capability.t;
  guard : Lock.t;
      (* held by [grow_scratch], [make_trace], and stop over [live] *)
  growth : Lock.t; (* held by [grow] *)
  mutable live : live option; (* [None] once stop began *)
}

(* The minimum version of a GC's compute firmware whose queues run 64-bit waits
   (Pm4.wait_64), by GC version: a version enters once a queue was seen
   comparing all 64 bits of a word the host moved across 2^32. GC 12.0.1: the
   R9700, MEC firmware 3010. *)
let wait64_from = [ ((12, 0, 1), 3010) ]
let ring_bytes = 16 lsl 20

(* The bytes of [p]'s first ring of a queue of kind [k]. *)
let first_ring (p : _ path) k = Int.min ring_bytes (p.most_ring k)

(* A path's refusal, as an open's error. *)
let why = function Lacking why | Full why | Refused why -> why
let segment_bytes = 2 lsl 20
let pointers_bytes = 4096

(* The stamp ring: RIG_AMD_STAMPS slots of four 8-byte words, then the record
   of a stamp of the measure at open. *)
let stamps_bytes = (4096 * 32) + 8

(* RIG_AMD_WAITS, the waits a submission's reserved room holds *)
external max_waits : unit -> int = "caml_rig_amd_max_waits" [@@noalloc]

(* The queues, as rig_amd_ring.c's [runs] accepts their parts. *)
let device_queues =
  Rig_edge.
    [
      { name = "COMPUTE:0"; runs = [ Words; Launch ] };
      { name = "COPY:0"; runs = [ Words; Copy ] };
    ]

(* Whether [g]'s compute queue reads AQL packets, which every die runs: on a GPU
   of several dies, else PM4 packets, which one die runs. *)
let reads_aql (g : Abi.Gpu.t) = g.xccs > 1

(* The queues' positions in the pointers: the compute queue's at the start,
   where an AQL queue's descriptor (amd_hsa_queue.h, amd_queue_t) holds them,
   the copy queue's after the descriptor. *)
let descriptor_bytes = 256
let read_at ~aql = function 0 when aql -> 128 | 0 -> 0 | _ -> descriptor_bytes
let write_at ~aql = function 0 when aql -> 56 | q -> read_at ~aql q + 8

(* The descriptor's fields an AQL queue's creator writes (amd_hsa_queue.h). *)
let max_cu_id = 72
let max_wave_id = 76
let read_dispatch_id_field_base_byte_offset = 136
let queue_properties = 180
let is_ptr64 = 1 lsl 1
let enable_profiling = 1 lsl 3

(* The descriptor's scratch fields. *)
let compute_tmpring_size = 140
let scratch_resource_descriptor = 144
let scratch_backing_memory_location = 160
let scratch_wave64_lane_byte_size = 176

(* Zeroes the [pointers_bytes] at [host], where a device's queues' positions
   lie, and where the compute queue reads AQL packets, writes its descriptor's
   fields its creator writes. *)
let init_pointers (gpu : Abi.Gpu.t) ~aql ~waves host =
  zero host pointers_bytes;
  if aql then begin
    let cus = gpu.compute_units * gpu.xccs in
    poke32 host queue_properties (is_ptr64 lor enable_profiling);
    poke32 host read_dispatch_id_field_base_byte_offset (read_at ~aql 0);
    poke32 host max_cu_id (cus - 1);
    poke32 host max_wave_id (waves - 1)
  end

(* Templates *)

(* Each packet the writer places, its values the arguments 0, 1 and 2 of a use,
   in rig_amd_stubs.h's order. On an AQL queue, whose PM4 words every die runs,
   a release writes once: on die 0, after the barrier of its packet, every die's
   work is done. *)
let templates (g : Abi.Gpu.t) ~interrupt ~waits64 ~aql =
  let once p = if aql then Pm4.pred_exec ~xcc_mask:1 p else p in
  [
    Pm4.wait g (Memory 0) Equal 1 ();
    Pm4.event_write Cs_partial_flush;
    Pm4.acquire_mem g System;
    (if waits64 then Pm4.wait_64 g 0 Greater_equal 1 () else []);
    once (Pm4.release_mem g System 0 (Low_32 1));
    once (Pm4.release_mem g System ~interrupt 0 (Data_64 1));
    Pm4.write_data (Memory 0) 1;
    Sdma.poll 0 Equal 1 ();
    Sdma.fence g 0 1;
    Sdma.trap;
    Sdma.copy_linear ~dst:0 ~src:1 ~bytes:2;
    Abi.Aql.indirect_buffer 0 ~dwords:1;
    Pm4.acquire_mem g Agent;
    Pm4.event_write Cs_partial_flush @ once (Pm4.copy_data Confirmed Clock 0);
    Sdma.timestamp 0;
    Sdma.const_fill ~dst:0 ~word:1 ~bytes:2;
  ]

let set_templates self g ~interrupt ~waits64 ~aql =
  let set i p =
    let words, holes = Template.flatten Packet (fun _ -> None) Fun.id p in
    set_template self i words holes
  in
  List.iteri set (templates g ~interrupt ~waits64 ~aql);
  set_copies self (Sdma.max_copy g) (Sdma.max_fill g)

let supported (g : Abi.Gpu.t) =
  let major, _, _ = g.target in
  if not (major = 11 || major = 12 || List.mem g.target [ (9, 4, 2); (9, 5, 0) ])
  then Error (strf "the device drives no %s GPU" (Abi.Gpu.processor g))
  else if Abi.Register.registers g = [] then
    Error (strf "no registers are known for the GC of %s" (Abi.Gpu.processor g))
  else Ok ()

let waits64 (p : _ path) =
  match List.assoc_opt p.gpu.gc wait64_from with
  | Some from -> p.mec >= from
  | None -> false

(* AQL scratch *)

(* The writes that point an AQL queue's descriptor to scratch [m] for kernels
   of [n] bytes per lane: offsets in the descriptor, all between
   [compute_tmpring_size] and [scratch_wave64_lane_byte_size] (rig_amd_stubs.h's
   RIG_AMD_DESC_SCRATCH), and 32-bit values. *)
let scratch_writes (g : Abi.Gpu.t) m n bytes =
  let base = mem_address m in
  let d = Abi.Scratch.descriptor g ~base bytes in
  let word i = Int32.to_int (String.get_int32_le d (4 * i)) land 0xffff_ffff in
  [
    (scratch_backing_memory_location, base land 0xffff_ffff);
    (scratch_backing_memory_location + 4, base lsr 32);
    (scratch_wave64_lane_byte_size, n);
    (compute_tmpring_size, Abi.Scratch.tmpring g n);
  ]
  @ List.init 4 (fun i -> (scratch_resource_descriptor + (4 * i), word i))

(* What installing the pending scratch stores, allocated ahead: the installed
   scratch retired, its value to be set, and the retired list it heads. *)
let retiring st =
  match st.installed with
  | Some (m, _) ->
      let r = { buffer = m; until = 0 } in
      (Some r, r :: st.retired)
  | None -> (None, st.retired)

(* Makes [p], the pending scratch the submission of [v] placed, installed: the
   installed one is freed once the word reaches [v]. *)
let install st (r, retired) p v =
  (match r with Some r -> r.until <- v | None -> ());
  st.retired <- retired;
  st.installed <- p;
  st.pending <- None

(* Frees the retired scratch the word passed, one at a time: each leaves the
   list before its free, so an exception leaves the rest for the next call. *)
let rec retire self st =
  let word = signaled_word self in
  match List.find_opt (fun r -> word >= r.until) st.retired with
  | None -> ()
  | Some r ->
      st.retired <- List.filter (fun r' -> r' != r) st.retired;
      free_mem r.buffer;
      retire self st

(* Installs a published scratch the queue took, and frees the retired ones the
   word passed. *)
let settle_scratch self st =
  (match st.pending with
  | Some _ as p ->
      let taken = scratch_taken self in
      if taken > 0 then install st (retiring st) p taken
  | None -> ());
  retire self st

(* [grow_scratch] under the device's guard, on a device not stopped. *)
let grow_held self (ops : ops) (g : Abi.Gpu.t) ~aql st n =
  settle_scratch self st;
  let have =
    match (st.pending, st.installed) with
    | Some (_, h), _ | None, Some (_, h) -> h
    | None, None -> 0
  in
  if have >= n then Served
  else
    let bytes = Abi.Scratch.size g n in
    match ops.alloc Device bytes with
    | None -> Short bytes
    | Some m -> (
        (* [m] is this call's until it is published: it is freed if what the
           publication stores raises as it is made, as for more scratch per die
           than a descriptor holds. The publication hands [m] to the
           submissions, so everything the outcome stores is made before it. *)
        match
          let writes = if aql then scratch_writes g m n bytes else [] in
          let ats, values = List.split writes in
          ( mem_address m,
            Array.of_list ats,
            Array.of_list values,
            Some (m, n),
            retiring st )
        with
        | exception e ->
            free_mem m;
            reraise e
        | base, ats, values, pending, ahead -> (
            let current = st.pending in
            let took = publish_scratch self base ats values in
            (* The publication this one replaces: placed by the submission of
               [took] since the settle above, or never placed and never in the
               queue's use, which is freed once [m] is recorded. *)
            match current with
            | Some _ when took > 0 ->
                install st ahead current took;
                st.pending <- pending;
                Served
            | Some (old, _) ->
                st.pending <- pending;
                free_mem old;
                Served
            | None ->
                st.pending <- pending;
                Served))

let grow_scratch g n =
  Lock.protect g.guard @@ fun () ->
  match g.live with
  | None -> Stopped
  | Some l -> grow_held g.self l.ops g.gpu ~aql:g.aql l.scratch n

(* Traces *)

(* The runs a device's trace buffers hold, and the bytes each shader engine
   traces into over all of them. *)
let trace_slots = 32
let trace_bytes = 256 lsl 20

(* The trace buffers, in GPU memory the host reads through the BAR where the
   path has some, else in host memory, and the end words in host memory. They
   are made under the device's guard, as the scratch grows, so a stop, which
   frees them, waits for them or refuses them. *)
let make_trace g =
  Lock.protect g.guard @@ fun () ->
  match g.live with
  | None -> Error "the device is stopped"
  | Some { traces = Some (t, _); _ } -> Ok t
  | Some ({ ops; traces = None; _ } as l) -> (
      let* () = ops.stable_power () in
      let engines = g.gpu.shader_engines * g.gpu.xccs in
      let window = trace_bytes / trace_slots in
      let n = window * trace_slots * engines in
      let buffers =
        match ops.alloc Mapped n with
        | Some m -> Some m
        | None -> ops.alloc Pinned n
      in
      match (buffers, ops.alloc Pinned (4 * trace_slots * engines)) with
      | Some b, Some e when mem_host b <> None && mem_host e <> None ->
          let t =
            {
              Abi.Capability.buffers = mem_address b;
              buffers_host = Option.get (mem_host b);
              window;
              slots = trace_slots;
              engines;
              ends = mem_address e;
              ends_host = Option.get (mem_host e);
            }
          in
          l.traces <- Some (t, [ b; e ]);
          Ok t
      | b, e ->
          Option.iter free_mem b;
          Option.iter free_mem e;
          Error (strf "no memory for %d bytes of trace buffers" n))

(* The capability's [scratch]: [g]'s scratch grown for [n] bytes per lane.
   Compiled code calls it and [trace] outside rig's calls, so each checks its
   thread's locks first. *)
let scratch g n =
  Lock.check "Rig_amd_abi.Capability.scratch";
  match grow_scratch g n with
  | Served -> Ok ()
  | Stopped -> Error "the device is stopped"
  | Short bytes -> Error (strf "no GPU memory for %d bytes of scratch" bytes)

let host_of what m =
  match mem_host m with
  | Some h -> Ok h
  | None -> Error (strf "the host does not address the %s" what)

let is_gpu ~vendor ~class_ =
  let base = class_ lsr 16 in
  vendor = 0x1002 && (base = 0x03 || base = 0x12)

(* Facts *)

let key = Type.Id.make ()

(* Both paths allocate whole pages of the GPU's 4 KiB, at addresses that start
   on one: KFD's at host addresses mmap reserves, the PCI path's in its GPU
   space ([Rig_pci.Memory.alloc]). *)
let region_alignment = 4096

let facts g =
  {
    Rig_edge.arch = Abi.Gpu.processor g.gpu;
    budget = g.budget;
    queues = device_queues;
    copy_engine = Some 1;
    completion = Store;
    waits = { stores = g.waits64; hosts = g.waits64; most = max_waits () };
    may_block = false;
    hang_ms = g.hang_ms;
    maps_host = g.maps_host;
    host_addresses = false;
    alignment = region_alignment;
    capability = Abi.declare g.capability;
    word = g.word;
    edge = Nativeint.of_int g.self;
  }

let capability g = g.capability

(* [g]'s live state. rig calls nothing that uses it after stop. *)
let live g =
  match g.live with
  | Some l -> l
  | None -> invalid_arg "Rig_amd: the device was stopped"

(* Memory *)

(* Counts a region that needs the device [src]'s HDP register flushed before a
   doorbell: [false] if the device keeps no more registers, or [src] has
   stopped. *)
let count_hdp g src delta = hdp_count g.self src delta

let alloc g (kind : Rig_edge.memory) n =
  if n < 1 then invalid_argf "Rig_amd.alloc: %d bytes, expected at least 1" n;
  let ops = (live g).ops in
  match kind with
  | Device | Pinned -> Option.map (region n) (ops.alloc kind n)
  | Mapped -> (
      match g.hdp with
      | None -> None
      | Some _ -> (
          match ops.alloc Mapped n with
          | None -> None
          | Some m ->
              ignore (count_hdp g g.self 1);
              Some (region ~flush:g.self n m)))

(* Gives back [m]: a free the path refuses loses the memory, which no caller can
   act on. *)
let give_back m = try free_mem m with Fault _ -> ()

(* The region goes before its count: an exception from the free leaves the
   count high, which costs flushes, where a count lowered first would let a
   doorbell skip the flush of a region still written through the BAR. *)
let free g r =
  give_back r.mem;
  Option.iter (fun src -> ignore (count_hdp g src (-1))) r.flush

(* The GPU names memory by address: the handle is the address. *)
let locate r =
  let a = mem_address r.mem in
  {
    Rig_edge.address = Some a;
    host = mem_host r.mem;
    handle = Nativeint.of_int a;
  }

let peer g g' = g.path = g'.path && g.reaches g'.index

let map_peer g _ r =
  match (live g).ops.map_peer r.mem with
  | None -> None
  | Some m -> (
      let view flush = Some (region ?flush r.bytes m) in
      match r.flush with
      | Some src when count_hdp g src 1 -> view (Some src)
      | Some _ ->
          free_mem m;
          None
      | None -> view None)

let map_host g a n =
  if n < 1 then invalid_argf "Rig_amd.map_host: %d bytes, expected at least 1" n;
  match (live g).ops.map_host with
  | None -> None
  | Some map -> Option.map (region n) (map a n)

(* Images *)

module Code_object = Abi.Code_object
module Names = Map.Make (String)

(* An image keeps the C launches its entries made, by function and as a list
   for the unload, until it is unloaded. Both change by stores of values made
   first, with nothing between the stores that allocates or polls, so an
   exception leaves them whole and in step for the unload. *)
type image = {
  co : Code_object.t;
  base : int; (* the address of the region it was laid over *)
  dev : t;
  lock : Lock.t;
  mutable launches : int Names.t; (* by function: its C launch *)
  mutable made : int list; (* the launches *)
}

let too_large g co =
  let large name =
    match Code_object.kernel co name with
    | Some k when k.group_segment > g.lds -> Some (name, k.group_segment)
    | _ -> None
  in
  List.find_map large (Code_object.kernels co)

let image g bin =
  let* co = Code_object.of_string bin in
  if not (Code_object.runs_on co g.gpu) then
    Error
      (strf "a code object for %s; the GPU is %s" (Code_object.target co)
         (Abi.Gpu.processor g.gpu))
  else
    match too_large g co with
    | Some (name, n) ->
        Error
          (strf "kernel %s takes %d bytes of local data share; the GPU has %d"
             name n g.lds)
    | None ->
        let lay r =
          let base = mem_address r.mem in
          ( {
              co;
              base;
              dev = g;
              lock = Lock.create ();
              launches = Names.empty;
              made = [];
            },
            Code_object.image co )
        in
        Ok (Rig_edge.Place (Code_object.size co, lay))

(* A dispatch's values: known to its launch, or the hand-over's argument [i], in
   the order of rig_amd_stubs.h's L_ARGS, L_SCRATCH, L_THREADS, L_GROUPS and
   L_GRID. *)
type value = Known of int | Arg of int

let word s i = Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff

(* The offsets of the implicit arguments [k] reads, in the order of
   rig_amd_stubs.h's H_BLOCKS, H_GROUP, H_REMAINDER, H_OFFSET, H_DIMS and H_LDS,
   or -1. *)
let hidden f (k : Code_object.kernel) =
  let at = Array.make 14 (-1) in
  let axis : Code_object.axis -> int = function X -> 0 | Y -> 1 | Z -> 2 in
  let set (h : Code_object.hidden) off =
    match h with
    | Block_count a -> at.(axis a) <- off
    | Group_size a -> at.(3 + axis a) <- off
    | Remainder a -> at.(6 + axis a) <- off
    | Global_offset a -> at.(9 + axis a) <- off
    | Grid_dims -> at.(12) <- off
    | Dynamic_lds_size -> at.(13) <- off
    | Other kind ->
        invalid_argf
          "Rig_amd.entry: kernel %s reads the implicit argument %s, which a \
           launch does not write"
          f kind
  in
  List.iter (fun (h, off) -> set h off) k.hidden;
  at

(* The values of [k]'s launch the C launch keeps beside its words: the LDS [k]
   takes itself, its most work-items and shared memory on a GPU whose groups
   take at most [lds] bytes, its arguments' bytes, and its explicit ones'. *)
let limits (k : Code_object.kernel) ~lds =
  [|
    k.group_segment;
    k.max_threads;
    lds - k.group_segment;
    k.kernarg_size;
    k.explicit_size;
  |]

let template p =
  Template.flatten Launch
    (function Known n -> Some (Int64.of_int n) | Arg _ -> None)
    (function Arg i -> i | Known _ -> assert false (* known: no hole *))
    p

(* The C launch of kernel [k], named [f], of an image at [base], on [g], whose
   groups take at most [lds] bytes of LDS.

   On a PM4 queue, its dispatch, whose word that holds the LDS its groups take,
   COMPUTE_PGM_RSRC2, the hand-over sets from the launch's shared memory. It is
   the one word that a group segment one granule larger changes.

   On an AQL queue, its kernel dispatch packet, whose workgroup sizes and group
   segment size the hand-over sets at their offsets, and which names the
   kernel's descriptor: the queue hands it its scratch and the packet's
   address. *)
let c_launch_of g (k : Code_object.kernel) ~base ~lds f =
  if reads_aql g then
    let words, holes =
      template
        (Aql.dispatch
           { k with group_segment = 0 }
           ~descriptor:(Known (base + k.descriptor))
           ~args:(Arg 0) ~threads:(1, 1, 1) ~grid:(Arg 8, Arg 9, Arg 10))
    in
    c_launch_aql words holes
      Aql.
        [|
          offset (Workgroup_size X);
          offset (Workgroup_size Y);
          offset (Workgroup_size Z);
          offset Group_segment_size;
        |]
      (limits k ~lds) (hidden f k)
  else
    let dispatch group =
      template
        (Pm4.dispatch g
           { k with group_segment = group }
           ~program:(Known (base + k.entry))
           ~scratch:(Arg 1) ~args:(Arg 0) ~packet:(Known 0)
           ~threads:(Arg 2, Arg 3, Arg 4) ~groups:(Arg 5, Arg 6, Arg 7) ())
    in
    let granule = Pm4.lds_granule g in
    let words, holes = dispatch 0 and more, _ = dispatch granule in
    let at =
      match
        List.filter
          (fun i -> word words i <> word more i)
          (List.init (String.length words / 4) Fun.id)
      with
      | [ at ] -> at
      | _ -> invalid_argf "Rig_amd.entry: kernel %s's LDS is not one word" f
    in
    c_launch_pm4 words holes
      [| at; word words at; word more at - word words at; granule |]
      (limits k ~lds) (hidden f k)

(* [m]'s launch of its kernel [k], named [f], and the scratch of its private
   segment, which grows only for a kernel the launch takes: [Error bytes] if the
   GPU has no memory for the [bytes] of scratch, having made nothing. A PM4
   dispatch hands a kernel no dispatch packet. The launch is this call's from
   [c_launch_of] on until [m] holds it: an exception before, such as an
   asynchronous one where the growth allocates, frees it. *)
let launch m f (k : Code_object.kernel) =
  let g = m.dev in
  if k.dispatch_ptr && not g.aql then
    invalid_argf
      "Rig_amd.entry: kernel %s reads its dispatch packet, which a launch on a \
       GPU of one die does not write"
      f;
  let l = c_launch_of g.gpu k ~base:m.base ~lds:g.lds f in
  match
    let grown =
      if k.private_segment > 0 then grow_scratch g k.private_segment
      else Served
    in
    match grown with
    | Served ->
        let made = Ok l in
        let launches = Names.add f l m.launches and kept = l :: m.made in
        m.launches <- launches;
        m.made <- kept;
        made
    | Short bytes -> Error bytes
    | Stopped -> raise (Fault "the device is stopped")
  with
  | Ok _ as made -> made
  | Error _ as short ->
      free_launch l;
      short
  | exception e ->
      free_launch l;
      reraise e

let entry m f =
  match Code_object.kernel m.co f with
  | None -> Rig_edge.Absent
  | Some k -> (
      let made =
        Lock.protect m.lock @@ fun () ->
        match Names.find_opt f m.launches with
        | Some l -> Ok l
        | None -> launch m f k
      in
      match made with
      | Ok l ->
          Entry
            {
              code = m.base + k.descriptor;
              launch = Nativeint.of_int l;
              params = Some k.explicit_size;
              threads = Some k.max_threads;
            }
      | Error bytes -> No_memory bytes)

(* The image's code is in a region rig frees; the launches [entry] made go
   here. rig unloads an image once, so past its take the unload allocates
   nothing and polls nowhere: an exception cannot end it with launches held.
   One that ends the take ends it before it changed anything. *)
let unload _ m =
  Lock.take m.lock;
  let made = m.made in
  m.launches <- Names.empty;
  m.made <- [];
  free_launches made;
  Lock.give m.lock

(* Opening *)

(* rig's array of timeline words: its address and its bytes. *)
external words : unit -> int * int = "caml_rig_edge_words"

(* [m] from [at] bytes on, which lie at the host address [host]. *)
let within (Mem (o, m)) at host =
  Mem (o, { m with address = m.address + at; host = Some host })

(* Makes the device's memory and queues, giving back what it took if one of them
   is refused. *)
let make (type m) ~word (p : m path) =
  if p.interrupt = 0 then
    invalid_arg "Rig_amd.make: the release's interrupt context is 0";
  if p.clock_hz < 1 then
    invalid_argf "Rig_amd.make: the clock's rate is %d Hz, expected at least 1"
      p.clock_hz;
  let count field n =
    if n < 1 then
      invalid_argf "Rig_amd.make: the GPU's %s is %d, expected at least 1"
        field n
  in
  count "xccs" p.gpu.xccs;
  count "shader_engines" p.gpu.shader_engines;
  count "compute_units" p.gpu.compute_units;
  count "scratch_slots" p.gpu.scratch_slots;
  let* () = supported p.gpu in
  let ops = ops p in
  let taken = ref [] and queues = ref false and copy_launch = ref 0 in
  (* Memory a queue the path could not stop may still read stays. *)
  let undo () =
    let stopped =
      (not !queues)
      ||
      match ops.stop ~fault:None with
      | `Stopped -> true
      | `Unknown -> false
      | exception Fault _ -> false
    in
    if stopped then begin
      if !copy_launch <> 0 then free_launch !copy_launch;
      List.iter give_back !taken
    end
  in
  let alloc what n =
    match ops.alloc Pinned n with
    | Some m ->
        taken := m :: !taken;
        Ok m
    | None -> Error (strf "no memory for its %s" what)
  in
  let waits64 = waits64 p and aql = reads_aql p.gpu in
  let open_device () =
    (* The word rig gave, in the device's mapping of rig's whole array of
       words: the path's first map of the array on the GPU makes it, and the
       others count against it. Where the path maps no host memory, or the
       word a caller gave lies outside the array, a word of the device's
       own, started at the given word's value, which rig frees once no
       queue waits on it. *)
    let word = Nativeint.to_int word in
    let start, bytes = words () in
    let in_array = word >= start && word + 8 <= start + bytes in
    let mapped =
      match ops.map_host with
      | None -> None
      | Some _ when not in_array -> None
      | Some map -> (
          match map start bytes with
          | Some m ->
              taken := m :: !taken;
              Some m
          | None -> None)
    in
    let* word =
      match mapped with
      | Some m -> Ok (within m (word - start) word)
      | None ->
          let* own = alloc "timeline word" 8 in
          let* host = host_of "timeline word" own in
          copy_word host word;
          Ok own
    in
    let* segment = alloc "argument segment" segment_bytes in
    let compute_kind = if aql then `Aql else `Pm4 in
    let* compute = alloc "compute ring" (first_ring p compute_kind) in
    let* copy = alloc "copy ring" (first_ring p `Sdma) in
    let* pointers = alloc "queue positions" pointers_bytes in
    let* stamps = alloc "stamp ring" stamps_bytes in
    let* stamps_host = host_of "stamp ring" stamps in
    zero stamps_host stamps_bytes;
    let* word_host = host_of "timeline word" word in
    let* segment_host = host_of "argument segment" segment in
    let* pointers_host = host_of "queue positions" pointers in
    let* compute_host = host_of "compute ring" compute in
    let* copy_host = host_of "copy ring" copy in
    init_pointers p.gpu ~aql ~waves:p.waves pointers_host;
    (* The kernel the device copies with, where Copy_kernel holds one for the
       GPU's processor. rig_amd_ring.c runs a copy of 2 MiB or more within this
       device's Device memory with it, on the compute engine, and every other
       copy on SDMA, whose dispatch costs less. *)
    let* copy_code =
      let processor = Abi.Gpu.processor p.gpu in
      match Copy_kernel.find processor with
      | None -> Ok []
      | Some bin ->
          let name = "copy_bytes" in
          let refused why =
            strf "the copy kernel %s for %s does not load: %s" name processor
              why
          in
          let* co = Result.map_error refused (Code_object.of_string bin) in
          let k = Option.get (Code_object.kernel co name) in
          let* code = alloc "private copy code" (Code_object.size co) in
          let* host = host_of "private copy code" code in
          write_code host (Code_object.image co);
          copy_launch :=
            c_launch_of p.gpu k ~base:(mem_address code) ~lds:p.lds name;
          Ok [ code ]
    in
    let queue q kind ring =
      let at = mem_address pointers in
      let* doorbell =
        Result.map_error why
          (p.queue kind ~ring:(mem_address ring) ~bytes:(first_ring p kind)
             ~read:(at + read_at ~aql q)
             ~write:(at + write_at ~aql q))
      in
      queues := true;
      Ok doorbell
    in
    let* compute_bell = queue 0 compute_kind compute in
    let* copy_bell = queue 1 `Sdma copy in
    (* The queues read none of the C state before a doorbell. *)
    let self = create () in
    set_memory self word_host (mem_address word);
    set_segment self segment_host (mem_address segment) segment_bytes;
    (* The GPU's own register takes the first of the empty table's slots, with
       no region counted, so that its own [`Mapped] memory always finds one and
       views of other GPUs' take at most the rest. *)
    Option.iter (set_hdp self) p.hdp;
    set_templates self p.gpu ~interrupt:p.interrupt ~waits64 ~aql;
    set_copy_launch self !copy_launch;
    let ring q kind host bell id =
      set_ring self q host (first_ring p kind) (p.most_ring kind)
        (pointers_host + write_at ~aql q)
        bell (ring_kind kind) id
    in
    ring 0 compute_kind compute_host compute_bell (mem_address compute);
    ring 1 `Sdma copy_host copy_bell (mem_address copy);
    if aql then set_descriptor self pointers_host (mem_address pointers);
    (* The values start after the word's: the device numbers on from it as
       an idle one would. *)
    renumber_device self (signaled_word self + 1);
    set_stamps self stamps_host (mem_address stamps) stamps_bytes p.clock_hz;
    let source = match p.clock with Source c -> c | Round_trip -> 0 in
    let* () =
      if set_clock self source then Ok ()
      else begin
        destroy self;
        Error "the GPU's clock could not be read against the host's"
      end
    in
    let rec g =
      {
        self;
        path = Type.Id.uid p.key;
        index = p.index;
        gpu = p.gpu;
        aql;
        lds = p.lds;
        waves = p.waves;
        hdp = p.hdp;
        budget = p.budget;
        hang_ms = p.hang_ms;
        waits64;
        maps_host = Option.is_some ops.map_host;
        word = region 8 word;
        reaches = p.reaches;
        capability =
          {
            gpu = p.gpu;
            clock_hz = p.clock_hz;
            compute =
              (if aql then Aql { scratch = (fun n -> scratch g n) } else Pm4);
            wgps = p.wgps;
            trace =
              (fun () ->
                Lock.check "Rig_amd_abi.Capability.trace";
                make_trace g);
          };
        guard = Lock.create ();
        growth = Lock.create ();
        live =
          Some
            {
              ops;
              own =
                copy_code
                @ [ pointers; stamps ] @ Option.to_list mapped;
              compute = compute_kind;
              parts =
                [
                  {
                    id = mem_address segment;
                    mems = [ segment ];
                    doorbell = None;
                  };
                  {
                    id = mem_address compute;
                    mems = [ compute ];
                    doorbell = Some compute_bell;
                  };
                  {
                    id = mem_address copy;
                    mems = [ copy ];
                    doorbell = Some copy_bell;
                  };
                ];
              copy_launch = !copy_launch;
              scratch = { installed = None; pending = None; retired = [] };
              traces = None;
            };
      }
    in
    Ok g
  in
  match open_device () with
  | Ok _ as g -> g
  | Error _ as e ->
      undo ();
      e
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      undo ();
      Printexc.raise_with_backtrace e bt

(* Work *)

external last : int -> int = "caml_rig_amd_last" [@@noalloc]

(* Timeline *)

external raise_word : int -> unit = "caml_rig_amd_raise"

let signaled g = signaled_word g.self

let sleep g ~seen ~still_ms =
  if signaled g = seen then (live g).ops.sleep ~ms:still_ms

(* Growth

   A growth makes a larger argument segment, or new queues of the path's on
   larger rings, each with positions of its own, and hands them to the C state,
   whose next room that needs them installs them at idle (rig_amd_ring.c's
   Growth). *)

external needed : int -> int * int * int = "caml_rig_amd_needed"

external hand : int -> int array -> int array -> int array -> bool
  = "caml_rig_amd_hand"

external growth_state : int -> int * int * int * int * int * int
  = "caml_rig_amd_growth"

external unhand : int -> int -> unit = "caml_rig_amd_unhand"

(* The bytes of memory a growth makes for [want] bytes: the first power of two
   from a page that holds twice them, so that growths come at doublings, and at
   most [most], a power of two of at least [want]. *)
let grown_bytes ?(most = max_int) want =
  let rec up n = if n >= 2 * want || n >= most then n else up (2 * n) in
  up 4096

(* The kinds of [l]'s queues, in the order of its rings. *)
let kinds l = [| (l.compute :> [ `Pm4 | `Aql | `Sdma ]); `Sdma |]

(* Destroys the queues and gives back the memory growths made, or the device
   opened with, that is neither installed nor handed: a replaced segment or
   queue, which no work reads as the device was idle when it was replaced and
   a queue read every packet up to its write position then, and what a growth
   handed that a later one replaced. Each leaves [l.parts] before it goes, so
   an exception leaves the rest for the next call. *)
let rec retire ?(keep = []) g l =
  let s, c, k, s', c', k' = growth_state g.self in
  let held = keep @ [ s; c; k; s'; c'; k' ] in
  match List.find_opt (fun p -> not (List.mem p.id held)) l.parts with
  | None -> ()
  | Some p ->
      l.parts <- List.filter (fun p' -> p' != p) l.parts;
      Option.iter l.ops.retire p.doorbell;
      List.iter give_back p.mems;
      retire ~keep g l

(* New [Pinned] memory for a growth, of each of [sizes] bytes, listed in
   [l.parts] as one part named by the first before anything else may raise,
   with its host addresses, or [Error n] for [n] bytes the path lacked, having
   given back what it took. *)
let spare l sizes =
  let rec take acc = function
    | [] -> Ok (List.rev acc)
    | n :: rest -> (
        match l.ops.alloc Pinned n with
        | Some m -> take (m :: acc) rest
        | None ->
            List.iter give_back acc;
            Error n)
  in
  let* ms = take [] sizes in
  let p = { id = mem_address (List.hd ms); mems = ms; doorbell = None } in
  l.parts <- p :: l.parts;
  let host m =
    match mem_host m with
    | Some h -> h
    | None -> raise (Fault "the host does not address the memory a growth made")
  in
  Ok (p, List.map host ms)

(* What a growth makes for queue [q] of kind [kind]: a ring of [bytes] bytes
   and positions of its own, at their host addresses, the positions' GPU
   address, and once [queue] made it, the queue's doorbell. *)
type made = {
  q : int;
  kind : [ `Pm4 | `Aql | `Sdma ];
  part : part;
  bytes : int;
  words : int;
  host : int;
  gpu : int;
}

(* The memory of a queue of kind [kind] on a ring of [bytes] bytes, its
   positions zeroed, or [Error bytes] for the bytes the path lacked. *)
let queue_memory (g : t) l q kind bytes =
  let* part, hosts = spare l [ bytes; pointers_bytes ] in
  let words = List.nth hosts 0 and host = List.nth hosts 1 in
  let gpu = mem_address (List.nth part.mems 1) in
  init_pointers g.gpu ~aql:(kind = `Aql) ~waves:g.waves host;
  Ok { q; kind; part; bytes; words; host; gpu }

(* Makes the path's queue on [m], or is the path's refusal [Lacking] or
   [Full]. Raises [Fault] for another refusal. *)
let make_queue l m =
  let aql = m.kind = `Aql and at = if m.kind = `Sdma then 1 else 0 in
  match
    l.ops.queue m.kind ~ring:m.part.id ~bytes:m.bytes
      ~read:(m.gpu + read_at ~aql at)
      ~write:(m.gpu + write_at ~aql at)
  with
  | Ok bell ->
      m.part.doorbell <- Some bell;
      Ok ()
  | Error (Lacking _ | Full _) as e -> e
  | Error (Refused why) -> raise (Fault why)

(* A growth replaces what one handed and no room installed rather than add to
   it. Everything that can run short comes before the take-back, so that a
   growth short of anything leaves what one handed alone:
   1. its memory;
   2. its queues, but those the path answers [Full] for, as when the
      installed queue and a handed one hold the kind's slots;
   3. the take-back of the queues it replaces, which ran no work, and their
      destruction, as of a queue an install replaced;
   4. the queues answered [Full], in the slots step 3 freed, for which the
      path needs no memory ([queue] of [path]), so any refusal is a [Fault].
   A path that answers [Full] then holds at most two queues of a kind: the
   installed one and one other. *)
let grow g =
  Lock.protect g.growth @@ fun () ->
  let l = live g in
  retire g l;
  let segment, compute, copy = needed g.self in
  if segment = 0 && compute = 0 && copy = 0 then Rig_edge.Grown
  else
    let memory =
      let* s =
        if segment = 0 then Ok None
        else
          let n = grown_bytes segment in
          Result.map
            (fun (p, hosts) -> Some (p, List.hd hosts, n))
            (spare l [ n ])
      in
      let ring q want =
        if want = 0 then Ok None
        else
          let kind = (kinds l).(q) in
          let n = grown_bytes ~most:(l.ops.most_ring kind) want in
          Result.map Option.some (queue_memory g l q kind n)
      in
      let* c = ring 0 compute in
      let* k = ring 1 copy in
      Ok (s, List.filter_map Fun.id [ c; k ])
    in
    match memory with
    | Error n ->
        retire g l;
        Short n
    | Ok (s, queues) -> (
        let rec make full = function
          | [] -> Ok full
          | m :: rest -> (
              match make_queue l m with
              | Ok () -> make full rest
              | Error (Full _) -> make (m :: full) rest
              | Error _ -> Error m.bytes)
        in
        match make [] queues with
        | Error n ->
            retire g l;
            Short n
        | Ok full ->
            List.iter (fun m -> unhand g.self m.q) queues;
            let keep =
              List.map (fun m -> m.part.id) queues
              @ Option.to_list (Option.map (fun (p, _, _) -> p.id) s)
            in
            retire ~keep g l;
            let again m =
              match make_queue l m with
              | Ok () -> ()
              | Error r -> raise (Fault (why r))
            in
            List.iter again full;
            let segment =
              match s with
              | Some (p, host, n) -> [| host; p.id; n |]
              | None -> [| 0; 0; 0 |]
            in
            let ring q =
              match List.find_opt (fun m -> m.q = q) queues with
              | Some m ->
                  let at = if m.kind = `Sdma then 1 else 0 in
                  [|
                    m.words;
                    m.bytes;
                    m.host + write_at ~aql:(m.kind = `Aql) at;
                    Option.get m.part.doorbell;
                    m.part.id;
                  |]
              | None -> [| 0; 0; 0; 0; 0 |]
            in
            let desc =
              match List.find_opt (fun m -> m.kind = `Aql) queues with
              | Some m -> [| m.host; m.gpu |]
              | None -> [| 0; 0 |]
            in
            if not (hand g.self segment (Array.append (ring 0) (ring 1)) desc)
            then raise (Fault "no host memory for a growth's record");
            Grown)

(* Loss *)

(* A queue the path could not destroy may still run, so its memory stays and its
   own releases raise the word. *)
let halt g l ~fault =
  (* No work on another device uses this GPU's memory once it is lost
     ([Rig]'s loss section), so their submissions stop storing to its HDP
     register before the path may unmap it. *)
  drop_hdp g.self;
  match l.ops.stop ~fault with
  | exception Fault _ -> ()
  | `Unknown -> ()
  | `Stopped ->
      raise_word g.self;
      if l.copy_launch <> 0 then free_launch l.copy_launch;
      let st = l.scratch in
      let buffers = Option.to_list st.installed @ Option.to_list st.pending in
      let traces = match l.traces with Some (_, ms) -> ms | None -> [] in
      List.iter give_back
        (List.map fst buffers
        @ List.map (fun r -> r.buffer) st.retired
        @ traces @ l.own
        @ List.concat_map (fun p -> p.mems) l.parts)

(* The C state's run part goes whatever the path's stop answered: the queues
   read none of it. *)
let stop g ~fault =
  let l =
    Lock.protect g.guard @@ fun () ->
    let l = g.live in
    g.live <- None;
    l
  in
  Option.iter (fun l -> halt g l ~fault) l;
  end_state g.self

(* Tests *)

(* The hooks read the run state, which the stop frees once it took [live]
   under the guard: they hold the guard over their read. *)
let refuse_stopped g what =
  if Option.is_none g.live then
    invalid_argf "Rig_amd.%s: the device stopped" what

let renumber g v =
  Lock.check "Rig_amd.renumber";
  Lock.protect g.guard @@ fun () ->
  refuse_stopped g "renumber";
  let last = last g.self in
  if signaled g <> last then
    invalid_arg "Rig_amd.renumber: the device's work runs";
  if v - 1 < last then
    invalid_argf "Rig_amd.renumber: value %d, expected at least %d" v (last + 1);
  renumber_device g.self v

external placed : int -> int -> int * int = "caml_rig_amd_placed"

let ring g q =
  Lock.check "Rig_amd.ring";
  let named (d : Rig_edge.queue) = d.name = q in
  match List.find_index named device_queues with
  | Some i ->
      Lock.protect g.guard @@ fun () ->
      refuse_stopped g "ring";
      placed g.self i
  | None -> invalid_argf "Rig_amd.ring: no queue %S" q
